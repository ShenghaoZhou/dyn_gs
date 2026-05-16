import numpy as np
import cv2
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from gs_dyn_obj.grouped_gs import GaussianSuperPrimitive
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.transforms.so3 import so3_exp_map
from gsplat import rasterization_2dgs
import matplotlib.pyplot as plt
import tyro
from dataclasses import dataclass
import io
from PIL import Image
from gs_dyn_obj.utils.init import d2n_tblr, unproject_depth
from scipy.spatial.transform import Rotation as R
import theseus as th

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    target_frame: int = 40
    near_plane: float = 0.01
    far_plane: float = 10.0
    device: str = "cuda"
    opt_lr: float = 1e-3
    opt_steps: int = 200
    warp_type: str = "forward" # forward or backward
    
    # MOGE options
    use_moge_depth: bool = False
    
    # Solver options
    use_lm: bool = False
    # LM specific
    damping: float = 0.1
    res_scale: int = 8 # Downsample for Jacobian memory

def depth_to_rgb(depth, min_val=0.0, max_val=2.0):
    """Consistent colormapping for depth visualization."""
    depth_norm = np.clip((depth - min_val) / (max_val - min_val), 0, 1)
    colormap = plt.get_cmap("magma")
    rgb = (colormap(depth_norm)[..., :3] * 255).astype(np.uint8)
    return rgb

def gs_to_planar_params(gs_params, T_CO):
    """
    Convert GS parameters to planar surface parameters (normal and distance) in camera view.
    """
    means = gs_params.means # [N, 3]
    quats = gs_params.quats # [N, 4]
    R_OG = quaternion_to_matrix(quats) # [N, 3, 3]
    n_O = R_OG[:, :, 2] # [N, 3]
    R_CO = T_CO[:3, :3]
    t_CO = T_CO[:3, 3]
    n_C = torch.einsum('ij,nj->ni', R_CO, n_O) # [N, 3]
    p_C = torch.einsum('ij,nj->ni', R_CO, means) + t_CO # [N, 3]
    d = -torch.sum(n_C * p_C, dim=1, keepdim=True) # [N, 1]
    return n_C, d

def render_custom_attribute(gs_params, attr, T_CW, K, width, height, near_plane=0.01, far_plane=100.0):
    """
    Render a custom attribute using gsplat rasterizer.
    """
    viewmats = T_CW.unsqueeze(0).unsqueeze(0).contiguous()
    Ks = K.unsqueeze(0).unsqueeze(0).contiguous()
    means = gs_params.means.unsqueeze(0).contiguous()
    quats = gs_params.quats.unsqueeze(0).contiguous()
    
    if gs_params.scales.shape[-1] == 2:
        scales = torch.cat([gs_params.scales, torch.zeros_like(gs_params.scales[..., :1])], dim=-1).unsqueeze(0).contiguous()
    else:
        scales = gs_params.scales.unsqueeze(0).contiguous()
        
    opacities = gs_params.opacity.squeeze(-1).unsqueeze(0).contiguous()
    colors = attr.unsqueeze(0).unsqueeze(0).contiguous()
    
    render_colors, render_alphas, _, _, _, _, _ = rasterization_2dgs(
        means, quats, scales, opacities, colors,
        viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane,
        render_mode="RGB"
    )
    return render_colors[0, 0].permute(2, 0, 1), render_alphas[0, 0].permute(2, 0, 1)

def manual_forward_homography_warp(image_ref_t, n_C, d, T_curr_ref, K, mask=None):
    """
    Explicitly compute the forward warp of each pixel using the plane-induced homography formula.
    image_ref_t: [C, H, W] torch tensor
    n_C: [N, 3] normals in ref camera view
    d: [N, 1] distances in ref camera view
    """
    C, H, W = image_ref_t.shape
    device = n_C.device
    
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )
    x0 = grid_x.reshape(-1)
    y0 = grid_y.reshape(-1)
    
    if mask is not None:
        m = mask.reshape(-1)
        x0 = x0[m]
        y0 = y0[m]
        colors = image_ref_t.reshape(C, -1).t()[m]
        n_C = n_C[m] if n_C.shape[0] > 1 else n_C
        d = d[m] if d.shape[0] > 1 else d
    else:
        colors = image_ref_t.reshape(C, -1).t()

    u0 = torch.stack([x0, y0, torch.ones_like(x0)], dim=1).float() # [N, 3] (homogeneous)
    
    R_curr_ref = T_curr_ref[:3, :3]
    t_curr_ref = T_curr_ref[:3, 3]
    
    K_inv = torch.inverse(K)
    dir0 = (K_inv @ u0.t()).t() # [N, 3]
    
    n_dot_dir = torch.sum(n_C * dir0, dim=1, keepdim=True)
    d_safe = torch.where(torch.abs(d) > 1e-6, d, torch.ones_like(d))
    p1_scaled = torch.matmul(R_curr_ref, dir0.t()).t() - (n_dot_dir / d_safe) * t_curr_ref.view(1, 3)
    
    u1_homog = torch.matmul(K, p1_scaled.t()).t()
    u1_pix = u1_homog[:, :2] / (u1_homog[:, 2:3] + 1e-8)
    z1_scaled = u1_homog[:, 2]
    
    valid = (u1_pix[:, 0] >= 0) & (u1_pix[:, 0] < W-1) & \
            (u1_pix[:, 1] >= 0) & (u1_pix[:, 1] < H-1) & \
            (z1_scaled > 1e-3)
    
    x = u1_pix[:, 0]
    y = u1_pix[:, 1]
    
    x0_idx = torch.floor(x).long()
    x1_idx = x0_idx + 1
    y0_idx = torch.floor(y).long()
    y1_idx = y0_idx + 1
    
    wa = (x1_idx.float() - x) * (y1_idx.float() - y)
    wb = (x - x0_idx.float()) * (y1_idx.float() - y)
    wc = (x1_idx.float() - x) * (y - y0_idx.float())
    wd = (x - x0_idx.float()) * (y - y0_idx.float())
    
    v_f = valid.float()
    wa = wa * v_f
    wb = wb * v_f
    wc = wc * v_f
    wd = wd * v_f
    
    warped_flat = torch.zeros((H * W, C), device=device)
    
    # Use index_add for potential vmap compatibility if needed by LM
    for x_i, y_i, w_i in zip([x0_idx, x1_idx, x0_idx, x1_idx], 
                             [y0_idx, y0_idx, y1_idx, y1_idx], 
                             [wa, wb, wc, wd]):
        xi = torch.clamp(x_i, 0, W-1)
        yi = torch.clamp(y_i, 0, H-1)
        idx = yi * W + xi
        warped_flat = warped_flat.index_add(0, idx, colors * w_i.unsqueeze(-1))

    return warped_flat.reshape(H, W, C).permute(2, 0, 1)

def manual_backward_homography_warp(image_ref_t, n_curr, d_curr, T_ref_curr, K):
    """
    Perform backward homography warping using rendered normal and distance maps.
    image_ref_t: [C, H_ref, W_ref] torch tensor (0-1)
    """
    H, W = n_curr.shape[1], n_curr.shape[2]
    device = n_curr.device
    
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )
    u1 = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).float().reshape(-1, 3) # [HW, 3]
    
    R_inv = T_ref_curr[:3, :3]
    t_inv = T_ref_curr[:3, 3]
    
    K_inv = torch.inverse(K)
    dir1 = (K_inv @ u1.t()).t() # [HW, 3]
    
    n1 = n_curr.permute(1, 2, 0).reshape(-1, 3) # [HW, 3]
    d1 = d_curr.reshape(-1, 1) # [HW, 1]
    
    d1_safe = torch.where(torch.abs(d1) > 1e-6, d1, torch.ones_like(d1))
    n1_dot_dir1 = torch.sum(n1 * dir1, dim=1, keepdim=True)
    p0_scaled = torch.matmul(R_inv, dir1.t()).t() - (n1_dot_dir1 / d1_safe) * t_inv.view(1, 3)
    
    u0_homog = torch.matmul(K, p0_scaled.t()).t()
    u0_pix = u0_homog[:, :2] / (u0_homog[:, 2:3] + 1e-8)
    
    grid = u0_pix.reshape(1, H, W, 2)
    grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
    
    warped = F.grid_sample(image_ref_t.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped[0]

def load_object_pose_world(data_root, frame_idx):
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists():
        return None
    with open(poses_file, "r") as f:
        lines = f.readlines()
    if frame_idx >= len(lines):
        return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])]) # x y z w
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4)
    T_WO[:3, :3] = r_WO
    T_WO[:3, 3] = t_WO
    return T_WO

def load_pose_T_CO(data_root, frame_idx):
    """
    Derive T_CO = T_CW @ T_WO
    using T_WC from extrinsics and T_WO from object_poses.txt
    """
    T_WO = load_object_pose_world(data_root, frame_idx)
    if T_WO is None:
        return None
    
    ext_file = Path(data_root) / "extrinsics" / f"{frame_idx:06d}.npy"
    if not ext_file.exists():
        return T_WO 
    
    T_WC = np.load(ext_file) 
    T_CW = np.linalg.inv(T_WC)
    
    T_CO = T_CW @ T_WO
    return T_CO

def optimize_pose(image_ref_t, gs_params, image_gt_t, K_torch, T_C_O_ref, T_C_O_init, cfg, gt_mask_t=None):
    """
    Optimize T_C_O_curr to minimize photometric error using AdamW.
    """
    H, W = image_gt_t.shape[1], image_gt_t.shape[2]
    
    log_R = torch.zeros((1, 3), device=cfg.device, requires_grad=True)
    trans = torch.zeros(3, device=cfg.device, requires_grad=True)
    
    optimizer = torch.optim.AdamW([
        {'params': [trans], 'lr': cfg.opt_lr * 1.5},
        {'params': [log_R], 'lr': cfg.opt_lr}
    ])
    
    losses = []
    best_loss = float('inf')
    best_T_C_O = T_C_O_init.clone()
    
    # Pre-render surface parameters once
    with torch.no_grad():
        if cfg.warp_type == "backward":
            n_init_val, d_init_val = gs_to_planar_params(gs_params, T_C_O_init)
            n_map, alpha_map = render_custom_attribute(gs_params, n_init_val, T_C_O_init, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_3ch_map, _ = render_custom_attribute(gs_params, d_init_val.repeat(1, 3), T_C_O_init, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_map = d_3ch_map[0:1]
        else:
            n_ref_val, d_ref_val = gs_to_planar_params(gs_params, T_C_O_ref)
            n_map, alpha_ref_map = render_custom_attribute(gs_params, n_ref_val, T_C_O_ref, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_ref_3ch_map, _ = render_custom_attribute(gs_params, d_ref_val.repeat(1, 3), T_C_O_ref, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_map = d_ref_3ch_map[0:1]
            
            _, alpha_map = render_custom_attribute(gs_params, n_ref_val, T_C_O_init, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            
            valid_mask_ref = (alpha_ref_map[0] > 0.5)
            n_ref_flat = n_map.permute(1, 2, 0).reshape(-1, 3)
            d_ref_flat = d_map.reshape(-1, 1)

    for step in range(cfg.opt_steps):
        optimizer.zero_grad()
        
        R_delta = so3_exp_map(log_R)[0]
        T_delta = torch.eye(4, device=cfg.device)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = trans
        
        T_C_O_curr = T_delta @ T_C_O_init
        T_ref_curr = T_C_O_ref @ torch.inverse(T_C_O_curr)
        
        if cfg.warp_type == "backward":
            warped = manual_backward_homography_warp(image_ref_t, n_map, d_map, T_ref_curr, K_torch)
        else:
            T_curr_ref = torch.inverse(T_ref_curr)
            warped = manual_forward_homography_warp(image_ref_t, n_ref_flat, d_ref_flat, T_curr_ref, K_torch, mask=valid_mask_ref)

        if gt_mask_t is not None:
            mask = gt_mask_t * alpha_map
        else:
            mask = alpha_map
            
        warped = warped * mask
        loss = F.l1_loss(warped, image_gt_t * mask)
        
        if torch.isnan(loss): break
        loss.backward()
        optimizer.step()
        
        curr_loss = loss.item()
        losses.append(curr_loss)
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_T_C_O = T_C_O_curr.detach().clone()
            
    return best_T_C_O, losses

def optimize_pose_lm(image_ref_t, gs_params, image_gt_t, K_torch, T_C_O_ref, T_C_O_init, cfg):
    """
    Optimize T_C_O_curr using Levenberg-Marquardt to minimize photometric error.
    """
    device = cfg.device
    H, W = image_gt_t.shape[1], image_gt_t.shape[2]
    
    # Downsample for Jacobian memory if needed
    opt_h, opt_w = H // cfg.res_scale, W // cfg.res_scale
    
    with torch.no_grad():
        if cfg.warp_type == "backward":
            n_init_val, d_init_val = gs_to_planar_params(gs_params, T_C_O_init)
            n_map, a_map = render_custom_attribute(gs_params, n_init_val, T_C_O_init, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_3ch_map, _ = render_custom_attribute(gs_params, d_init_val.repeat(1, 3), T_C_O_init, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_map = d_3ch_map[0:1]
        else:
            n_ref_val, d_ref_val = gs_to_planar_params(gs_params, T_C_O_ref)
            n_map, a_ref_map = render_custom_attribute(gs_params, n_ref_val, T_C_O_ref, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_ref_3ch_map, _ = render_custom_attribute(gs_params, d_ref_val.repeat(1, 3), T_C_O_ref, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_map = d_ref_3ch_map[0:1]
            
            # For forward warp mask
            _, a_map = render_custom_attribute(gs_params, n_ref_val, T_C_O_init, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            
        img_ref_o = F.interpolate(image_ref_t.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
        img_gt_o = F.interpolate(image_gt_t.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
        n_map_o = F.interpolate(n_map.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
        d_map_o = F.interpolate(d_map.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
        alpha_o = F.interpolate(a_map.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
        
        K_o = K_torch.clone()
        K_o[0,0] *= (opt_w/W); K_o[1,1] *= (opt_h/H)
        K_o[0,2] *= (opt_w/W); K_o[1,2] *= (opt_h/H)

    # Convert 3x4 to 7D (translation + quaternion [w, x, y, z]) for Theseus SE3 init
    init_np = T_C_O_init.detach().cpu().numpy() if torch.is_tensor(T_C_O_init) else T_C_O_init
    R_init = init_np[:3, :3]
    t_init = init_np[:3, 3]
    q_xyzw = R.from_matrix(R_init).as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    T_init_7d = torch.cat([torch.from_numpy(t_init), torch.from_numpy(q_wxyz)], dim=0).float().unsqueeze(0)
    
    # Theseus setup
    T_curr_var = th.SE3(T_init_7d, name="T_CO_curr")
    
    aux_vars = [
        th.Variable(img_ref_o.unsqueeze(0), name="img_ref"),
        th.Variable(img_gt_o.unsqueeze(0), name="img_gt"),
        th.Variable(n_map_o.unsqueeze(0), name="n_map"),
        th.Variable(d_map_o.unsqueeze(0), name="d_map"),
        th.Variable(alpha_o.unsqueeze(0), name="alpha"),
        th.Variable(T_C_O_ref.unsqueeze(0).to("cpu"), name="T_ref"),
        th.Variable(K_o.unsqueeze(0).to("cpu"), name="K")
    ]
    
    def error_fn(optim_vars, aux_vars):
        T_curr = optim_vars[0].to_matrix()
        img_r, img_g, n_m, d_m, alpha, T_ref, K = [v.tensor[0] for v in aux_vars]
        
        B = T_curr.shape[0]
        errors = []
        for i in range(B):
            T_C_O_curr = T_curr[i]
            T_ref_curr = T_ref @ torch.inverse(T_C_O_curr)
            
            if cfg.warp_type == "backward":
                warped = manual_backward_homography_warp(img_r, n_m, d_m, T_ref_curr, K)
            else:
                T_curr_ref = torch.inverse(T_ref_curr)
                n_flat = n_m.permute(1, 2, 0).reshape(-1, 3)
                d_flat = d_m.reshape(-1, 1)
                warped = manual_forward_homography_warp(img_r, n_flat, d_flat, T_curr_ref, K)
            
            mask = alpha 
            err = (warped - img_g) * mask
            errors.append(err.reshape(1, -1) / np.sqrt(opt_h * opt_w))
            
        return torch.cat(errors, dim=0)

    objective = th.Objective()
    cf = th.AutoDiffCostFunction([T_curr_var], error_fn, 3 * opt_h * opt_w, aux_vars=aux_vars, name="photometric")
    objective.add(cf)
    
    optimizer = th.LevenbergMarquardt(objective, max_iterations=cfg.opt_steps)
    layer = th.TheseusLayer(optimizer).to(device)
    
    inputs = {"T_CO_curr": T_curr_var.tensor.to(device)}
    updated, info = layer.forward(inputs, optimizer_kwargs={"verbose": True, "damping": cfg.damping, "track_err_history": True})
    
    objective.to(device)
    best_T_34 = updated["T_CO_curr"][0]
    res = torch.eye(4, device=device)
    res[:3, :] = best_T_34[:3, :]
    
    err_history = getattr(info, "err_history", None)
    return res, err_history, objective

def setup_blueprint():
    import rerun.blueprint as rrb
    blueprint = rrb.Blueprint(
        rrb.Tabs(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.Spatial2DView(name="GT Image", contents=["gt/image"]),
                        rrb.Spatial2DView(name="GT Depth", contents=["gt/depth"]),
                        rrb.Spatial2DView(name="GT Normal", contents=["gt/normal"]),
                    ),
                    rrb.Vertical(
                        rrb.Spatial2DView(name="GS Depth @ Target", contents=["gs_init/depth"]),
                        rrb.Spatial2DView(name="GS Normal @ Target", contents=["gs_init/normal"]),
                    ),
                    rrb.Grid(
                        rrb.Spatial2DView(name="Warped Optimized", contents=["warped/optimized"]),
                        rrb.Spatial2DView(name="Opt Loss", contents=["opt/loss_plot"]),
                        name="Optimization"
                    ),
                ),
                name="2D Warping"
            ),
            rrb.Horizontal(
                rrb.Spatial3DView(name="GT PC", contents=["world/object_gt"]),
                rrb.Spatial3DView(name="GS PC from Depth", contents=["world/gs_depth_pc"]),
                rrb.Spatial3DView(name="GS Original", contents=["world/object_gs"]),
                name="Comparison 3D"
            ),
            rrb.Spatial3DView(
                name="Point Clouds",
                contents=["world/**"]
            )
        )
    )
    return blueprint

def main(cfg: Config):
    rr.init("exp_warp_two_view_moge", spawn=False)
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    rr.send_blueprint(setup_blueprint())
    
    device = torch.device(cfg.device)
    data_dir = Path(cfg.data_root)
    
    # 1. Load data for init frame
    init_stem = f"{cfg.init_frame:06d}"
    image_init = np.array(cv2.imread(str(data_dir / "images" / f"{init_stem}.png"))[..., ::-1])
    mask_init = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{init_stem}.png"), cv2.IMREAD_GRAYSCALE))
    
    # Depth choice
    if cfg.use_moge_depth:
        depth_init = np.load(data_dir / "moge_depth" / f"{init_stem}.npy")
    else:
        depth_init = np.load(data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{init_stem}.npy")
    
    # Load MoGe normal instead of derived from depth
    normal_init = np.load(data_dir / "moge_normal" / f"{init_stem}.npy")
    if normal_init.shape[0] == 3 and normal_init.shape[1] != 3:
        normal_init = normal_init.transpose(1, 2, 0)
    
    K_init = np.load(data_dir / "intrinsics" / f"{init_stem}.npy")
    T_C_O_init = load_pose_T_CO(cfg.data_root, cfg.init_frame)
    if T_C_O_init is None:
        # Fallback to standard load_pose if logic fails
        with open(Path(cfg.data_root) / "object_poses.txt", "r") as f:
            line = f.readlines()[cfg.init_frame].split()
            t = np.array([float(line[1]), float(line[2]), float(line[3])])
            q = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])])
            T_C_O_init = np.eye(4); T_C_O_init[:3,:3] = R.from_quat(q).as_matrix(); T_C_O_init[:3,3] = t

    # Initialize GS using MoGe normal
    gsp = GaussianSuperPrimitive(image_init, mask_init, depth_init, T_C_O_init, K_init, normal=normal_init)
    gs_params = gsp.gs_params
    
    # 2. Load data for target frame
    target_stem = f"{cfg.target_frame:06d}"
    image_target_gt = np.array(cv2.imread(str(data_dir / "images" / f"{target_stem}.png"))[..., ::-1])
    depth_target_gt = np.load(data_dir / "depth_dyn" / f"{target_stem}.npy")
    K_target = np.load(data_dir / "intrinsics" / f"{target_stem}.npy")
    
    ext_file_target = data_dir / "extrinsics" / f"{target_stem}.npy"
    T_WC_target_gt = np.load(ext_file_target)
    T_WO_target_gt = load_object_pose_world(cfg.data_root, cfg.target_frame)
    T_C_O_target_gt = np.linalg.inv(T_WC_target_gt) @ T_WO_target_gt

    K_torch = torch.from_numpy(K_target).float().to(device)
    T_C_O_target_gt_torch = torch.from_numpy(T_C_O_target_gt).float().to(device)
    H, W = image_target_gt.shape[:2]
    
    # GT Normal for visualization (from GT depth)
    depth_target_gt_torch = torch.from_numpy(depth_target_gt).float().to(device)
    xyz_target_gt = unproject_depth(depth_target_gt_torch, K_torch, H, W)
    normal_target_gt, _ = d2n_tblr(xyz_target_gt.permute(2, 0, 1).unsqueeze(0))
    normal_target_gt = -normal_target_gt[0].permute(1, 2, 0)
    normal_vis_gt = (normal_target_gt.cpu().numpy() + 1.0) / 2.0
    
    # 3. GS Render at target frame (for init/GT comparison)
    with torch.no_grad():
        n_gs, d_gs = gs_to_planar_params(gs_params, T_C_O_target_gt_torch)
        n_map_gs, alpha_gs = render_custom_attribute(gs_params, n_gs, T_C_O_target_gt_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_3ch_gs, _ = render_custom_attribute(gs_params, d_gs.repeat(1, 3), T_C_O_target_gt_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_map_gs = d_3ch_gs[0]
        color_map_gs, _ = render_custom_attribute(gs_params, gs_params.colors, T_C_O_target_gt_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        
    normal_vis_gs = (n_map_gs.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
    depth_vis_gs = depth_to_rgb(d_map_gs.cpu().numpy())
    depth_vis_gt = depth_to_rgb(depth_target_gt)
    image_gs_rendered = (color_map_gs.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    
    # Log GT and GS initial render
    rr.log("gt/image", rr.Image(image_target_gt))
    rr.log("gt/depth", rr.Image(depth_vis_gt))
    rr.log("gt/normal", rr.Image((normal_vis_gt.clip(0, 1) * 255).astype(np.uint8)))
    rr.log("gs_init/depth", rr.Image(depth_vis_gs))
    rr.log("gs_init/normal", rr.Image((normal_vis_gs.clip(0, 1) * 255).astype(np.uint8)))
    
    # --- 3D Point Clouds ---
    pts_C_gt = xyz_target_gt.reshape(-1, 3).cpu().numpy()
    colors_gt = image_target_gt.reshape(-1, 3)
    valid_gt = (depth_target_gt.reshape(-1) > 0.01)
    pts_W_gt = (T_WC_target_gt[:3, :3] @ pts_C_gt[valid_gt].T).T + T_WC_target_gt[:3, 3]
    rr.log("world/object_gt", rr.Points3D(pts_W_gt, colors=colors_gt[valid_gt], radii=0.001))
    
    pts_C_gs_depth = unproject_depth(d_map_gs, K_torch, H, W).reshape(-1, 3).cpu().numpy()
    colors_gs_pc = image_gs_rendered.reshape(-1, 3)
    valid_gs_depth = (d_map_gs.reshape(-1).cpu().numpy() > 0.01) & (alpha_gs.reshape(-1).cpu().numpy() > 0.5)
    pts_W_gs_pc = (T_WC_target_gt[:3, :3] @ pts_C_gs_depth[valid_gs_depth].T).T + T_WC_target_gt[:3, 3]
    rr.log("world/gs_depth_pc", rr.Points3D(pts_W_gs_pc, colors=colors_gs_pc[valid_gs_depth], radii=0.001))
    
    gs_pts_O = gs_params.means.cpu().numpy()
    gs_colors = (gs_params.colors.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    pts_W_gs_orig = (T_WO_target_gt[:3, :3] @ gs_pts_O.T).T + T_WO_target_gt[:3, 3]
    rr.log("world/object_gs", rr.Points3D(pts_W_gs_orig, colors=gs_colors, radii=0.001))
    
    rr.log("world/camera_gt", rr.Pinhole(image_from_camera=K_target, width=W, height=H))
    rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WC_target_gt[:3, :3], translation=T_WC_target_gt[:3, 3]))

    # 4. Pose Optimization
    image_ref_t = torch.from_numpy(image_init).float().to(device).permute(2, 0, 1) / 255.0
    image_gt_t = torch.from_numpy(image_target_gt).float().to(device).permute(2, 0, 1) / 255.0
    T_C_O_ref_torch = torch.from_numpy(T_C_O_init).float().to(device)
    
    if cfg.use_lm:
        print(f"Starting LM Optimization ({cfg.warp_type})...")
        T_C_O_est, err_history, objective = optimize_pose_lm(
            image_ref_t, gs_params, image_gt_t, K_torch, T_C_O_ref_torch, T_C_O_ref_torch, cfg
        )
        if err_history is not None:
            losses = err_history[0].detach().cpu().numpy().tolist()
        else:
            with torch.no_grad():
                losses = [objective.error().norm().item()**2 / 2.0]
    else:
        print(f"Starting AdamW Optimization ({cfg.warp_type})...")
        T_C_O_est, losses = optimize_pose(
            image_ref_t, gs_params, image_gt_t, K_torch, T_C_O_ref_torch, T_C_O_ref_torch, cfg
        )
    
    print(f"Final Loss: {losses[-1]:.6f}")
    
    # Visualize final warping
    with torch.no_grad():
        if cfg.warp_type == "backward":
            n_curr, d_curr = gs_to_planar_params(gs_params, T_C_O_est)
            n_map, alpha_map = render_custom_attribute(gs_params, n_curr, T_C_O_est, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_3ch_map, _ = render_custom_attribute(gs_params, d_curr.repeat(1, 3), T_C_O_est, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_map = d_3ch_map[0:1]
            T_ref_curr = T_C_O_ref_torch @ torch.inverse(T_C_O_est)
            warped = manual_backward_homography_warp(image_ref_t, n_map, d_map, T_ref_curr, K_torch)
            warped = warped * alpha_map
        else:
            n_ref, d_ref = gs_to_planar_params(gs_params, T_C_O_ref_torch)
            n_ref_map, alpha_ref_map = render_custom_attribute(gs_params, n_ref, T_C_O_ref_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_ref_3ch_map, _ = render_custom_attribute(gs_params, d_ref.repeat(1, 3), T_C_O_ref_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_ref_map = d_ref_3ch_map[0:1]
            mask_ref = alpha_ref_map > 0.5
            n_ref_flat = n_ref_map.permute(1, 2, 0).reshape(-1, 3)
            d_ref_flat = d_ref_map.reshape(-1, 1)
            T_curr_ref = T_C_O_est @ torch.inverse(T_C_O_ref_torch)
            warped = manual_forward_homography_warp(image_ref_t, n_ref_flat, d_ref_flat, T_curr_ref, K_torch, mask=mask_ref)
            _, alpha_opt = render_custom_attribute(gs_params, n_ref, T_C_O_est, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            warped = warped * alpha_opt

    warped_np = (warped.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    rr.log("warped/optimized", rr.Image(warped_np))
    
    # Log loss plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(losses)
    ax.set_title("Optimization Loss")
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    rr.log("opt/loss_plot", rr.Image(np.array(Image.open(buf))))
    plt.close(fig)
    
    print("\nPose Comparison at Frame 40:")
    print("GT T_CO:\n", T_C_O_target_gt)
    print("Est T_CO:\n", T_C_O_est.cpu().numpy())
    
    T_WC_est = T_WO_target_gt @ np.linalg.inv(T_C_O_est.cpu().numpy())
    rr.log("world/camera_est", rr.Pinhole(image_from_camera=K_target, width=W, height=H))
    rr.log("world/camera_est", rr.Transform3D(mat3x3=T_WC_est[:3, :3], translation=T_WC_est[:3, 3]))
    rr.log("world/camera_est/frame", rr.Transform3D(relation=rr.TransformRelation.ParentFromChild))

    T_WC_init = T_WO_target_gt @ np.linalg.inv(T_C_O_init)
    rr.log("world/camera_init", rr.Pinhole(image_from_camera=K_target, width=W, height=H))
    rr.log("world/camera_init", rr.Transform3D(mat3x3=T_WC_init[:3, :3], translation=T_WC_init[:3, 3]))
    rr.log("world/camera_init/frame", rr.Transform3D(relation=rr.TransformRelation.ParentFromChild))

    print("Finished.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
