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
import flow_vis
from dataclasses import dataclass
import io
from PIL import Image
from gs_dyn_obj.utils.init import d2n_tblr, unproject_depth
from scipy.spatial.transform import Rotation as R
import time
from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    target_frame: int = 40
    near_plane: float = 0.01
    far_plane: float = 10.0
    device: str = "cuda"
    opt_lr: float = 3e-3
    opt_steps: int = 1000
    ufm_model: str = "base" # base, refine, base-980, refine-980
    use_gt_pose: bool = False # Use GT camera-to-object pose for triangulation

def depth_to_rgb(depth, min_val=0.0, max_val=2.0):
    """Consistent colormapping for depth visualization."""
    depth_norm = np.clip((depth - min_val) / (max_val - min_val), 0, 1)
    colormap = plt.get_cmap("magma")
    rgb = (colormap(depth_norm)[..., :3] * 255).astype(np.uint8)
    return rgb

def gs_to_planar_params(gs_params, T_CW):
    """
    Convert GS parameters to planar surface parameters (normal and distance) in camera view.
    """
    means = gs_params.means # [N, 3]
    quats = gs_params.quats # [N, 4]
    R_WG = quaternion_to_matrix(quats) # [N, 3, 3]
    n_W = R_WG[:, :, 2] # [N, 3]
    R_CW = T_CW[:3, :3]
    t_CW = T_CW[:3, 3]
    n_C = torch.einsum('ij,nj->ni', R_CW, n_W) # [N, 3]
    p_C = torch.einsum('ij,nj->ni', R_CW, means) + t_CW # [N, 3]
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
    
    R = T_curr_ref[:3, :3]
    t = T_curr_ref[:3, 3]
    
    K_inv = torch.inverse(K)
    dir0 = (K_inv @ u0.t()).t() # [N, 3]
    
    n_dot_dir = torch.sum(n_C * dir0, dim=1, keepdim=True)
    d_safe = torch.where(torch.abs(d) > 1e-6, d, torch.ones_like(d))
    p1_scaled = torch.matmul(R, dir0.t()).t() - (n_dot_dir / d_safe) * t.view(1, 3)
    
    u1_homog = torch.matmul(K, p1_scaled.t()).t()
    u1_pix = u1_homog[:, :2] / (u1_homog[:, 2:3] + 1e-8)
    z1_scaled = u1_homog[:, 2]
    
    valid = (u1_pix[:, 0] >= 0) & (u1_pix[:, 0] < W-1) & \
            (u1_pix[:, 1] >= 0) & (u1_pix[:, 1] < H-1) & \
            (z1_scaled > 0)
    
    u1_pix = u1_pix[valid]
    colors = colors[valid]
    z1_scaled = z1_scaled[valid]
    
    sort_idx = torch.argsort(z1_scaled, descending=True)
    u1_pix = u1_pix[sort_idx]
    colors = colors[sort_idx]
    
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
    
    mask00 = (x0_idx >= 0) & (x0_idx < W) & (y0_idx >= 0) & (y0_idx < H)
    mask10 = (x1_idx >= 0) & (x1_idx < W) & (y0_idx >= 0) & (y0_idx < H)
    mask01 = (x0_idx >= 0) & (x0_idx < W) & (y1_idx >= 0) & (y1_idx < H)
    mask11 = (x1_idx >= 0) & (x1_idx < W) & (y1_idx >= 0) & (y1_idx < H)
    
    warped = torch.zeros((H, W, C), device=device)
    
    for m, y_idx, x_idx, w in zip([mask00, mask10, mask01, mask11], 
                                  [y0_idx, y0_idx, y1_idx, y1_idx], 
                                  [x0_idx, x1_idx, x0_idx, x1_idx], 
                                  [wa, wb, wc, wd]):
        if m.any():
            warped.index_put_((y_idx[m], x_idx[m]), colors[m] * w[m].unsqueeze(-1), accumulate=True)

    return warped.permute(2, 0, 1)

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
    T_WO = load_object_pose_world(data_root, frame_idx)
    if T_WO is None: return None
    ext_file = Path(data_root) / "extrinsics" / f"{frame_idx:06d}.npy"
    if not ext_file.exists(): return T_WO 
    T_WC = np.load(ext_file) 
    T_CW = np.linalg.inv(T_WC)
    T_CO = T_CW @ T_WO
    return T_CO

def predict_ufm_correspondences(model, source_image, target_image):
    device = next(model.parameters()).device
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            result = model.predict_correspondences_batched(
                source_image=torch.from_numpy(source_image).to(device),
                target_image=torch.from_numpy(target_image).to(device),
            )
        flow_output = result.flow.flow_output[0].cpu().float()
        covisibility = result.covisibility.mask[0].cpu().float()
    return flow_output, covisibility

def optimize_pose(image_ref_t, gs_params, image_gt_t, K_torch, T_C_O_ref, T_C_O_init, cfg, gt_mask_t=None):
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
    with torch.no_grad():
        n_ref_val, d_ref_val = gs_to_planar_params(gs_params, T_C_O_ref)
        n_map, alpha_ref_map = render_custom_attribute(gs_params, n_ref_val, T_C_O_ref, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_ref_3ch_map, _ = render_custom_attribute(gs_params, d_ref_val.repeat(1, 3), T_C_O_ref, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_map = d_ref_3ch_map[0:1]
        valid_mask_ref = (alpha_ref_map[0] > 0.5)
        n_ref_flat = n_map.permute(1, 2, 0).reshape(-1, 3)
        d_ref_flat = d_map.reshape(-1, 1)
        
        # Pre-render static alpha map at initial pose guess
        _, alpha_map = render_custom_attribute(gs_params, n_ref_val, T_C_O_init, K_torch, W, H, cfg.near_plane, cfg.far_plane)

    for step in range(cfg.opt_steps):
        optimizer.zero_grad()
        R_delta = so3_exp_map(log_R)[0]
        T_delta = torch.eye(4, device=cfg.device)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = trans
        T_C_O_curr = T_delta @ T_C_O_init
        T_ref_curr = T_C_O_ref @ torch.inverse(T_C_O_curr)
        T_curr_ref = torch.inverse(T_ref_curr)
        warped = manual_forward_homography_warp(image_ref_t, n_ref_flat, d_ref_flat, T_curr_ref, K_torch, mask=valid_mask_ref)
        
        if gt_mask_t is not None: mask = gt_mask_t * alpha_map
        else: mask = alpha_map
        
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
            
    with torch.no_grad():
        T_ref_curr = T_C_O_ref @ torch.inverse(best_T_C_O)
        T_curr_ref = torch.inverse(T_ref_curr)
        warped_final = manual_forward_homography_warp(image_ref_t, n_ref_flat, d_ref_flat, T_curr_ref, K_torch, mask=valid_mask_ref)
        
    return best_T_C_O, losses, warped_final

def triangulate_points(pts0, pts1, K, T_rel):
    """
    pts0: [N, 2]
    pts1: [N, 2]
    K: [3, 3]
    T_rel: [4, 4] T_C1_C0
    Returns: [N, 3] in C0 frame
    """
    P0 = K @ np.eye(3, 4)
    P1 = K @ T_rel[:3, :]
    pts4D = cv2.triangulatePoints(P0, P1, pts0.T, pts1.T)
    pts3D = (pts4D[:3, :] / pts4D[3, :]).T
    return pts3D

def main(cfg: Config):
    rr.init("exp_triangulate_ufm", spawn=False)
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    
    device = torch.device(cfg.device)
    data_dir = Path(cfg.data_root)
    
    # 1. Load data for init frame
    init_stem = f"{cfg.init_frame:06d}"
    image_init = np.array(cv2.imread(str(data_dir / "images" / f"{init_stem}.png"))[..., ::-1])
    mask_init = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{init_stem}.png"), cv2.IMREAD_GRAYSCALE))
    depth_init = np.load(data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{init_stem}.npy")
    K_init = np.load(data_dir / "intrinsics" / f"{init_stem}.npy")
    T_C_O_init = load_pose_T_CO(cfg.data_root, cfg.init_frame)
    T_WO_init = load_object_pose_world(cfg.data_root, cfg.init_frame)

    # Initialize GS (for pose optimization if needed, and for visualization)
    gsp = GaussianSuperPrimitive(image_init, mask_init, depth_init, T_C_O_init, K_init)
    gs_params = gsp.gs_params
    
    # 2. Load data for target frame
    target_stem = f"{cfg.target_frame:06d}"
    image_target_gt = np.array(cv2.imread(str(data_dir / "images" / f"{target_stem}.png"))[..., ::-1])
    K_target = np.load(data_dir / "intrinsics" / f"{target_stem}.npy")
    
    T_WO_target_gt = load_object_pose_world(cfg.data_root, cfg.target_frame)
    ext_file_target = data_dir / "extrinsics" / f"{target_stem}.npy"
    T_WC_target_gt = np.load(ext_file_target)
    T_CW_target_gt = np.linalg.inv(T_WC_target_gt)
    T_C_O_target_gt = T_CW_target_gt @ T_WO_target_gt

    K_torch = torch.from_numpy(K_target).float().to(device)
    T_C_O_ref_torch = torch.from_numpy(T_C_O_init).float().to(device)
    H, W = image_target_gt.shape[:2]

    # 3. Get Relative Pose
    warped_opt = None
    if cfg.use_gt_pose:
        T_C_O_est = torch.from_numpy(T_C_O_target_gt).float().to(device)
    else:
        print("Optimizing relative pose...")
        image_ref_t = torch.from_numpy(image_init).float().to(device).permute(2, 0, 1) / 255.0
        image_gt_t = torch.from_numpy(image_target_gt).float().to(device).permute(2, 0, 1) / 255.0
        T_C_O_est, _, warped_opt = optimize_pose(image_ref_t, gs_params, image_gt_t, K_torch, T_C_O_ref_torch, T_C_O_ref_torch, cfg)

    # T_rel: T_C1_C0 = T_C1_O @ T_O_C0
    T_C_O0 = T_C_O_init
    T_C_O1 = T_C_O_est.cpu().numpy()
    T_rel = T_C_O1 @ np.linalg.inv(T_C_O0)

    # Print GT vs Estimated Pose
    print("\nPose Comparison at Target Frame:")
    print("GT T_CO:\n", T_C_O_target_gt)
    print("Est T_CO:\n", T_C_O1)

    # 4. Get UFM Flow
    print(f"Loading UFM {cfg.ufm_model} model...")
    model_repo_map = {
        "base"              : "infinity1096/UFM-Base",
        "refine"            : "infinity1096/UFM-Refine",
        "base-980"          : "infinity1096/UFM-Base-980",
        "refine-980"        : "infinity1096/UFM-Refine-980",
    }
    if "base" in cfg.ufm_model:
        model = UniFlowMatchConfidence.from_pretrained(model_repo_map[cfg.ufm_model])
    elif "refine" in cfg.ufm_model:
        model = UniFlowMatchClassificationRefinement.from_pretrained(model_repo_map[cfg.ufm_model])
    else: raise ValueError("Invalid model variant")
    model.to(device); model.eval()
    
    print("Predicting UFM correspondences...")
    flow_ufm, covis_ufm = predict_ufm_correspondences(model, image_init, image_target_gt)
    
    # 5. Triangulation
    print("Triangulating dense points...")
    mask_idx = np.where(mask_init > 0)
    pts0 = np.stack([mask_idx[1], mask_idx[0]], axis=-1).astype(np.float32) # [N, 2]
    
    # Get flow at these pixels
    flow_np = flow_ufm.permute(1, 2, 0).numpy()
    flow_at_pts = flow_np[mask_idx[0], mask_idx[1]]
    pts1 = pts0 + flow_at_pts
    
    # Filter by covisibility
    covis_np = covis_ufm.numpy()
    covis_at_pts = covis_np[mask_idx[0], mask_idx[1]]
    valid = covis_at_pts > 0.5
    
    pts0_v = pts0[valid]
    pts1_v = pts1[valid]
    colors_v = image_init[mask_idx[0][valid], mask_idx[1][valid]]
    
    pts3D_C0 = triangulate_points(pts0_v, pts1_v, K_init, T_rel)
    
    # Transform to Object Frame: X_O = T_O_C0 @ X_C0
    T_O_C0 = np.linalg.inv(T_C_O0)
    pts3D_O = (T_O_C0[:3, :3] @ pts3D_C0.T).T + T_O_C0[:3, 3]
    
    # Transform to World Frame at target frame: X_W = T_WO_1 @ X_O
    pts3D_W = (T_WO_target_gt[:3, :3] @ pts3D_O.T).T + T_WO_target_gt[:3, 3]

    # 6. Visualization
    flow_vis_image = flow_vis.flow_to_color(flow_np)
    rr.log("ufm/flow", rr.Image(flow_vis_image))
    rr.log("ufm/covisibility", rr.Image(covis_ufm.numpy()))
    
    rr.log("world/triangulated", rr.Points3D(pts3D_W, colors=colors_v, radii=0.001))
    
    # Log GT for comparison
    depth_target_gt = np.load(data_dir / "depth_dyn" / f"{target_stem}.npy")
    xyz_target_gt = unproject_depth(torch.from_numpy(depth_target_gt).float().to(device), torch.from_numpy(K_target).float().to(device), H, W)
    pts_C_gt = xyz_target_gt.reshape(-1, 3).cpu().numpy()
    colors_gt = image_target_gt.reshape(-1, 3)
    valid_gt = (depth_target_gt.reshape(-1) > 0.01)
    pts_W_gt = (T_WC_target_gt[:3, :3] @ pts_C_gt[valid_gt].T).T + T_WC_target_gt[:3, 3]
    rr.log("world/gt_pc", rr.Points3D(pts_W_gt, colors=colors_gt[valid_gt], radii=0.001))

    # Log GS for comparison
    gs_pts_O = gs_params.means.cpu().numpy()
    gs_colors = (gs_params.colors.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    pts_W_gs = (T_WO_target_gt[:3, :3] @ gs_pts_O.T).T + T_WO_target_gt[:3, 3]
    rr.log("world/gs_orig", rr.Points3D(pts_W_gs, colors=gs_colors, radii=0.001))

    # --- Log Cameras in World Frame ---
    H_init, W_init = image_init.shape[:2]
    H_target, W_target = image_target_gt.shape[:2]

    # 1. Initial Camera (at init_frame)
    T_WC_init = T_WO_init @ np.linalg.inv(T_C_O_init)
    rr.log("world/camera_init", rr.Pinhole(image_from_camera=K_init, width=W_init, height=H_init))
    rr.log("world/camera_init", rr.Transform3D(mat3x3=T_WC_init[:3, :3], translation=T_WC_init[:3, 3]))
    rr.log("world/camera_init/image", rr.Image(image_init))
    
    # 2. GT Camera (at target_frame)
    rr.log("world/camera_gt", rr.Pinhole(image_from_camera=K_target, width=W_target, height=H_target))
    rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WC_target_gt[:3, :3], translation=T_WC_target_gt[:3, 3]))
    rr.log("world/camera_gt/image", rr.Image(image_target_gt))

    # 3. Origin Camera (starting guess for target_frame)
    T_WC_origin = T_WO_target_gt @ np.linalg.inv(T_C_O_init)
    rr.log("world/camera_origin", rr.Pinhole(image_from_camera=K_target, width=W_target, height=H_target))
    rr.log("world/camera_origin", rr.Transform3D(mat3x3=T_WC_origin[:3, :3], translation=T_WC_origin[:3, 3]))
    rr.log("world/camera_origin/image", rr.Image(image_target_gt))

    # 4. Optimized Camera (at target_frame)
    T_WC_optimized = T_WO_target_gt @ np.linalg.inv(T_C_O_est.cpu().numpy())
    rr.log("world/camera_optimized", rr.Pinhole(image_from_camera=K_target, width=W_target, height=H_target))
    rr.log("world/camera_optimized", rr.Transform3D(mat3x3=T_WC_optimized[:3, :3], translation=T_WC_optimized[:3, 3]))
    if warped_opt is not None:
        warped_np = (warped_opt.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        rr.log("world/camera_optimized/image", rr.Image(warped_np))
    else:
        rr.log("world/camera_optimized/image", rr.Image(image_target_gt))

    print("Triangulation complete. Check Rerun.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
