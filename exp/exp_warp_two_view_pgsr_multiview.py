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
    patch_size: int = 3
    sample_num: int = 10000

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

# --- PGSR Loss Functions ---

def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)

def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B, P, 1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid

def lncc(ref, nea):
    # ref: [batch_size, total_patch_size]
    # nea: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = (cross * cross) / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    # ncc = torch.mean(ncc, dim=1, keepdim=True) # Return per-patch loss
    return ncc

def compute_multi_view_ncc_loss(
    ref_image_gray, 
    nea_image_gray, 
    ref_normal, 
    ref_distance, 
    T_nea_ref, # Relative transform from ref view to nea view
    K, 
    patch_size=3, 
    valid_indices=None,
    pixels=None
):
    """
    Direct version of compute_multi_view_loss that takes explicit transformations.
    """
    device = ref_image_gray.device
    total_patch_size = (patch_size * 2 + 1) ** 2
    
    R_rel = T_nea_ref[:3, :3]
    t_rel = T_nea_ref[:3, 3]
    
    # Process inputs if they are maps
    if len(ref_normal.shape) == 3:
        ref_normal = ref_normal.permute(1, 2, 0).reshape(-1, 3)
    if len(ref_distance.shape) == 3:
        ref_distance = ref_distance.reshape(-1)
    elif len(ref_distance.shape) == 2:
        ref_distance = ref_distance.reshape(-1)
        
    if valid_indices is not None:
        ref_local_n = ref_normal[valid_indices]
        ref_local_d = ref_distance[valid_indices]
        pixels = pixels[valid_indices]
    else:
        ref_local_n = ref_normal
        ref_local_d = ref_distance

    if pixels is None:
        raise ValueError("Pixels must be provided.")

    # Homography: H = K * (R - t*n^T/d) * K_inv
    K_inv = torch.inverse(K)
    
    H_ref_to_nea = R_rel[None] - \
        torch.matmul(t_rel[None, :, None].expand(ref_local_d.shape[0], 3, 1), 
                    ref_local_n[:, :, None].permute(0, 2, 1)) / ref_local_d[..., None, None]
    
    H_ref_to_nea = torch.matmul(K[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_nea)
    H_ref_to_nea = H_ref_to_nea @ K_inv
    
    offsets = patch_offsets(patch_size, device)
    ori_pixels_patch = pixels.reshape(-1, 1, 2) + offsets.float()
    
    H_ref, W_ref = ref_image_gray.shape[-2:]
    pixels_patch = ori_pixels_patch.clone()
    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W_ref - 1) - 1.0
    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H_ref - 1) - 1.0
    
    ref_gray_val = F.grid_sample(ref_image_gray[None, None], pixels_patch.view(1, -1, 1, 2), align_corners=True)
    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)
    
    grid = patch_warp(H_ref_to_nea.reshape(-1, 3, 3), ori_pixels_patch)
    
    H_nea, W_nea = nea_image_gray.shape[-2:]
    grid[:, :, 0] = 2 * grid[:, :, 0] / (W_nea - 1) - 1.0
    grid[:, :, 1] = 2 * grid[:, :, 1] / (H_nea - 1) - 1.0
    
    sampled_nea_gray_val = F.grid_sample(nea_image_gray[None, None], grid.reshape(1, -1, 1, 2), align_corners=True)
    sampled_nea_gray_val = sampled_nea_gray_val.reshape(-1, total_patch_size)
    
    ncc_val = lncc(ref_gray_val, sampled_nea_gray_val)
    return ncc_val

# --- Optimization Logic ---

def load_object_pose_world(data_root, frame_idx):
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists(): return None
    with open(poses_file, "r") as f:
        lines = f.readlines()
    if frame_idx >= len(lines): return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])]) # x y z w
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4); T_WO[:3, :3] = r_WO; T_WO[:3, 3] = t_WO
    return T_WO

def load_pose_T_CO(data_root, frame_idx):
    T_WO = load_object_pose_world(data_root, frame_idx)
    if T_WO is None: return None
    ext_file = Path(data_root) / "extrinsics" / f"{frame_idx:06d}.npy"
    if not ext_file.exists(): return T_WO 
    T_WC = np.load(ext_file) 
    T_CW = np.linalg.inv(T_WC)
    return T_CW @ T_WO

def setup_blueprint():
    import rerun.blueprint as rrb
    blueprint = rrb.Blueprint(
        rrb.Tabs(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Vertical(
                        rrb.Spatial2DView(name="GT Image", contents=["gt/image"]),
                        rrb.Spatial2DView(name="GT Normal", contents=["gt/normal"]),
                    ),
                    rrb.Vertical(
                        rrb.Spatial2DView(name="GS Normal @ Target", contents=["gs_init/normal"]),
                        rrb.Spatial2DView(name="Warped Optimized", contents=["warped/optimized"]),
                    ),
                    rrb.Spatial2DView(name="Opt Loss", contents=["opt/loss_plot"]),
                    name="Optimization"
                ),
                name="2D Comparison"
            ),
            rrb.Spatial3DView(
                name="Point Clouds",
                contents=["world/**"]
            )
        )
    )
    return blueprint

def main(cfg: Config):
    rr.init("exp_warp_two_view_pgsr_multiview", spawn=False)
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    rr.send_blueprint(setup_blueprint())
    
    device = torch.device(cfg.device)
    data_dir = Path(cfg.data_root)
    
    # 1. Load data
    init_stem = f"{cfg.init_frame:06d}"
    image_init = np.array(cv2.imread(str(data_dir / "images" / f"{init_stem}.png"))[..., ::-1])
    mask_init = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{init_stem}.png"), cv2.IMREAD_GRAYSCALE))
    depth_init = np.load(data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{init_stem}.npy")
    K_init = np.load(data_dir / "intrinsics" / f"{init_stem}.npy")
    T_C_O_init = load_pose_T_CO(cfg.data_root, cfg.init_frame)

    gsp = GaussianSuperPrimitive(image_init, mask_init, depth_init, T_C_O_init, K_init)
    gs_params = gsp.gs_params
    
    target_stem = f"{cfg.target_frame:06d}"
    image_target_gt = np.array(cv2.imread(str(data_dir / "images" / f"{target_stem}.png"))[..., ::-1])
    K_target = np.load(data_dir / "intrinsics" / f"{target_stem}.npy")
    T_WC_target_gt = np.load(data_dir / "extrinsics" / f"{target_stem}.npy")
    T_WO_target_gt = load_object_pose_world(cfg.data_root, cfg.target_frame)
    T_C_O_target_gt = np.linalg.inv(T_WC_target_gt) @ T_WO_target_gt

    K_torch = torch.from_numpy(K_target).float().to(device)
    H, W = image_target_gt.shape[:2]
    
    # 2. GS and GT Visualization
    with torch.no_grad():
        T_C_O_target_gt_torch = torch.from_numpy(T_C_O_target_gt).float().to(device)
        n_gs, d_gs = gs_to_planar_params(gs_params, T_C_O_target_gt_torch)
        n_map_gs, alpha_gs = render_custom_attribute(gs_params, n_gs, T_C_O_target_gt_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_map_3ch_gs, _ = render_custom_attribute(gs_params, d_gs.repeat(1, 3), T_C_O_target_gt_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_map_gs = d_map_3ch_gs[0]
        color_map_gs, _ = render_custom_attribute(gs_params, gs_params.colors, T_C_O_target_gt_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        
    normal_vis_gs = (n_map_gs.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
    image_gs_rendered = (color_map_gs.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
        
    rr.log("gt/image", rr.Image(image_target_gt))
    rr.log("gs_init/normal", rr.Image((normal_vis_gs.clip(0, 1) * 255).astype(np.uint8)))

    # --- 3D Logging ---
    # GT Objects
    depth_target_gt = np.load(data_dir / "depth_dyn" / f"{target_stem}.npy")
    depth_target_gt_torch = torch.from_numpy(depth_target_gt).float().to(device)
    xyz_target_gt = unproject_depth(depth_target_gt_torch, K_torch, H, W)
    normal_target_gt, _ = d2n_tblr(xyz_target_gt.permute(2, 0, 1).unsqueeze(0))
    normal_target_gt = -normal_target_gt[0].permute(1, 2, 0)
    rr.log("gt/normal", rr.Image(((normal_target_gt.cpu().numpy() + 1.0) / 2.0 * 255).astype(np.uint8)))

    pts_C_gt = xyz_target_gt.reshape(-1, 3).cpu().numpy()
    colors_gt = image_target_gt.reshape(-1, 3)
    valid_gt = (depth_target_gt.reshape(-1) > 0.01)
    pts_W_gt = (T_WC_target_gt[:3, :3] @ pts_C_gt[valid_gt].T).T + T_WC_target_gt[:3, 3]
    rr.log("world/object_gt", rr.Points3D(pts_W_gt, colors=colors_gt[valid_gt], radii=0.001))
    
    # GS from depth PC
    pts_C_gs_depth = unproject_depth(d_map_gs, K_torch, H, W).reshape(-1, 3).cpu().numpy()
    colors_gs_pc = image_gs_rendered.reshape(-1, 3)
    valid_gs_depth = (d_map_gs.reshape(-1).cpu().numpy() > 0.01) & (alpha_gs.reshape(-1).cpu().numpy() > 0.5)
    pts_W_gs_pc = (T_WC_target_gt[:3, :3] @ pts_C_gs_depth[valid_gs_depth].T).T + T_WC_target_gt[:3, 3]
    rr.log("world/gs_depth_pc", rr.Points3D(pts_W_gs_pc, colors=colors_gs_pc[valid_gs_depth], radii=0.001))
    
    # GS Original
    gs_pts_O = gs_params.means.cpu().numpy()
    gs_colors = (gs_params.colors.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    pts_W_gs_orig = (T_WO_target_gt[:3, :3] @ gs_pts_O.T).T + T_WO_target_gt[:3, 3]
    rr.log("world/object_gs", rr.Points3D(pts_W_gs_orig, colors=gs_colors, radii=0.001))
    
    # Camera GT
    rr.log("world/camera_gt", rr.Pinhole(image_from_camera=K_target, width=W, height=H))
    rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WC_target_gt[:3, :3], translation=T_WC_target_gt[:3, 3]))

    # Camera Init (Guess for frame 40)
    T_WC_init = T_WO_target_gt @ np.linalg.inv(T_C_O_init)
    rr.log("world/camera_init", rr.Pinhole(image_from_camera=K_target, width=W, height=H))
    rr.log("world/camera_init", rr.Transform3D(mat3x3=T_WC_init[:3, :3], translation=T_WC_init[:3, 3]))
    rr.log("world/camera_init/frame", rr.Transform3D(relation=rr.TransformRelation.ParentFromChild))

    # 3. Optimization Setup
    image_ref_t = torch.from_numpy(image_target_gt).float().to(device).permute(2, 0, 1) / 255.0
    image_nea_t = torch.from_numpy(image_init).float().to(device).permute(2, 0, 1) / 255.0
    
    image_ref_gray = image_ref_t.mean(0)
    image_nea_gray = image_nea_t.mean(0)
    
    T_C_O_init_torch = torch.from_numpy(T_C_O_init).float().to(device)
    T_C_O_ref_torch = T_C_O_init_torch.clone() # Use init as starting guess for target pose

    # Optimization variables
    log_R_delta = torch.zeros((1, 3), device=device, requires_grad=True)
    t_delta = torch.zeros((1, 3), device=device, requires_grad=True)
    
    optimizer = torch.optim.AdamW([log_R_delta, t_delta], lr=cfg.opt_lr)
    
    losses = []
    
    print("Starting AdamW Optimization with PGSR NCC Loss...")
    for i in tqdm(range(cfg.opt_steps)):
        optimizer.zero_grad()
        
        R_delta = so3_exp_map(log_R_delta)[0]
        T_delta = torch.eye(4, device=device)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = t_delta[0]
        
        T_C_O_curr = T_delta @ T_C_O_init_torch
        
        # Render normal and distance maps at current pose
        n_curr, d_curr = gs_to_planar_params(gs_params, T_C_O_curr)
        n_map, alpha_map = render_custom_attribute(gs_params, n_curr, T_C_O_curr, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_map_3ch, _ = render_custom_attribute(gs_params, d_curr.repeat(1, 3), T_C_O_curr, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_map = d_map_3ch[0]
        
        # Relative transform from target cam (ref) to init cam (nea)
        # P_nea = T_C_init_O * T_C_target_O^-1 * P_ref
        T_nea_ref = T_C_O_init_torch @ torch.inverse(T_C_O_curr)
        
        # Sample pixels from mask
        mask_flat = (alpha_map[0] > 0.5).reshape(-1)
        valid_indices = torch.where(mask_flat)[0]
        if len(valid_indices) > cfg.sample_num:
            perm = torch.randperm(len(valid_indices), device=device)[:cfg.sample_num]
            valid_indices = valid_indices[perm]
            
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        pixels = torch.stack([grid_x, grid_y], dim=-1).float().reshape(-1, 2)
        
        if len(valid_indices) == 0:
            print("Zero mask area, stopping.")
            break
            
        ncc_loss_per_patch = compute_multi_view_ncc_loss(
            image_ref_gray, image_nea_gray, n_map, d_map, T_nea_ref, K_torch,
            patch_size=cfg.patch_size, valid_indices=valid_indices, pixels=pixels
        )
        
        loss = ncc_loss_per_patch.mean()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        if i % 10 == 0:
            pass # rr.Scalar is not supported in this version
            
    # 4. Results
    print(f"Final Loss: {losses[-1]:.6f}")
    
    with torch.no_grad():
        R_delta_f = so3_exp_map(log_R_delta)[0]
        T_delta_f = torch.eye(4, device=device)
        T_delta_f[:3, :3] = R_delta_f
        T_delta_f[:3, 3] = t_delta[0]
        T_C_O_est = T_delta_f @ T_C_O_init_torch
        
        n_est, d_est = gs_to_planar_params(gs_params, T_C_O_est)
        n_map_est, alpha_map_est = render_custom_attribute(gs_params, n_est, T_C_O_est, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_map_3ch_est, _ = render_custom_attribute(gs_params, d_est.repeat(1, 3), T_C_O_est, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_map_est = d_map_3ch_est[0]
        
        T_nea_ref_est = T_C_O_init_torch @ torch.inverse(T_C_O_est)
        
        # Log final pose estimation in Rerun
        T_WC_est = T_WO_target_gt @ np.linalg.inv(T_C_O_est.cpu().numpy())
        rr.log("world/camera_est", rr.Pinhole(image_from_camera=K_target, width=W, height=H))
        rr.log("world/camera_est", rr.Transform3D(mat3x3=T_WC_est[:3, :3], translation=T_WC_est[:3, 3]))

        # Log visual comparison
        normal_vis_est = (n_map_est.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
        rr.log("warped/optimized", rr.Image((normal_vis_est.clip(0, 1) * 255).astype(np.uint8)))

    # Log loss plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(losses); ax.set_title("AdamW NCC Loss"); ax.set_yscale('log')
    buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
    rr.log("opt/loss_plot", rr.Image(np.array(Image.open(buf))))
    plt.close(fig)

    print("\nPose Comparison at Target Frame:")
    print("GT T_CO:\n", T_C_O_target_gt)
    print("Est T_CO:\n", T_C_O_est.cpu().numpy())

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
