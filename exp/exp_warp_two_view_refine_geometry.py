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
import time

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
    refine_lr: float = 1e-2
    refine_steps: int = 100
    patch_size_refine: int = 8
    sample_patches: int = 2000 # Max patches to optimize for performance

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
    # 1. Load Object Pose in World Frame T_WO
    T_WO = load_object_pose_world(data_root, frame_idx)
    if T_WO is None:
        return None
    
    # 2. Load Camera Pose in World Frame T_WC
    ext_file = Path(data_root) / "extrinsics" / f"{frame_idx:06d}.npy"
    if not ext_file.exists():
        return T_WO 
    
    T_WC = np.load(ext_file) 
    T_CW = np.linalg.inv(T_WC)
    
    T_CO = T_CW @ T_WO
    return T_CO

# --- Patch Geometry Refinement Helpers ---

def patch_offsets(patch_size, device):
    """
    Generate offsets for a square patch.
    patch_size: size of one side (e.g. 8)
    """
    half = patch_size // 2
    offsets = torch.arange(-half, -half + patch_size, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)

def patch_warp(H, uv):
    """
    H: [B, 3, 3] homographies
    uv: [B, P, 2] reference pixels
    returns: [B, P, 2] warped pixels
    """
    B, P = uv.shape[:2]
    ones = torch.ones((B, P, 1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)
    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid

def lncc_loss(ref, nea):
    """
    ref: [B, P]
    nea: [B, P]
    """
    B, P = ref.shape
    mu_ref = ref.mean(dim=1, keepdim=True)
    mu_nea = nea.mean(dim=1, keepdim=True)
    
    ref_zero = ref - mu_ref
    nea_zero = nea - mu_nea
    
    num = (ref_zero * nea_zero).sum(dim=1)
    den = torch.sqrt((ref_zero**2).sum(dim=1) * (nea_zero**2).sum(dim=1) + 1e-8)
    
    ncc = num / den
    return 1.0 - ncc # Loss form

def optimize_patch_geometry(image_ref_t, image_tgt_t, n_ref_map, d_ref_map, alpha_ref_map, T_C_O_ref, T_C_O_opt, K, cfg):
    """
    Refine normal and distance for each 8x8 patch in the object area.
    """
    device = image_ref_t.device
    C, H, W = image_ref_t.shape
    ps = cfg.patch_size_refine
    
    # 1. Identify object patches
    # Grid the mask
    mask = (alpha_ref_map[0] > 0.5)
    
    # Create patch grid
    y_starts = torch.arange(0, H - ps + 1, ps, device=device)
    x_starts = torch.arange(0, W - ps + 1, ps, device=device)
    grid_y, grid_x = torch.meshgrid(y_starts, x_starts, indexing='ij')
    patch_centers_y = grid_y + ps // 2
    patch_centers_x = grid_x + ps // 2
    
    # Filter patches that are mostly inside the mask
    # We check the center or the whole patch coverage
    mask_patches = F.unfold(mask.float().unsqueeze(0).unsqueeze(0), kernel_size=ps, stride=ps) # [1, ps*ps, N_patches]
    coverage = mask_patches.mean(dim=1)[0] # [N_patches]
    valid_patch_mask = coverage > 0.5
    
    valid_indices = torch.where(valid_patch_mask)[0]
    if len(valid_indices) > cfg.sample_patches:
        perm = torch.randperm(len(valid_indices), device=device)[:cfg.sample_patches]
        valid_indices = valid_indices[perm]
    
    num_patches = len(valid_indices)
    if num_patches == 0:
        print("No valid object patches found for refinement.")
        return n_ref_map, d_ref_map
    
    # Get original patch center coordinates
    all_patch_centers_x = patch_centers_x.flatten()[valid_indices]
    all_patch_centers_y = patch_centers_y.flatten()[valid_indices]
    
    # 2. Initialize (n, d) from the GS maps
    # We take the value at the center of each patch
    n_init = n_ref_map[:, all_patch_centers_y, all_patch_centers_x].t().contiguous() # [N, 3]
    d_init = d_ref_map[0, all_patch_centers_y, all_patch_centers_x].unsqueeze(-1).contiguous() # [N, 1]
    
    # Parameterize n using spherical coordinates or just keep as vector and normalize
    # Let's use 3D vector and normalize
    n_param = n_init.clone().detach().requires_grad_(True)
    d_param = d_init.clone().detach().requires_grad_(True)
    
    optimizer = torch.optim.AdamW([n_param, d_param], lr=cfg.refine_lr)
    
    # Relative pose Ref -> Tgt
    T_OC_ref = torch.inverse(T_C_O_ref)
    T_tgt_ref = T_C_O_opt @ T_OC_ref
    R_rel = T_tgt_ref[:3, :3]
    t_rel = T_tgt_ref[:3, 3]
    K_inv = torch.inverse(K)
    losses = []
    
    # Pre-compute reference patches
    offsets = patch_offsets(ps, device) # [1, ps*ps, 2]
    patch_coords_ref = torch.stack([all_patch_centers_x, all_patch_centers_y], dim=-1).unsqueeze(1).float() + offsets.float() # [N, ps*ps, 2]
    
    # Sample ref image at these patches
    # F.grid_sample takes normalized coords [-1, 1]
    ref_coords_norm = patch_coords_ref.clone()
    ref_coords_norm[..., 0] = 2.0 * ref_coords_norm[..., 0] / (W - 1) - 1.0
    ref_coords_norm[..., 1] = 2.0 * ref_coords_norm[..., 1] / (H - 1) - 1.0
    
    # Use grayscale for NCC if preferred, or RGB. Let's use RGB flattened.
    # ref_patches: [N, ps*ps, 3]
    ref_patches = F.grid_sample(
        image_ref_t.unsqueeze(0), 
        ref_coords_norm.view(1, -1, 1, 2), 
        align_corners=True
    ).reshape(3, -1).t().view(num_patches, ps*ps, 3)
    
    # Flatten patches for NCC: [N, ps*ps*3]
    ref_patches_flat = ref_patches.reshape(num_patches, -1)
    
    image_tgt_gray = image_tgt_t.mean(dim=0, keepdim=True)
    image_ref_gray = image_ref_t.mean(dim=0, keepdim=True)
    # Actually let's just use gray patches for NCC to be more robust?
    ref_patches_gray = ref_patches.mean(dim=-1) # [N, ps*ps]
    
    pbar = tqdm(range(cfg.refine_steps), desc="Refining Geometry")
    for step in pbar:
        optimizer.zero_grad()
        
        # Normalize normals
        n_unit = F.normalize(n_param, dim=1)
        
        # Homography H = K (R - (t n^T)/d) K_inv
        # Batched version:
        # H: [N, 3, 3]
        d_safe = torch.clamp(d_param, min=1e-3)
        
        # (t n^T) / d : [N, 3, 3]
        tnT_d = torch.matmul(t_rel.view(3, 1), n_unit.view(num_patches, 1, 3)) / d_safe.view(num_patches, 1, 1)
        H_m = R_rel.unsqueeze(0) - tnT_d
        H_batch = K @ H_m @ K_inv
        
        # Warp ref patch coords to tgt frame
        warped_coords = patch_warp(H_batch, patch_coords_ref) # [N, ps*ps, 2]
        
        # Sample tgt image
        tgt_coords_norm = warped_coords.clone()
        tgt_coords_norm[..., 0] = 2.0 * tgt_coords_norm[..., 0] / (W - 1) - 1.0
        tgt_coords_norm[..., 1] = 2.0 * tgt_coords_norm[..., 1] / (H - 1) - 1.0
        
        tgt_patches = F.grid_sample(
            image_tgt_t.unsqueeze(0), 
            tgt_coords_norm.reshape(1, -1, 1, 2), 
            align_corners=True
        ).reshape(3, -1).t().reshape(num_patches, ps*ps, 3).mean(dim=-1) # [N, ps*ps]
        
        loss = lncc_loss(ref_patches_gray, tgt_patches).mean()
        
        loss.backward()
        optimizer.step()
        
        curr_loss = loss.item()
        losses.append(curr_loss)
        if step % 10 == 0:
            pbar.set_postfix({"loss": f"{curr_loss:.6f}"})
            
    # After optimization, create the refined maps
    with torch.no_grad():
        n_refined_map = n_ref_map.clone()
        d_refined_map = d_ref_map.clone()
        
        n_final = F.normalize(n_param, dim=1)
        # We need to fill the patches. For simplicity, just update the centers or the whole block?
        # Let's update the whole block for each valid patch.
        # This is slightly inefficient but clear.
        for i, idx in enumerate(valid_indices):
            cx = all_patch_centers_x[i].item()
            cy = all_patch_centers_y[i].item()
            x0, y0 = cx - ps//2, cy - ps//2
            n_refined_map[:, y0:y0+ps, x0:x0+ps] = n_final[i].view(3, 1, 1)
            d_refined_map[:, y0:y0+ps, x0:x0+ps] = d_param[i].view(1, 1, 1)
            
    return n_refined_map, d_refined_map, losses

def optimize_pose(image_ref_t, gs_params, image_gt_t, K_torch, T_C_O_ref, T_C_O_init, cfg, gt_mask_t=None):
    """
    Optimize T_C_O_curr to minimize photometric error.
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
                        rrb.Spatial2DView(name="GS Initial Normal", contents=["gs_init/normal"]),
                        rrb.Spatial2DView(name="GS Refined Normal", contents=["gs_refined/normal"]),
                    ),
                    rrb.Grid(
                        rrb.Spatial2DView(name="Warped Optimized", contents=["warped/optimized"]),
                        rrb.Spatial2DView(name="Warped Refined", contents=["warped/refined"]),
                        rrb.Spatial2DView(name="Pose Loss", contents=["opt/pose_loss_plot"]),
                        rrb.Spatial2DView(name="Refine Loss", contents=["opt/refine_loss_plot"]),
                        name="Optimization"
                    ),
                ),
                name="2D Warping"
            ),
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="GT Normal", contents=["gt/normal"]),
                    rrb.Spatial2DView(name="GS Initial Normal", contents=["gs_init/normal"]),
                    rrb.Spatial2DView(name="GS Refined Normal", contents=["gs_refined/normal"]),
                ),
                name="Comparison 2D"
            ),
            rrb.Horizontal(
                rrb.Spatial3DView(name="GT PC", contents=["world/object_gt"]),
                rrb.Spatial3DView(name="GS Depth PC", contents=["world/gs_depth_pc"]),
                rrb.Spatial3DView(name="GS Refined PC", contents=["world/gs_refined_pc"]),
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
    rr.init("exp_warp_two_view_refine_geometry", spawn=False)
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    
    # Send blueprint
    rr.send_blueprint(setup_blueprint())
    
    device = torch.device(cfg.device)
    data_dir = Path(cfg.data_root)
    
    # 1. Load data for init frame (35)
    init_stem = f"{cfg.init_frame:06d}"
    image_init = np.array(cv2.imread(str(data_dir / "images" / f"{init_stem}.png"))[..., ::-1])
    mask_init = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{init_stem}.png"), cv2.IMREAD_GRAYSCALE))
    depth_init = np.load(data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{init_stem}.npy")
    K_init = np.load(data_dir / "intrinsics" / f"{init_stem}.npy")
    T_C_O_init = load_pose_T_CO(cfg.data_root, cfg.init_frame)
    if T_C_O_init is None:
        # Fallback to standard load_pose if logic fails
        with open(Path(cfg.data_root) / "object_poses.txt", "r") as f:
            line = f.readlines()[cfg.init_frame].split()
            t = np.array([float(line[1]), float(line[2]), float(line[3])])
            q = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])])
            T_C_O_init = np.eye(4); T_C_O_init[:3,:3] = R.from_quat(q).as_matrix(); T_C_O_init[:3,3] = t

    # Initialize GS
    gsp = GaussianSuperPrimitive(image_init, mask_init, depth_init, T_C_O_init, K_init)
    gs_params = gsp.gs_params
    
    # 2. Load data for target frame (40)
    target_stem = f"{cfg.target_frame:06d}"
    image_target_gt = np.array(cv2.imread(str(data_dir / "images" / f"{target_stem}.png"))[..., ::-1])
    depth_target_gt = np.load(data_dir / "depth_dyn" / f"{target_stem}.npy")
    K_target = np.load(data_dir / "intrinsics" / f"{target_stem}.npy")
    
    # Load GT Poses in World
    ext_file_target = data_dir / "extrinsics" / f"{target_stem}.npy"
    T_WC_target_gt = np.load(ext_file_target)
    T_CW_target_gt = np.linalg.inv(T_WC_target_gt)
    
    T_WO_target_gt = load_object_pose_world(cfg.data_root, cfg.target_frame)
    T_C_O_target_gt = T_CW_target_gt @ T_WO_target_gt

    K_torch = torch.from_numpy(K_target).float().to(device)
    T_C_O_target_gt_torch = torch.from_numpy(T_C_O_target_gt).float().to(device)
    H, W = image_target_gt.shape[:2]
    
    # Compute GT normal for visualization
    depth_target_gt_torch = torch.from_numpy(depth_target_gt).float().to(device)
    xyz_target_gt = unproject_depth(depth_target_gt_torch, K_torch, H, W)
    normal_target_gt, _ = d2n_tblr(xyz_target_gt.permute(2, 0, 1).unsqueeze(0))
    normal_target_gt = -normal_target_gt[0].permute(1, 2, 0) # point to camera
    normal_vis_gt = (normal_target_gt.cpu().numpy() + 1.0) / 2.0
    
    # 3. GS Render at target frame
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
    
    # Log GT and initial GS
    rr.log("gt/image", rr.Image(image_target_gt))
    rr.log("gt/depth", rr.Image(depth_vis_gt))
    rr.log("gt/normal", rr.Image((normal_vis_gt.clip(0, 1) * 255).astype(np.uint8)))
    
    rr.log("gs_init/depth", rr.Image(depth_vis_gs))
    rr.log("gs_init/normal", rr.Image((normal_vis_gs.clip(0, 1) * 255).astype(np.uint8)))
    
    # --- 3D Point Clouds in WORLD Frame ---
    # GT Point Cloud in World Frame
    pts_C_gt = xyz_target_gt.reshape(-1, 3).cpu().numpy()
    colors_gt = image_target_gt.reshape(-1, 3)
    valid_gt = (depth_target_gt.reshape(-1) > 0.01)
    pts_C_gt = pts_C_gt[valid_gt]
    colors_gt = colors_gt[valid_gt]
    pts_W_gt = (T_WC_target_gt[:3, :3] @ pts_C_gt.T).T + T_WC_target_gt[:3, 3]
    rr.log("world/object_gt", rr.Points3D(pts_W_gt, colors=colors_gt, radii=0.001))
    
    # GS Point Cloud from rendered depth in World Frame
    pts_C_gs_depth = unproject_depth(d_map_gs, K_torch, H, W).reshape(-1, 3).cpu().numpy()
    colors_gs_pc = image_gs_rendered.reshape(-1, 3)
    valid_gs_depth = (d_map_gs.reshape(-1).cpu().numpy() > 0.01) & (alpha_gs.reshape(-1).cpu().numpy() > 0.5)
    pts_C_gs_depth = pts_C_gs_depth[valid_gs_depth]
    colors_gs_pc = colors_gs_pc[valid_gs_depth]
    pts_W_gs_pc = (T_WC_target_gt[:3, :3] @ pts_C_gs_depth.T).T + T_WC_target_gt[:3, 3]
    rr.log("world/gs_depth_pc", rr.Points3D(pts_W_gs_pc, colors=colors_gs_pc, radii=0.001))
    
    # GS Original Point Cloud (now transformed to WORLD Frame)
    gs_pts_O = gs_params.means.cpu().numpy()
    gs_colors = (gs_params.colors.cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    # Transform from Object Frame to World Frame
    pts_W_gs_orig = (T_WO_target_gt[:3, :3] @ gs_pts_O.T).T + T_WO_target_gt[:3, 3]
    rr.log("world/object_gs", rr.Points3D(pts_W_gs_orig, colors=gs_colors, radii=0.001))
    
    # Log Cameras in World
    rr.log("world/camera_gt", rr.Pinhole(image_from_camera=K_target, width=W, height=H))
    rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WC_target_gt[:3, :3], translation=T_WC_target_gt[:3, 3]))
    rr.log("world/camera_gt/image", rr.Image(image_target_gt))
    
    # Coordinate frame visualization
    rr.log("world/camera_gt/frame", rr.Transform3D(relation=rr.TransformRelation.ParentFromChild))

    # 4. Pose Optimization
    image_ref_t = torch.from_numpy(image_init).float().to(device).permute(2, 0, 1) / 255.0
    image_gt_t = torch.from_numpy(image_target_gt).float().to(device).permute(2, 0, 1) / 255.0
    
    T_C_O_ref_torch = torch.from_numpy(T_C_O_init).float().to(device)
    
    # Initialize optimization from reference pose (frame 35)
    # We want to find the pose at frame 40.
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
            
            # For visualization, we also want the alpha map at the optimized pose
            _, alpha_opt = render_custom_attribute(gs_params, n_ref, T_C_O_est, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            warped = warped * alpha_opt

    warped_np = (warped.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    rr.log("warped/optimized", rr.Image(warped_np))
    
    # Log loss plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(losses)
    ax.set_title("Pose Optimization Loss")
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    rr.log("opt/pose_loss_plot", rr.Image(np.array(Image.open(buf))))
    plt.close(fig)
    
    # Print GT vs Estimated Pose
    print("\nPose Comparison at Frame 40:")
    print("GT T_CO:\n", T_C_O_target_gt)
    print("Est T_CO:\n", T_C_O_est.cpu().numpy())
    
    # Calculate and Log Init Camera Pose in World (starting guess for frame 40)
    T_WC_init = T_WO_target_gt @ np.linalg.inv(T_C_O_init)
    rr.log("world/camera_init", rr.Pinhole(image_from_camera=K_target, width=W, height=H))
    rr.log("world/camera_init", rr.Transform3D(mat3x3=T_WC_init[:3, :3], translation=T_WC_init[:3, 3]))
    rr.log("world/camera_init/frame", rr.Transform3D(relation=rr.TransformRelation.ParentFromChild))

    # Calculate Est Camera Pose in World: T_WC_est = T_WO_target_gt @ inv(T_C_O_est)
    T_WC_est = T_WO_target_gt @ np.linalg.inv(T_C_O_est.cpu().numpy())
    rr.log("world/camera_est", rr.Pinhole(image_from_camera=K_target, width=W, height=H))
    rr.log("world/camera_est", rr.Transform3D(mat3x3=T_WC_est[:3, :3], translation=T_WC_est[:3, 3]))
    rr.log("world/camera_est/frame", rr.Transform3D(relation=rr.TransformRelation.ParentFromChild))

    # 5. Geometry Refinement
    print("\nStarting Geometry Refinement...")
    # We need initial normal/distance maps at ref frame for the patches
    with torch.no_grad():
        n_ref, d_ref = gs_to_planar_params(gs_params, T_C_O_ref_torch)
        n_ref_map, alpha_ref_map = render_custom_attribute(gs_params, n_ref, T_C_O_ref_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_ref_3ch_map, _ = render_custom_attribute(gs_params, d_ref.repeat(1, 3), T_C_O_ref_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        d_ref_map = d_ref_3ch_map[0:1]
        
    n_refined_map, d_refined_map, refine_losses = optimize_patch_geometry(
        image_ref_t, image_gt_t, n_ref_map, d_ref_map, alpha_ref_map, T_C_O_ref_torch, T_C_O_est, K_torch, cfg
    )
    
    # Log refinement loss plot
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(refine_losses)
    ax.set_title("Geometry Refinement Loss (NCC)")
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    rr.log("opt/refine_loss_plot", rr.Image(np.array(Image.open(buf))))
    plt.close(fig)
    
    # 6. Final Refined Visualization
    with torch.no_grad():
        # Warp using refined patches
        # For simplicity, we use a global warp logic but with the refined maps
        # Note: manual_forward_homography_warp already takes maps.
        mask_ref = alpha_ref_map > 0.5
        n_refined_flat = n_refined_map.permute(1, 2, 0).reshape(-1, 3)
        d_refined_flat = d_refined_map.reshape(-1, 1)
        
        T_curr_ref = T_C_O_est @ torch.inverse(T_C_O_ref_torch)
        warped_refined = manual_forward_homography_warp(image_ref_t, n_refined_flat, d_refined_flat, T_curr_ref, K_torch, mask=mask_ref)
        
        # Alpha at optimized pose
        _, alpha_opt = render_custom_attribute(gs_params, n_ref, T_C_O_est, K_torch, W, H, cfg.near_plane, cfg.far_plane)
        warped_refined = warped_refined * alpha_opt
        
    warped_refined_np = (warped_refined.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
    rr.log("warped/refined", rr.Image(warped_refined_np))
    
    # Log refined normals and depth
    refined_normal_vis = (n_refined_map.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
    rr.log("gs_refined/normal", rr.Image((refined_normal_vis.clip(0, 1) * 255).astype(np.uint8)))
    
    # Log Refined 3D Point Cloud
    # Transform refined depth map to world
    pts_C_refined = unproject_depth(d_refined_map[0], K_torch, H, W).reshape(-1, 3).cpu().numpy()
    valid_refined = (d_refined_map[0].reshape(-1).cpu().numpy() > 0.01) & (alpha_ref_map.reshape(-1).cpu().numpy() > 0.5)
    pts_C_refined = pts_C_refined[valid_refined]
    
    # Colors from ref image sampled at these points
    colors_refined = image_init.reshape(-1, 3)[valid_refined]
    
    # Transform to world: P_W = T_WC_ref @ P_C_ref
    T_WO_init = load_object_pose_world(cfg.data_root, cfg.init_frame)
    ext_file_init = data_dir / "extrinsics" / f"{cfg.init_frame:06d}.npy"
    T_WC_init_actual = np.load(ext_file_init)
    
    pts_W_refined = (T_WC_init_actual[:3, :3] @ pts_C_refined.T).T + T_WC_init_actual[:3, 3]
    rr.log("world/gs_refined_pc", rr.Points3D(pts_W_refined, colors=colors_refined, radii=0.001))

    print("Finished.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
