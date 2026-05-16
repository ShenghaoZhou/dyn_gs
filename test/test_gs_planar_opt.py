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

@dataclass
class Config:
    data_root: str = "data/dtc_sample_dense"
    near_plane: float = 0.01
    far_plane: float = 10.0
    device: str = "cuda"
    opt_lr: float = 1e-4
    opt_steps: int = 100
    patience: int = 10
    convergence_threshold: float = 1e-9
    warp_type: str = "backward" # "backward" or "forward"

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

def manual_forward_homography_warp(image_ref_t, n_C, d, T_curr_ref, K, mask=None, colors_override=None):
    """
    Explicitly compute the forward warp of each pixel using the plane-induced homography formula.
    image_ref_t: [C, H, W] torch tensor
    n_C: [N, 3] normals in ref camera view
    d: [N, 1] distances in ref camera view
    """
    C, H, W = image_ref_t.shape
    if colors_override is not None:
        # If pre-filtered values are passed, H and W should be the TARGET image dimensions
        # We assume H, W are known or passed. Let's use 512, 512 as placeholder if missing, 
        # but here we'll just use the ones from image_ref_t if it's the right size.
        pass 

    device = n_C.device
    
    # 1. Get reference pixel coordinates and colors
    if colors_override is not None:
        # If pre-filtered values are passed, we need their original pixel positions
        # To keep it simple, let's assume n_C and d were flattened from HxW
        # and we know which indices were valid.
        # But wait, it's easier to just pass the mask to this function.
        # Let's revert to a cleaner version.
        pass

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
    
    # 2. Extract relative rotation and translation from ref to curr
    R = T_curr_ref[:3, :3]
    t = T_curr_ref[:3, 3]
    
    # 3. Apply the Homography formula u_i ~ H * u_0
    K_inv = torch.inverse(K)
    dir0 = (K_inv @ u0.t()).t() # [N, 3]
    
    n_dot_dir = torch.sum(n_C * dir0, dim=1, keepdim=True)
    d_safe = torch.where(torch.abs(d) > 1e-6, d, torch.ones_like(d))
    p1_scaled = torch.matmul(R, dir0.t()).t() - (n_dot_dir / d_safe) * t.view(1, 3)
    
    # Project to pixel coordinates in frame i
    u1_homog = torch.matmul(K, p1_scaled.t()).t()
    u1_pix = u1_homog[:, :2] / (u1_homog[:, 2:3] + 1e-8)
    z1_scaled = u1_homog[:, 2]
    
    # 4. Z-buffered Splatting (Forward Warp)
    valid = (u1_pix[:, 0] >= 0) & (u1_pix[:, 0] < W-1) & \
            (u1_pix[:, 1] >= 0) & (u1_pix[:, 1] < H-1) & \
            (z1_scaled > 0)
    
    u1_pix = u1_pix[valid]
    colors = colors[valid]
    z1_scaled = z1_scaled[valid]
    
    sort_idx = torch.argsort(z1_scaled, descending=True)
    u1_pix = u1_pix[sort_idx]
    colors = colors[sort_idx]
    
    u1_int = u1_pix.long()
    
    # To make this differentiable, we use bilinear splatting
    x = u1_pix[:, 0]
    y = u1_pix[:, 1]
    
    x0 = torch.floor(x).long()
    x1 = x0 + 1
    y0 = torch.floor(y).long()
    y1 = y0 + 1
    
    # Bilinear weights
    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x - x0.float()) * (y1.float() - y)
    wc = (x1.float() - x) * (y - y0.float())
    wd = (x - x0.float()) * (y - y0.float())
    
    # Boundary masks for the 4 neighbors
    mask00 = (x0 >= 0) & (x0 < W) & (y0 >= 0) & (y0 < H)
    mask10 = (x1 >= 0) & (x1 < W) & (y0 >= 0) & (y0 < H)
    mask01 = (x0 >= 0) & (x0 < W) & (y1 >= 0) & (y1 < H)
    mask11 = (x1 >= 0) & (x1 < W) & (y1 >= 0) & (y1 < H)
    
    warped = torch.zeros((H, W, C), device=device)
    
    # Accumulate colors using bilinear weights
    # Note: index_put_ with accumulate=True is differentiable w.r.t. values (wa*colors), 
    # but NOT w.r.t. indices (y0, x0). 
    # To get gradients for Pose, we need wa, wb, etc. to be differentiable.
    # Fortunately, they ARE differentiable w.r.t x and y!
    
    # We use a trick to avoid multiple passes if possible, but 4 passes is clearest
    for m, y_idx, x_idx, w in zip([mask00, mask10, mask01, mask11], 
                                  [y0, y0, y1, y1], 
                                  [x0, x1, x0, x1], 
                                  [wa, wb, wc, wd]):
        if m.any():
            # index_put_ accumulates values into the tensor
            warped.index_put_((y_idx[m], x_idx[m]), colors[m] * w[m].unsqueeze(-1), accumulate=True)

    return warped.permute(2, 0, 1)

def optimize_pose(image_ref_t, gs_params, image_gt_t, K_torch, T_C_W_ref, T_C_W_init, cfg, gt_mask_t=None):
    """
    Optimize T_C_W_curr to minimize photometric error.
    """
    H, W = image_gt_t.shape[1], image_gt_t.shape[2]
    
    # Pose parameterization
    # We optimize relative pose delta from T_C_W_init
    log_R = torch.zeros((1, 3), device=cfg.device, requires_grad=True)
    trans = torch.zeros(3, device=cfg.device, requires_grad=True)
    
    # Separate LRs for translation and rotation are often better
    # trans_lr = cfg.opt_lr
    # rot_lr = cfg.opt_lr * 0.3
    optimizer = torch.optim.AdamW([
        {'params': [trans], 'lr': cfg.opt_lr * 1.5},
        {'params': [log_R], 'lr': cfg.opt_lr}
    ])
    
    losses = []
    best_loss = float('inf')
    best_T_C_W = T_C_W_init.clone()
    patience_counter = 0
    
    # --- Pre-render surface parameters once ---
    with torch.no_grad():
        if cfg.warp_type == "backward":
            # For backward warp, we render normal/depth maps at the INITIAL target pose
            n_init_val, d_init_val = gs_to_planar_params(gs_params, T_C_W_init)
            n_map, alpha_map = render_custom_attribute(gs_params, n_init_val, T_C_W_init, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_3ch_map, _ = render_custom_attribute(gs_params, d_init_val.repeat(1, 3), T_C_W_init, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_map = d_3ch_map[0:1]
        else:
            # For forward warp, we render normal/depth maps at the REFERENCE pose
            n_ref_val, d_ref_val = gs_to_planar_params(gs_params, T_C_W_ref)
            n_map, alpha_ref_map = render_custom_attribute(gs_params, n_ref_val, T_C_W_ref, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_ref_3ch_map, _ = render_custom_attribute(gs_params, d_ref_val.repeat(1, 3), T_C_W_ref, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_map = d_ref_3ch_map[0:1]
            
            # For forward warp, we also need to render the alpha map at the INITIAL target pose 
            # for the photometric loss masking (or we can use backward alpha_map if preferred)
            _, alpha_map = render_custom_attribute(gs_params, n_ref_val, T_C_W_init, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            
            # Prepare flattened versions for forward warp
            valid_mask_ref = (alpha_ref_map[0] > 1e-3)
            n_ref_flat = n_map.permute(1, 2, 0).reshape(-1, 3)
            d_ref_flat = d_map.reshape(-1, 1)

    for step in range(cfg.opt_steps):
        optimizer.zero_grad()
        
        # Build Delta_T
        R_delta = so3_exp_map(log_R)[0]
        T_delta = torch.eye(4, device=cfg.device)
        T_delta[:3, :3] = R_delta
        T_delta[:3, 3] = trans
        
        T_C_W_curr = T_delta @ T_C_W_init
        
        # 1. Compute T_ref_curr = T_ref_c.
        T_ref_curr = T_C_W_ref @ torch.inverse(T_C_W_curr)
        
        # 2. Warping
        if cfg.warp_type == "backward":
            warped = manual_backward_homography_warp(image_ref_t, n_map, d_map, T_ref_curr, K_torch)
        else:
            T_curr_ref = torch.inverse(T_ref_curr)
            warped = manual_forward_homography_warp(image_ref_t, n_ref_flat, d_ref_flat, T_curr_ref, K_torch, mask=valid_mask_ref)

        # Use GT mask instead of rendered alpha_map if available
        if gt_mask_t is not None:
            warped = warped * gt_mask_t
        else:
            warped = warped * alpha_map 
        
        # 4. Loss
        if gt_mask_t is not None:
            # When using GT mask, we compare warped reference against current frame within the mask
            loss = F.l1_loss(warped, image_gt_t * gt_mask_t)
        else:
            loss = F.l1_loss(warped, image_gt_t * alpha_map) 
        
        if torch.isnan(loss):
            print("NaN loss detected")
            break
            
        loss.backward()
        optimizer.step()
        
        curr_loss = loss.item()
        losses.append(curr_loss)
        
        if curr_loss < best_loss:
            best_loss = curr_loss
            best_T_C_W = T_C_W_curr.detach().clone()
            patience_counter = 0
        else:
            patience_counter += 1
            
        # if patience_counter >= cfg.patience:
        #     break
            
        # if step > 0 and abs(losses[-2] - curr_loss) < cfg.convergence_threshold:
        #     break
            
    return best_T_C_W, losses

def main(cfg: Config):
    rr.init("gs_planar_opt")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    
    device = torch.device(cfg.device)
    data_dir = Path(cfg.data_root)
    
    K = np.load(data_dir / "intrinsics.npy")
    K_torch = torch.from_numpy(K).float().to(device)
    
    image_paths = sorted((data_dir / "image").glob("*.jpg"), key=lambda p: int(p.stem))
    
    # 1. Initialize from first frame
    ref_stem = image_paths[0].stem
    image_ref = np.array(cv2.imread(str(image_paths[0]))[..., ::-1])
    mask_ref = np.array(cv2.imread(str(data_dir / "mask" / f"{ref_stem}.png"), cv2.IMREAD_GRAYSCALE))
    depth_ref = np.load(data_dir / "obj_depth_gt" / f"{ref_stem}.npy")
    T_W_C_ref_gt = np.load(data_dir / "pose" / f"{ref_stem}.npy")
    T_C_W_ref_gt = np.linalg.inv(T_W_C_ref_gt)
    T_C_W_ref_torch = torch.from_numpy(T_C_W_ref_gt).float().to(device)
    
    # Initialize GS
    gsp = GaussianSuperPrimitive(image_ref, mask_ref, depth_ref, T_C_W_ref_gt, K)
    gs_params = gsp.gs_params
    
    H, W = image_ref.shape[:2]
    image_ref_t = torch.from_numpy(image_ref).float().to(device).permute(2, 0, 1) / 255.0
    

    
    # Tracking state
    T_C_W_prev = T_C_W_ref_torch.clone()
    image_prev_t = image_ref_t.clone()
    
    gt_cam_traj = []
    est_cam_traj = []
    
    for idx, img_path in enumerate(tqdm(image_paths)):
        stem = img_path.stem
        
        # reference frame for frame i is frame i-1
        T_C_W_prev_local = T_C_W_prev.clone()
        image_ref_t_local = image_prev_t.clone()

        rr.set_time("frame_idx", sequence=idx)
        rr.set_time("timestamp", sequence=int(stem))
        
        # Log the reference image for this tracking step
        rr.log("ref/image", rr.Image(image_ref_t_local.permute(1, 2, 0).cpu().numpy().clip(0, 1)))
        
        image_gt = np.array(cv2.imread(str(img_path))[..., ::-1])
        mask_path = data_dir / "mask" / f"{stem}.png"
        mask = np.array(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE))
        
        T_W_C_gt = np.load(data_dir / "pose" / f"{stem}.npy")
        T_C_W_gt = np.linalg.inv(T_W_C_gt)
        gt_cam_traj.append(T_W_C_gt[:3, 3])
        
        image_gt_t = torch.from_numpy(image_gt).float().to(device).permute(2, 0, 1) / 255.0
        # GT mask for current frame 
        gt_mask_t = torch.from_numpy(mask).float().to(device).unsqueeze(0) / 255.0
        
        if idx == 0:
            T_C_W_est = T_C_W_ref_torch
        else:
            # 2. Optimize pose using the LAST frame as reference
            # For frame i, we use frame i-1 as the reference image and reference pose
            T_C_W_init = T_C_W_prev # Start search from previous frame's estimated pose
            
            # Log initial alpha map for debugging to see if it starts far from the object
            with torch.no_grad():
                n_init, d_init = gs_to_planar_params(gs_params, T_C_W_init)
                _, alpha_init = render_custom_attribute(gs_params, n_init, T_C_W_init, K_torch, W, H, cfg.near_plane, cfg.far_plane)
                rr.log("opt/init_alpha_map", rr.Image(alpha_init.permute(1, 2, 0).cpu().numpy().clip(0, 1)))

            T_C_W_est, losses = optimize_pose(
                image_ref_t_local, gs_params, image_gt_t, K_torch, T_C_W_prev_local, T_C_W_init, cfg,
                gt_mask_t=gt_mask_t
            )
            
            # Log loss plot
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.plot(losses)
            ax.set_title(f"Pose Opt Loss (ref: frame {idx-1})")
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            rr.log("opt/loss_plot", rr.Image(np.array(Image.open(buf))))
            plt.close(fig)

        # Log visualization for current estimation
        rr.log("gt/image", rr.Image(image_gt))
        
        with torch.no_grad():
            n_curr, d_curr = gs_to_planar_params(gs_params, T_C_W_est)
            n_map, alpha_map = render_custom_attribute(gs_params, n_curr, T_C_W_est, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            
            d_3ch_map, _ = render_custom_attribute(gs_params, d_curr.repeat(1, 3), T_C_W_est, K_torch, W, H, cfg.near_plane, cfg.far_plane)
            d_map = d_3ch_map[0:1]
            
            # Visualize warping from the local reference frame (frame i-1)
            if cfg.warp_type == "backward":
                T_ref_curr = T_C_W_prev_local @ torch.inverse(T_C_W_est)
                warped = manual_backward_homography_warp(image_ref_t_local, n_map, d_map, T_ref_curr, K_torch)
            else:
                T_curr_ref = T_C_W_est @ torch.inverse(T_C_W_prev_local)
                # Render n_ref for visualization
                n_ref_val, d_ref_val = gs_to_planar_params(gs_params, T_C_W_prev_local)
                n_ref_map, alpha_ref_map = render_custom_attribute(gs_params, n_ref_val, T_C_W_prev_local, K_torch, W, H, cfg.near_plane, cfg.far_plane)
                d_ref_3ch_map, _ = render_custom_attribute(gs_params, d_ref_val.repeat(1, 3), T_C_W_prev_local, K_torch, W, H, cfg.near_plane, cfg.far_plane)
                d_ref_map = d_ref_3ch_map[0:1]
                
                n_ref_flat = n_ref_map.permute(1, 2, 0).reshape(-1, 3)
                d_ref_flat = d_ref_map.reshape(-1, 1)
                warped = manual_forward_homography_warp(image_ref_t_local, n_ref_flat, d_ref_flat, T_curr_ref, K_torch, mask=alpha_ref_map[0] > 1e-3)

            # Use current frame GT mask for visualization
            warped = warped * gt_mask_t
            
            warped_np = warped.permute(1, 2, 0).cpu().numpy()
            rr.log("opt/warped_image", rr.Image(warped_np.clip(0, 1)))
            
            normal_vis = (n_map.permute(1, 2, 0).cpu().numpy() + 1.0) / 2.0
            rr.log("opt/normal_map", rr.Image(normal_vis.clip(0, 1)))
            rr.log("opt/alpha_map", rr.Image(alpha_map.permute(1, 2, 0).cpu().numpy().clip(0, 1)))

        # Update pose for NEXT iteration
        T_C_W_prev = T_C_W_est.detach()
        # Update reference image for NEXT iteration
        image_prev_t = image_gt_t.detach()
        
        T_W_C_est = torch.inverse(T_C_W_est).detach().cpu().numpy()
        est_cam_traj.append(T_W_C_est[:3, 3])
            
        rr.log("world/camera_gt", rr.Transform3D(
            translation=T_W_C_gt[:3, 3], mat3x3=T_W_C_gt[:3, :3],
            relation=rr.TransformRelation.ParentFromChild
        ))
        rr.log("world/camera_est", rr.Transform3D(
            translation=T_W_C_est[:3, 3], mat3x3=T_W_C_est[:3, :3],
            relation=rr.TransformRelation.ParentFromChild
        ))
        rr.log("world/gt_traj", rr.LineStrips3D([np.array(gt_cam_traj)], colors=[[0, 255, 0]]))
        rr.log("world/est_traj", rr.LineStrips3D([np.array(est_cam_traj)], colors=[[255, 0, 0]]))
        
        t_err = np.linalg.norm(T_W_C_est[:3, 3] - T_W_C_gt[:3, 3])
        rr.log("error/t_err", rr.Scalars(t_err))

    print("Finished.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
