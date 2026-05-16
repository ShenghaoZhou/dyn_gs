import numpy as np
import cv2
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from gs_dyn_obj.grouped_gs import GaussianSuperPrimitive
from gs_dyn_obj.gs_rendering_gsplat import render_2dgs
from pytorch3d.transforms import quaternion_to_matrix
import matplotlib.pyplot as plt

def depth_to_rgb(depth, min_val=0.0, max_val=2.0):
    """Consistent colormapping for depth visualization."""
    depth_norm = np.clip((depth - min_val) / (max_val - min_val), 0, 1)
    colormap = plt.get_cmap("magma")
    rgb = (colormap(depth_norm)[..., :3] * 255).astype(np.uint8)
    return rgb

def gs_to_planar_params(gs_params, T_CW):
    """
    Convert GS parameters to planar surface parameters (normal and distance) in camera view.
    Used here to demonstrate the connection between GS and plane-induced homography.
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

def manual_backward_homography_warp(image_ref, n_curr, d_curr, T_ref_curr, K):
    """
    Even faster and higher quality implementation using F.grid_sample.
    Instead of splatting frame 0 pixels forward, we loop over each pixel in the 
    CURRENT frame, use its rendered planar parameters (n, d) to find the 
    corresponding pixel in the REFERENCE frame via homography, and sample it.
    """
    H, W = n_curr.shape[1], n_curr.shape[2]
    device = n_curr.device
    
    # 1. Create a grid of current pixel coordinates [H, W, 2]
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=device), 
        torch.arange(W, device=device), 
        indexing='ij'
    )
    u1 = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).float().reshape(-1, 3) # [HW, 3]
    
    # 2. Extract relative rotation and translation from current to reference
    # T_ref_curr = [R_inv | t_inv]
    R_inv = T_ref_curr[:3, :3]
    t_inv = T_ref_curr[:3, 3]
    
    # 3. Compute reference coordinates u0 = H_inv * u1
    # For a target pixel u1 with normal n1 and distance d1 in current view:
    # P1 = z1 * K_inv * u1
    # n1^T * P1 + d1 = 0 => z1 = -d1 / (n1^T * K_inv * u1)
    # P0 = R_inv * P1 + t_inv
    #    = z1 * R_inv * K_inv * u1 + t_inv
    #    = (-d1 / (n1^T * K_inv * u1)) * R_inv * K_inv * u1 + t_inv
    
    K_inv = torch.inverse(K)
    dir1 = (K_inv @ u1.t()).t() # [HW, 3] Directions in current camera space
    
    # Gather rendered normals and distances
    n1 = n_curr.permute(1, 2, 0).reshape(-1, 3) # [HW, 3]
    d1 = d_curr.reshape(-1, 1) # [HW, 1]
    
    # Avoid division by zero for background pixels where d1 is zero
    # These pixels will later be masked out using alpha
    d1_safe = torch.where(torch.abs(d1) > 1e-6, d1, torch.ones_like(d1))
    
    # Compute P0 (scaled by 1/z1 for stability, just like in forward warp)
    # u0 ~ R_inv * dir1 - (t_inv * (n1^T * dir1) / d1)
    n1_dot_dir1 = torch.sum(n1 * dir1, dim=1, keepdim=True)
    p0_scaled = torch.matmul(R_inv, dir1.t()).t() - (n1_dot_dir1 / d1_safe) * t_inv.view(1, 3)
    
    # Project to reference pixel coordinates
    u0_homog = torch.matmul(K, p0_scaled.t()).t()
    u0_pix = u0_homog[:, :2] / (u0_homog[:, 2:3] + 1e-8)
    
    # 4. Use grid_sample to sample the reference image
    # Normalize u0_pix to [-1, 1] for grid_sample
    grid = u0_pix.reshape(1, H, W, 2)
    grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
    
    # Samples expect [B, C, H, W]
    image_ref_t = torch.from_numpy(image_ref).float().to(device).permute(2, 0, 1).unsqueeze(0) / 255.0
    warped = F.grid_sample(image_ref_t, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return warped[0]

def render_custom_attribute(gs_params, attr, T_CW, K, width, height, near_plane=0.01, far_plane=100.0):
    """
    Render a custom attribute using gsplat rasterizer.
    """
    from gsplat import rasterization_2dgs
    viewmats = T_CW.unsqueeze(0).unsqueeze(0).contiguous()
    Ks = K.unsqueeze(0).unsqueeze(0).contiguous()
    means = gs_params.means.unsqueeze(0).contiguous()
    quats = gs_params.quats.unsqueeze(0).contiguous()
    
    # 2DGS expects 3D scales but uses only first two. Last one should be small or zero.
    if gs_params.scales.shape[-1] == 2:
        scales = torch.cat([gs_params.scales, torch.zeros_like(gs_params.scales[..., :1])], dim=-1).unsqueeze(0).contiguous()
    else:
        scales = gs_params.scales.unsqueeze(0).contiguous()
        
    opacities = gs_params.opacity.squeeze(-1).unsqueeze(0).contiguous()
    
    # Attr is the 'color' to render
    colors = attr.unsqueeze(0).unsqueeze(0).contiguous()
    
    render_colors, render_alphas, _, _, _, _, _ = rasterization_2dgs(
        means, quats, scales, opacities, colors,
        viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane,
        render_mode="RGB"
    )
    return render_colors[0, 0].permute(2, 0, 1), render_alphas[0, 0].permute(2, 0, 1)

def main():
    # Initialize Rerun and connect to the proxy
    rr.init("gs_homography_warp")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "data/dtc_sample"
    data_dir = Path(data_root)
    
    # Load intrinsics
    K = np.load(data_dir / "intrinsics.npy")
    K_torch = torch.from_numpy(K).float().to(device)
    
    # Get sequence of frames
    image_paths = sorted((data_dir / "image").glob("*.jpg"), key=lambda p: int(p.stem))
    if not image_paths:
        print(f"No images found in {data_dir / 'image'}")
        return
        
    # 1. Initialize GS from the first frame
    ref_stem = image_paths[0].stem
    print(f"Initializing GS from reference frame {ref_stem}...")
    
    image_ref = np.array(cv2.imread(str(image_paths[0]))[..., ::-1])
    mask_ref = np.array(cv2.imread(str(data_dir / "mask" / f"{ref_stem}.png"), cv2.IMREAD_GRAYSCALE))
    depth_ref = np.load(data_dir / "obj_depth_gt" / f"{ref_stem}.npy")
    T_W_C_ref = np.load(data_dir / "pose" / f"{ref_stem}.npy")
    T_C_W_ref = np.linalg.inv(T_W_C_ref)
    
    # GaussianSuperPrimitive creates one 2DGS per masked pixel
    gsp = GaussianSuperPrimitive(image_ref, mask_ref, depth_ref, T_C_W_ref, K)
    gs_params = gsp.gs_params
    
    H, W = image_ref.shape[:2]
    
    # Log reference frame for context
    rr.log("ref/image", rr.Image(image_ref), static=True)
    
    # 2. Loop over the sequence and warp frame 0 pixels manually
    print("Processing sequence with manual backward homography warping...")
    for idx, img_path in enumerate(tqdm(image_paths)):
        stem = img_path.stem
        rr.set_time("frame_idx", sequence=idx)
        rr.set_time("timestamp", sequence=int(stem))
        
        # Load current frame target pose
        image_gt = np.array(cv2.imread(str(img_path))[..., ::-1])
        T_W_C_curr = np.load(data_dir / "pose" / f"{stem}.npy")
        T_C_W_curr = np.linalg.inv(T_W_C_curr)
        T_C_W_curr_torch = torch.from_numpy(T_C_W_curr).float().to(device)
        
        # --- Backward Mapping Concept ---
        # Instead of pushing pixels from frame 0, we pull them for each pixel in frame i.
        # This requires knowing the plane (n, d) at each pixel in the current view.
        
        # 3. Render the curr view's normal and distance maps
        n_curr, d_curr = gs_to_planar_params(gs_params, T_C_W_curr_torch)
        
        n_map, alpha_map = render_custom_attribute(gs_params, n_curr, T_C_W_curr_torch, K_torch, W, H)
        d_3ch_map, _ = render_custom_attribute(gs_params, d_curr.repeat(1, 3), T_C_W_curr_torch, K_torch, W, H)
        d_map = d_3ch_map[0:1] # Distance is scalar
        
        # 4. Perform manual backward homography warp using grid_sample
        # T_ref_curr = T_ref_world * T_world_curr = T_W_C_ref.inv() * T_W_C_curr
        T_ref_curr = np.linalg.inv(T_W_C_ref) @ T_W_C_curr
        T_ref_curr_torch = torch.from_numpy(T_ref_curr).float().to(device)
        
        warped_render = manual_backward_homography_warp(
            image_ref, n_map, d_map, T_ref_curr_torch, K_torch
        )
        # Apply alpha mask to clean up background
        warped_render = warped_render * alpha_map
        
        # 5. Visualization
        rr.log("gt/image", rr.Image(image_gt))
        
        # Warped results
        warped_rgb = warped_render.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        rr.log("warped/image", rr.Image(warped_rgb))
        
        # Geometry context
        n_vis = (n_map.detach().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
        rr.log("warped/normal_map", rr.Image(n_vis.clip(0, 1)))
        
        # Log camera pose for context in 3D
        rr.log("world/camera", rr.Transform3D(
            translation=T_W_C_curr[:3, 3],
            mat3x3=T_W_C_curr[:3, :3],
            relation=rr.TransformRelation.ParentFromChild
        ))

    print("Finished processing sequence.")

if __name__ == "__main__":
    main()
