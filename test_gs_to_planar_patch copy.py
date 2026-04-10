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

def manual_homography_warp(image_ref, n_C, d, T_i0, K, mask):
    """
    Explicitly compute the forward warp of each pixel using the plane-induced homography formula.
    H = K * (R - t * n^T / d) * K_inv
    """
    H, W = image_ref.shape[:2]
    device = n_C.device
    
    # 1. Get reference pixel coordinates and colors
    y0, x0 = torch.where(mask)
    u0 = torch.stack([x0, y0, torch.ones_like(x0)], dim=1).float() # [N, 3] (homogeneous)
    colors = torch.from_numpy(image_ref[mask.cpu().numpy()]).float().to(device) / 255.0 # [N, 3]
    
    # 2. Extract relative rotation and translation from ref to curr
    R = T_i0[:3, :3]
    t = T_i0[:3, 3]
    
    # 3. Apply the Homography formula u_i ~ H * u_0
    # Specifically, for a pixel u0 corresponding to a 3D point P0 on a plane (n, d):
    # P1 = R * P0 + t = R * (z0 * K_inv * u0) + t
    # Since n^T * P0 + d = 0 => z0 * n^T * K_inv * u0 + d = 0 => z0 = -d / (n^T * K_inv * u0)
    # Substituting z0: P1 = (-d / (n^T * K_inv * u0)) * (R * K_inv * u0) + t
    # The projected pixel u1 = K * P1 / P1_z
    
    K_inv = torch.inverse(K)
    dir0 = (K_inv @ u0.t()).t() # [N, 3] Directions in camera-0 space
    
    # Compute P1 (scaled by 1/z0 relative to the standard formula for stability)
    # P1_scaled = R * dir0 - (t * (n^T * dir0) / d)
    # Note: we use our previously computed n_C and d in ref camera space.
    n_dot_dir = torch.sum(n_C * dir0, dim=1, keepdim=True)
    p1_scaled = torch.matmul(R, dir0.t()).t() - (n_dot_dir / d) * t.view(1, 3)
    
    # Project to pixel coordinates in frame i
    u1_homog = torch.matmul(K, p1_scaled.t()).t()
    u1_pix = u1_homog[:, :2] / u1_homog[:, 2:3]
    z1_scaled = u1_homog[:, 2] # Depth in target view (relative)
    
    # 4. Z-buffered Splatting (Forward Warp)
    # Filter points that fall outside the image or behind the camera
    valid = (u1_pix[:, 0] >= 0) & (u1_pix[:, 0] < W-1) & \
            (u1_pix[:, 1] >= 0) & (u1_pix[:, 1] < H-1) & \
            (z1_scaled > 0)
    
    u1_pix = u1_pix[valid]
    colors = colors[valid]
    z1_scaled = z1_scaled[valid]
    
    # Sort by depth so that foreground pixels are drawn over background pixels
    # (Since we use a simple overwrite splat, we want front-most to be indexed last, 
    # but index_put is non-deterministic for multiple indices. 
    # To be safe, we sort far-to-near and then write).
    sort_idx = torch.argsort(z1_scaled, descending=True)
    u1_pix = u1_pix[sort_idx]
    colors = colors[sort_idx]
    
    # Integer rounding for "hard" splat
    u1_int = u1_pix.long()
    
    # Construct the warped image
    warped = torch.zeros((H, W, 3), device=device)
    # y = u1_int[:, 1], x = u1_int[:, 0]
    warped[u1_int[:, 1], u1_int[:, 0]] = colors
    
    # Simple gap filling: splat to a 2x2 neighborhood to avoid tiny holes
    # (This is a "concrete" way to improve splat quality without a full rasterizer)
    for dx in [0, 1]:
        for dy in [0, 1]:
            u_neighbor = (u1_pix + torch.tensor([dx, dy], device=device)).long()
            valid_n = (u_neighbor[:, 0] < W) & (u_neighbor[:, 1] < H)
            warped[u_neighbor[valid_n, 1], u_neighbor[valid_n, 0]] = colors[valid_n]

    return warped.permute(2, 0, 1)

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
    
    # Pre-calculate planar parameters in reference camera space
    # (These are static as they are tied to the first frame pixels)
    T_C_W_ref_torch = torch.from_numpy(T_C_W_ref).float().to(device)
    n_C, d = gs_to_planar_params(gs_params, T_C_W_ref_torch)
    
    H, W = image_ref.shape[:2]
    mask_ref_torch = torch.from_numpy(mask_ref > 0).to(device)
    
    # Log reference frame for context
    rr.log("ref/image", rr.Image(image_ref), static=True)
    
    # 2. Loop over the sequence and warp frame 0 pixels manually
    print("Processing sequence with manual homography warping...")
    for idx, img_path in enumerate(tqdm(image_paths)):
        stem = img_path.stem
        rr.set_time("frame_idx", sequence=idx)
        rr.set_time("timestamp", sequence=int(stem))
        
        # Load current frame target pose
        image_gt = np.array(cv2.imread(str(img_path))[..., ::-1])
        T_W_C_curr = np.load(data_dir / "pose" / f"{stem}.npy")
        T_C_W_curr = np.linalg.inv(T_W_C_curr)
        
        # Relative pose from ref camera to current camera: T_curr_ref = T_curr_world * T_world_ref
        T_curr_ref = T_C_W_curr @ T_W_C_ref
        T_curr_ref_torch = torch.from_numpy(T_curr_ref).float().to(device)
        
        # 3. Perform manual homography forward warp
        warped_render = manual_homography_warp(
            image_ref, n_C, d, T_curr_ref_torch, K_torch, mask_ref_torch
        )
        
        # 4. Visualization
        rr.log("gt/image", rr.Image(image_gt))
        
        # Warped results
        warped_rgb = warped_render.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)
        rr.log("warped/image", rr.Image(warped_rgb))
        
        # Log camera pose for context in 3D
        rr.log("world/camera", rr.Transform3D(
            translation=T_W_C_curr[:3, 3],
            mat3x3=T_W_C_curr[:3, :3],
            relation=rr.TransformRelation.ParentFromChild
        ))

    print("Finished processing sequence.")

if __name__ == "__main__":
    main()
