import numpy as np
import cv2
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
from gs_dyn_obj.grouped_gs import GaussianSuperPrimitive
from gs_dyn_obj.gs_rendering_gsplat import render_2dgs
from pytorch3d.transforms import quaternion_to_matrix
import matplotlib.pyplot as plt
from gsplat import rasterization_2dgs

def depth_to_rgb(depth, min_val=0.0, max_val=2.0):
    """Consistent colormapping for depth visualization."""
    depth_norm = np.clip((depth - min_val) / (max_val - min_val), 0, 1)
    colormap = plt.get_cmap("magma")
    rgb = (colormap(depth_norm)[..., :3] * 255).astype(np.uint8)
    return rgb

def gs_to_planar_params(gs_params, T_CW):
    """
    Convert GS parameters to planar surface parameters (normal and distance) in camera view.
    Args:
        gs_params: Object with 'means' and 'quats'
        T_CW: [4, 4] camera-from-world transform
    Returns:
        n_C: [N, 3] normal in camera space
        d: [N, 1] distance such that n_C^T p_C + d = 0
    """
    means = gs_params.means # [N, 3]
    quats = gs_params.quats # [N, 4]
    
    # Get rotation matrices from quaternions (World from GS-local)
    R_WG = quaternion_to_matrix(quats) # [N, 3, 3]
    
    # Normal in world space (third column of R)
    n_W = R_WG[:, :, 2] # [N, 3]
    
    # Camera rotation and translation
    R_CW = T_CW[:3, :3]
    t_CW = T_CW[:3, 3]
    
    # Normal in camera space: n_C = R_CW @ n_W
    n_C = torch.einsum('ij,nj->ni', R_CW, n_W) # [N, 3]
    
    # GS center in camera space: p_C = R_CW @ p_W + t_CW
    p_C = torch.einsum('ij,nj->ni', R_CW, means) + t_CW # [N, 3]
    
    # Plane equation: n_C^T p_C + d = 0  => d = -n_C^T p_C
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
    rr.init("gs_to_planar_patch")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = "data/dtc_sample"
    data_dir = Path(data_root)
    
    # Load first frame data
    K = np.load(data_dir / "intrinsics.npy")
    image_paths = sorted((data_dir / "image").glob("*.jpg"), key=lambda p: int(p.stem))
    if not image_paths:
        print(f"No images found in {data_dir / 'image'}")
        return
        
    stem = image_paths[0].stem
    print(f"Loading frame {stem}...")
    
    image = np.array(cv2.imread(str(image_paths[0]))[..., ::-1])
    mask = np.array(cv2.imread(str(data_dir / "mask" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
    depth_gt = np.load(data_dir / "obj_depth_gt" / f"{stem}.npy")
    T_W_C = np.load(data_dir / "pose" / f"{stem}.npy")
    T_C_W = np.linalg.inv(T_W_C)
    
    # Initialize 2DGS (logic from GaussianSuperPrimitive)
    print("Initializing 2DGS...")
    gsp = GaussianSuperPrimitive(image, mask, depth_gt, T_C_W, K)
    gs_params = gsp.gs_params
    
    # Move to GPU
    K_torch = torch.from_numpy(K).float().to(device)
    T_C_W_torch = torch.from_numpy(T_C_W).float().to(device)
    
    H, W = image.shape[:2]
    
    # 1. Render RGB, Depth, Normal using gsplat backend
    print("Rendering with gsplat...")
    render_image, render_depth, render_normal, render_alpha = render_2dgs(
        gs_params.means, gs_params.quats, gs_params.scales,
        gs_params.colors, gs_params.opacity,
        viewmat=T_C_W_torch,
        K=K_torch,
        width=W, height=H
    )
    
    # 2. Recover planar parameters n_C, d from GS parameters
    print("Recovering planar parameters...")
    n_C, d = gs_to_planar_params(gs_params, T_C_W_torch)
    
    # 3. Render recovered normal and distance maps for comparison
    print("Rendering recovered maps...")
    rec_normal_map, _ = render_custom_attribute(gs_params, n_C, T_C_W_torch, K_torch, W, H)
    
    # Render distance map (repeat d to 3 channels for RGB renderer)
    d_3 = d.repeat(1, 3)
    rec_dist_map_3ch, _ = render_custom_attribute(gs_params, d_3, T_C_W_torch, K_torch, W, H)
    rec_dist_map = rec_dist_map_3ch[0:1] # Distance is scalar
    
    # 4. Visualization in Rerun
    print("Logging to Rerun...")
    # GT inputs
    rr.log("gt/image", rr.Image(image))
    rr.log("gt/depth", rr.Image(depth_to_rgb(depth_gt)))
    
    # Rendered outputs
    rr.log("render/image", rr.Image(render_image.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)))
    rr.log("render/depth", rr.DepthImage(render_depth.detach().cpu().squeeze().numpy()))
    # gsplat normals are in camera space. Map [-1, 1] to [0, 1] for visualization.
    normal_vis = (render_normal.detach().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
    rr.log("render/normal", rr.Image(normal_vis.clip(0, 1)))
    
    # Recovered outputs
    rec_normal_vis = (rec_normal_map.detach().cpu().permute(1, 2, 0).numpy() + 1.0) / 2.0
    rr.log("recovered/normal", rr.Image(rec_normal_vis.clip(0, 1)))
    rr.log("recovered/dist", rr.DepthImage(rec_dist_map.detach().cpu().squeeze().numpy()))
    
    # Difference maps
    normal_diff = torch.norm(render_normal - rec_normal_map, dim=0).detach().cpu().numpy()
    rr.log("diff/normal_norm", rr.DepthImage(normal_diff))
    
    print("Finished.")

if __name__ == "__main__":
    main()
