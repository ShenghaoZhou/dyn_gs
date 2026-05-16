import os
import cv2
import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
import io
from PIL import Image
import matplotlib.pyplot as plt

from data import HOT3DDataLoader
from gs_dyn_obj.grouped_gs import GaussianSuperPrimitive, RGB2SH
from gs_dyn_obj.gs_param import GSParam
from gs_dyn_obj.utils.init import unproject_depth, d2n_tblr
from gs_dyn_obj.utils.vis import vis_2dgs_rerun
from gs_dyn_obj.gs_rendering import render_2dgs
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, quaternion_to_matrix

from run_single_view_loss import compute_single_view_loss
from run_multi_view_loss import compute_multi_view_loss

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    data_seq: str = "clip-003312"
    device: str = "cuda"
    lr: float = 1e-3
    num_steps_per_frame: int = 150 # Increased to accommodate pyramid
    kf_every: int = 5
    window_size: int = 4
    sample_num: int = 10000
    single_view_weight: float = 0.015
    multi_view_weight: float = 0.1
    densify_error_th: float = 0.05
    densify_every: int = 5
    near_plane: float = 0.01
    far_plane: float = 10.0
    use_gt_depth: bool = False
    use_pgsr: bool = True
    prune_opacity_th: float = 0.01
    prune_screen_size_th: float = 20.0
    
    # Pyramid Parameters
    pyr_levels: int = 2
    pyr_interval: int = 30 # Steps per pyramid level (coarsest to finest)
    
    # TSDF Parameters
    voxel_size: float = 0.002
    sdf_trunc: float = 0.01

class MiniCam:
    def __init__(self, K, extrin, width, height, ncc_scale=1.0):
        self.K = torch.from_numpy(K).float().cuda() if isinstance(K, np.ndarray) else K
        # world_view_transform should be the transpose of the world-to-camera (or object-to-camera) matrix
        extrin_t = torch.from_numpy(extrin).float().cuda() if isinstance(extrin, np.ndarray) else extrin
        self.world_view_transform = extrin_t.t().contiguous()
        self.width = width
        self.height = height
        self.ncc_scale = ncc_scale
        
    def get_k(self, scale=1.0):
        K = self.K.clone()
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] *= scale
        K[1, 2] *= scale
        return K
    
    def get_inv_k(self, scale=1.0):
        return torch.inverse(self.get_k(scale))
    
    def get_calib_matrix_nerf(self, scale=1.0):
        return self.get_k(scale), self.world_view_transform

def get_scaled_cam(cam, scale):
    new_K = cam.get_k(scale)
    new_w = int(cam.width * scale)
    new_h = int(cam.height * scale)
    # Extrinsics remain the same
    new_cam = MiniCam(new_K, cam.world_view_transform.t().cpu().numpy(), new_w, new_h, cam.ncc_scale)
    return new_cam

def get_lapla_norm(img, kernel=None):
    # img: [C, H, W]
    device = img.device
    laplacian_kernel = (
        torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float32
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    laplacian_kernel = laplacian_kernel.repeat(1, img.shape[0], 1, 1)
    laplacian = F.conv2d(img[None], laplacian_kernel, padding="same")
    laplacian_norm = torch.linalg.vector_norm(
        laplacian, ord=1, dim=1, keepdim=True)
    
    # Zero edges
    laplacian_norm[..., :, 0] = 0
    laplacian_norm[..., :, -1] = 0
    laplacian_norm[..., 0, :] = 0
    laplacian_norm[..., -1, :] = 0
    
    if kernel is not None:
        return F.conv2d(laplacian_norm, kernel, padding="same")[0, 0].clamp(0, 1)
    return laplacian_norm[0, 0].clamp(0, 1)

def load_model_infer(data_dir, frame_idx, target_size=None):
    depth_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
    mask_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
    
    depth = np.load(depth_path) if depth_path.exists() else None
    mask = np.array(Image.open(mask_path).convert("L")) if mask_path.exists() else None
    
    if target_size is not None:
        if depth is not None and depth.shape[:2] != target_size[::-1]:
            depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        if mask is not None and mask.shape[:2] != target_size[::-1]:
            mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
            
    return depth, mask

def build_rotation_from_normal(normal):
    # normal: [N, 3]
    device = normal.device
    num_pts = normal.shape[0]
    
    # 1. Create orthogonal basis
    # Pick a random vector for each point
    ref_axis1 = torch.randn(num_pts, 3, device=device)
    ref_axis1 = F.normalize(ref_axis1, dim=1)
    
    # Gram-Schmidt to get first axis orthogonal to normal
    rotation_axis1 = ref_axis1 - (normal * ref_axis1).sum(dim=1, keepdim=True) * normal
    rotation_axis1 = F.normalize(rotation_axis1, dim=1)
    
    # Second axis is cross product
    rotation_axis2 = torch.cross(normal, rotation_axis1, dim=1)
    rotation_axis2 = F.normalize(rotation_axis2, dim=1)
    
    # 2. Build rotation matrix [axis1, axis2, normal]
    # We want columns to be our basis vectors
    R = torch.stack([rotation_axis1, rotation_axis2, normal], dim=-1)
    
    # Ensure right-handedness (should be true by construction, but safer)
    det = torch.linalg.det(R)
    # If det is -1, flip one axis
    flip_mask = det < 0
    R[flip_mask, :, 1] *= -1
    
    return matrix_to_quaternion(R)

def main(cfg: Config):
    rr.init("obj_gs_reconst_pyr")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    
    data_loader = HOT3DDataLoader(Path(cfg.data_root) / cfg.data_seq, depth_model="GT" if cfg.use_gt_depth else None)
    device = torch.device(cfg.device)
    
    # Laplacian disc kernel
    radius = 3
    disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1, device=device)
    ky, kx = torch.meshgrid(
        torch.arange(-radius, radius + 1),
        torch.arange(-radius, radius + 1),
        indexing="ij",
    )
    disc_kernel[0, 0, torch.sqrt(kx**2 + ky**2) <= radius + 0.5] = 1
    disc_kernel = disc_kernel / disc_kernel.sum()

    # 1. Initialize from first frame
    frame_0 = data_loader[0]
    H, W = frame_0["image"].shape[:2]
    
    if cfg.use_gt_depth:
        depth_0 = frame_0.get("depth")
        mask_0 = frame_0["obj_mask"]
    else:
        depth_0, mask_0 = load_model_infer(data_loader.processed_data_dir, 0, target_size=(W, H))
        if depth_0 is None or mask_0 is None:
            depth_0 = frame_0.get("depth")
            mask_0 = frame_0["obj_mask"]
    
    T_CW_0 = frame_0["extrin"]
    K_0 = frame_0["K"]
    T_WO_0 = data_loader.get_obj_pose(0)
    
    # Initialize GSP in world frame
    gsp = GaussianSuperPrimitive(frame_0["image"], mask_0, depth_0, T_CW_0, K_0)
    gs_params = gsp.gs_params 
    
    # Transform to Object Frame
    T_OW_0 = np.linalg.inv(T_WO_0)
    T_OW_0_t = torch.from_numpy(T_OW_0).float().to(device)
    
    gs_params.means = torch.einsum('ij,nj->ni', T_OW_0_t[:3, :3], gs_params.means) + T_OW_0_t[:3, 3]
    quat_OW_0 = matrix_to_quaternion(T_OW_0_t[:3, :3])
    gs_params.quats = quaternion_multiply(quat_OW_0.unsqueeze(0), gs_params.quats)
    
    gs_params.scales = torch.log(gs_params.scales.clamp_min(1e-6))
    gs_params.opacity = torch.log(gs_params.opacity / (1 - gs_params.opacity))
    gs_params.colors = RGB2SH(gs_params.colors)
    
    # Make parameters optimizable
    gs_params.means.requires_grad = True
    gs_params.quats.requires_grad = True
    gs_params.opacity.requires_grad = True
    
    optimizer = torch.optim.Adam([
        {"params": [gs_params.means], "lr": 0.001},
        {"params": [gs_params.quats], "lr": 0.001},
        {"params": [gs_params.scales], "lr": 0.005},
        {"params": [gs_params.colors], "lr": 0.0025},
        {"params": [gs_params.opacity], "lr": 0.05},
    ])
    
    keyframes = [] 
    
    for idx in tqdm(range(len(data_loader))):
        rr.set_time("frame_idx", sequence=idx)
        frame = data_loader[idx]
        T_CW = frame["extrin"]
        T_WO = data_loader.get_obj_pose(idx)
        K = frame["K"]
        image = frame["image"]
        obj_mask = frame["obj_mask"]
        
        T_CO = T_CW @ T_WO
        T_CO_t = torch.from_numpy(T_CO).float().to(device)
        
        image_t = torch.from_numpy(image).float().to(device).permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(obj_mask > 0).bool().to(device)
        
        cam = MiniCam(K, T_CO, image.shape[1], image.shape[0])

        # --- Create Image & Mask Pyramid ---
        image_pyr = [image_t]
        mask_pyr = [mask_t.unsqueeze(0).float()]
        for _ in range(cfg.pyr_levels - 1):
            image_pyr.append(F.avg_pool2d(image_pyr[-1], 2))
            mask_pyr.append(F.avg_pool2d(mask_pyr[-1], 2))
        
        # Binary masks for loss calculation
        for i in range(len(mask_pyr)):
            mask_pyr[i] = (mask_pyr[i][0] > 0.5)
        
        # Optimization steps for current frame with Pyramid
        for step in range(cfg.num_steps_per_frame):
            # Calculate current pyramid level (from coarsest to finest)
            curr_pyr_lvl = max(0, (cfg.pyr_levels - 1) - (step // cfg.pyr_interval))
            
            target_image = image_pyr[curr_pyr_lvl]
            target_mask = mask_pyr[curr_pyr_lvl]
            
            scale = 1.0 / (2**curr_pyr_lvl)
            curr_w, curr_h = target_image.shape[2], target_image.shape[1]
            curr_K = cam.get_k(scale)
            curr_cam = get_scaled_cam(cam, scale)
            
            optimizer.zero_grad()
            
            # Render at current pyramid resolution
            render_image, render_depth, render_normal, render_alpha = render_2dgs(
                gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales),
                gs_params.colors, torch.sigmoid(gs_params.opacity),
                viewmat=T_CO_t, K=curr_K,
                width=curr_w, height=curr_h,
                near_plane=cfg.near_plane, far_plane=cfg.far_plane
            )
            
            # Photometric loss
            loss_photo = F.l1_loss(render_image * target_mask, target_image * target_mask)
            
            # Mask loss
            loss_mask = F.l1_loss(render_alpha * (~target_mask).float(), torch.zeros_like(render_alpha))
            
            # Single-view PGSR loss
            loss_single = torch.tensor(0.0, device=device)
            if cfg.use_pgsr:
                loss_single = compute_single_view_loss(
                    render_normal, render_depth, target_image, curr_cam, 
                    weight=cfg.single_view_weight, mask=target_mask
                )
            
            # Multi-view PGSR loss (always against previous keyframes at original resolution if needed)
            # Actually, compute_multi_view_loss uses pixels and cam.
            # If we use curr_cam, the warping should be handled by the library.
            loss_multi = torch.tensor(0.0, device=device)
            if cfg.use_pgsr and len(keyframes) > 0:
                kf_idx = np.random.randint(max(0, len(keyframes) - cfg.window_size), len(keyframes))
                kf_cam, kf_image_gray = keyframes[kf_idx]
                
                # Sample pixels from current (possibly downsampled) view
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(curr_h, device=device), 
                    torch.arange(curr_w, device=device), 
                    indexing='ij'
                )
                pixels = torch.stack([grid_x, grid_y], dim=-1).float().reshape(-1, 2)
                
                mask_indices = torch.where(target_mask.reshape(-1))[0]
                if len(mask_indices) > cfg.sample_num:
                    perm = torch.randperm(len(mask_indices), device=device)[:cfg.sample_num]
                    mask_indices = mask_indices[perm]
                
                if len(mask_indices) > 0:
                    # Current target is target_image (scaled)
                    # Keyframe is kf_image_gray (always full resolution)
                    ncc, ncc_mask = compute_multi_view_loss(
                        target_image.mean(0).unsqueeze(0), kf_image_gray.unsqueeze(0), 
                        render_normal, render_depth,
                        curr_cam, kf_cam,
                        pixels=pixels[mask_indices],
                        valid_indices=torch.arange(len(mask_indices), device=device)
                    )
                    loss_multi = ncc.mean() * cfg.multi_view_weight
            
            total_loss = loss_photo + loss_mask + loss_single + loss_multi
            total_loss.backward()
            optimizer.step()
        
        # Densification (happens at full resolution for best quality)
        if idx > 0 and idx % cfg.densify_every == 0:
            with torch.no_grad():
                render_image, render_depth, _, render_alpha = render_2dgs(
                    gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales),
                    gs_params.colors, torch.sigmoid(gs_params.opacity),
                    viewmat=T_CO_t, K=cam.K,
                    width=cam.width, height=cam.height
                )
                
                if cfg.use_gt_depth:
                    depth_curr = frame.get("depth")
                    mask_curr = frame["obj_mask"]
                else:
                    depth_curr, mask_curr = load_model_infer(data_loader.processed_data_dir, idx, target_size=(cam.width, cam.height))
                
                if depth_curr is not None:
                    depth_curr_t = torch.from_numpy(depth_curr).float().to(device)
                    
                    if not cfg.use_gt_depth:
                        mask_bool = torch.from_numpy(mask_curr).to(device) > 0
                        overlap_mask = (render_depth[0] > 0) & (depth_curr_t > 0) & mask_bool
                        if overlap_mask.sum() > 100:
                            scales = render_depth[0][overlap_mask] / depth_curr_t[overlap_mask]
                            median_scale = torch.median(scales)
                            depth_curr_t = depth_curr_t * median_scale
                            
                    init_proba = get_lapla_norm(image_t, disc_kernel)
                    penalty = get_lapla_norm(render_image, disc_kernel)
                    
                    sample_mask = (torch.rand_like(init_proba) < (init_proba - penalty) * 2.0)
                    sample_mask = sample_mask & mask_t & (depth_curr_t > 0)
                    
                    render_depth_val = render_depth[0]
                    depth_closer = (depth_curr_t < (render_depth_val - 0.01)) | (render_depth_val == 0)
                    sample_mask = sample_mask | (depth_closer & mask_t & (depth_curr_t > 0) & (torch.rand_like(init_proba) < 0.2))
                    
                    if sample_mask.sum() > 0:
                        y, x = torch.where(sample_mask)
                        if len(y) > 2000:
                            perm = torch.randperm(len(y), device=device)[:2000]
                            y, x = y[perm], x[perm]
                        
                        z = depth_curr_t[y, x]
                        fx, fy = cam.K[0, 0], cam.K[1, 1]
                        cx, cy = cam.K[0, 2], cam.K[1, 2]
                        
                        xc = (x.float() - cx) * z / fx
                        yc = (y.float() - cy) * z / fy
                        pts_c = torch.stack([xc, yc, z], dim=-1)
                        
                        full_pts_c = unproject_depth(depth_curr_t, cam.K, cam.height, cam.width)
                        normals_c, valid_n = d2n_tblr(full_pts_c.permute(2, 0, 1).unsqueeze(0))
                        normals_c = -normals_c[0] 
                        normals_c = F.normalize(normals_c, dim=0)
                        
                        new_normals_c = normals_c[:, y, x].permute(1, 0)
                        
                        T_OC_t = torch.inverse(T_CO_t)
                        pts_o = torch.einsum('ij,nj->ni', T_OC_t[:3, :3], pts_c) + T_OC_t[:3, 3]
                        new_normals_o = torch.einsum('ij,nj->ni', T_OC_t[:3, :3], new_normals_c)
                        new_normals_o = F.normalize(new_normals_o, dim=1)
                        
                        new_means = pts_o
                        new_colors = RGB2SH(image_t[:, y, x].permute(1, 0))
                        new_quats = build_rotation_from_normal(new_normals_o)
                        
                        sampled_init_proba = init_proba[y, x].clamp_min(1e-6)
                        pixel_scales = (1.0 / torch.sqrt(sampled_init_proba)).clamp(2.0, cam.width / 5.0)
                        new_sizes = pixel_scales * (z / ((fx+fy)/2))
                        new_scales = torch.log(new_sizes.view(-1, 1).repeat(1, 2).clamp(1e-6, 1e6))
                        
                        new_opacity = torch.ones((len(new_means), 1), device=device) * 0.1
                        new_opacity = torch.log(new_opacity / (1 - new_opacity)) 
                        
                        gs_params.means = torch.nn.Parameter(torch.cat([gs_params.means.data, new_means], dim=0))
                        gs_params.quats = torch.nn.Parameter(torch.cat([gs_params.quats.data, new_quats], dim=0))
                        gs_params.scales = torch.nn.Parameter(torch.cat([gs_params.scales.data, new_scales], dim=0))
                        gs_params.colors = torch.nn.Parameter(torch.cat([gs_params.colors.data, new_colors], dim=0))
                        gs_params.opacity = torch.nn.Parameter(torch.cat([gs_params.opacity.data, new_opacity], dim=0))
                        
                        optimizer = torch.optim.Adam([
                            {"params": [gs_params.means], "lr": 0.001},
                            {"params": [gs_params.quats], "lr": 0.001},
                            {"params": [gs_params.scales], "lr": 0.005},
                            {"params": [gs_params.colors], "lr": 0.0025},
                            {"params": [gs_params.opacity], "lr": 0.05},
                        ])
                        print(f"Densified GS: added {len(new_means)} points. Total: {len(gs_params.means)}")

        # Update keyframes (always full resolution)
        if idx % cfg.kf_every == 0:
            keyframes.append((cam, image_t.mean(0).detach()))
            if len(keyframes) > 100: 
                keyframes.pop(0)

        # Pruning
        with torch.no_grad():
            means_c = torch.einsum('ij,nj->ni', T_CO_t[:3, :3], gs_params.means) + T_CO_t[:3, 3]
            dist = means_c[:, 2].clamp_min(0.01) 
            
            curr_scales = torch.exp(gs_params.scales)
            max_scales = curr_scales.max(dim=-1)[0]
            f = cam.K[0, 0]
            screen_size = f * max_scales / dist
            
            valid_mask = (torch.sigmoid(gs_params.opacity.squeeze(-1)) > cfg.prune_opacity_th) & (screen_size < cfg.prune_screen_size_th * cam.width)
            
            n_before = len(gs_params.means)
            if valid_mask.sum() == 0:
                valid_mask = torch.ones_like(valid_mask)
            
            gs_params.means = torch.nn.Parameter(gs_params.means[valid_mask])
            gs_params.quats = torch.nn.Parameter(gs_params.quats[valid_mask])
            gs_params.scales = torch.nn.Parameter(gs_params.scales[valid_mask])
            gs_params.colors = torch.nn.Parameter(gs_params.colors[valid_mask])
            gs_params.opacity = torch.nn.Parameter(gs_params.opacity[valid_mask])
            n_after = len(gs_params.means)
            if n_before != n_after:
                optimizer = torch.optim.Adam([
                    {"params": [gs_params.means], "lr": 0.001},
                    {"params": [gs_params.quats], "lr": 0.001},
                    {"params": [gs_params.scales], "lr": 0.005},
                    {"params": [gs_params.colors], "lr": 0.0025},
                    {"params": [gs_params.opacity], "lr": 0.05},
                ])

        # Logging
        if idx % 1 == 0:
            # Re-render at full resolution for logging
            with torch.no_grad():
                render_image_full, _, render_normal_full, _ = render_2dgs(
                    gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales),
                    gs_params.colors, torch.sigmoid(gs_params.opacity),
                    viewmat=T_CO_t, K=cam.K,
                    width=cam.width, height=cam.height
                )
            
            rr.log("color", rr.Image(image))
            rr.log("render", rr.Image(render_image_full.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1)))
            rr.log("normal", rr.Image(((render_normal_full.permute(1, 2, 0).detach().cpu().numpy() + 1) / 2 * 255).astype(np.uint8)))
            
            with torch.no_grad():
                sh_colors = gs_params.colors.detach()
                C0 = 0.28209479177387814
                colors_rgb = (sh_colors * C0 + 0.5).clamp(0, 1).cpu().numpy()
                rr.log("object/gs", rr.Points3D(
                    gs_params.means.detach().cpu().numpy(), 
                    colors=(colors_rgb * 255).astype(np.uint8)
                ))
            rr.log("object/cam", rr.Transform3D(translation=T_CO[:3, 3], mat3x3=T_CO[:3, :3], relation=rr.TransformRelation.ChildFromParent))
            rr.log("object/cam/image", rr.Pinhole(resolution=(image.shape[1], image.shape[0]), image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF))
            
    # --- TSDF Fusion ---
    print("Running TSDF Fusion...")
    import open3d as o3d
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.voxel_size,
        sdf_trunc=cfg.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for idx in tqdm(range(len(data_loader)), desc="Fusing"):
        frame = data_loader[idx]
        T_CW = frame["extrin"]
        T_WO = data_loader.get_obj_pose(idx)
        T_CO = T_CW @ T_WO
        T_CO_t = torch.from_numpy(T_CO).float().to(device)
        K = frame["K"]
        image = frame["image"]
        
        with torch.no_grad():
            render_image, render_depth, _, _ = render_2dgs(
                gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales),
                gs_params.colors, torch.sigmoid(gs_params.opacity),
                viewmat=T_CO_t, K=torch.from_numpy(K).float().to(device),
                width=image.shape[1], height=image.shape[0],
                near_plane=cfg.near_plane, far_plane=cfg.far_plane
            )
            color_np = (render_image.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
            depth_np = render_depth[0].cpu().numpy()
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                image.shape[1], image.shape[0], K[0,0], K[1,1], K[0,2], K[1,2]
            )
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(color_np),
                o3d.geometry.Image(depth_np),
                depth_scale=1.0,
                depth_trunc=cfg.far_plane,
                convert_rgb_to_intensity=False
            )
            volume.integrate(rgbd, intrinsic, T_CO)

    print("Extracting Mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    rr.log("object/mesh", rr.Mesh3D(
        vertex_positions=np.asarray(mesh.vertices),
        triangle_indices=np.asarray(mesh.triangles),
        vertex_colors=np.asarray(mesh.vertex_colors),
        vertex_normals=np.asarray(mesh.vertex_normals)
    ), static=True)

    print("Reconstruction Complete.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
