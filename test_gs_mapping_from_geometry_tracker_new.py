import numpy as np
import cv2
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import rerun.blueprint as rrb
import random
import threading
import time
import json
import open3d as o3d

from geometric_tracker import GeometricTracker
from obj_gs_mapping import GSMapping, MappingConfig, MiniCam, get_scaled_cam, build_rotation_from_normal
from gs_dyn_obj.gs_param import GSParam
from gs_dyn_obj.grouped_gs import RGB2SH
from gs_dyn_obj.gs_rendering import render_2dgs
from gs_dyn_obj.utils.init import unproject_depth, d2n_tblr

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    init_frame: int = 0
    n_frames: int = 50
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    no_vis: bool = False
    use_ray_dist: bool = True
    run_tsdf: bool = True
    voxel_size: float = 0.002
    sdf_trunc: float = 0.01

def load_object_pose_world(data_root, frame_idx):
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists(): return None
    with open(poses_file, "r") as f: lines = f.readlines()
    if frame_idx >= len(lines): return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])]) # x y z w
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4)
    T_WO[:3, :3] = r_WO; T_WO[:3, 3] = t_WO
    return T_WO

def load_frame_data(data_dir, frame_idx):
    stem = f"{frame_idx:06d}"
    img_path = data_dir / "images" / f"{frame_idx:05d}.jpg"
    if not img_path.exists():
        img_path = data_dir / "images" / f"{stem}.png"
    if not img_path.exists(): return None
    
    img = np.array(cv2.imread(str(img_path))[..., ::-1])
    mask_est_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
    depth_est_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
    
    if not mask_est_path.exists():
        mask_est_path = data_dir / "obj_masks" / f"{stem}.png"
    
    if not mask_est_path.exists(): return None
    
    mask = np.array(cv2.imread(str(mask_est_path), cv2.IMREAD_GRAYSCALE))
    depth = np.load(depth_est_path) if depth_est_path.exists() else None
    
    if depth is not None and depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Try to load K and T from various sources
    K_path = data_dir / "intrinsics" / f"{stem}.npy"
    if not K_path.exists():
        # Load from metadata.json if available
        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                K = np.array(meta["K"])
                T_CW_gt = np.eye(4) 
        else:
            K = np.eye(3); T_CW_gt = np.eye(4)
    else:
        K = np.load(K_path)
        T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        
    T_WO_gt = load_object_pose_world(data_dir, frame_idx)
    return {"image": img, "mask": mask, "depth": depth, "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": frame_idx}

def init_gs_from_tracker_points(points, colors, device, normals=None, ray_o=None, ray_d=None, ray_dist=None):
    num_pts = points.shape[0]
    means = torch.from_numpy(points).float().to(device); means.requires_grad = True
    colors_sh = RGB2SH(torch.from_numpy(colors).float().to(device)); colors_sh.requires_grad = True
    if normals is not None:
        quats = build_rotation_from_normal(torch.from_numpy(normals).float().to(device))
    else:
        quats = torch.zeros((num_pts, 4), device=device); quats[:, 0] = 1.0
    quats.requires_grad = True
    scales = torch.log(torch.ones((num_pts, 2), device=device) * 0.002); scales.requires_grad = True
    opacity = torch.logit(torch.ones((num_pts, 1), device=device) * 0.5); opacity.requires_grad = True
    
    # Ray parameters
    if ray_o is not None: ray_o = ray_o.to(device)
    if ray_d is not None: ray_d = ray_d.to(device)
    if ray_dist is not None: 
        ray_dist = ray_dist.to(device)
        ray_dist.requires_grad = True
        
    return GSParam(means, quats, scales, colors_sh, opacity, ray_o, ray_d, ray_dist)

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("test_gs_mapping_consolidated", spawn=False)
        if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
    
    data_dir = Path(cfg.data_root) / cfg.clip_id
    device = torch.device(cfg.device)
    
    # Mapping Config (using defaults from MappingConfig)
    map_cfg = MappingConfig(
        device=cfg.device,
        num_steps_per_frame=100, 
        pyr_levels=1,
        pyr_interval=30,
        fix_color_and_scales=True,
        use_ray_dist=cfg.use_ray_dist
    )
    
    f0 = load_frame_data(data_dir, cfg.init_frame)
    if f0 is None:
        print(f"Failed to load initial frame {cfg.init_frame} for {cfg.clip_id}")
        return

    # Initial GS
    H, W = f0["image"].shape[:2]
    mask0 = f0["mask"] > 0
    depth0 = f0["depth"]
    K0 = f0["K"]
    
    # Setup initial means from depth
    y, x = np.where(mask0)
    if len(y) == 0:
        print(f"Empty mask in first frame for {cfg.clip_id}")
        return
        
    perm = np.random.permutation(len(y))[:8000]
    y, x = y[perm], x[perm]
    z = depth0[y, x]
    pts_c = np.stack([(x - K0[0, 2]) * z / K0[0, 0], (y - K0[1, 2]) * z / K0[1, 1], z], axis=-1)
    
    # Assume object is at 50cm in front for first frame
    T_CW_gt0 = f0.get("T_CW_gt", np.eye(4))
    T_WO_gt0 = f0.get("T_WO_gt", None)
    if T_WO_gt0 is not None:
        T_C0O = T_CW_gt0 @ T_WO_gt0
    else:
        T_C0O = np.eye(4); T_C0O[2, 3] = 0.5
        
    T_OC0 = np.linalg.inv(T_C0O)
    pts_o = (pts_c @ T_OC0[:3, :3].T) + T_OC0[:3, 3]
    colors0 = f0["image"][y, x] / 255.0
    
    # Normals from depth
    full_pts_c = unproject_depth(torch.from_numpy(depth0).float().to(device), torch.from_numpy(K0).float().to(device), H, W)
    normals_c, _ = d2n_tblr(full_pts_c.permute(2, 0, 1).unsqueeze(0))
    normals_c = -F.normalize(normals_c[0][:, y, x].permute(1, 0), dim=1).cpu().numpy()
    normals_o = (normals_c @ T_OC0[:3, :3].T)
    
    # Ray parameters for initialization
    ray_o_o = None
    ray_d_o = None
    ray_dist = None
    
    if cfg.use_ray_dist:
        ray_o_o = torch.from_numpy(T_OC0[:3, 3]).float().view(1, 3).repeat(len(pts_o), 1)
        # ray_d in camera space
        fx, fy, cx, cy = K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]
        ray_d_c = np.stack([(x - cx) / fx, (y - cy) / fy, np.ones_like(x)], axis=-1)
        ray_d_c = ray_d_c / np.linalg.norm(ray_d_c, axis=-1, keepdims=True)
        # ray_d in object space
        ray_d_o = torch.from_numpy(ray_d_c @ T_OC0[:3, :3].T).float()
        # ray_dist (distance along ray)
        ray_dist = torch.norm(torch.from_numpy(pts_c).float(), dim=-1, keepdim=True)

    gs_params = init_gs_from_tracker_points(pts_o, colors0, device, normals=normals_o, ray_o=ray_o_o, ray_d=ray_d_o, ray_dist=ray_dist)
    mapper = GSMapping(map_cfg, gs_params)
    
    mapping_thread = threading.Thread(target=mapper.run)
    mapping_thread.start()
    
    # TSDF Volume
    volume = None
    if cfg.run_tsdf:
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=cfg.voxel_size,
            sdf_trunc=cfg.sdf_trunc,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
    
    psnrs = []
    estimated_poses = {}
    pbar = tqdm(range(cfg.n_frames), desc="Consolidated Mapping")
    
    # Pre-load all poses if possible to avoid file I/O in the loop
    gt_poses = {}
    for i in range(cfg.n_frames):
        idx = cfg.init_frame + i
        T_WO = load_object_pose_world(data_dir, idx)
        if T_WO is not None:
            # We need Object-to-Camera (T_CO)
            # T_CW is Camera-to-World? No, load_frame_data loads "K", but not T_CW.
            # Usually in this dataset, we have GT T_CO directly or we can compute it.
            # Let's check if we can get T_CO from load_frame_data.
            pass

    for i in pbar:
        idx = cfg.init_frame + i
        fd = load_frame_data(data_dir, idx)
        if fd is None: break
        
        # In this dataset, fd["T_CO"] is often available if it's a test script
        T_CW_gt = fd.get("T_CW_gt", np.eye(4))
        T_WO_gt = fd.get("T_WO_gt", None)
        if T_WO_gt is not None:
            T_CiO = T_CW_gt @ T_WO_gt
        else:
            T_CiO = np.eye(4); T_CiO[2, 3] = 0.5
            
        estimated_poses[idx] = T_CiO
        
        frame_data = {
            "image": fd["image"],
            "mask": fd["mask"],
            "depth": fd["depth"],
            "K": fd["K"],
            "T_CiO": T_CiO,
            "frame_idx": idx
        }
        mapper.update(frame_data)
        
        # Wait for processing
        while mapper.data_queue.qsize() > 0:
            time.sleep(0.05)
        time.sleep(0.5) # Buffer for 100 steps
        
        # Evaluate
        with torch.no_grad():
            T_CO_t = torch.from_numpy(T_CiO).float().to(device)
            K_t = torch.from_numpy(fd["K"]).float().to(device)
            render_image, _, _, _ = render_2dgs(
                mapper.gs_params.means, F.normalize(mapper.gs_params.quats), 
                torch.exp(mapper.gs_params.scales), mapper.gs_params.colors, 
                torch.sigmoid(mapper.gs_params.opacity),
                viewmat=T_CO_t, K=K_t, width=W, height=H
            )
            target_image = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0
            mask_t = torch.from_numpy(fd["mask"] > 0).to(device)
            
            if mask_t.any():
                mse = torch.mean((render_image[:, mask_t] - target_image[:, mask_t])**2)
                psnr = -10.0 * torch.log10(mse + 1e-10)
                psnrs.append(psnr.item())
                pbar.set_postfix({"PSNR": f"{psnr.item():.2f}", "GS": len(mapper.gs_params.means)})

            if not cfg.no_vis:
                rr.set_time("frame_idx", sequence=idx)
                rr.log("input/image", rr.Image(fd["image"]))
                rr.log("input/mask", rr.Image(fd["mask"]))
                
                # Rendered image
                render_np = (render_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                rr.log("render/image", rr.Image(render_np))
                
                # Point cloud
                means_np = mapper.gs_params.means.detach().cpu().numpy()
                # Simplified color: convert SH[0] to RGB
                colors_sh0 = mapper.gs_params.colors.detach().cpu()
                if colors_sh0.dim() == 3: colors_sh0 = colors_sh0[:, 0, :]
                colors_rgb = (colors_sh0 * 0.28209 + 0.5).clamp(0, 1).numpy()
                rr.log("world/points", rr.Points3D(means_np, colors=colors_rgb))

    mapper.stop()
    mapping_thread.join()
    
    if psnrs:
        print(f"\nAverage Consolidated PSNR: {np.mean(psnrs):.2f} dB")
    else:
        print("No PSNR results generated.")

    # Final High-Quality Rendering
    print("\nPerforming final high-quality rendering sweep...")
    final_psnrs = []
    for i in tqdm(range(cfg.n_frames), desc="Final Rendering"):
        idx = cfg.init_frame + i
        fd = load_frame_data(data_dir, idx)
        if fd is None: break
        
        # Use the stored poses
        T_CiO = estimated_poses.get(idx, np.eye(4))
        
        with torch.no_grad():
            T_CO_t = torch.from_numpy(T_CiO).float().to(device)
            K_t = torch.from_numpy(fd["K"]).float().to(device)
            
            # Ensure parameters are on the correct device and normalized
            means = mapper.gs_params.means
            quats = F.normalize(mapper.gs_params.quats)
            scales = torch.exp(mapper.gs_params.scales)
            colors = mapper.gs_params.colors
            opacity = torch.sigmoid(mapper.gs_params.opacity)
            
            render_image, render_depth, _, _ = render_2dgs(
                means, quats, scales, colors, opacity,
                viewmat=T_CO_t, K=K_t, width=W, height=H
            )
            
            # TSDF Integration
            if cfg.run_tsdf:
                color_np = (render_image.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                depth_np = render_depth[0].cpu().numpy()
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    W, H, 
                    fd["K"][0,0], fd["K"][1,1], fd["K"][0,2], fd["K"][1,2]
                )
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d.geometry.Image(color_np),
                    o3d.geometry.Image(depth_np),
                    depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
                )
                volume.integrate(rgbd, intrinsic, T_CiO)
            
            target_image = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0
            mask_t = torch.from_numpy(fd["mask"] > 0).to(device)
            
            if mask_t.any():
                render_image = render_image.clamp(0, 1)
                mse = torch.mean((render_image[:, mask_t] - target_image[:, mask_t])**2)
                psnr = -10.0 * torch.log10(mse + 1e-10)
                final_psnrs.append(psnr.item())
            
            # Log to Rerun
            if not cfg.no_vis:
                rr.set_time("frame_idx", sequence=idx)
                render_np = (render_image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                rr.log("final/render", rr.Image(render_np))
                
    if final_psnrs:
        print(f"\nAverage Final PSNR: {np.mean(final_psnrs):.2f} dB")

    # Export TSDF Mesh
    if cfg.run_tsdf:
        save_dir = Path("output") / cfg.clip_id
        save_dir.mkdir(parents=True, exist_ok=True)
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(save_dir / "object_mesh.ply"), mesh)
        print(f"Saved TSDF mesh to {save_dir / 'object_mesh.ply'}")
        
        # Log to Rerun
        if not cfg.no_vis:
            rr.log("world/mesh", rr.Mesh3D(
                vertex_positions=np.asarray(mesh.vertices),
                triangle_indices=np.asarray(mesh.triangles),
                vertex_colors=np.asarray(mesh.vertex_colors),
                vertex_normals=np.asarray(mesh.vertex_normals)
            ), static=True)

if __name__ == "__main__":
    main(tyro.cli(Config))
