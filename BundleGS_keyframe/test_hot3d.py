import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import json
import time
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from BundleGS_keyframe.bundlesdf_gs import BundleSdfGS
from gs_dyn_obj.gs_rendering import render_2dgs, render_3dgs

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    init_frame: int = 0
    n_frames_track: int = 30
    n_frames_eval: int = 10
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    no_vis: bool = False
    use_gt_depth: bool = False
    use_photometric: bool = True
    stride: int = 1
    gs_type: str = "2d" # "2d" or "3d"
    photometric_mode: str = "lm" # "hybrid", "adam", or "lm"
    multiprocess: bool = False
    fix_color: bool = True
    fix_scale: bool = True
    num_steps: int = 150 # Even higher for better accuracy
    mask_loss_weight: float = 20.0 # Aggressive mask enforcement
    use_pgsr: bool = False
    multi_view_ncc_weight: float = 1.0
    pyr_levels: int = 2
    use_ray_dist: bool = True
    densify_every: int = 10 # Slightly less frequent for stability

def load_object_pose_world(data_root, frame_idx):
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists(): 
        logging.warning(f"Poses file not found: {poses_file}")
        return None
    with open(poses_file, "r") as f: lines = f.readlines()
    if frame_idx >= len(lines): 
        logging.warning(f"Frame index {frame_idx} out of range (len={len(lines)})")
        return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])]) # x y z w
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4)
    T_WO[:3, :3] = r_WO; T_WO[:3, 3] = t_WO
    return T_WO

def load_frame_data(data_dir, frame_idx, use_gt_depth=False):
    img_path = Path(data_dir) / "images" / f"{frame_idx:06d}.png"
    mask_path = Path(data_dir) / "obj_masks" / f"{frame_idx:06d}.png"
    if use_gt_depth:
        depth_path = Path(data_dir) / "depth_dyn" / f"{frame_idx:06d}.npy"
    else:
        depth_path = Path(data_dir) / "model_infer" / f"depth_{frame_idx:05d}.npy"
    
    if not img_path.exists():
        img_path = Path(data_dir) / "images" / f"{frame_idx:06d}.jpg"
        if not img_path.exists(): return None
            
    img = np.array(cv2.imread(str(img_path))[..., ::-1])
    mask = np.array(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)) if mask_path.exists() else None
    depth = np.load(depth_path).astype(np.float32) if depth_path.exists() else None
    
    if mask is None:
        logging.warning(f"Mask not found for frame {frame_idx} at {mask_path}")
        return None
        
    if depth is not None and depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask is not None and mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    K = np.load(Path(data_dir) / "intrinsics" / f"{frame_idx:06d}.npy")
    T_CW_gt = np.load(Path(data_dir) / "extrinsics" / f"{frame_idx:06d}.npy")
    T_WO_gt = load_object_pose_world(data_dir, frame_idx)
    
    return {"image": img, "mask": mask, "depth": depth, "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": frame_idx}

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("BundleGS_HOT3D", spawn=False)
        if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
        rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    data_dir = Path(cfg.data_root) / cfg.clip_id
    device = torch.device(cfg.device)
    
    # Setup Tracker and Mapping configs
    from BundleGS.bundlesdf_gs import GeoTrackerConfig
    from obj_gs_mapping import MappingConfig
    
    tracker_cfg = GeoTrackerConfig(
        gs_type=cfg.gs_type,
        use_photometric_refinement=cfg.use_photometric,
        photometric_mode=cfg.photometric_mode
    )
    mapping_cfg = MappingConfig(
        gs_type=cfg.gs_type,
        device=device,
        num_steps_per_frame=cfg.num_steps,
        fix_color=cfg.fix_color,
        fix_scale=cfg.fix_scale,
        use_pgsr=cfg.use_pgsr,
        multi_view_ncc_weight=cfg.multi_view_ncc_weight,
        pyr_levels=cfg.pyr_levels,
        use_ray_dist=cfg.use_ray_dist,
        densify_every=cfg.densify_every,
        mask_loss_weight=cfg.mask_loss_weight
    )
    
    # Initialize BundleSdfGS
    print(f"\n[INFO] Initializing BundleSdfGS (multiprocessing={cfg.multiprocess})")
    tracker = BundleSdfGS(tracker_cfg, mapping_cfg, use_multiprocessing=cfg.multiprocess)
    
    all_dist_errors = []
    all_rot_errors = []
    traj_obj_est_C, traj_obj_gt_C = [], []

    # Phase 1: Tracking and Mapping
    print("\n>>> Phase 1: Tracking and Mapping")
    t0 = time.time()
    for i in range(cfg.n_frames_track):
        idx = cfg.init_frame + i * cfg.stride
        fd = load_frame_data(data_dir, idx, use_gt_depth=cfg.use_gt_depth)
        if fd is None: break
        
        T_WO_gt = fd["T_WO_gt"]
        T_CW_gt = fd["T_CW_gt"]
        T_CO_gt = T_CW_gt @ T_WO_gt if T_WO_gt is not None else None
        
        color_rgb = fd["image"]
        mask = fd["mask"]
        depth = fd["depth"]
        K = fd["K"]
        
        H, W = color_rgb.shape[:2]
        print(f"\n--- Processing Frame {idx} ({i+1}/{cfg.n_frames_track}) ---")
        
        # Initialize in Object space at frame 0 if GT is available
        T_CO_est = tracker.run(color_rgb, mask, depth, K, T_CO_init=T_CO_gt if i == 0 else None)
        
        # Log to Rerun
        if not cfg.no_vis:
            rr.set_time("frame_idx", sequence=idx)
            rr.log("input/image", rr.Image(color_rgb))
            
            T_OC_est = np.linalg.inv(T_CO_est)
            rr.log("object/tracker/camera_est", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
            traj_obj_est_C.append(T_OC_est[:3, 3])
            rr.log("object/tracker/traj_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[[0, 255, 0]], radii=0.003))
            
            if T_CO_gt is not None:
                T_OC_gt = np.linalg.inv(T_CO_gt)
                traj_obj_gt_C.append(T_OC_gt[:3, 3])
                rr.log("object/camera_gt", rr.Transform3D(mat3x3=T_OC_gt[:3, :3], translation=T_OC_gt[:3, 3]))
                rr.log("object/traj_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[255, 255, 255]], radii=0.003))

        if T_CO_gt is not None:
            dist_error = np.linalg.norm(T_CO_est[:3, 3] - T_CO_gt[:3, 3])
            rot_error = np.rad2deg(np.arccos(np.clip((np.trace(T_CO_est[:3, :3] @ T_CO_gt[:3, :3].T) - 1) / 2, -1, 1)))
            all_dist_errors.append(dist_error)
            all_rot_errors.append(rot_error)
            print(f"Tracking Result - ATE: {dist_error:.4f}m, RotErr: {rot_error:.4f}deg")

        # Visualization of GS Render
        if not cfg.no_vis and tracker.obj_gs is not None:
            with torch.no_grad():
                render_mode = "3dgs" if cfg.gs_type == "3d" else "normal"
                render_image, _, _, _ = tracker.obj_gs.render(
                    torch.from_numpy(T_CO_est).float().to(device), 
                    torch.from_numpy(K).float().to(device), 
                    W, H, mode=render_mode,
                    near_plane=tracker.tracker_cfg.near_plane, 
                    far_plane=tracker.tracker_cfg.far_plane
                )
                render_np = (render_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                rr.log("render/image", rr.Image(render_np))
                
                means_np = tracker.obj_gs.gs_params.means.detach().cpu().numpy()
                colors_sh = tracker.obj_gs.gs_params.colors.detach().cpu().numpy()
                if colors_sh.ndim == 3: colors_sh = colors_sh[:, 0, :]
                colors_rgb = np.clip(colors_sh * 0.28209479177387814 + 0.5, 0, 1)
                rr.log("object/gs_points", rr.Points3D(means_np, colors=colors_rgb, radii=0.004))
                print(f"[DEBUG] GS Model count: {len(means_np)}")

    t1 = time.time()
    avg_fps = cfg.n_frames_track / (t1 - t0)
    print(f"\n[INFO] Phase 1 finished. Avg FPS: {avg_fps:.2f}")

    # Phase 2: Evaluation
    print("\n>>> Phase 2: Evaluation")
    eval_psnrs = []
    
    # Map relative keyframe indices back to absolute indices
    abs_keyframes = [cfg.init_frame + k * cfg.stride for k in tracker.keyframes]
    pbar = tqdm(abs_keyframes, desc="Eval")

    for abs_idx in pbar:
        fd = load_frame_data(data_dir, abs_idx, use_gt_depth=cfg.use_gt_depth)
        if fd is None: continue
        
        T_CiO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"] if fd["T_WO_gt"] is not None else np.eye(4)
        K = fd["K"]
        H, W = fd["image"].shape[:2]
        
        with torch.no_grad():
            T_t = torch.from_numpy(T_CiO_gt).float().cuda()
            K_t = torch.from_numpy(K).float().cuda()
            
            # Activations are handled inside render_2dgs/render_3dgs
            if cfg.gs_type == "3d":
                render_image, _, _, _ = render_3dgs(
                    tracker.obj_gs.gs_params.means, tracker.obj_gs.gs_params.quats,
                    tracker.obj_gs.gs_params.scales, tracker.obj_gs.gs_params.colors,
                    tracker.obj_gs.gs_params.opacity,
                    viewmat=T_t, K=K_t, width=W, height=H
                )
            else:
                render_image, _, _, _ = render_2dgs(
                    tracker.obj_gs.gs_params.means, tracker.obj_gs.gs_params.quats,
                    tracker.obj_gs.gs_params.scales, tracker.obj_gs.gs_params.colors,
                    tracker.obj_gs.gs_params.opacity,
                    viewmat=T_t, K=K_t, width=W, height=H
                )
            
            target_image = torch.from_numpy(fd["image"]).float().cuda().permute(2, 0, 1) / 255.0
            mask_t = torch.from_numpy(fd["mask"] > 0).cuda()
            
            if mask_t.any():
                render_image = render_image.clamp(0, 1)
                mse = torch.mean((render_image[:, mask_t] - target_image[:, mask_t])**2)
                psnr = -10.0 * torch.log10(mse + 1e-10)
                eval_psnrs.append(psnr.item())
                pbar.set_description(f"Eval PSNR: {psnr.item():.2f}dB")
            
            if not cfg.no_vis:
                rr.set_time("frame_idx", sequence=fd["frame_idx"])
                rr.log("input/image", rr.Image(fd["image"]))
                render_np = (render_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rr.log("render/image", rr.Image(render_np))
                
                T_OCi_gt = np.linalg.inv(T_CiO_gt)
                rr.log("object/camera_gt", rr.Transform3D(mat3x3=T_OCi_gt[:3, :3], translation=T_OCi_gt[:3, 3]))

    metrics = {
        "ate": np.mean(all_dist_errors) if all_dist_errors else None,
        "rot_err": np.mean(all_rot_errors) if all_rot_errors else None,
        "psnr": np.mean(eval_psnrs) if eval_psnrs else None,
        "fps": avg_fps
    }

    if all_dist_errors:
        print("\n" + "="*40)
        print(f"BundleGS Results for {cfg.clip_id}:")
        print(f"  ATE: {metrics['ate']:.4f} m")
        print(f"  Rot Error: {metrics['rot_err']:.2f} deg")
        if eval_psnrs:
            print(f"  Average PSNR: {metrics['psnr']:.2f} dB")
        print(f"  FPS: {metrics['fps']:.2f}")
        print("="*40)
    
    return metrics

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
