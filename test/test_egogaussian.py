import multiprocessing
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
ROOT = Path(__file__).parent.absolute()
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "third_party" / "PGSR"))

# Add on-the-fly-nvs to path for COLMAP loader
OTF_PATH = Path(__file__).parent / "third_party" / "on-the-fly-nvs"
sys.path.append(str(OTF_PATH))
from dataloaders.read_write_model import read_model
sys.path.remove(str(OTF_PATH))

@dataclass
class Config:
    data_root: str = "data/EgoGaussian-Data/Submission/HOI4D"
    video_id: str = "Video1"
    init_frame: int = 0
    n_frames_track: int = 100
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
    num_steps: int = 150 
    mask_loss_weight: float = 20.0 
    use_pgsr: bool = False
    multi_view_ncc_weight: float = 1.0
    pyr_levels: int = 2
    use_ray_dist: bool = True
    densify_every: int = 10 

def load_colmap_data(video_dir):
    sparse_dir = Path(video_dir) / "sparse" / "0"
    if not sparse_dir.exists():
        logging.error(f"Sparse directory not found: {sparse_dir}")
        return None, None
    cameras, images, _ = read_model(str(sparse_dir), ext=".bin")
    return cameras, images

def get_colmap_frame(frame_idx, cameras_dict, images_dict):
    img_name = f"{frame_idx:05d}.png"
    for img_id, img in images_dict.items():
        if img.name == img_name:
            cam = cameras_dict[img.camera_id]
            R_mat = img.qvec2rotmat()
            t = img.tvec
            T_CW = np.eye(4)
            T_CW[:3, :3] = R_mat
            T_CW[:3, 3] = t
            
            if cam.model == "PINHOLE":
                fx, fy, cx, cy = cam.params
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            elif cam.model == "SIMPLE_PINHOLE":
                f, cx, cy = cam.params
                K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
            else:
                logging.warning(f"Unsupported camera model: {cam.model}. Assuming PINHOLE-like params.")
                fx, fy, cx, cy = cam.params[:4]
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            return T_CW, K
    return None, None

def load_frame_data(video_dir, frame_idx, cameras_dict, images_dict):
    img_path = Path(video_dir) / "images" / f"{frame_idx:05d}.png"
    mask_path = Path(video_dir) / "obj_masks" / f"{frame_idx:05d}.png"
    depth_path = Path(video_dir) / "model_infer" / f"depth_{frame_idx:05d}.npy"
    
    if not img_path.exists():
        img_path = Path(video_dir) / "images" / f"{frame_idx:05d}.jpg"
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
        
    T_CW_gt, K = get_colmap_frame(frame_idx, cameras_dict, images_dict)
    
    if T_CW_gt is None:
        logging.warning(f"COLMAP pose not found for frame {frame_idx}")
        return None
        
    return {
        "image": img, 
        "mask": mask, 
        "depth": depth, 
        "K": K, 
        "T_CW_gt": T_CW_gt, 
        "frame_idx": frame_idx
    }

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("BundleGS_EgoGaussian", spawn=False)
        if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
        rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    video_dir = Path(cfg.data_root) / cfg.video_id
    device = torch.device(cfg.device)
    
    from BundleGS.bundlesdf_gs import BundleSdfGS
    from gs_dyn_obj.gs_rendering import render_2dgs, render_3dgs
    
    cameras_dict, images_dict = load_colmap_data(video_dir)
    if cameras_dict is None: return

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
    
    traj_obj_est_C = []
    last_T_CO_est = None

    # Phase 1: Tracking and Mapping
    print("\n>>> Phase 1: Tracking and Mapping")
    t0 = time.time()
    for i in range(cfg.n_frames_track):
        idx = cfg.init_frame + i * cfg.stride
        fd = load_frame_data(video_dir, idx, cameras_dict, images_dict)
        if fd is None: break
        
        color_rgb = fd["image"]
        mask = fd["mask"]
        depth = fd["depth"]
        K = fd["K"]
        
        H, W = color_rgb.shape[:2]
        print(f"\n--- Processing Frame {idx} ({i+1}/{cfg.n_frames_track}) ---")
        
        # Initialize T_CO_init as identity if i==0 and no GT object pose is available
        T_CO_est, _ = tracker.run(color_rgb, mask, depth, K, T_CO_init=np.eye(4) if i == 0 else None)
        last_T_CO_est = T_CO_est
        
        # Log to Rerun
        if not cfg.no_vis:
            rr.set_time("frame_idx", sequence=idx)
            rr.log("input/image", rr.Image(color_rgb))
            
            T_OC_est = np.linalg.inv(T_CO_est)
            rr.log("object/tracker/camera_est", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
            traj_obj_est_C.append(T_OC_est[:3, 3])
            rr.log("object/tracker/traj_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[[0, 255, 0]], radii=0.003))

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
    avg_fps = (i + 1) / (t1 - t0)
    print(f"\n[INFO] Phase 1 finished. Avg FPS: {avg_fps:.2f}")

    tracker.on_finish()
    if cfg.multiprocess:
        tracker._update_gs_from_mapper()

    # Phase 2: Evaluation (PSNR on keyframes)
    print("\n>>> Phase 2: Evaluation")
    eval_psnrs = []
    
    abs_keyframes = [cfg.init_frame + k * cfg.stride for k in tracker.keyframes]
    pbar = tqdm(abs_keyframes, desc="Eval")

    for abs_idx in pbar:
        fd = load_frame_data(video_dir, abs_idx, cameras_dict, images_dict)
        if fd is None: continue
        
        # Since we don't have GT T_CO, we evaluate on the estimated poses for keyframes
        # Or we can just skip this if the goal is only GT-based evaluation.
        # But here we evaluate how well the GS model fits the keyframes.
        # We need the T_CO that was estimated for this keyframe.
        # tracker.poses stores them.
        k_idx = tracker.keyframes.index((abs_idx - cfg.init_frame) // cfg.stride)
        T_CiO_est = tracker.poses[k_idx]
        
        K = fd["K"]
        H, W = fd["image"].shape[:2]
        
        with torch.no_grad():
            T_t = torch.from_numpy(T_CiO_est).float().to(device)
            K_t = torch.from_numpy(K).float().to(device)
            
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
            
            target_image = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0
            mask_t = torch.from_numpy(fd["mask"] > 0).to(device)
            
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

    metrics = {
        "psnr": np.mean(eval_psnrs) if eval_psnrs else None,
        "fps": avg_fps
    }

    print("\n" + "="*40)
    print(f"BundleGS Results for {cfg.video_id}:")
    if eval_psnrs:
        print(f"  Average PSNR: {metrics['psnr']:.2f} dB")
    print(f"  FPS: {metrics['fps']:.2f}")
    print("="*40)
    
    return metrics

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    cfg = tyro.cli(Config)
    main(cfg)
