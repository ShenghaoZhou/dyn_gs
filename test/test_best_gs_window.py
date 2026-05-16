import os
import sys
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import rerun as rr
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

from BundleGS.bundlesdf_gs import BundleSdfGS, GeoTrackerConfig
from obj_gs_mapping import MappingConfig
from gs_dyn_obj.gs_rendering import render_2dgs, render_3dgs

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    init_frame: int = 0
    # Search space as comma-separated strings
    n_track_list: str = "10,20,30"
    n_eval_list: str = "10,20"
    
    device: str = "cuda"
    no_vis: bool = True
    use_gt_depth: bool = False
    use_photometric: bool = True
    stride: int = 1
    gs_type: str = "3d" # "2d" or "3d"
    num_steps_per_frame: int = 150
    densify_every: int = 5

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
    
    if mask is None: return None
    if depth is not None and depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    if mask is not None and mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    K = np.load(Path(data_dir) / "intrinsics" / f"{frame_idx:06d}.npy")
    T_CW_gt = np.load(Path(data_dir) / "extrinsics" / f"{frame_idx:06d}.npy")
    T_WO_gt = load_object_pose_world(data_dir, frame_idx)
    
    return {"image": img, "mask": mask, "depth": depth, "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": frame_idx}

def compute_psnr(tracker, fd, device, gs_type):
    T_CiO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"] if fd["T_WO_gt"] is not None else np.eye(4)
    K = fd["K"]
    H, W = fd["image"].shape[:2]
    
    with torch.no_grad():
        T_t = torch.from_numpy(T_CiO_gt).float().to(device)
        K_t = torch.from_numpy(K).float().to(device)
        
        if gs_type == "3d":
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
            return psnr.item()
    return None

def run_experiment(cfg: Config, n_track: int, n_eval: int):
    data_dir = Path(cfg.data_root) / cfg.clip_id
    device = torch.device(cfg.device)
    
    tracker_cfg = GeoTrackerConfig(
        gs_type=cfg.gs_type,
        n_init_frames=5,
        use_photometric_refinement=cfg.use_photometric
    )
    mapping_cfg = MappingConfig(
        gs_type=cfg.gs_type,
        device=device,
        num_steps_per_frame=cfg.num_steps_per_frame,
        densify_every=cfg.densify_every
    )
    
    tracker = BundleSdfGS(tracker_cfg, mapping_cfg, use_multiprocessing=False)
    
    dist_errors = []
    rot_errors = []
    track_psnrs = []
    
    print(f"\n[Exp] n_track={n_track}, n_eval={n_eval}")
    
    # Tracking Phase
    for i in range(n_track):
        idx = cfg.init_frame + i * cfg.stride
        fd = load_frame_data(data_dir, idx, use_gt_depth=cfg.use_gt_depth)
        if fd is None: break
        
        T_CO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"] if fd["T_WO_gt"] is not None else None
        T_CO_est = tracker.run(fd["image"], fd["mask"], fd["depth"], fd["K"], T_CO_init=T_CO_gt if i == 0 else None)
        
        if T_CO_gt is not None:
            dist_err = np.linalg.norm(T_CO_est[:3, 3] - T_CO_gt[:3, 3])
            rot_err = np.rad2deg(np.arccos(np.clip((np.trace(T_CO_est[:3, :3] @ T_CO_gt[:3, :3].T) - 1) / 2, -1, 1)))
            dist_errors.append(dist_err)
            rot_errors.append(rot_err)

    # Eval Track PSNR on keyframes
    abs_keyframes = [cfg.init_frame + k * cfg.stride for k in tracker.keyframes]
    for abs_idx in abs_keyframes:
        fd = load_frame_data(data_dir, abs_idx, use_gt_depth=cfg.use_gt_depth)
        if fd is None: continue
        psnr = compute_psnr(tracker, fd, device, cfg.gs_type)
        if psnr is not None: track_psnrs.append(psnr)
            
    # Evaluation Phase (following frames)
    eval_psnrs = []
    for i in range(n_eval):
        idx = cfg.init_frame + (n_track + i) * cfg.stride
        fd = load_frame_data(data_dir, idx, use_gt_depth=cfg.use_gt_depth)
        if fd is None: break
        psnr = compute_psnr(tracker, fd, device, cfg.gs_type)
        if psnr is not None: eval_psnrs.append(psnr)
            
    avg_track_psnr = np.mean(track_psnrs) if track_psnrs else 0
    avg_eval_psnr = np.mean(eval_psnrs) if eval_psnrs else 0
    avg_ate = np.mean(dist_errors) if dist_errors else 0
    avg_rot = np.mean(rot_errors) if rot_errors else 0
    
    return {
        "n_track": n_track,
        "n_eval": n_eval,
        "track_psnr": avg_track_psnr,
        "eval_psnr": avg_eval_psnr,
        "ate": avg_ate,
        "rot": avg_rot
    }

def main(cfg: Config):
    n_track_list = [int(x) for x in cfg.n_track_list.split(",")]
    n_eval_list = [int(x) for x in cfg.n_eval_list.split(",")]
    
    results = []
    
    for n_track in n_track_list:
        for n_eval in n_eval_list:
            if n_eval < 5:
                print(f"Skipping n_eval={n_eval} (must be >= 5)")
                continue
            res = run_experiment(cfg, n_track, n_eval)
            results.append(res)
            print(f"  Result: Track PSNR={res['track_psnr']:.2f}, Eval PSNR={res['eval_psnr']:.2f}, ATE={res['ate']:.4f}")

    # Summary Table
    print("\n" + "="*80)
    print(f"{'n_track':<8} | {'n_eval':<8} | {'Track PSNR':<12} | {'Eval PSNR':<12} | {'ATE (m)':<10} | {'Rot (deg)':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['n_track']:<8} | {r['n_eval']:<8} | {r['track_psnr']:<12.2f} | {r['eval_psnr']:<12.2f} | {r['ate']:<10.4f} | {r['rot']:<10.2f}")
    print("="*80)

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
