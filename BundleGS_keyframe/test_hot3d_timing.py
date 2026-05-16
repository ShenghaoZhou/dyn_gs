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
from collections import defaultdict

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from BundleGS.bundlesdf_gs import BundleSdfGS
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
    no_vis: bool = True
    use_gt_depth: bool = False
    use_photometric: bool = True
    stride: int = 1
    gs_type: str = "2d" # "2d" or "3d"
    photometric_mode: str = "lm" # "hybrid", "adam", or "lm"
    multiprocess: bool = False
    fix_color: bool = False
    fix_scale: bool = False
    num_steps: int = 100
    use_pgsr: bool = False
    multi_view_ncc_weight: float = 1.0
    pyr_levels: int = 2
    use_ray_dist: bool = True

def load_object_pose_world(data_root, frame_idx):
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists(): return None
    with open(poses_file, "r") as f: lines = f.readlines()
    if frame_idx >= len(lines): return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])])
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

class Timer:
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}

    def start(self, name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_times[name] = time.perf_counter()

    def stop(self, name):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if name in self.start_times:
            elapsed = time.perf_counter() - self.start_times[name]
            self.timings[name].append(elapsed)

    def print_summary(self):
        print("\n" + "="*50)
        print(f"{'Step':<30} | {'Avg (ms)':<10} | {'Total (s)':<10}")
        print("-" * 55)
        for name, values in sorted(self.timings.items(), key=lambda x: sum(x[1]), reverse=True):
            avg_ms = np.mean(values) * 1000
            total_s = np.sum(values)
            print(f"{name:<30} | {avg_ms:<10.2f} | {total_s:<10.4f}")
        print("="*50 + "\n")

def main(cfg: Config):
    timer = Timer()
    
    if not cfg.no_vis:
        rr.init("BundleGS_HOT3D_Timing", spawn=False)
        if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)

    data_dir = Path(cfg.data_root) / cfg.clip_id
    device = torch.device(cfg.device)
    
    from BundleGS.bundlesdf_gs import GeoTrackerConfig
    from obj_gs_mapping import MappingConfig
    
    tracker_cfg = GeoTrackerConfig(
        gs_type=cfg.gs_type,
        n_init_frames=5,
        use_photometric_refinement=cfg.use_photometric,
        photometric_mode=cfg.photometric_mode
    )
    mapping_cfg = MappingConfig(
        gs_type=cfg.gs_type,
        device=device,
        num_steps_per_frame=cfg.num_steps,
        densify_every=5,
        fix_color=cfg.fix_color,
        fix_scale=cfg.fix_scale,
        use_pgsr=cfg.use_pgsr,
        multi_view_ncc_weight=cfg.multi_view_ncc_weight,
        pyr_levels=cfg.pyr_levels,
        use_ray_dist=cfg.use_ray_dist
    )
    
    tracker = BundleSdfGS(tracker_cfg, mapping_cfg, use_multiprocessing=cfg.multiprocess)
    
    # Phase 1: Tracking and Mapping
    print("\n>>> Phase 1: Tracking and Mapping")
    for i in range(cfg.n_frames_track):
        idx = cfg.init_frame + i * cfg.stride
        
        timer.start("Data Loading")
        fd = load_frame_data(data_dir, idx, use_gt_depth=cfg.use_gt_depth)
        timer.stop("Data Loading")
        
        if fd is None: break
        
        T_CO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"] if fd["T_WO_gt"] is not None else None
        color_rgb = fd["image"]
        mask = fd["mask"]
        depth = fd["depth"]
        K = fd["K"]
        
        H, W = color_rgb.shape[:2]
        
        timer.start("Tracker Run (Total)")
        # Note: Internal timing will be added to bundlesdf_gs.py if possible, 
        # otherwise we just time the whole run call here.
        T_CO_est = tracker.run(color_rgb, mask, depth, K, T_CO_init=T_CO_gt if i == 0 else None)
        timer.stop("Tracker Run (Total)")
        
        if not cfg.no_vis:
            timer.start("Visualization (Rerun)")
            rr.set_time("frame_idx", sequence=idx)
            rr.log("input/image", rr.Image(color_rgb))
            T_OC_est = np.linalg.inv(T_CO_est)
            rr.log("object/tracker/camera_est", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
            timer.stop("Visualization (Rerun)")

    tracker.print_timings()
    timer.print_summary()

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
