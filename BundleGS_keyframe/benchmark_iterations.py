import os
import sys
import numpy as np
import cv2
import torch
import time
from pathlib import Path
from dataclasses import dataclass
import tyro

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from BundleGS.bundlesdf_gs import BundleSdfGS, GeoTrackerConfig
from obj_gs_mapping import MappingConfig
from BundleGS.test_hot3d import load_frame_data

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    init_frame: int = 0
    n_frames_track: int = 30
    gs_type: str = "3d"
    photometric_mode: str = "lm"
    fix_color: bool = True
    fix_scale: bool = True
    num_steps: int = 100

def main(cfg: Config):
    data_dir = Path(cfg.data_root) / cfg.clip_id
    device = torch.device("cuda")
    
    tracker_cfg = GeoTrackerConfig(
        gs_type=cfg.gs_type,
        n_init_frames=5,
        use_photometric_refinement=True,
        photometric_mode=cfg.photometric_mode
    )
    mapping_cfg = MappingConfig(
        gs_type=cfg.gs_type,
        device=device,
        num_steps_per_frame=cfg.num_steps,
        densify_every=5,
        fix_color=cfg.fix_color,
        fix_scale=cfg.fix_scale
    )
    
    tracker = BundleSdfGS(tracker_cfg, mapping_cfg, use_multiprocessing=False)
    
    times = []
    for i in range(cfg.n_frames_track):
        idx = cfg.init_frame + i
        fd = load_frame_data(data_dir, idx)
        if fd is None: break
        
        T_CO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"] if fd["T_WO_gt"] is not None else None
        
        t0 = time.time()
        tracker.run(fd["image"], fd["mask"], fd["depth"], fd["K"], T_CO_init=T_CO_gt if i == 0 else None)
        t1 = time.time()
        times.append(t1 - t0)
        
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    print(f"Iterations: {cfg.num_steps}, Avg Time: {avg_time*1000:.2f}ms, FPS: {fps:.2f}")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
