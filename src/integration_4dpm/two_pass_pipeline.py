"""
Option C: Two-Pass Hybrid Pipeline.

Pass 1: Executes 4D_PM end-to-end to obtain global camera poses, object deformation
        trajectories, and 4D super-primitive point clouds with object permanence.
Pass 2: Initializes dynamic Gaussian primitives anchored on 4D_PM's solved geometry,
        applies the 4D motion trajectory, and optimizes appearance / spherical harmonics
        via differentiable Gaussian Splatting for photorealistic novel-view synthesis.
"""

import os
import sys
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from .env_bridge import PM_ROOT, PM_PYTHON, run_4dpm_script


def run_4dpm_pass1(
    clip_id: str,
    data_root: str,
    output_dir: str = "dump/4dpm_twopass",
    kf_interval: int = 10,
    window_size: int = 15,
    num_frames: int = 150
) -> Path:
    """
    Executes Pass 1 (4D_PM end-to-end optimization).
    Returns path to the output cache pickle.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cache_path = Path(output_dir) / f"{clip_id}_opt.pickle"

    if cache_path.exists():
        print(f"[4D_PM Pass 1] Found existing 4D_PM cache at {cache_path}")
        return cache_path

    # Check if this clip already has a pre-existing 4D_PM dump in 4D_PM/dump
    clean_id = clip_id.replace('-', '_')
    candidate_dumps = [
        PM_ROOT / "dump" / f"hot3d_{clip_id}.pickle",
        PM_ROOT / "dump" / f"hot3d_{clean_id}.pickle",
        PM_ROOT / "dump" / f"{clip_id}.pickle",
        PM_ROOT / "dump" / f"{clean_id}.pickle",
    ]
    for existing_dump in candidate_dumps:
        if existing_dump.exists():
            print(f"[4D_PM Pass 1] Using pre-computed 4D_PM dump from {existing_dump}")
            import shutil
            shutil.copyfile(existing_dump, cache_path)
            return cache_path

    print(f"[4D_PM Pass 1] Running 4D_PM end-to-end optimization for {clip_id}...")
    candidate_paths = [
        Path(data_root) / clip_id,
        PM_ROOT / "data" / "hot3d" / "extracted" / clip_id,
        Path("/media/shzhou/RPNG_FLASH_2/dynamic_gs/hot3d_data_process/data/hot3d_clips_processed") / clip_id
    ]
    clip_data_path = None
    for p in candidate_paths:
        if p.exists():
            clip_data_path = p
            break

    if clip_data_path is None:
        raise FileNotFoundError(f"Clip data not found for {clip_id} in {candidate_paths}")

    # Create temporary config
    config_text = f"""
alltracker:
  inference_iters: 4
  conf_thr: 0.5
  visibility_thr: 0.5
  num_supp_frames: 10
  model_path: "{str(PM_ROOT / 'checkpoints' / 'alltracker.pth')}"
static_classifier:
  residual_thr: 0.04
  dynamic_segmentation_thr: 0.1
pi3:
  confidence_thr: 0.1
  path: "{str(PM_ROOT / 'third_party' / 'Pi3')}"
dataset:
  type: 'hot3d'
  path: '{str(clip_data_path.resolve())}'
  resize_to: [512, 512]
  start_frame: 0
  frame_stride: 1
  fov_deg: 98.0
frontend:
  pre_erode: 1
  min_segment: 100
  downsample_pow: 1
  num_pts: 150
  num_pts_active: 80
  include_normals: False
sam_params:
  checkpoint: "{str(PM_ROOT / 'checkpoints' / 'sam2.1_hiera_large.pt')}"
  model_cfg: '/configs/sam2.1/sam2.1_hiera_l.yaml'
  nms_score_type: 'stabilty'
  select_smallest: False
  nms: True
  box_nms_thresh: 0.8
  iou_threshold: 0.5
  stability_threshold: 0.9
  active_stability_threshold: 0.9
  active_iou_threshold: 0.5
alignment:
  gauge_freedom: 'last'
kf_interval: {kf_interval}
window_size: {window_size}
paused: False
"""
    tmp_config_path = Path(output_dir) / f"{clip_id}_config.yaml"
    with open(tmp_config_path, "w") as f:
        f.write(config_text)

    script = f"""
import os, sys
sys.path.insert(0, '{str(PM_ROOT)}')
from frontend.utils import load_config
from optimise import run_optimization, save_cache

config = load_config('{str(tmp_config_path.resolve())}')
out = run_optimization(config, verbose=False)
save_cache(out, '{str(cache_path.resolve())}')
"""
    run_4dpm_script(script, timeout=600)

    if not cache_path.exists():
        raise RuntimeError(f"Pass 1 failed: cache not found at {cache_path}")

    return cache_path


def load_4dpm_geometry_and_poses(cache_path: Path) -> Dict[str, Any]:
    """
    Extracts solved poses, keyframes, pointmaps, and segments from 4D_PM cache.
    """
    script = f"""
import sys, pickle
sys.path.insert(0, '{str(PM_ROOT)}')
import numpy as np

with open('{str(cache_path.resolve())}', 'rb') as f:
    data = pickle.load(f)

opt_vars = data['opt_vars']
frontend = data['frontend_output']

kf_poses = [p.detach().cpu().numpy() for p in opt_vars['kf_poses']]
kf_ids = frontend['kf_ids']

# Extract 3D points from keyframe 0
kf0_pts = frontend['points'][0] if 'points' in frontend else None
if kf0_pts is not None:
    pts_np = kf0_pts.detach().cpu().numpy()
else:
    pts_np = np.zeros((100, 3), dtype=np.float32)

export_data = {{
    'kf_ids': kf_ids,
    'kf_poses': kf_poses,
    'initial_points': pts_np
}}

with open('{str(cache_path.resolve())}.summary.pkl', 'wb') as f:
    pickle.dump(export_data, f)
"""
    run_4dpm_script(script, timeout=60)
    summary_path = Path(str(cache_path) + ".summary.pkl")
    with open(summary_path, "rb") as f:
        summary = pickle.load(f)
    if summary_path.exists(): summary_path.unlink()
    return summary
