"""
Option A: 4D_PM Multi-View Keyframe Prior Module.

Uses 4D_PM's Pi3 foundation model + SAM 2 to generate multi-view consistent
metric 3D pointmaps, surface normals, and super-primitive segmentations
for initializing the Dynamic Gaussian Splats and geometric tracker.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from .env_bridge import PM_ROOT, PM_PYTHON, run_4dpm_script


def compute_or_load_4dpm_prior(
    clip_dir: str,
    clip_id: str,
    cache_dir: str = "dump/4dpm_priors",
    num_init_kfs: int = 5,
    kf_interval: int = 2,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """
    Computes or loads the multi-view Pi3 geometry prior for the specified clip.
    Returns:
        dict containing:
            - 'depth': (H, W) float32 metric depth for frame 0
            - 'points3d': (H, W, 3) float32 3D pointmap in camera 0 frame
            - 'normals': (H, W, 3) float32 surface normals
            - 'valid_mask': (H, W) bool mask of valid geometry
    """
    cache_path = Path(cache_dir) / f"prior_{clip_id}.npz"
    if cache_path.exists():
        print(f"[4D_PM Prior] Loading cached prior from {cache_path}")
        data = np.load(cache_path)
        return {k: data[k] for k in data.files}

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"[4D_PM Prior] Computing multi-view Pi3 prior for {clip_id} ({num_init_kfs} keyframes)...")

    # Script executed inside 4D_PM environment
    script = f"""
import os, sys
sys.path.insert(0, '{str(PM_ROOT)}')
import numpy as np
import torch
import cv2
from pathlib import Path
from frontend.pi3.wrapper import run_pi3
from tool.etc import image_tt

clip_path = Path('{str(Path(clip_dir).resolve())}')
cache_out = Path('{str(cache_path.resolve())}')

# Check image directory structure
img_dir = clip_path / 'images'
if not img_dir.exists():
    img_dir = clip_path

# Load first N keyframes
img_files = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')))
if not img_files:
    raise RuntimeError(f'No images found in {{img_dir}}')

kfs_images = []
num_kfs = min({num_init_kfs}, len(img_files))
kf_indices = [i * {kf_interval} for i in range(num_kfs) if i * {kf_interval} < len(img_files)]

for idx in kf_indices:
    img = cv2.imread(str(img_files[idx]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    kfs_images.append(img)

# Run Pi3
H_orig, W_orig = kfs_images[0].shape[:2]
target_size = (512, 512)
kfs_resized = [cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR) for img in kfs_images]
kfs_tensor = torch.stack([image_tt(img, 'cuda:0') for img in kfs_resized], dim=0)

with torch.no_grad():
    res = run_pi3(kfs_tensor, confidence_thr=0.1, ortho_poses=True, out_shape=target_size)
    pts0 = res['local_points'][0].cpu().numpy()  # (512, 512, 3)
    mask0 = res['pi_masks'][0].cpu().numpy()     # (512, 512)

# Resize pointmap back to original image size
pts0_full = cv2.resize(pts0, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
mask0_full = cv2.resize(mask0.astype(np.float32), (W_orig, H_orig), interpolation=cv2.INTER_NEAREST) > 0.5
depth0 = pts0_full[..., 2]

# Compute normals via cross product of point tangents
gy, gx = np.gradient(pts0_full, axis=(0, 1))
normals = np.cross(gx, gy)
norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-6
normals = normals / norm
# Ensure normals point toward camera
inward = np.sum(normals * pts0_full, axis=-1) > 0
normals[inward] = -normals[inward]

np.savez_compressed(
    cache_out,
    depth=depth0.astype(np.float32),
    points3d=pts0_full.astype(np.float32),
    normals=normals.astype(np.float32),
    valid_mask=mask0_full
)
print('Successfully saved 4D_PM prior to', cache_out)
"""
    run_4dpm_script(script, timeout=180)

    if not cache_path.exists():
        raise RuntimeError(f"Failed to generate 4D_PM prior at {cache_path}")

    data = np.load(cache_path)
    return {k: data[k] for k in data.files}
