"""
any4d_multiview_engine.py
=========================
Multi-View Any4D Engine for full_system_ow.

Key features:
1. Native Multi-View ($M$-view, default 4) batching matching any4d_4v_combined.
2. Conditioned on known Camera Extrinsics (T_WC) and Intrinsics (K).
3. Produces high-quality metric depth maps Z_t (replacing MapAnything depth).
4. Produces dense 3D scene flow and Kabsch SE(3) pose estimates for the dynamic object.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Path resolution for Any4D
_HERE = Path(__file__).parent.absolute()
_WORKSPACE_ROOT = _HERE.parent.absolute()
_ANY4D_ROOT = (_WORKSPACE_ROOT / "Any4D").resolve()
if str(_ANY4D_ROOT) not in sys.path:
    sys.path.insert(0, str(_ANY4D_ROOT))

logger = logging.getLogger("Any4DMultiView")

# DINOv2 normalization constants
_DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def scale_intrinsics(K: np.ndarray, orig_hw: Tuple[int, int], target_hw: Tuple[int, int]) -> np.ndarray:
    """Scales camera intrinsic matrix K from original resolution to target resolution."""
    orig_h, orig_w = orig_hw
    target_h, target_w = target_hw
    sx = target_w / orig_w
    sy = target_h / orig_h
    K_scaled = K.copy().astype(np.float32)
    K_scaled[0, 0] *= sx
    K_scaled[0, 2] *= sx
    K_scaled[1, 1] *= sy
    K_scaled[1, 2] *= sy
    return K_scaled


def normalize_image(img_hwc_uint8: np.ndarray, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize image and apply DINOv2 normalization -> Tensor(3, H, W)."""
    h, w = target_hw
    resized = cv2.resize(img_hwc_uint8, (w, h), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    t = (t - _DINO_MEAN) / _DINO_STD
    return t


def resize_mask(mask_hw: np.ndarray, target_hw: Tuple[int, int]) -> torch.Tensor:
    """Resize mask -> Bool Tensor(H, W)."""
    h, w = target_hw
    resized = cv2.resize(mask_hw.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(resized > 0)


def ransac_kabsch(pts_src: np.ndarray, pts_dst: np.ndarray, max_iters=100, inlier_thresh=0.03) -> Tuple[Optional[np.ndarray], int]:
    """
    Solves for rigid transformation T = [R, t] using RANSAC with Kabsch SVD.
    Filters out background and occluding hand outliers.
    """
    N = pts_src.shape[0]
    if N < 4:
        return None, 0
    best_T = None
    best_inliers = 0
    for _ in range(max_iters):
        idx = np.random.choice(N, 4, replace=False)
        c_src = pts_src[idx].mean(0)
        c_dst = pts_dst[idx].mean(0)
        A = (pts_src[idx] - c_src).T @ (pts_dst[idx] - c_dst)
        U, _, Vt = np.linalg.svd(A)
        d = np.linalg.det(Vt.T @ U.T)
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T
        t = c_dst - R @ c_src
        pred = (pts_src @ R.T) + t
        res = np.linalg.norm(pred - pts_dst, axis=1)
        inliers = int(np.sum(res < inlier_thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_T = np.eye(4)
            best_T[:3, :3] = R
            best_T[:3, 3] = t

    # Refit with all inliers
    if best_T is not None and best_inliers >= 4:
        pred = (pts_src @ best_T[:3, :3].T) + best_T[:3, 3]
        inlier_mask = np.linalg.norm(pred - pts_dst, axis=1) < inlier_thresh
        if np.sum(inlier_mask) >= 4:
            c_src = pts_src[inlier_mask].mean(0)
            c_dst = pts_dst[inlier_mask].mean(0)
            A = (pts_src[inlier_mask] - c_src).T @ (pts_dst[inlier_mask] - c_dst)
            U, _, Vt = np.linalg.svd(A)
            d = np.linalg.det(Vt.T @ U.T)
            D = np.diag([1.0, 1.0, d])
            R = Vt.T @ D @ U.T
            t = c_dst - R @ c_src
            best_T[:3, :3] = R
            best_T[:3, 3] = t
    return best_T, best_inliers


class Any4DMultiViewEngine:
    """
    Multi-View Any4D inference engine maintaining an M-view temporal sliding window.
    """
    def __init__(
        self,
        checkpoint_path: str = "Any4D/checkpoints/any4d_4v_combined.pth",
        config_dir: str = "Any4D/configs",
        device: str = "cuda",
        window_size: int = 4,
        target_resolution: Tuple[int, int] = (336, 518),  # (H, W), multiples of 14
        use_known_poses: bool = True,
        min_scene_flow_mag: float = 0.005,
        min_mask_pts: int = 25,
    ):
        self.device = torch.device(device)
        self.window_size = window_size
        self.target_resolution = target_resolution
        self.use_known_poses = use_known_poses
        self.min_scene_flow_mag = min_scene_flow_mag
        self.min_mask_pts = min_mask_pts

        self.checkpoint_path = str((_WORKSPACE_ROOT / checkpoint_path).resolve())
        self.config_dir = str((_WORKSPACE_ROOT / config_dir).resolve())

        self.model = None
        self.ref_frame = None
        self.ref_pts3d = None
        self.ref_mask_resized = None
        self.frame_history: List[Dict[str, Any]] = []
        self.orig_hw: Optional[Tuple[int, int]] = None
        self._is_initialized = False

    def lazy_load_model(self):
        if self.model is not None:
            return
        print(f"[Any4D Engine] Initializing Any4D from {self.checkpoint_path} on {self.device}...")
        import hydra
        from any4d.models import init_model

        hydra.core.global_hydra.GlobalHydra.instance().clear()
        hydra.initialize_config_dir(version_base=None, config_dir=self.config_dir)
        overrides = [
            "machine=local",
            "model=any4d",
            "model.encoder.uses_torch_hub=false",
            "model/task=images_only",
        ]
        cfg = hydra.compose(config_name="train", overrides=overrides)
        model = init_model(cfg.model.model_str, cfg.model.model_config)
        
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        model.to(self.device)
        model.eval()
        self.model = model
        print("[Any4D Engine] Model loaded successfully.")

    def set_reference_frame(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        extrin: np.ndarray,
        K: np.ndarray,
        T_WO_gt: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Registers frame 0 as the reference anchor, runs initial inference,
        and returns the Any4D metric depth for frame 0.
        """
        self.lazy_load_model()
        H, W = image_rgb.shape[:2]
        self.orig_hw = (H, W)

        self.ref_frame = {
            "frame_idx": 0,
            "image": image_rgb,
            "mask": mask,
            "extrin": extrin,
            "K": K,
            "T_WO_gt": T_WO_gt
        }
        self.frame_history = [self.ref_frame]
        self.ref_mask_resized = resize_mask(mask, self.target_resolution).to(self.device)

        # Run Any4D on reference frame (replicated M times for batch consistency)
        depth_0, pts3d_0, _, _ = self._run_multiview_batch([self.ref_frame] * self.window_size)
        self.ref_pts3d = pts3d_0
        self._is_initialized = True
        return depth_0

    def process_frame(
        self,
        frame_idx: int,
        image_rgb: np.ndarray,
        mask: Optional[np.ndarray],
        extrin: np.ndarray,
        K: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Processes frame t with Any4D multi-view window.
        Returns:
            depth_z_full: (H, W) Metric camera depth map (replaces MapAnything).
            T_CO_est: (4, 4) Rigid camera-from-object pose estimate from 3D scene flow.
            pts3d_obj: (N, 3) Metric 3D points in camera frame.
            pts2d_obj: (N, 2) 2D pixel projections in full resolution.
        """
        if not self._is_initialized or self.model is None:
            raise RuntimeError("Any4D engine must be initialized with set_reference_frame first.")

        curr_frame = {
            "frame_idx": frame_idx,
            "image": image_rgb,
            "mask": mask if mask is not None else np.ones(self.orig_hw, dtype=bool),
            "extrin": extrin,
            "K": K
        }
        self.frame_history.append(curr_frame)

        # Form M-view window: [ref_frame, past_kfs..., curr_frame]
        window = [self.ref_frame]
        remaining = self.window_size - 1
        recent = self.frame_history[1:]
        if len(recent) < remaining:
            # Pad with current frame
            window.extend([recent[-1]] * (remaining - len(recent)))
            window.extend(recent)
        else:
            window.extend(recent[-remaining:])

        return self._run_multiview_batch(window)

    def _run_multiview_batch(
        self,
        window_frames: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Executes multi-view forward pass over window_frames and extracts metrics."""
        from any4d.utils.inference import loss_of_one_batch_multi_view

        tH, tW = self.target_resolution
        orig_h, orig_w = self.orig_hw

        views = []
        for i, frame in enumerate(window_frames):
            img_norm = normalize_image(frame["image"], self.target_resolution).unsqueeze(0).to(self.device)
            v_mask = resize_mask(frame["mask"], self.target_resolution).to(self.device)
            
            view_dict = {
                "img": img_norm,
                "data_norm_type": ["dinov2"],
                "non_ambiguous_mask": torch.ones((tH, tW), dtype=torch.bool, device=self.device),
                "binary_mask": v_mask,
                "is_metric_scale": torch.ones(1, dtype=torch.bool, device=self.device),
            }

            if self.use_known_poses and frame.get("extrin") is not None:
                # extrin is T_CW (Camera-from-World)
                # Any4D expects T_WC (Camera-to-World)
                T_WC = np.linalg.inv(frame["extrin"])
                view_dict["camera_poses"] = torch.from_numpy(T_WC).float().unsqueeze(0).to(self.device)

            if frame.get("K") is not None:
                K_scaled = scale_intrinsics(frame["K"], (orig_h, orig_w), self.target_resolution)
                view_dict["intrinsics"] = torch.from_numpy(K_scaled).float().unsqueeze(0).to(self.device)

            views.append(view_dict)

        with torch.no_grad():
            pred_result = loss_of_one_batch_multi_view(
                views,
                self.model,
                None,
                self.device,
                use_amp=True
            )

        # Target view is the last view in the multi-view window
        target_idx = len(window_frames)
        target_pred = pred_result[f"pred{target_idx}"]

        # 1. Metric Camera Depth Extraction (Z_t)
        if "pts3d_cam" in target_pred:
            pts3d_cam = target_pred["pts3d_cam"][0]  # (tH, tW, 3)
            depth_z_model = pts3d_cam[..., 2].clamp(min=0.01).cpu().numpy()
        elif "depth_along_ray" in target_pred and "ray_directions" in target_pred:
            ray_d = target_pred["ray_directions"][0]
            depth_ray = target_pred["depth_along_ray"][0]
            depth_z_model = (ray_d * depth_ray)[..., 2].clamp(min=0.01).cpu().numpy()
        else:
            depth_z_model = np.ones((tH, tW), dtype=np.float32)

        depth_z_full = cv2.resize(depth_z_model, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)

        # 2. 3D Scene Flow & Object Tracking (Ref -> Target)
        ref_pred = pred_result["pred1"]
        pts3d_ref = ref_pred["pts3d"][0].cpu()  # (tH, tW, 3)

        if "scene_flow" in target_pred:
            scene_flow_t = target_pred["scene_flow"][0].cpu()  # (tH, tW, 3)
        else:
            scene_flow_t = torch.zeros_like(pts3d_ref)

        pts3d_cur = pts3d_ref + scene_flow_t  # (tH, tW, 3)

        # Object pixels selection
        ref_mask = self.ref_mask_resized.cpu()
        sel = ref_mask
        if sel.sum() < self.min_mask_pts:
            return depth_z_full, None, None, None

        pts_ref_np = pts3d_ref[sel].numpy()
        pts_cur_np = pts3d_cur[sel].numpy()

        # Solve rigid relative transform T_rel via RANSAC Kabsch SVD
        T_rel, inlier_count = ransac_kabsch(pts_ref_np, pts_cur_np, max_iters=100, inlier_thresh=0.03)
        if T_rel is None or inlier_count < min(15, int(sel.sum() * 0.2)):
            return depth_z_full, None, None, None

        # Object motion in world: T_WO_t = T_rel @ T_WO_ref
        curr_extrin = window_frames[-1]["extrin"]
        T_WO_ref = self.ref_frame.get("T_WO_gt")
        if T_WO_ref is None:
            T_WO_ref = np.eye(4)

        T_WO_est = T_rel @ T_WO_ref

        # Temporal jump check against last valid estimate (max 20cm per frame)
        if getattr(self, "last_valid_T_WO", None) is not None:
            delta_t = np.linalg.norm(T_WO_est[:3, 3] - self.last_valid_T_WO[:3, 3])
            if delta_t > 0.20:
                # Suspect jump, do not return pose
                return depth_z_full, None, None, None
        self.last_valid_T_WO = T_WO_est
        T_CO_est = curr_extrin @ T_WO_est

        # 3. Dense 2D-3D Correspondences for Tracker
        target_K = window_frames[-1]["K"]
        fx, fy, cx, cy = target_K[0, 0], target_K[1, 1], target_K[0, 2], target_K[1, 2]
        pts3d_cam_t = (pts_cur_np @ curr_extrin[:3, :3].T) + curr_extrin[:3, 3]
        valid_z = pts3d_cam_t[:, 2] > 0.05
        
        pts3d_valid = pts3d_cam_t[valid_z]
        u = pts3d_valid[:, 0] / pts3d_valid[:, 2] * fx + cx
        v = pts3d_valid[:, 1] / pts3d_valid[:, 2] * fy + cy
        pts2d_valid = np.stack([u, v], axis=-1).astype(np.float32)

        return depth_z_full, T_CO_est, pts3d_valid.astype(np.float32), pts2d_valid
