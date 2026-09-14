import os
import sys
import numpy as np
import cv2
import poselib
import torch
import torch.nn.functional as F
import multiprocessing
import threading
import time
import yaml
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
multiprocessing.set_start_method('spawn', force=True)

sys.path.insert(0, str(Path(__file__).parent))

from geometric_tracker import GeometricTracker, run_ba as run_ba_geometric
from gs_dyn_obj.obj_gs import ObjectGS
from gs_dyn_obj.gs_param import GSParam
from gs_dyn_obj.gs_rendering import render_2dgs, render_3dgs
from obj_gs_mapping import GSMapping, MappingConfig, MiniCam, gen_virtul_cam, compute_single_view_loss
# Add PGSR to sys.path for ssim
sys.path.append(str(Path(__file__).parent.parent / "third_party" / "PGSR"))
from utils.loss_utils import ssim

@dataclass
class GeoTrackerConfig:
    feature_type: str = "orb" # "loftr", "superpoint", "orb", "grid"
    n_features: int = 2000 # Increased for better robustness
    min_pnp_inliers: int = 20
    max_pose_jump: float = 10.0 # Effectively disable rejection to match benchmark
    use_photometric_refinement: bool = True
    n_init_frames: int = 10 # Number of frames for photometric initialization
    
    # Keyframe logic
    min_kf_interval: int = 3
    max_keyframes: int = 20
    min_kf_rot: float = 5.0 # degrees
    kf_overlap_thresh: float = 0.8
    
    # Depth/Geometry
    near_plane: float = 0.1
    far_plane: float = 5.0
    triangulate: bool = True
    triangulate_thresh: float = 0.05
    grid_spacing: int = 6 # Denser grid for better tracking
    ransac_thresh: float = 1.0 # Stricter RANSAC like benchmark
    
    # Advanced Filtering (Disable for uncalibrated monocular depth)
    use_occlusion_check: bool = False
    use_informed_filtering: bool = False
    informed_thresh: float = 50.0 # Relaxed pixels
    monocular_scale_alignment: bool = True # Use RelPose to fix scale at start
    align_depth: bool = False
    align_with_bias: bool = True
    
    gs_type: str = "2d" # "2d" or "3d"
    photometric_mode: str = "lm" # "hybrid", "adam", or "lm"
    use_match_projections: bool = False
    debug: bool = False

class BundleSdfGS:
    def __init__(self, tracker_cfg: GeoTrackerConfig, cfg_mapping: MappingConfig, use_multiprocessing: bool = True, initial_gs: GSParam = None):
        self.tracker_cfg = tracker_cfg
        self.cfg_mapping = cfg_mapping
        self.use_multiprocessing = use_multiprocessing
        self.initial_gs_param = initial_gs
        
        self.tracker = GeometricTracker(tracker_cfg)
        # Ensure tracker's internal cfg has the flags
        self.tracker.cfg.use_occlusion_check = tracker_cfg.use_occlusion_check
        self.tracker.cfg.use_informed_filtering = tracker_cfg.use_informed_filtering
        self.tracker.cfg.informed_thresh = tracker_cfg.informed_thresh
        self.obj_gs = None
        self.cnt = -1
        self.keyframes = []
        self.keyframes_data = [] # To store fd for evaluation # Indices of keyframes
        self.poses = {} # Combined poses (C0-from-Ci)
        
        # Shared state for mapper
        self.kf_queue = multiprocessing.Queue(maxsize=10)
        self.stop_event = multiprocessing.Event()
        self.lock = threading.Lock()
        
        self.p_dict = multiprocessing.Manager().dict()
        self.p_dict['latest_gs'] = None
        self.p_dict['gs_ready'] = False
        self.p_dict['last_finished_frame'] = -1
        
        self.timings = defaultdict(list)
        
        self.last_synced_frame = -1
        self.gs_ready = False
        self.prev_gray = None
        self.mapping_process = None
        self.mapper = None
        
        # Optical Flow for prediction
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev_gray = None
        self.prev_mask = None
        
    @property
    def mapper_last_finished_frame(self):
        return self.p_dict.get('last_finished_frame', -1)
        
    def _update_gs_from_mapper(self):
        """Update tracker's GS model from mapper if available."""
        if self.p_dict.get('latest_gs') is not None:
            with self.lock:
                logging.info("Updating GS model from mapper")
                self.obj_gs.gs_params = self.p_dict['latest_gs'].clone().to("cuda")
                self.p_dict['latest_gs'] = None
                self.gs_ready = True
                
                # Refresh photometric reference to be consistent with new geometry
                if self.obj_gs.image_ref is not None:
                    K = self.tracker.K_dict.get(self.keyframes[-1])
                    if K is not None:
                        with torch.no_grad():
                            H, W = self.obj_gs.image_ref.shape[1], self.obj_gs.image_ref.shape[2]
                            render_mode = "3dgs" if self.tracker_cfg.gs_type == "3d" else "normal"
                            img_ref, _, _, alpha_ref = self.obj_gs.gs_params.render(
                                self.obj_gs.T_C_O_ref, torch.from_numpy(K).float().cuda(), 
                                width=W, height=H, mode=render_mode
                            )
                            self.obj_gs.update_reference(img_ref, self.obj_gs.T_C_O_ref, alpha_ref)
                
                # Update geometric tracker's 3D points from the optimized GS means
                new_points = self.obj_gs.gs_params.means.detach().cpu().numpy()
                self.tracker.update_object_points(new_points)

    def print_timings(self):
        print("\n" + "-"*30)
        print("BundleSdfGS Internal Timings")
        print("-"*30)
        for name, values in sorted(self.timings.items(), key=lambda x: sum(x[1]), reverse=True):
            avg_ms = np.mean(values) * 1000
            print(f"{name:<25}: {avg_ms:>8.2f} ms")
        print("-"*30 + "\n")

    def init_gs_model(self, color, depth, mask, K, T_WO_init=None, T_CW_init=None, pre_initialized_gs=None):
        """
        Initialize the 3D Gaussian Splatting model from the first frame.
        We now use T_WO parameterization (Object in World).
        """
        if pre_initialized_gs is not None:
            self.obj_gs = ObjectGS(pre_initialized_gs, T_WO_init if T_WO_init is not None else np.eye(4), 1.0)
            self.T_WO = T_WO_init if T_WO_init is not None else np.eye(4)
            T_C0O_init = T_CW_init @ T_WO_init if (T_CW_init is not None and T_WO_init is not None) else np.eye(4)
            self.T_C0O_anchor = T_C0O_init
            gs_params = pre_initialized_gs
            
            # Start Mapper and return
            if self.use_multiprocessing:
                self._start_mapping_process(gs_params)
            else:
                self._init_sync_mapper(gs_params)
            return True
            H, W = color.shape[:2]
            iy, ix = np.where(mask > 0)
            
            # Use random sampling for a more robust depth prior
            max_pts = 10000
            if len(iy) > max_pts:
                perm = np.random.choice(len(iy), max_pts, replace=False)
                iy, ix = iy[perm], ix[perm]
                
            pts2d = np.stack([ix, iy], axis=-1).astype(np.float32)
            z = depth[iy, ix]
            valid = z > 0.01
            pts2d, ix, iy, z = pts2d[valid], ix[valid], iy[valid], z[valid]
        
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts3d = np.stack([(pts2d[:, 0] - cx) * z / fx, (pts2d[:, 1] - cy) * z / fy, z], axis=-1)
        
        # Camera-from-Object at initialization
        T_C0O_init = T_CW_init @ T_WO_init if (T_CW_init is not None and T_WO_init is not None) else np.eye(4)
        self.T_C0O_anchor = T_C0O_init # Still keep for some internal relative logic if needed
        
        # Transform points to Object space
        T_OC0 = np.linalg.inv(T_C0O_init)
        pts3d = (pts3d @ T_OC0[:3, :3].T) + T_OC0[:3, 3]
            
        colors = color[iy, ix] / 255.0
        
        # Compute normals for GS orientation
        from gs_dyn_obj.utils.init import unproject_depth, d2n_tblr
        import torch
        depth_t = torch.from_numpy(depth).float().cuda()
        K_t = torch.from_numpy(K).float().cuda()
        full_pts_c = unproject_depth(depth_t, K_t, H, W)
        normals_c_full, _ = d2n_tblr(full_pts_c.permute(2, 0, 1).unsqueeze(0))
        normals_c = -F.normalize(normals_c_full[0][:, iy, ix].permute(1, 0), dim=1).cpu().numpy()
        
        if T_C0O_init is not None:
            normals_o = (normals_c @ T_OC0[:3, :3].T)
        else:
            normals_o = normals_c

        # Compute Ray parameters for constrained optimization
        ray_o_o = None
        ray_d_o = None
        ray_dist = None
        if self.cfg_mapping.use_ray_dist:
            # Ray origin in object space
            T_OC = np.linalg.inv(self.T_C0O_anchor)
            ray_o_o = torch.from_numpy(T_OC[:3, 3]).float().cuda().view(1, 3).repeat(len(pts3d), 1)
            
            # ray_d in camera space
            ray_d_c = np.stack([(pts2d[:, 0] - cx) / fx, (pts2d[:, 1] - cy) / fy, np.ones_like(z)], axis=-1)
            ray_d_c = ray_d_c / np.linalg.norm(ray_d_c, axis=-1, keepdims=True)
            
            # ray_d in object space
            ray_d_o = torch.from_numpy(ray_d_c @ T_OC[:3, :3].T).float().cuda()
            
            # ray_dist (distance along ray in camera space)
            # pts_c_torch = torch.from_numpy(pts3d_c).float().cuda() # pts3d is already in object space
            # We can use the depth 'z' as an initial guess for distance if ray_d_z is roughly 1
            # But more accurately: dist = norm(pts_c)
            pts_c = np.stack([(pts2d[:, 0] - cx) * z / fx, (pts2d[:, 1] - cy) * z / fy, z], axis=-1)
            ray_dist = torch.norm(torch.from_numpy(pts_c).float().cuda(), dim=-1, keepdim=True)
            ray_dist.requires_grad = True

        if pre_initialized_gs is None:
            from obj_gs_mapping import init_gs_from_tracker_points
            gs_params = init_gs_from_tracker_points(
                pts3d, colors, "cuda", normals=normals_o, 
                ray_o=ray_o_o, ray_d=ray_d_o, ray_dist=ray_dist,
                gs_type=self.tracker_cfg.gs_type
            )
            self.obj_gs = ObjectGS(gs_params, T_WO_init if T_WO_init is not None else np.eye(4), 1.0)
            self.T_WO = T_WO_init if T_WO_init is not None else np.eye(4)
        
        if self.use_multiprocessing:
            self._start_mapping_process(gs_params)
        else:
            self._init_sync_mapper(gs_params)
        return True

    def _start_mapping_process(self, gs_params):
        # Start Mapper Process
        gs_params_cpu = gs_params.clone().to("cpu")
        for attr in ["means", "quats", "scales", "colors", "opacity", "ray_dist"]:
            p = getattr(gs_params_cpu, attr)
            if p is not None:
                setattr(gs_params_cpu, attr, p.detach().requires_grad_(False))
        
        self.mapping_process = multiprocessing.Process(
            target=run_gs_mapping_process, 
            args=(self.cfg_mapping, gs_params_cpu, self.kf_queue, self.p_dict, self.stop_event)
        )
        self.mapping_process.start()

    def _init_sync_mapper(self, gs_params):
        # Initialize Synchronous Mapper
        from obj_gs_mapping import GSMapping
        self.mapper = GSMapping(self.cfg_mapping, gs_params)
        self.mapper.device = torch.device("cuda")
        self.mapper.gs_params.to(self.mapper.device)
        
        # Enable grad for optimization
        params_to_enable = ["quats", "opacity"]
        if self.cfg_mapping.use_ray_dist and self.mapper.gs_params.ray_dist is not None:
            params_to_enable.append("ray_dist")
        else:
            params_to_enable.append("means")
            
        if not self.cfg_mapping.fix_scale:
            params_to_enable.append("scales")
        if not self.cfg_mapping.fix_color:
            if self.mapper.gs_params.shs is not None:
                params_to_enable.append("shs")
            else:
                params_to_enable.append("colors")
            
        for attr in params_to_enable:
            p = getattr(self.mapper.gs_params, attr)
            if p is not None:
                p.requires_grad_(True)

        self.mapper.setup_optimizer()
        self.mapper.keyframes = []
        self.mapping_frame_count = 0
        
        # Pre-compute disc kernel for densification
        radius = 3
        self.mapper.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1, device=self.mapper.device)
        ky, kx = torch.meshgrid(torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1), indexing="ij")
        self.mapper.disc_kernel[0, 0, torch.sqrt(kx**2 + ky**2) <= radius + 0.5] = 1
        self.mapper.disc_kernel = self.mapper.disc_kernel / self.mapper.disc_kernel.sum()
        return True

    def render_current_view(self, K, W, H, T_CiO=None):
        """
        Renders the current object GS model from a given camera pose.
        If T_CiO is None, uses the latest tracked pose.
        """
        if self.obj_gs is None or self.obj_gs.gs_params is None:
            return None, None
        
        if T_CiO is None:
            T_CiO = self.poses[self.cnt]
        
        # Use the appropriate renderer based on gs_type
        T_CiO_torch = torch.from_numpy(T_CiO).float().to(self.obj_gs.gs_params.means.device)
        K_torch = torch.from_numpy(K).float().to(self.obj_gs.gs_params.means.device)
        if self.tracker_cfg.gs_type == "2d":
            img, _, _, alpha = render_2dgs(
                self.obj_gs.gs_params.means, self.obj_gs.gs_params.quats, self.obj_gs.gs_params.scales,
                self.obj_gs.gs_params.colors, self.obj_gs.gs_params.opacity,
                T_CiO_torch, K_torch, W, H, shs=self.obj_gs.gs_params.shs
            )
        else:
            img, _, _, alpha = render_3dgs(
                self.obj_gs.gs_params.means, self.obj_gs.gs_params.quats, self.obj_gs.gs_params.scales,
                self.obj_gs.gs_params.colors, self.obj_gs.gs_params.opacity,
                T_CiO_torch, K_torch, W, H, shs=self.obj_gs.gs_params.shs
            )
        return (img, alpha)

    def run(self, color, mask, depth, K, T_CW=None, T_WO_init=None, return_render=False, any4d_hint=None):
        """
        Main tracking loop. Now supports T_OW parameterization and Any4D multi-view hint.
        T_CW: Camera-from-World (extrinsics)
        T_WO_init: Object-in-World at start
        any4d_hint: Optional tuple (T_CO_any4d, pts3d_any4d, pts2d_any4d)
        """
        t_start = time.perf_counter()
        self.cnt += 1
        H, W = color.shape[:2]
        
        if self.cnt == 0:
            self.prev_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            self.T_CW = T_CW if T_CW is not None else np.eye(4)
            self.T_WO = T_WO_init if T_WO_init is not None else np.eye(4)
            
            T_C0O_init = self.T_CW @ self.T_WO
            self.tracker.poses[0] = np.eye(4)
            self.tracker.K_dict[0] = K
            self.tracker.add_new_points_from_depth(0, color, mask, depth, K, np.eye(4), align=True, spacing=4)
            self.keyframes.append(0)
            self.tracker.keyframes.append(0)
            self.init_gs_model(color, depth, mask, K, self.T_WO, self.T_CW, pre_initialized_gs=self.initial_gs_param)
            self.poses[0] = T_C0O_init
            self.prev_mask = mask.copy()
            self.prev_depth = depth.copy()
            self.prev_color = color.copy()
            # Initial frame setup complete
        else:
            t0 = time.perf_counter()
            self.T_CW = T_CW if T_CW is not None else np.eye(4)
            curr_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            
            self._update_gs_from_mapper()
            
            if self.prev_gray is None:
                self.prev_gray = curr_gray
                return self.poses.get(self.cnt-1, np.eye(4)), None
                
            curr_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            flow = self.dis.calc(self.prev_gray, curr_gray, None)
            self.prev_gray = curr_gray
            self.timings["Pre-processing (Flow)"].append(time.perf_counter() - t0)

            if self.cnt > 1:
                T_pprev_rel = self.tracker.poses[self.cnt-2]
                T_prev_rel = self.tracker.poses[self.cnt-1]
                V = T_prev_rel @ np.linalg.inv(T_pprev_rel)
                
                v_t = np.linalg.norm(V[:3, 3])
                v_R = np.rad2deg(np.arccos(np.clip((np.trace(V[:3, :3]) - 1) / 2, -1, 1)))
                if v_t > 0.5 or v_R > 20.0:
                    print(f"[BundleSdfGS] High velocity detected: {v_t:.4f}m, {v_R:.2f}deg. Discarding motion model.")
                    self.tracker.poses[self.cnt-2] = self.tracker.poses[self.cnt-1].copy()
                    if any4d_hint is not None and any4d_hint[0] is not None:
                        T_guess_CiO = any4d_hint[0]
                        T_guess_rel = T_guess_CiO @ np.linalg.inv(self.poses[0])
                    else:
                        T_guess_CiO = self.T_CW @ self.T_WO
                        T_guess_rel = T_guess_CiO @ np.linalg.inv(self.poses[0])
                else:
                    T_guess_rel = V @ T_prev_rel
                    # Guard constant velocity with Any4D 3D scene flow prior
                    if any4d_hint is not None and any4d_hint[0] is not None:
                        T_a4d_rel = any4d_hint[0] @ np.linalg.inv(self.poses[0])
                        cv_a4d_diff = np.linalg.norm(T_guess_rel[:3, 3] - T_a4d_rel[:3, 3])
                        if cv_a4d_diff > 0.08:
                            print(f"[BundleSdfGS] CV diverged from Any4D by {cv_a4d_diff:.3f}m. Using Any4D motion prior.")
                            T_guess_rel = T_a4d_rel
            else:
                if any4d_hint is not None and any4d_hint[0] is not None:
                    T_guess_rel = any4d_hint[0] @ np.linalg.inv(self.poses[0])
                else:
                    T_guess_rel = self.tracker.poses[self.cnt-1].copy()

            # Stage 1: Photometric Initialization / Refinement
            use_photometric = self.tracker_cfg.use_photometric_refinement and (self.cnt < self.tracker_cfg.n_init_frames or self.gs_ready)
            
            if use_photometric and self.obj_gs is not None and self.obj_gs.gs_params is not None:
                if not self.use_multiprocessing and self.mapper is not None:
                    self.obj_gs.gs_params = self.mapper.gs_params
                
                t0_photo = time.perf_counter()
                if self.cnt > 1:
                    if any4d_hint is not None and any4d_hint[0] is not None:
                        T_CO_guess = any4d_hint[0]
                        T_WO_guess = np.linalg.inv(self.T_CW) @ T_CO_guess
                    else:
                        T_CO_guess = T_guess_rel @ self.poses[0]
                        T_WO_guess = np.linalg.inv(self.T_CW) @ T_CO_guess
                else:
                    T_WO_guess = self.T_WO
                
                self.obj_gs.T_W_O = T_WO_guess
                if self.tracker_cfg.photometric_mode == "hybrid":
                    T_WO_photo, _ = self.obj_gs.optimize_wrt_image_lm_hybrid(
                        T_C_W=self.T_CW, 
                        image=color,
                        K=K,
                        mask=mask,
                        mask_ref=self.prev_mask,
                        adam_steps=5, 
                        lm_steps=15,
                        damping=10.0,
                        pyramid_levels=[(4, 10), (2, 10), (1, 20)],
                        gs_type=self.tracker_cfg.gs_type
                    )
                elif self.tracker_cfg.photometric_mode == "adam":
                    T_WO_photo, _ = self.obj_gs.optimize_wrt_image(
                        T_C_W=self.T_CW, 
                        image=color,
                        K=K,
                        mask=mask,
                        mask_ref=self.prev_mask,
                        lr=1e-3,
                        num_steps=30,
                        gs_type=self.tracker_cfg.gs_type
                    )
                elif self.tracker_cfg.photometric_mode == "lm":
                    if self.obj_gs.gs_params.means.numel() > 0:
                        T_WO_photo, _ = self.obj_gs.optimize_wrt_image_lm(
                            T_C_W=self.T_CW, 
                            image=color,
                            K=K,
                            mask=mask,
                            mask_ref=self.prev_mask,
                            damping=0.01,
                            pyramid_levels=[(4, 10), (2, 10), (1, 20)],
                            gs_type=self.tracker_cfg.gs_type
                        )
                    else:
                        T_WO_photo = T_WO_guess
                else:
                    raise ValueError(f"Unknown photometric_mode: {self.tracker_cfg.photometric_mode}")
                
                T_WO_refined = T_WO_photo.detach().cpu().numpy() if hasattr(T_WO_photo, "detach") else T_WO_photo
                T_CiO_refined = self.T_CW @ T_WO_refined
                T_guess_new = T_CiO_refined @ np.linalg.inv(self.poses[0])

                diff_t = np.linalg.norm(T_guess_new[:3, 3] - T_guess_rel[:3, 3])
                diff_R = np.rad2deg(np.arccos(np.clip((np.trace(T_guess_new[:3, :3] @ T_guess_rel[:3, :3].T) - 1) / 2, -1, 1)))
                
                # Check against Any4D if present to prevent photometric drift
                reject_photo = (diff_t > 0.5 or diff_R > 20.0)
                if not reject_photo and any4d_hint is not None and any4d_hint[0] is not None:
                    T_a4d_rel = any4d_hint[0] @ np.linalg.inv(self.poses[0])
                    diff_a4d_t = np.linalg.norm(T_guess_new[:3, 3] - T_a4d_rel[:3, 3])
                    diff_a4d_R = np.rad2deg(np.arccos(np.clip((np.trace(T_guess_new[:3, :3] @ T_a4d_rel[:3, :3].T) - 1) / 2, -1, 1)))
                    if diff_a4d_t > 0.08 or diff_a4d_R > 12.0:
                        print(f"[BundleSdfGS] Rejecting photometric refinement: drifted {diff_a4d_t:.3f}m, {diff_a4d_R:.1f}deg from Any4D")
                        reject_photo = True

                if not reject_photo:
                    print(f"[BundleSdfGS] Photometric refinement adjusted guess: {diff_t:.6f}m, {diff_R:.6f}deg")
                    T_guess_rel = T_guess_new
                
                self.tracker.poses[self.cnt] = T_guess_rel
                self.timings["Photometric Refinement"].append(time.perf_counter() - t0_photo)
            
            T_guess = T_guess_rel

            # Stage 2: Geometric Tracking
            t0_geo = time.perf_counter()
            success, n_inliers = self.tracker.step_informed_with_occlusion(self.cnt, color, mask, depth, K, T_guess, flow_prev_curr=flow)
            self.timings["Geometric Tracking"].append(time.perf_counter() - t0_geo)
            self.last_pnp_success = success
            self.last_n_inliers = n_inliers
            
            jump = np.linalg.norm(self.tracker.poses[self.cnt][:3, 3] - T_guess[:3, 3])
            max_jump = 1.0 if n_inliers < 50 else 2.0
            if not success or n_inliers < self.tracker_cfg.min_pnp_inliers or jump > max_jump:
                if success and jump > max_jump:
                    print(f"[GeoTracker] Rejecting PnP (jump: {jump:.4f}m > {max_jump}m)")
                
                # Any4D Fallback check
                if any4d_hint is not None and any4d_hint[0] is not None:
                    T_CO_a4d = any4d_hint[0]
                    print(f"[BundleSdfGS] Recovering with Any4D multi-view pose at frame {self.cnt} (PnP inliers: {n_inliers})")
                    T_rel_a4d = T_CO_a4d @ np.linalg.inv(self.poses[0])
                    self.poses[self.cnt] = T_CO_a4d
                    self.tracker.poses[self.cnt] = T_rel_a4d
                elif not success:
                    self.tracker.poses[self.cnt] = T_guess
                    self.poses[self.cnt] = T_guess @ self.poses[0]
                else:
                    self.poses[self.cnt] = self.tracker.poses[self.cnt] @ self.poses[0]
            else:
                self.poses[self.cnt] = self.tracker.poses[self.cnt] @ self.poses[0]
            
            # Sync final pose back to obj_gs for rendering and future frames
            if self.obj_gs is not None:
                self.T_WO = np.linalg.inv(self.T_CW) @ self.poses[self.cnt]
                self.obj_gs.T_W_O = self.T_WO
            
        # Keyframe Logic
        is_kf = False
        if self.cnt == 0:
            is_kf = True
        elif self.cnt - self.keyframes[-1] >= self.tracker_cfg.min_kf_interval:
            T_curr = self.poses[self.cnt]
            is_redundant = False
            for kf_idx in self.keyframes:
                T_kf = self.poses[kf_idx]
                R_diff = T_curr[:3, :3].T @ T_kf[:3, :3]
                rot_diff_deg = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)))
                trans_diff = np.linalg.norm(T_curr[:3, 3] - T_kf[:3, 3])
                
                active_tids_curr = set(tid for tid, t in self.tracker.tracks.items() if self.cnt in t['obs'])
                active_tids_kf = set(tid for tid, t in self.tracker.tracks.items() if kf_idx in t['obs'])
                overlap = len(active_tids_curr.intersection(active_tids_kf)) / max(len(active_tids_curr), 1)
                
                if rot_diff_deg < self.tracker_cfg.min_kf_rot or overlap > self.tracker_cfg.kf_overlap_thresh:
                    is_redundant = True
                    break
            if not is_redundant: is_kf = True
            if self.cnt % 5 == 0: is_kf = True
            if self.cnt < self.tracker_cfg.n_init_frames: is_kf = True

        # Final Pose for this frame: T_CiO
        T_CiO = self.tracker.poses[self.cnt] @ self.poses[0]
        self.poses[self.cnt] = T_CiO
        self.T_WO = np.linalg.inv(self.T_CW) @ T_CiO
        
        # Render if requested
        render_image_pkg = None
        if return_render:
            render_image_pkg = self.render_current_view(K, color.shape[1], color.shape[0], T_CiO)

        # Update keyframe if needed
        if is_kf:
            self.keyframes.append(self.cnt)
            self.tracker.keyframes.append(self.cnt)
            # Update tracker with new points
            # If GS is ready, use optimized GS depth but ONLY for high-confidence regions
            curr_depth = depth
            # Robustly use only monocular depth for points to avoid GS reconstruction noise
            curr_depth = depth
            
            self.tracker.add_new_points_from_depth(
                self.cnt, color, mask, curr_depth, K, 
                self.tracker.poses[self.cnt], 
                align=(self.cnt >= self.tracker_cfg.n_init_frames)
            )
            
            # Update mapper with object-centric pose
            kf_data = {"frame_idx": self.cnt, "image": color, "mask": mask, "depth": depth, "K": K, "T_CiO": T_CiO}
            if self.use_multiprocessing:
                self.kf_queue.put(kf_data)
            else:
                self.run_mapping_step_sync(kf_data)
        
        self.timings["Total run()"].append(time.perf_counter() - t_start)
        self.prev_mask = mask.copy()
        return T_CiO, render_image_pkg

    def force_densify(self, color, mask, depth, K, T_CiO):
        """Signals the mapper to perform an immediate densification step."""
        kf_data = {
            "frame_idx": self.cnt, 
            "image": color, 
            "mask": mask, 
            "depth": depth, 
            "K": K, 
            "T_CiO": T_CiO,
            "force_densify": True
        }
        if self.use_multiprocessing:
            self.kf_queue.put(kf_data)
        else:
            self.run_mapping_step_sync(kf_data)

    def run_mapping_step_sync(self, frame_data):
        """Synchronous mapping step."""
        t0_map = time.perf_counter()
        self.mapper.optimize_frame(frame_data, num_steps=self.cfg_mapping.num_steps_per_frame, frame_count=self.mapping_frame_count)
        self.timings["Mapping (Sync)"].append(time.perf_counter() - t0_map)
        self.gs_ready = True
        self.mapping_frame_count += 1
        
        # After mapping, refresh tracker's internal state to match new geometry
        with torch.no_grad():
            if self.obj_gs.image_ref is not None:
                K = self.tracker.K_dict.get(self.keyframes[-1])
                if K is not None:
                    H, W = self.obj_gs.image_ref.shape[1], self.obj_gs.image_ref.shape[2]
                    render_mode = "3dgs" if self.tracker_cfg.gs_type == "3d" else "normal"
                    img_ref, _, _, alpha_ref = self.obj_gs.gs_params.render(
                        self.obj_gs.T_C_O_ref, torch.from_numpy(K).float().cuda(), 
                        width=W, height=H, mode=render_mode
                    )
                    self.obj_gs.update_reference(img_ref, self.obj_gs.T_C_O_ref, alpha_ref)
            
            # Update geometric tracker's 3D points
            new_points = self.obj_gs.gs_params.means.detach().cpu().numpy()
            self.tracker.update_object_points(new_points)

    def run_ba_python(self):
        if len(self.keyframes) < 2: return
        if self.tracker_cfg.triangulate:
            self.tracker.triangulate_tracks(self.keyframes[-1], self.tracker_cfg.triangulate_thresh)
        run_ba_geometric(self.keyframes, self.tracker.poses, self.tracker.tracks, self.tracker.K_dict, fix_first=(self.keyframes[0] == 0))

    def on_finish(self):
        self.stop_event.set()
        if self.mapping_process:
            self.mapping_process.join()

def run_gs_mapping_process(cfg, initial_gs, kf_queue, p_dict, stop_event):
    """Target function for the GS Mapping process."""
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    
    from obj_gs_mapping import GSMapping, MiniCam, gen_virtul_cam, compute_single_view_loss
    import queue
    import torch
    import sys
    
    print(f"[Mapper] Process started. PID: {os.getpid()}", flush=True)
    
    mapper = GSMapping(cfg, initial_gs)
    print(f"[Mapper] Process started on {cfg.device}")
    mapper.device = torch.device(cfg.device)
    mapper.gs_params.to(mapper.device)
    
    # Re-enable grad for optimization in this process based on config
    params_to_enable = ["quats", "opacity"]
    if cfg.use_ray_dist and mapper.gs_params.ray_dist is not None:
        params_to_enable.append("ray_dist")
    else:
        params_to_enable.append("means")
        
    if not cfg.fix_scale:
        params_to_enable.append("scales")
    if not cfg.fix_color:
        if mapper.gs_params.shs is not None:
            params_to_enable.append("shs")
        else:
            params_to_enable.append("colors")
        
    for attr in params_to_enable:
        p = getattr(mapper.gs_params, attr)
        if p is not None:
            p.requires_grad_(True)
            
    mapper.setup_optimizer()
    
    # Pre-compute disc kernel for densification
    radius = 3
    mapper.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1, device=mapper.device)
    ky, kx = torch.meshgrid(torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1), indexing="ij")
    mapper.disc_kernel[0, 0, torch.sqrt(kx**2 + ky**2) <= radius + 0.5] = 1
    mapper.disc_kernel = mapper.disc_kernel / mapper.disc_kernel.sum()
    
    frame_count = 0
    while not stop_event.is_set():
        try:
            frame_data = None
            while not kf_queue.empty(): frame_data = kf_queue.get_nowait()
            if frame_data is None:
                try: frame_data = kf_queue.get(timeout=0.1)
                except queue.Empty: continue
            
            print(f"[Mapper] Processing frame {frame_data['frame_idx']} (Queue size: {kf_queue.qsize()})")
            mapper.optimize_frame(frame_data, num_steps=cfg.num_steps_per_frame, frame_count=frame_count)
            
            # Detach before putting in shared dict
            latest_gs = mapper.gs_params.clone().to("cpu")
            for attr in ["means", "quats", "scales", "colors", "opacity"]:
                p = getattr(latest_gs, attr)
                if p is not None:
                    setattr(latest_gs, attr, p.detach().requires_grad_(False))
            p_dict['latest_gs'] = latest_gs
            
            p_dict['last_finished_frame'] = frame_data["frame_idx"]
            frame_count += 1
            if frame_count % 1 == 0:
                print(f"[Mapper] Finished frame {frame_data['frame_idx']}, GS count: {len(mapper.gs_params.means)}")
        except Exception as e:
            print(f"[Mapper] Fatal Error: {e}")
            import traceback
            traceback.print_exc()
            break
