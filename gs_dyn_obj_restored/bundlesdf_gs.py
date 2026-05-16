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
multiprocessing.set_start_method('spawn', force=True)

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

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
    feature_type: str = "loftr" # "loftr", "superpoint", "orb", "grid"
    n_features: int = 1000
    min_pnp_inliers: int = 20
    max_pose_jump: float = 1.0 # meters
    use_photometric_refinement: bool = True
    n_init_frames: int = 5 # Number of frames for photometric initialization
    
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
    grid_spacing: int = 8
    ransac_thresh: float = 3.0
    
    # Advanced Filtering (Disable for uncalibrated monocular depth)
    use_occlusion_check: bool = False
    use_informed_filtering: bool = False
    informed_thresh: float = 50.0 # Relaxed pixels
    monocular_scale_alignment: bool = True # Use RelPose to fix scale at start
    
    gs_type: str = "2d" # "2d" or "3d"
    photometric_mode: str = "lm" # "hybrid", "adam", or "lm"
    
    # Depth Alignment Logic
    align_depth: bool = False
    align_with_bias: bool = True

class BundleSdfGS:
    def __init__(self, tracker_cfg: GeoTrackerConfig, cfg_mapping: MappingConfig, use_multiprocessing: bool = True):
        self.tracker_cfg = tracker_cfg
        self.cfg_mapping = cfg_mapping
        self.use_multiprocessing = use_multiprocessing
        
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
        
        self.gs_ready = False
        self.mapping_frame_count = 0
        
        self.depth_scale = 1.0
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

    def init_gs_model(self, color, depth, mask, K, T_CO_init=None):
        """
        Initialize the 3D Gaussian Splatting model from the first frame.
        If T_CO_init is provided, the model is initialized in Object space.
        Otherwise, it's initialized in the Camera frame of frame 0.
        """
        H, W = color.shape[:2]
        from geometric_tracker import detect_features_on_mask, sample_grid_on_mask
        pts2d = sample_grid_on_mask(mask, 2) # Denser init
        ix, iy = np.round(pts2d[:, 0]).astype(int), np.round(pts2d[:, 1]).astype(int)
        z = depth[iy, ix]
        valid = z > 0.01
        pts2d, ix, iy, z = pts2d[valid], ix[valid], iy[valid], z[valid]
        
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        pts3d = np.stack([(pts2d[:, 0] - cx) * z / fx, (pts2d[:, 1] - cy) * z / fy, z], axis=-1)
        
        # Save anchor transformation for coordinate system alignment
        self.T_C0O_anchor = T_CO_init if T_CO_init is not None else np.eye(4)
        
        # Transform points to Object space if anchor is provided
        if T_CO_init is not None:
            T_OC0 = np.linalg.inv(T_CO_init)
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
        
        if T_CO_init is not None:
            normals_o = (normals_c @ T_OC0[:3, :3].T)
        else:
            normals_o = normals_c

        from obj_gs_mapping import init_gs_from_tracker_points
        gs_params = init_gs_from_tracker_points(pts3d, colors, "cuda", normals=normals_o, gs_type=self.tracker_cfg.gs_type)
        self.obj_gs = ObjectGS(gs_params, T_CO_init if T_CO_init is not None else np.eye(4), 1.0)
        
        if self.use_multiprocessing:
            # Start Mapper Process
            gs_params_cpu = gs_params.clone().to("cpu")
            for attr in ["means", "quats", "scales", "colors", "opacity"]:
                p = getattr(gs_params_cpu, attr)
                if p is not None:
                    setattr(gs_params_cpu, attr, p.detach().requires_grad_(False))
            
            self.mapping_process = multiprocessing.Process(
                target=run_gs_mapping_process, 
                args=(self.cfg_mapping, gs_params_cpu, self.kf_queue, self.p_dict, self.stop_event)
            )
            self.mapping_process.start()
        else:
            # Initialize Synchronous Mapper
            from obj_gs_mapping import GSMapping
            self.mapper = GSMapping(self.cfg_mapping, gs_params)
            self.mapper.device = torch.device("cuda")
            self.mapper.gs_params.to(self.mapper.device)
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

    def run(self, color, mask, depth, K, T_CO_init=None, **kwargs):
        """
        Main tracking loop.
        """
        self.cnt += 1
        depth = depth * self.depth_scale
        H, W = color.shape[:2]
        
        if self.cnt == 0:
            self.prev_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            self.T_C0O_anchor = T_CO_init.copy() if T_CO_init is not None else np.eye(4)
            self.tracker.poses[0] = np.eye(4)
            self.tracker.K_dict[0] = K
            self.tracker.add_new_points_from_depth(0, color, mask, depth, K, np.eye(4), align=True, spacing=4)
            self.keyframes.append(0)
            self.tracker.keyframes.append(0)
            self.init_gs_model(color, depth, mask, K, self.T_C0O_anchor)
            self.poses[0] = self.T_C0O_anchor
            self.prev_mask = mask.copy()
            self.prev_depth = depth.copy()
            self.prev_color = color.copy()
            return self.poses[0], None
        else:
            curr_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            
            # Frame 1: Scale Alignment if enabled
            if self.cnt == 1 and self.tracker_cfg.monocular_scale_alignment and poselib is not None:
                try:
                    # Get correspondences via flow
                    from geometric_tracker import interpolate_flow, sample_grid_on_mask
                    pts0 = sample_grid_on_mask(self.prev_mask, self.tracker_cfg.grid_spacing)
                    flow01 = self.dis.calc(self.prev_gray, curr_gray, None)
                    pts1 = pts0 + interpolate_flow(flow01, pts0)
                    
                    # Filter by current mask
                    ix, iy = np.round(pts1[:, 0]).astype(int), np.round(pts1[:, 1]).astype(int)
                    valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
                    valid[valid] &= (mask[iy[valid], ix[valid]] > 0)
                    pts0, pts1 = pts0[valid], pts1[valid]
                    
                    if len(pts0) > 20:
                        d0 = self.prev_depth[np.round(pts0[:, 1]).astype(int), np.round(pts0[:, 0]).astype(int)]
                        d1 = depth[np.round(pts1[:, 1]).astype(int), np.round(pts1[:, 0]).astype(int)]
                        
                        cam0 = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [K[0,0], K[1,1], K[0,2], K[1,2]]}
                        res_rel, _ = poselib.estimate_monodepth_relative_pose(pts0, pts1, d0, d1, cam0, cam0, {'max_reproj_error': self.tracker_cfg.ransac_thresh})
                        
                        if res_rel:
                            self.depth_scale = res_rel.scale
                            print(f"[BundleSdfGS] Estimated depth scale from RelPose: {self.depth_scale:.4f}")
                            # Scale the current depth map as well for the rest of this frame
                            depth = depth * self.depth_scale
                            
                            for tid, t in self.tracker.tracks.items():
                                t['pt3d'] *= self.depth_scale
                            self.obj_gs.gs_params.means.data *= self.depth_scale
                            self.obj_gs.gs_params.scales.data += np.log(self.depth_scale)
                except Exception as e:
                    print(f"[BundleSdfGS] Scale alignment failed: {e}")

            self._update_gs_from_mapper()
            
            if self.prev_gray is None:
                self.prev_gray = curr_gray
                return self.poses.get(self.cnt-1, np.eye(4))
                
            curr_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            flow = self.dis.calc(self.prev_gray, curr_gray, None)
            self.prev_gray = curr_gray

            if self.cnt > 1:
                T_pprev_rel = self.tracker.poses[self.cnt-2]
                T_prev_rel = self.tracker.poses[self.cnt-1]
                T_guess_rel = T_prev_rel @ (np.linalg.inv(T_pprev_rel) @ T_prev_rel)
            else:
                T_guess_rel = self.tracker.poses[self.cnt-1].copy()

            # Stage 1: Photometric Initialization / Refinement
            use_photometric = self.tracker_cfg.use_photometric_refinement and (self.cnt < self.tracker_cfg.n_init_frames or self.gs_ready)
            
            if use_photometric and self.obj_gs is not None and self.obj_gs.gs_params is not None:
                # Ensure tracker uses up-to-date GS model (mapper is used in sync mode, obj_gs in multi-proc)
                if not self.use_multiprocessing and self.mapper is not None:
                    self.obj_gs.gs_params = self.mapper.gs_params
                
                # Use object GS for photometric refinement
                # We optimize for T_CiC0 (relative motion to frame 0)
                # The object is fixed at self.T_C0O_anchor in the C0 frame.
                self.obj_gs.T_W_O = self.T_C0O_anchor
                if self.tracker_cfg.photometric_mode == "hybrid":
                    T_W_O_photo, _ = self.obj_gs.optimize_wrt_image_lm_hybrid(
                        T_C_W=T_guess_rel, 
                        image=color,
                        K=K,
                        mask=mask,
                        mask_ref=self.prev_mask,
                        adam_steps=5, 
                        lm_steps=15,
                        damping=500.0,
                        pyramid_levels=[(1, 20)], # Flat pyramid for speed
                        gs_type=self.tracker_cfg.gs_type
                    )
                elif self.tracker_cfg.photometric_mode == "adam":
                    T_W_O_photo, _ = self.obj_gs.optimize_wrt_image(
                        T_C_W=T_guess_rel, 
                        image=color,
                        K=K,
                        mask=mask,
                        mask_ref=self.prev_mask,
                        lr=1e-3,
                        num_steps=20, # Equivalent to total steps in hybrid
                        gs_type=self.tracker_cfg.gs_type
                    )
                elif self.tracker_cfg.photometric_mode == "lm":
                    T_W_O_photo, _ = self.obj_gs.optimize_wrt_image_lm(
                        T_C_W=T_guess_rel, 
                        image=color,
                        K=K,
                        mask=mask,
                        mask_ref=self.prev_mask,
                        damping=500.0,
                        pyramid_levels=[(1, 20)],
                        gs_type=self.tracker_cfg.gs_type
                    )
                else:
                    raise ValueError(f"Unknown photometric_mode: {self.tracker_cfg.photometric_mode}")
                # The returned T_W_O_photo is the refined object pose in C0 frame.
                # The total refined Ci-from-O pose is T_CiC0_guess @ T_C0O_refined.
                # We want the new T_CiC0 = T_CiO @ inv(T_C0O_anchor)
                T_W_O_refined_np = T_W_O_photo.detach().cpu().numpy() if hasattr(T_W_O_photo, "detach") else T_W_O_photo
                T_CiO_refined = T_W_O_refined_np # The optimizer refined the full T_CiO
                T_guess_new = T_CiO_refined @ np.linalg.inv(self.T_C0O_anchor)
                
                diff_t = np.linalg.norm(T_guess_new[:3, 3] - T_guess_rel[:3, 3])
                diff_R = np.rad2deg(np.arccos(np.clip((np.trace(T_guess_new[:3, :3] @ T_guess_rel[:3, :3].T) - 1) / 2, -1, 1)))
                if diff_t > 1e-6 or diff_R > 1e-4:
                    print(f"[BundleSdfGS] Photometric refinement adjusted guess: {diff_t:.6f}m, {diff_R:.6f}deg")
                
                T_guess_rel = T_guess_new
                self.tracker.poses[self.cnt] = T_guess_rel
            
            T_guess = T_guess_rel

            # Stage 2: Geometric Tracking
            success, n_inliers = self.tracker.step_informed_with_occlusion(self.cnt, color, mask, depth, K, T_guess, flow_prev_curr=flow)
            
            if not success or n_inliers < self.tracker_cfg.min_pnp_inliers:
                self.tracker.add_new_points_from_depth(self.cnt, color, mask, depth, K, self.tracker.poses.get(self.cnt, T_guess), align=True)
                if not success:
                    self.poses[self.cnt] = T_guess
                    self.tracker.poses[self.cnt] = T_guess
                else:
                    self.poses[self.cnt] = self.tracker.poses[self.cnt]
            else:
                self.poses[self.cnt] = self.tracker.poses[self.cnt]
            
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

        # Final Pose for this frame: T_CiO = T_CiC0 @ T_C0O
        T_CiO = self.tracker.poses[self.cnt] @ self.T_C0O_anchor
        self.poses[self.cnt] = T_CiO
        
        # Update keyframe if needed
        if is_kf:
            self.keyframes.append(self.cnt)
            self.tracker.keyframes.append(self.cnt)
            self.tracker.add_new_points_from_depth(self.cnt, color, mask, depth, K, self.tracker.poses[self.cnt], align=(self.cnt >= self.tracker_cfg.n_init_frames))
            
            # Update mapper with object-centric pose
            kf_data = {"frame_idx": self.cnt, "image": color, "mask": mask, "depth": depth, "K": K, "T_CiO": T_CiO}
            if self.use_multiprocessing:
                self.kf_queue.put(kf_data)
            else:
                self.run_mapping_step_sync(kf_data)
        
        self.prev_mask = mask.copy()
        
        # Render if requested
        render_image_pkg = None
        if kwargs.get('return_render', False) and self.obj_gs is not None:
            with torch.no_grad():
                render_mode = "3dgs" if self.tracker_cfg.gs_type == "3d" else "normal"
                img, depth_r, normal_r, alpha = self.obj_gs.render(
                    torch.from_numpy(T_CiO).float().to("cuda"),
                    torch.from_numpy(K).float().to("cuda"),
                    W, H, mode=render_mode
                )
                render_image_pkg = (img, alpha)

        return T_CiO, render_image_pkg

    def run_mapping_step_sync(self, frame_data):
        """Synchronous mapping step."""
        self.mapper.optimize_frame(frame_data, num_steps=self.cfg_mapping.num_steps_per_frame, frame_count=self.mapping_frame_count)
        
        # Periodic densification and pruning
        if self.mapping_frame_count > 0 and self.mapping_frame_count % self.cfg_mapping.densify_every == 0:
            kf_cam = MiniCam(frame_data["K"], frame_data["T_CiO"], frame_data["image"].shape[1], frame_data["image"].shape[0])
            image_t = torch.from_numpy(frame_data["image"]).float().to("cuda").permute(2, 0, 1) / 255.0
            mask_t = torch.from_numpy(frame_data["mask"] > 0).to("cuda")
            T_CO_t = torch.from_numpy(frame_data["T_CiO"]).float().to("cuda")
            
            self.mapper.densify(image_t, frame_data["depth"], mask_t, kf_cam, T_CO_t)
            if hasattr(self.mapper, "prune"):
                self.mapper.prune(T_CO_t, kf_cam)
                
        self.gs_ready = True
        self.mapping_frame_count += 1

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
    from obj_gs_mapping import GSMapping, MiniCam, gen_virtul_cam, compute_single_view_loss
    import queue
    
    mapper = GSMapping(cfg, initial_gs)
    mapper.device = torch.device(cfg.device)
    mapper.gs_params.to(mapper.device)
    
    # Re-enable grad for optimization in this process based on config
    params_to_enable = ["means", "quats", "opacity"]
    if not cfg.fix_scale:
        params_to_enable.append("scales")
    if not cfg.fix_color:
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
            
            image_t = torch.from_numpy(frame_data["image"]).float().to(mapper.device).permute(2, 0, 1) / 255.0
            mask_t = torch.from_numpy(frame_data["mask"] > 0).bool().to(mapper.device)
            T_CO_t = torch.from_numpy(frame_data["T_CiO"]).float().to(mapper.device)
            K_t = torch.from_numpy(frame_data["K"]).float().to(mapper.device)
            cam = MiniCam(frame_data["K"], frame_data["T_CiO"], image_t.shape[2], image_t.shape[1])
            
            for _ in range(cfg.num_steps_per_frame):
                if stop_event.is_set(): break
                mapper.optimizer.zero_grad()
                render_mode = "3dgs" if cfg.gs_type == "3d" else "normal"
                res = mapper.gs_params.render(T_CO_t, K_t, cam.width, cam.height, mode=render_mode)
                r_img, _, _, r_alpha = res
                loss = F.l1_loss(r_img * mask_t, image_t * mask_t) + F.l1_loss(r_alpha[0], mask_t.float())
                loss.backward()
                mapper.optimizer.step()
                
            if frame_count % cfg.densify_every == 0:
                mapper.densify(image_t, frame_data["depth"], mask_t, cam, T_CO_t)
                mapper.prune(T_CO_t, cam)
                # Detach before putting in shared dict
                latest_gs = mapper.gs_params.clone().to("cpu")
                for attr in ["means", "quats", "scales", "colors", "opacity"]:
                    p = getattr(latest_gs, attr)
                    if p is not None:
                        setattr(latest_gs, attr, p.detach().requires_grad_(False))
                p_dict['latest_gs'] = latest_gs
            
            p_dict['last_finished_frame'] = frame_data["frame_idx"]
            frame_count += 1
            if frame_count % 5 == 0:
                logging.info(f"Mapper Progress: {frame_count} frames, GS count: {len(mapper.gs_params.means)}")
        except Exception as e:
            logging.error(f"Mapper Process Error: {e}")
            break
