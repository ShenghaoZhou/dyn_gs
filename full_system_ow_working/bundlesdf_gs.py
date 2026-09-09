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
import queue
from dataclasses import dataclass, replace
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
# NOTE: the old `from utils.loss_utils import ssim` here (plus a sys.path.append
# for third_party/PGSR) was dead -- `ssim` is never used in this module and the
# PGSR directory does not exist. Removed; loss_utils.py has never existed in the
# repo, so importing it crashed startup. See gs_dyn_obj/utils/ssim.py for ssim.

@dataclass
class GeoTrackerConfig:
    feature_type: str = "orb" # "loftr", "superpoint", "orb", "grid"
    n_features: int = 2000 # Increased for better robustness
    min_pnp_inliers: int = 20
    # Max |pose jump| of PnP vs the predicted guess, in meters.
    # The original value was 10.0 with the comment "Effectively disable rejection
    # to match benchmark" -- i.e. rejection had been deliberately switched off.
    # That meant every PnP estimate was stored as truth and then reused as the
    # next frame's velocity, so one bad estimate poisoned every later frame.
    # 0.05 m is ~2x the GT per-frame motion on this clip (median 4.6 mm, max
    # 22 mm) and ~40x smaller than the 1.2-1.9 m jumps the tracker actually
    # produced. BundleSdfGS also widens this adaptively from observed accepted
    # motion (see _jump_limit).
    max_pose_jump: float = 0.05
    # Translation / rotation limits for the photometric-refinement gate. These
    # used to live only in GlobalConfig and were never read -- the comparison in
    # step() was hardcoded to 0.5 m / 20.0 deg. They are now the floors for the
    # adaptive limits returned by _jump_limit().
    max_photo_jump_t: float = 0.05
    max_photo_jump_R: float = 3.0
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
    # Depth alignment for the tracker. Kept OFF and wired explicitly from the
    # runner: estimate_depth_alignment() regresses the depth buffer against
    # t['pt3d'], which was itself lifted from the same buffer and then overwritten
    # by GS means from the same buffer, so it is circular and cannot recover an
    # unknown absolute scale -- only per-frame drift against already-tracked
    # points. The mapper has its own separate align_depth flag
    # (obj_gs_mapping.MappingConfig); that is the one the pipeline calibrates on.
    #
    # The old monocular_scale_alignment flag was removed: it was read by nobody
    # except the warning below, so it advertised a calibration that never ran.
    align_depth: bool = False
    align_with_bias: bool = True
    
    # When PnP rejects a frame, withhold that frame from the GS keyframe buffer.
    #
    # Measured OFF. A paired 5-clip A/B (same code, same seed, same clip, only
    # this flag differing) found withholding rejected keyframes made things
    # worse everywhere: 0/5 clips improved on scale-free ATE/travel, and
    # aggregate mean ATE/travel went 0.215 -> 0.371 (+72%), mean Umeyama RMSE
    # 0.1015 -> 0.1247 (+23%). The cost is map sparsity -- withheld frames are
    # views the mapper never sees -- and it exceeds whatever the poisoning
    # would have caused. A rejected pose is bad data, but it is still a view.
    # Kept as a switch so the A/B stays reproducible; default is OFF.
    gate_kf_commit: bool = False
    gs_type: str = "2d" # "2d" or "3d"
    photometric_mode: str = "lm" # "hybrid", "adam", or "lm"
    use_match_projections: bool = False
    # Snap the geometric tracker's track points onto optimized GS means after each
    # mapping update. Set False to keep the two representations independent.
    update_tracker_points_from_gs: bool = True
    debug: bool = False

def prepare_gs_optimizer(mapper, cfg, device):
    """Enable per-attribute grads, set up the optimizer, and build the densification kernel.

    Shared by the synchronous and asynchronous mapper paths so their optimization
    setup cannot drift apart.
    """
    params_to_enable = ["quats", "opacity"]
    if cfg.use_ray_dist and mapper.gs_params.ray_dist is not None:
        params_to_enable.append("ray_dist")
    else:
        params_to_enable.append("means")
    if not cfg.fix_scale:
        params_to_enable.append("scales")
    if not cfg.fix_color:
        params_to_enable.append("shs" if mapper.gs_params.shs is not None else "colors")
    for attr in params_to_enable:
        p = getattr(mapper.gs_params, attr)
        if p is not None:
            p.requires_grad_(True)

    mapper.setup_optimizer()

    radius = 3
    disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1, device=device)
    ky, kx = torch.meshgrid(torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1), indexing="ij")
    disc_kernel[0, 0, torch.sqrt(kx ** 2 + ky ** 2) <= radius + 0.5] = 1
    mapper.disc_kernel = disc_kernel / disc_kernel.sum()


class BundleSdfGS:
    def __init__(self, tracker_cfg: GeoTrackerConfig, cfg_mapping: MappingConfig, use_multiprocessing: bool = True, initial_gs: GSParam = None):
        self.tracker_cfg = tracker_cfg
        self.cfg_mapping = cfg_mapping
        self.use_multiprocessing = use_multiprocessing
        self.initial_gs_param = initial_gs
        
        # Give the tracker its own copy of the config. step() mutates
        # tracker.cfg.max_pose_jump every frame to apply the adaptive jump limit
        # (_jump_limit). If the tracker shared the declarative tracker_cfg, that
        # mutation would leak back and ratchet the floor upward frame over frame,
        # silently disabling rejection again.
        self.tracker = GeometricTracker(replace(tracker_cfg))
        # Ensure tracker's internal cfg has the flags
        self.tracker.cfg.use_occlusion_check = tracker_cfg.use_occlusion_check
        self.tracker.cfg.use_informed_filtering = tracker_cfg.use_informed_filtering
        self.tracker.cfg.informed_thresh = tracker_cfg.informed_thresh
        self.obj_gs = None
        self.cnt = -1
        self.keyframes = []
        self.keyframes_data = [] # To store fd for evaluation # Indices of keyframes
        self.poses = {} # Combined poses (C0-from-Ci)

        # Frame indices whose PnP estimate was actually ACCEPTED, plus the
        # translation magnitude of each accepted step. Velocity prediction and
        # the pose-jump limit are derived from these only.
        #
        # Rationale: the tracker is fully autoregressive. T_prev_rel is the most
        # recently STORED pose, and a rejected frame writes the biased guess back
        # as if it were measured (see the rejection branch below). With the
        # baseline velocity cap of 0.5 m/frame, drift of ~0.024 m/frame accumulated
        # undetected for 100+ frames because it never approached the 1.0 m
        # rejection threshold, and each step was then fed forward as "truth".
        # Restricting both the velocity and the limit to accepted frames breaks
        # that loop: a rejected frame freezes the estimate instead of amplifying it.
        self.accepted_rel = [] # frame indices whose PnP estimate was accepted
        self.accepted_step_t = []
        self.accepted_step_r = []
        self.pnp_rejections = 0
        # Keyframes that the periodic/init rule wanted to take but the gate
        # suppressed (only nonzero when gate_kf_commit is True, which defaults
        # False -- see GeoTrackerConfig). Counted for the summary so a drop in
        # keyframe count is not mistaken for the mapper stalling.
        self.kf_skipped_reject = 0
        
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
        self.mapping_process = None
        self.mapper = None
        
        # Optical Flow for prediction
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev_gray = None
        self.prev_mask = None

        # Surface config combinations that silently weaken the pipeline.
        # The old monocular_scale_alignment/align_depth warning here fired on
        # every single run because the runner never passed either flag through,
        # so it was pure noise. That wiring is now explicit (see
        # align_depth_tracker in the runner) and the dead flag is gone.
        if not (tracker_cfg.use_occlusion_check or tracker_cfg.use_informed_filtering):
            logging.warning(
                "[BundleSdfGS] use_occlusion_check and use_informed_filtering are both off: "
                "geometric tracking is pure optical-flow projection + RANSAC PnP, "
                "with no depth consistency check or track recovery."
            )
        
    def _jump_limit(self):
        """Adaptive pose-jump limits from observed ACCEPTED inter-frame motion.

        Returns (max_t_m, max_r_deg). The config values are FLOORS, not caps:
        observed real motion may widen the limit, but never shrink it below the
        operator-chosen threshold.

        The multipliers are chosen against GT for this clip: per-frame translation
        is 4.6 mm median / 22 mm max, so 8x the median (3.7 cm) clears the fastest
        real frame while staying 30x+ below the 1.2-1.9 m jumps the pipeline was
        producing. Rotation is 0.60 deg median / 2.04 deg max, so 3x the median
        with a 3 deg floor.
        """
        floor_t = max(self.tracker_cfg.max_pose_jump, self.tracker_cfg.max_photo_jump_t)
        floor_r = self.tracker_cfg.max_photo_jump_R
        st = self.accepted_step_t[-50:]
        sr = self.accepted_step_r[-50:]
        med_t = float(np.median(st)) if st else 0.01
        med_r = float(np.median(sr)) if sr else 0.5
        return max(floor_t, 8.0 * med_t), max(floor_r, 3.0 * med_r)

    @property
    def mapper_last_finished_frame(self):
        return self.p_dict.get('last_finished_frame', -1)

    def _gs_points_in_tracker_frame(self):
        """Return GS means in the frame the geometric tracker stores track points in.

        GS means are in object space, but the tracker seeds its first points with
        T_CiO = I (see run()), so its 'object' frame is really the frame-0 camera.
        Comparing the two directly made the 5cm nearest-neighbour gate fail almost
        entirely, so the GS -> tracker update never happened.
        """
        T_C0O = self.T_C0O_anchor if getattr(self, "T_C0O_anchor", None) is not None else self.poses.get(0, np.eye(4))
        means_o = self.obj_gs.gs_params.means.detach().cpu().numpy()
        return means_o @ T_C0O[:3, :3].T + T_C0O[:3, 3]

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
                if self.tracker_cfg.update_tracker_points_from_gs:
                    self.tracker.update_object_points(self._gs_points_in_tracker_frame())

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
        
        prepare_gs_optimizer(self.mapper, self.cfg_mapping, self.mapper.device)
        self.mapper.keyframes = []
        self.mapping_frame_count = 0
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

    def _enqueue_keyframe(self, kf_data):
        """Hand a keyframe to the mapper without ever blocking the tracker.

        The tracker is the real-time path, so it must never stall on the mapper.
        When the queue is full we evict the oldest pending keyframe and keep the
        newest -- the mapper only processes the last drained frame anyway.
        """
        if not self.use_multiprocessing:
            self.run_mapping_step_sync(kf_data)
            return

        try:
            self.kf_queue.put(kf_data, block=False)
            return
        except queue.Full:
            pass

        try:
            dropped = self.kf_queue.get_nowait()
            logging.warning(
                f"[BundleSdfGS] Mapper queue full; dropped keyframe "
                f"{dropped.get('frame_idx')} to accept keyframe {kf_data['frame_idx']}"
            )
        except queue.Empty:
            pass

        try:
            self.kf_queue.put(kf_data, block=False)
        except queue.Full:
            logging.warning(f"[BundleSdfGS] Dropped keyframe {kf_data['frame_idx']} (mapper too slow)")

    def run(self, color, mask, depth, K, T_CW=None, T_WO_init=None, return_render=False):
        """
        Main tracking loop. Now supports T_OW parameterization.
        T_CW: Camera-from-World (extrinsics)
        T_WO_init: Object-in-World at start
        """
        t_start = time.perf_counter()
        self.cnt += 1
        # Frame 0 has no PnP (no previous pose to compare against), so the
        # geometric stage never runs and the keyframe commit below would read an
        # undefined name. Default to "not rejected" and let the geometric stage
        # overwrite it for cnt > 0.
        pnp_rejected = False
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
            curr_gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            
            self._update_gs_from_mapper()
            
            if self.prev_gray is None:
                self.prev_gray = curr_gray
                return self.poses.get(self.cnt-1, np.eye(4)), None
                
            flow = self.dis.calc(self.prev_gray, curr_gray, None)
            self.prev_gray = curr_gray
            self.timings["Pre-processing (Flow)"].append(time.perf_counter() - t0)

            if len(self.accepted_rel) >= 2:
                # Velocity from the two most recent ACCEPTED frames, not the two
                # most recently STORED ones. A rejected frame stores its biased
                # guess as if it were a measurement, so the old indexing
                # (poses[cnt-1] / poses[cnt-2]) folded that bias into V and then
                # re-used it as the next frame's prior -- the autoregressive
                # amplification behind the frame-130 blowup.
                a_prev, a_pprev = self.accepted_rel[-1], self.accepted_rel[-2]
                T_prev_rel = self.tracker.poses[a_prev]
                T_pprev_rel = self.tracker.poses[a_pprev]
                V = T_prev_rel @ np.linalg.inv(T_pprev_rel)

                # Report only. The old code halved the step and then kept using
                # it anyway, which still forwarded the bad estimate.
                v_t = np.linalg.norm(V[:3, 3])
                v_R = np.rad2deg(np.arccos(np.clip((np.trace(V[:3, :3]) - 1) / 2, -1, 1)))
                if v_t > 0.5 or v_R > 20.0:
                    print(f"[BundleSdfGS] Unusual accepted-step velocity: {v_t:.4f}m, {v_R:.2f}deg")

                T_guess_rel = V @ T_prev_rel
            else:
                # Not enough accepted evidence to extrapolate: predict no motion
                # rather than inventing one. Matches the old behaviour for the
                # first frames, where poses[0] is the identity anchor.
                T_guess_rel = np.eye(4)

            # One motion-adaptive limit, applied to both rejection gates below and
            # to the tracker's internal PnP gate (assigned before the call).
            # Setting self.tracker.cfg is sufficient: geometric_tracker reads it
            # with getattr(self.cfg, "max_pose_jump", ...) at each call, so the
            # per-frame value is picked up. Previously the three sites disagreed
            # (tracker 0.1/1.0, here 1.0/2.0, photometric 0.5/20.0), and the
            # 1.0 m effective threshold was 220x the GT per-frame motion, so it
            # never fired.
            jump_t_lim, jump_r_lim = self._jump_limit()
            self.tracker.cfg.max_pose_jump = jump_t_lim

            # Stage 1: Photometric Initialization / Refinement
            use_photometric = self.tracker_cfg.use_photometric_refinement and (self.cnt < self.tracker_cfg.n_init_frames or self.gs_ready)
            
            if use_photometric and self.obj_gs is not None and self.obj_gs.gs_params is not None:
                # Ensure tracker uses up-to-date GS model (mapper is used in sync mode, obj_gs in multi-proc)
                if not self.use_multiprocessing and self.mapper is not None:
                    self.obj_gs.gs_params = self.mapper.gs_params
                
                t0_photo = time.perf_counter()
                # The object is parameterized as T_WO (Object in World)
                # We optimize for T_WO directly.
                self.T_CW = T_CW if T_CW is not None else np.eye(4)
                
                # T_WO_guess is current self.T_WO (updated from last frame's velocity if available)
                if self.cnt > 1:
                    # Update T_WO_guess using velocity in World Frame if possible, 
                    # but BundleSdfGS uses relative camera motion usually.
                    # Here we keep it simple and use T_guess_rel @ self.poses[0] @ inv(T_CW)
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

                
                # Safety check: reject if photometric refinement moved too much.
                # Shares the adaptive limit with the geometric gate below -- one
                # frame cannot move further than that whether the estimate came
                # from LM on the GS or from RANSAC PnP. The comparison was
                # hardcoded to 0.5 m / 20 deg and logged with logging.info, which
                # is invisible at the WARNING root level, so these rejections
                # never appeared in the logs.
                diff_t = np.linalg.norm(T_guess_new[:3, 3] - T_guess_rel[:3, 3])
                diff_R = np.rad2deg(np.arccos(np.clip((np.trace(T_guess_new[:3, :3] @ T_guess_rel[:3, :3].T) - 1) / 2, -1, 1)))

                if diff_t > jump_t_lim or diff_R > jump_r_lim:
                    print(
                        f"[BundleSdfGS] Rejecting photometric refinement "
                        f"({diff_t:.4f}m, {diff_R:.2f}deg > {jump_t_lim:.4f}m / {jump_r_lim:.2f}deg); "
                        f"keeping geometric guess"
                    )
                else:
                    print(f"[BundleSdfGS] Photometric refinement adjusted guess: {diff_t:.6f}m, {diff_R:.6f}deg")
                    T_guess_rel = T_guess_new
                
                self.tracker.poses[self.cnt] = T_guess_rel
                self.timings["Photometric Refinement"].append(time.perf_counter() - t0_photo)
            
            T_guess = T_guess_rel

            # Stage 2: Geometric Tracking
            t0_geo = time.perf_counter()
            success, n_inliers = self.tracker.step_informed_with_occlusion(self.cnt, color, mask, depth, K, T_guess, flow_prev_curr=flow)
            self.timings["Geometric Tracking"].append(time.perf_counter() - t0_geo)
            
            jump = np.linalg.norm(self.tracker.poses[self.cnt][:3, 3] - T_guess[:3, 3])
            pnp_rejected = (not success) or (n_inliers < self.tracker_cfg.min_pnp_inliers) or (jump > jump_t_lim)
            if pnp_rejected:
                self.pnp_rejections += 1
                print(
                    f"[BundleSdfGS] Rejecting PnP (inliers={n_inliers}, jump={jump:.4f}m "
                    f"> max_jump={jump_t_lim:.4f}m); falling back to velocity guess"
                )
                # step_informed_with_occlusion stores the rejected pose in tracker.poses,
                # so write the guess back explicitly -- reading tracker.poses[cnt] here
                # would silently re-accept the pose we just rejected.
                self.tracker.poses[self.cnt] = T_guess
                self.poses[self.cnt] = T_guess
                # DON'T add points yet as the pose is suspect. Critically, cnt is
                # NOT appended to accepted_rel: the frozen estimate becomes the
                # anchor for the next frame's velocity.
            else:
                self.poses[self.cnt] = self.tracker.poses[self.cnt]
                # Record the accepted step for velocity prediction and for
                # calibrating next frame's jump limit.
                if self.accepted_rel:
                    T_prev = self.tracker.poses[self.accepted_rel[-1]]
                    self.accepted_step_t.append(float(np.linalg.norm(
                        self.tracker.poses[self.cnt][:3, 3] - T_prev[:3, 3])))
                    self.accepted_step_r.append(float(np.rad2deg(np.arccos(np.clip(
                        (np.trace(self.tracker.poses[self.cnt][:3, :3].T @ T_prev[:3, :3]) - 1) / 2, -1, 1)))))
                self.accepted_rel.append(self.cnt)
                # If we are low on points AND PnP was successful, add more
                if len(self.tracker.tracks) < 300:
                    self.tracker.add_new_points_from_depth(self.cnt, color, mask, depth, K, self.tracker.poses[self.cnt], align=True)
            
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
            # Deliberately NOT gated on pnp_rejected. Two earlier attempts gated
            # this and both starved the tracker: is_kf drives add_new_points_from_depth
            # below, which is the tracker's ONLY replenishment path. Withhold it and
            # len(pts2d) falls under min_pnp_inliers, so geometric_tracker returns
            # (False, 0) -- the "inliers=0" logs -- every subsequent frame rejects,
            # the pose freezes at the velocity guess, and nothing ever recovers.
            # Measured: 67/150 rejections -> 140/150 with mean ATE 0.19 -> 0.45.
            #
            # What IS gated on pnp_rejected is the GS keyframe commit below, which
            # is the actual poison: the mapper optimizes Gaussians toward
            # frame_data["T_CiO"], so a velocity-guess pose bakes the error into
            # the means and update_object_points() then snaps surviving tracks
            # onto those wrong means.
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
            # Update tracker with new points. Runs on the keyframe cadence and is
            # deliberately UNGATED on pnp_rejected -- see the is_kf comment above.
            # Injecting at a suspect pose is mildly wrong but harmless; skipping it
            # starves the tracker and rejects every subsequent frame.
            self.tracker.add_new_points_from_depth(
                self.cnt, color, mask, depth, K,
                self.tracker.poses[self.cnt],
                align=(self.cnt >= self.tracker_cfg.n_init_frames)
            )

            # This gate defaults to OFF -- see GeoTrackerConfig.gate_kf_commit.
            # The hypothesis was that committing a velocity-guess pose poisons the
            # map, because the mapper optimizes Gaussians toward frame_data["T_CiO"]
            # and update_object_points() then snaps surviving tracks onto those
            # means. A paired 5-clip A/B tested exactly that and it did not hold:
            # withholding the frame cost the mapper a view every time and never
            # reduced the error, so the poisoned-view harm was smaller than the
            # sparsity harm. Two side observations from the same run:
            #   * Rejection counts did not diverge. Gate ON 17/89/8/72/54 vs
            #     gate OFF 13/116/17/85/51 (same clip order), comparable on every
            #     clip, and the gate-OFF runs did NOT enter the unrecoverable
            #     state. So the unrecoverable death spiral traced to gating is_kf
            #     (which starves add_new_points_from_depth), NOT to committing a
            #     rejected keyframe. is_kf stays ungated for that reason.
            #   * The depth scale error tracked rejection rate, not the gate.
            #     Low-rejection clips (8-17/150) came out near metric, s=0.86-1.11.
            #     High-rejection clips (72-116/150) came out compressed,
            #     s=0.17-0.52 -- both variants of clip-003318 sat at s~0.17-0.18.
            #     One mid-rejection clip (51-54/150) came out EXPANDED, s=1.58-2.71,
            #     so the sign of the error is not fixed, only its magnitude. Either
            #     way, rejecting banks the velocity guess, which on those clips is
            #     smaller than the true motion, and that is the dominant error
            #     source across the dataset. This gate does not touch it.
            if self.tracker_cfg.gate_kf_commit and pnp_rejected:
                self.kf_skipped_reject += 1
                print(
                    f"[BundleSdfGS] Frame {self.cnt} is a keyframe but PnP was "
                    f"rejected: tracker points injected, GS keyframe withheld"
                )
            else:
                self.keyframes.append(self.cnt)
                self.tracker.keyframes.append(self.cnt)
                # Update mapper with object-centric pose
                kf_data = {"frame_idx": self.cnt, "image": color, "mask": mask, "depth": depth, "K": K, "T_CiO": T_CiO}
                self._enqueue_keyframe(kf_data)
        
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
        self._enqueue_keyframe(kf_data)

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
            if self.tracker_cfg.update_tracker_points_from_gs:
                self.tracker.update_object_points(self._gs_points_in_tracker_frame())

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
    prepare_gs_optimizer(mapper, cfg, mapper.device)
    
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
