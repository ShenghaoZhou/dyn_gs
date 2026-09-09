import sys
import os
from pathlib import Path
from typing import Literal

# Add current folder to sys.path for self-contained imports
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Imports from project
from dataclasses import dataclass
import tyro
import torch
import numpy as np
import cv2
import rerun as rr
import time
import queue
import torch.nn.functional as F
import torch.multiprocessing as mp
from collections import defaultdict
import math
from tqdm import tqdm

# Imports for Dynamic System (BundleGS)
from bundlesdf_gs import BundleSdfGS, GeoTrackerConfig
from obj_gs_mapping import MappingConfig, GSMapping, init_gs_from_tracker_points, unproject_depth, d2n_tblr, start_mapping_process, MiniCam
from gs_dyn_obj.obj_gs import ObjectGS
from gs_dyn_obj.gs_param import GSParam
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
import rerun.blueprint as rrb

# Background Scene Model (from run_full_system_debug.py)
from gs_scene.scene_model import SceneModel
try:
    from old_ref.keyframe_window import KeyFrameWindow
except ImportError:
    class KeyFrameWindowFallback:
        def __init__(self, chunk_size, rr_log=True):
            self.chunk_size = chunk_size
            self.count = 0
        def try_add_frame(self, frame, model):
            if self.count % self.chunk_size == 0:
                self.count += 1
                return True
            self.count += 1
            return False
    KeyFrameWindow = KeyFrameWindowFallback

@dataclass
class GlobalConfig:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    device: str = "cuda"
    num_frames: int = 150
    init_frame: int = 0
    n_features: int = 2000
    feature_type: Literal["orb", "gftt", "grid"] = "grid"
    ransac_thresh: float = 1.0
    informed_thresh: float = 50.0
    # Max pose jump vs the predicted guess, in meters. 1.0 m was ~220x the GT
    # per-frame motion on this clip (median 4.6 mm, max 22 mm), so the rejection
    # gate never fired and drift accumulated unopposed. 0.05 m is ~2x GT max.
    # BundleSdfGS._jump_limit() widens this adaptively from observed accepted
    # motion, so real fast motion is still allowed through.
    max_pose_jump: float = 0.05
    num_opt_steps: int = 50
    mask_loss_weight: float = 20.0
    pyr_levels: int = 2
    window_size: int = 10
    grid_spacing: int = 2
    gs_type: Literal["2d", "3d"] = "2d"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    serve: bool = False
    no_vis: bool = False
    # Where to dump per-frame T_CiO estimate + GT. Empty string disables it.
    # The raw mean ATE does not say whether the error is one outlier or a broad
    # offset, so the matrices are written out for offline inspection.
    traj_dump: str = "renders/trajectory_est_gt.npz"
    debug: bool = False
    separate_render: bool = True
    use_alpha_blending: bool = True
    
    # Static BG specific
    chunk_size: int = 8
    kf_every: int = 5
    lr_anchor: float = 1e-3
    lr_gs: float = 1e-3
    use_guided_mvs: bool = False
    use_anchors: bool = False
    use_exposure: bool = False
    pyr_levels_static: int = 2
    num_steps_static: int = 10 
    anchor_dist_threshold: float = 2.0
    
    # Dynamic specific
    use_photometric: bool = True
    photometric_mode: str = "lm"
    n_init_frames: int = 10
    num_steps_dyn: int = 150
    densify_every: int = 5
    prune_every: int = 1
    min_kf_rot: float = 5.0
    kf_overlap_thresh: float = 0.8
    min_kf_interval: int = 3
    # Photometric-refinement gate. These were previously declared here and never
    # read -- the comparison in BundleSdfGS.step() was hardcoded to 0.5 m / 20
    # deg. They now reach the tracker config and act as floors for the adaptive
    # limit. Calibrated against GT per-frame motion: translation 22 mm max,
    # rotation 2.04 deg max (the run was producing 36-60 deg steps).
    max_photo_jump_t: float = 0.05
    max_photo_jump_R: float = 3.0
    fix_color: bool = True
    fix_scale: bool = True
    use_ray_dist: bool = True
    multi_view_ncc_weight: float = 0.0
    kf_every_dyn: int = 5
    do_refine: bool = True
    disable_bg: bool = False # Enable BG by default
    densify_error_threshold: float = 10.0
    
    # Tracker specific
    use_informed_filtering: bool = True
    use_occlusion_check: bool = True
    align_depth: bool = True
    align_with_bias: bool = True
    multiprocess_dyn: bool = False # Default to False as in test_hot3d.py
    use_pgsr: bool = False

def load_frame_data_v2(data_dir, frame_idx):
    import cv2
    import numpy as np
    from scipy.spatial.transform import Rotation as R
    """Helper to load frame data consistent with run_full_system_debug.py"""
    img_path = data_dir / "images" / f"{frame_idx:06d}.png"
    if not img_path.exists():
        img_path = data_dir / "images" / f"{frame_idx:06d}.jpg"
    
    # Search for mask
    mask_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
    if not mask_path.exists():
        mask_path = data_dir / "dyn_obj_masked_infer" / f"mask_{frame_idx:05d}.png"
    if not mask_path.exists():
        mask_path = data_dir / "obj_masks" / f"{frame_idx:06d}.png"
    
    # Search for depth
    depth_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
    if not depth_path.exists():
        depth_path = data_dir / "dyn_obj_masked_infer" / f"depth_{frame_idx:05d}.npy"
    if not depth_path.exists():
        depth_path = data_dir / "depth" / f"{frame_idx:06d}.npy"
    
    k_path = data_dir / "intrinsics" / f"{frame_idx:06d}.npy"
    extrin_path = data_dir / "extrinsics" / f"{frame_idx:06d}.npy"
    hand_mask_path = data_dir / "hand_masks" / f"{frame_idx:06d}.png"
    
    if not img_path.exists(): return None
    
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
    if mask is not None and mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
    depth = np.load(depth_path).astype(np.float32) if depth_path.exists() else None
    if depth is not None and depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        
    K = np.load(k_path)
    extrin = np.load(extrin_path)
    hand_mask = cv2.imread(str(hand_mask_path), cv2.IMREAD_GRAYSCALE) if hand_mask_path.exists() else None
    if hand_mask is not None and hand_mask.shape[:2] != (h, w):
        hand_mask = cv2.resize(hand_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
    T_WO_gt = None
    obj_pose_path = data_dir / "object_poses.txt"
    if obj_pose_path.exists():
        with open(obj_pose_path, "r") as f:
            lines = f.readlines()
            if frame_idx < len(lines):
                parts = [float(x) for x in lines[frame_idx].split()]
                if len(parts) >= 8:
                    t = np.array(parts[1:4])
                    q = np.array(parts[4:8])
                    T_WO_gt = np.eye(4)
                    T_WO_gt[:3, :3] = R.from_quat(q).as_matrix()
                    T_WO_gt[:3, 3] = t
                
    return {
        "image": image, "mask": mask, "depth": depth, "K": K, "extrin": extrin,
        "hand_mask": hand_mask, "T_WO_gt": T_WO_gt, "frame_idx": frame_idx
    }

def data_loader_worker(cfg, static_q, dynamic_q):
    data_dir = Path(cfg.data_root) / cfg.clip_id
    count = 0
    for i in range(cfg.num_frames):
        idx = cfg.init_frame + i
        fd = load_frame_data_v2(data_dir, idx)
        if fd is None: 
            print(f"[Data Loader] Frame {idx} not found, stopping.")
            break
        if not cfg.disable_bg:
            static_q.put(fd)
        dynamic_q.put(fd)
        count += 1
    print(f"[Data Loader] Loaded {count} frames.")
    static_q.put(None)
    dynamic_q.put(None)

def static_scene_worker(cfg, bg_queue, data_q):
    from gs_scene.scene_model import SceneModel
    try:
        from old_ref.keyframe_window import KeyFrameWindow
    except ImportError:
        class KeyFrameWindowFallback:
            def __init__(self, chunk_size, rr_log=True):
                self.chunk_size = chunk_size
                self.count = 0
            def try_add_frame(self, frame, model):
                if self.count % self.chunk_size == 0:
                    self.count += 1
                    return True
                self.count += 1
                return False
        KeyFrameWindow = KeyFrameWindowFallback
    
    if not cfg.no_vis:
        rr.init("FullSystem2", recording_id="dyn_gs_unified")
        if not cfg.serve: rr.connect_grpc(cfg.rerun_url)

    data_dir = Path(cfg.data_root) / cfg.clip_id
    f0 = load_frame_data_v2(data_dir, cfg.init_frame)
    if f0 is None: return
    h, w = f0["image"].shape[:2]
    scene_model = SceneModel(width=w, height=h, num_steps=cfg.num_steps_static, use_anchors=cfg.use_anchors, use_exposure=cfg.use_exposure, pyr_levels=cfg.pyr_levels_static, anchor_dist_threshold=cfg.anchor_dist_threshold, use_guided_mvs=cfg.use_guided_mvs)
    keyframe_window = KeyFrameWindow(cfg.chunk_size, rr_log=not cfg.no_vis)
    
    while True:
        fd = data_q.get()
        if fd is None: break
        idx = fd["frame_idx"]
        img_np = fd["image"]
        depth = fd["depth"] if fd["depth"] is not None else np.zeros((h, w), dtype=np.float32)
        obj_mask = fd["mask"]; hand_mask = fd["hand_mask"]
        dynamic_mask = np.zeros((h, w), dtype=bool)
        if obj_mask is not None: dynamic_mask |= (obj_mask > 0)
        if hand_mask is not None: dynamic_mask |= (hand_mask > 0)
        static_mask = ~dynamic_mask
        
        if keyframe_window.try_add_frame({"image": img_np, "extrin": fd["extrin"], "K": fd["K"], "depth": depth, "obj_mask": obj_mask, "hand_mask": hand_mask}, scene_model):
            with torch.enable_grad():
                scene_model.update(img_np, depth, fd["extrin"], fd["K"], mask=static_mask)
                if bg_queue is not None:
                    shs = torch.cat([scene_model.gaussian_params["f_dc"]["val"], scene_model.gaussian_params["f_rest"]["val"]], dim=1)
                    bg_queue.put({"means": scene_model.gaussian_params["xyz"]["val"].detach().cpu().numpy(), "quats": scene_model.gaussian_params["rotation"]["val"].detach().cpu().numpy(), "scales": scene_model.gaussian_params["scaling"]["val"].detach().cpu().numpy(), "colors": scene_model.gaussian_params["f_dc"]["val"].detach().cpu().numpy().squeeze(1), "opacity": scene_model.gaussian_params["opacity"]["val"].detach().cpu().numpy(), "shs": shs.detach().cpu().numpy()})
            if not cfg.no_vis:
                rr.set_time("frame_idx", sequence=idx)
                xyz = scene_model.xyz.detach().cpu().numpy(); colors = scene_model.colors.detach().cpu().numpy().squeeze()
                if len(xyz) > 0: rr.log("static/gs_points", rr.Points3D(xyz, colors=colors))

def umeyama(src, dst):
    """Estimate the optimal similarity transform (s, R, t) mapping src -> dst.

    src, dst: (N, 3) point arrays. Returns scale s, rotation R and translation t
    such that dst ~= s * (R @ src.T).T + t. Used to separate a constant offset or
    scale error from genuine drift: if the aligned error is much smaller than the
    raw error, the estimate is a rigid-ish translation of the truth rather than
    diverging.
    """
    n = src.shape[0]
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    z_s, z_d = src - mu_s, dst - mu_d
    cov = z_d.T @ z_s / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:   # reject reflection
        S[-1, -1] = -1
    R = U @ S @ Vt
    var_s = np.mean(np.sum(z_s ** 2, axis=1))
    s = float(np.trace(np.diag(D) @ S) / var_s) if var_s > 0 else 1.0
    t = mu_d - s * (R @ mu_s)
    return s, R, t


def dynamic_worker(cfg: GlobalConfig, bg_queue, data_q):
    print(f"[Dynamic Worker] Starting on {cfg.device}")
    device = torch.device(cfg.device)
    
    # Initialize Tracker and Mapping configs for BundleSdfGS
    tracker_cfg = GeoTrackerConfig(
        feature_type=cfg.feature_type,
        n_features=cfg.n_features,
        use_photometric_refinement=cfg.use_photometric,
        photometric_mode=cfg.photometric_mode,
        n_init_frames=cfg.n_init_frames,
        max_pose_jump=cfg.max_pose_jump,
        max_photo_jump_t=cfg.max_photo_jump_t,
        max_photo_jump_R=cfg.max_photo_jump_R,
        informed_thresh=cfg.informed_thresh,
        ransac_thresh=cfg.ransac_thresh,
        grid_spacing=cfg.grid_spacing,
        gs_type=cfg.gs_type,
        use_informed_filtering=cfg.use_informed_filtering,
        use_occlusion_check=cfg.use_occlusion_check,
        min_kf_rot=cfg.min_kf_rot,
        kf_overlap_thresh=cfg.kf_overlap_thresh,
        min_kf_interval=cfg.min_kf_interval
    )
    
    map_cfg = MappingConfig(
        gs_type=cfg.gs_type,
        device=cfg.device,
        num_steps_per_frame=cfg.num_steps_dyn,
        fix_color=cfg.fix_color,
        fix_scale=cfg.fix_scale,
        use_ray_dist=cfg.use_ray_dist,
        multi_view_ncc_weight=cfg.multi_view_ncc_weight,
        pyr_levels=cfg.pyr_levels,
        densify_every=cfg.densify_every,
        prune_every=cfg.prune_every,
        mask_loss_weight=cfg.mask_loss_weight,
        align_depth=cfg.align_depth,
        align_with_bias=cfg.align_with_bias,
        use_pgsr=cfg.use_pgsr
    )
    
    # Fetch first frame for dense initialization from depth prior
    f0 = data_q.get()
    if f0 is None: return
    
    # Dense initialization logic using Object-World coordinates
    T_WO_gt_f0 = f0["T_WO_gt"]
    T_CW_f0 = f0["extrin"]
    T_CO_gt_f0 = T_CW_f0 @ T_WO_gt_f0 if T_WO_gt_f0 is not None else np.eye(4)
    
    color_f0 = f0["image"]
    depth_f0 = f0["depth"]
    mask_f0 = f0.get("mask")
    if mask_f0 is None: mask_f0 = np.ones_like(depth_f0, dtype=bool)
    else: mask_f0 = mask_f0 > 0
    K_f0 = f0["K"]
    
    # Sample dense points from depth prior
    iy, ix = np.where(mask_f0 & (depth_f0 > 0.01))
    max_pts = 10000
    if len(iy) > max_pts:
        perm = np.random.choice(len(iy), max_pts, replace=False)
        iy, ix = iy[perm], ix[perm]
    
    z = depth_f0[iy, ix]
    fx, fy, cx, cy = K_f0[0, 0], K_f0[1, 1], K_f0[0, 2], K_f0[1, 2]
    pts3d_c = np.stack([(ix - cx) * z / fx, (iy - cy) * z / fy, z], axis=-1)
    
    # Transform to Object space
    T_OC0 = np.linalg.inv(T_CO_gt_f0)
    pts3d_o = (pts3d_c @ T_OC0[:3, :3].T) + T_OC0[:3, 3]
    colors_f0 = color_f0[iy, ix] / 255.0
    
    # Normals for GS orientation
    depth_t = torch.from_numpy(depth_f0).float().cuda()
    K_t = torch.from_numpy(K_f0).float().cuda()
    full_pts_c = unproject_depth(depth_t, K_t, depth_f0.shape[0], depth_f0.shape[1])
    normals_c_full, _ = d2n_tblr(full_pts_c.permute(2, 0, 1).unsqueeze(0))
    normals_c = -F.normalize(normals_c_full[0][:, iy, ix].permute(1, 0), dim=1).cpu().numpy()
    normals_o = (normals_c @ T_OC0[:3, :3].T)
    
    # Ray parameters for GS refinement
    ray_o_o = torch.from_numpy(T_OC0[:3, 3]).float().cuda().view(1, 3).repeat(len(pts3d_o), 1)
    ray_d_c = np.stack([(ix - cx) / fx, (iy - cy) / fy, np.ones_like(z)], axis=-1)
    ray_d_c = ray_d_c / np.linalg.norm(ray_d_c, axis=-1, keepdims=True)
    ray_d_o = torch.from_numpy(ray_d_c @ T_OC0[:3, :3].T).float().cuda()
    ray_dist = torch.norm(torch.from_numpy(pts3d_c).float().cuda(), dim=-1, keepdim=True)
    
    initial_gs = init_gs_from_tracker_points(
        pts3d_o, colors_f0, "cuda", normals=normals_o,
        ray_o=ray_o_o, ray_d=ray_d_o, ray_dist=ray_dist,
        gs_type=cfg.gs_type
    )

    # Initialize BundleSdfGS with the dense prior
    tracker = BundleSdfGS(tracker_cfg, map_cfg, use_multiprocessing=cfg.multiprocess_dyn, initial_gs=initial_gs)
    
    ates = []
    # Full T_CiO matrices (estimate and GT) for offline trajectory inspection.
    traj_est, traj_gt = [], []
    last_static_gs_data = None
    traj_obj_est_C, traj_obj_gt_C = [], []
    
    if not cfg.no_vis:
        rr.init("FullSystem2", recording_id="dyn_gs_unified")
        if not cfg.serve: rr.connect_grpc(cfg.rerun_url)

    current_fd = f0
    while True:
        if current_fd is not None:
            fd = current_fd
            current_fd = None
        else:
            fd = data_q.get()
        if fd is None: break
        idx = fd["frame_idx"]
        loop_start = time.time()
        
        # Poll background GS data
        if not cfg.disable_bg:
            while not bg_queue.empty():
                try: last_static_gs_data = bg_queue.get_nowait()
                except: break
        
        # Get GT for initialization if first frame
        T_WO_gt = fd["T_WO_gt"]
        T_CW_gt = fd["extrin"]
        T_CO_gt = T_CW_gt @ T_WO_gt if T_WO_gt is not None else None
        
        # Run BundleSdfGS
        # Returns T_CiO (Camera-from-Object) and optionally a render
        res = tracker.run(
            fd["image"], fd["mask"], fd["depth"], fd["K"], 
            T_CW=fd["extrin"],
            T_WO_init=T_WO_gt if idx == cfg.init_frame else None,
            return_render=True
        )
        if res is None: continue
        T_CO_est, img_fg_pkg = res
        if isinstance(img_fg_pkg, (list, tuple)) and len(img_fg_pkg) == 2:
            img_fg_torch, alpha_fg_torch = img_fg_pkg
        elif isinstance(img_fg_pkg, torch.Tensor):
            img_fg_torch = img_fg_pkg
            alpha_fg_torch = (img_fg_torch.sum(dim=0, keepdim=True) > 0.001).float()
        else:
            img_fg_torch, alpha_fg_torch = None, None
        
        # Background Rendering and Compositing
        h, w = fd["image"].shape[:2]
        img_bg = torch.zeros(3, h, w, device=device)
        alpha_bg = torch.zeros(1, h, w, device=device)
        
        if not cfg.disable_bg and last_static_gs_data is not None:
            from gs_dyn_obj.gs_param import GSParam
            static_gs = GSParam(
                means=torch.from_numpy(last_static_gs_data["means"]).float().to(device),
                quats=torch.from_numpy(last_static_gs_data["quats"]).float().to(device),
                scales=torch.from_numpy(last_static_gs_data["scales"]).float().to(device),
                colors=torch.from_numpy(last_static_gs_data["colors"]).float().to(device),
                opacity=torch.from_numpy(last_static_gs_data["opacity"]).float().to(device),
                shs=torch.from_numpy(last_static_gs_data["shs"]).float().to(device) if last_static_gs_data.get("shs") is not None else None
            )
            # Render background from current camera pose (extrin)
            img_bg, _, _, alpha_bg = static_gs.render(torch.from_numpy(fd["extrin"]).float().to(device), torch.from_numpy(fd["K"]).float().to(device), w, h, mode="3dgs")

        if img_fg_torch is not None:
            img_fg = img_fg_torch.to(device)
            alpha_fg = alpha_fg_torch.to(device)
        else:
            img_fg = torch.zeros(3, h, w, device=device)
            alpha_fg = torch.zeros(1, h, w, device=device)
        
        if cfg.use_alpha_blending:
            img_render = img_fg + img_bg * (1 - alpha_fg)
        else:
            img_render = img_bg.clone()
            mask_obj = fd["mask"]
            if mask_obj is not None:
                mask_obj_t = torch.from_numpy(mask_obj > 0).to(device)
                img_render[:, mask_obj_t] = img_fg[:, mask_obj_t]
            else:
                img_render[:, alpha_fg[0] > 0.1] = img_fg[:, alpha_fg[0] > 0.1]
        
        img_render = torch.clamp(img_render, 0, 1)
        
        # Metrics and Logging
        ate = np.linalg.norm(T_CO_est[:3, 3] - T_CO_gt[:3, 3]) if T_CO_gt is not None else 0.0
        ates.append(ate)
        traj_est.append(T_CO_est)
        traj_gt.append(T_CO_gt)
        target_image = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0
        psnr = -10.0 * torch.log10(torch.mean((img_render - target_image)**2) + 1e-10)
        
        # Object-area Photometric Error Check for Densification
        mask_obj = fd["mask"]
        if mask_obj is not None:
            mask_obj_t = torch.from_numpy(mask_obj > 0).to(device)
            if mask_obj_t.any() and img_fg_torch is not None:
                # Calculate object-specific L1 loss
                obj_l1 = F.l1_loss(img_fg[:, mask_obj_t], target_image[:, mask_obj_t])
                if obj_l1 > cfg.densify_error_threshold:
                    print(f"[ObjectGS] High photometric error (L1={obj_l1:.4f} > {cfg.densify_error_threshold}), triggering densification.")
                    tracker.force_densify(fd["image"], fd["mask"], fd["depth"], fd["K"], T_CO_est)
        
        if idx % 10 == 0:
            print(f"Frame {idx}: ATE={ate:.4f}, PSNR={psnr.item():.2f}, FPS={1.0/(time.time()-loop_start):.2f}")

        if not cfg.no_vis:
            rr.set_time("frame_idx", sequence=idx)
            img_np = (img_render.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            rr.log("input/image", rr.Image(fd["image"]).compress(jpeg_quality=50))
            rr.log("output/combined", rr.Image(img_np).compress(jpeg_quality=50))
            rr.log("output/ate", rr.Scalars(ate))
            rr.log("output/psnr", rr.Scalars(psnr.item()))
            
            # Object-centric visualization
            T_OC_est = np.linalg.inv(T_CO_est)
            rr.log("object/tracker/camera", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
            traj_obj_est_C.append(T_OC_est[:3, 3])
            rr.log("object/tracker/traj_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[[0, 255, 0]], radii=0.003))
            
            if T_CO_gt is not None:
                T_OC_gt = np.linalg.inv(T_CO_gt)
                traj_obj_gt_C.append(T_OC_gt[:3, 3])
                rr.log("object/camera_gt", rr.Transform3D(mat3x3=T_OC_gt[:3, :3], translation=T_OC_gt[:3, 3]))
                rr.log("object/traj_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[255, 255, 255]], radii=0.003))

            if tracker.obj_gs is not None:
                pts_O = tracker.obj_gs.gs_params.means.detach().cpu().numpy()
                if tracker.obj_gs.gs_params.shs is not None:
                    cols_sh = tracker.obj_gs.gs_params.shs.detach().cpu().numpy()
                    if cols_sh.ndim == 3: cols_sh = cols_sh[:, 0, :]
                    cols_rgb = np.clip(cols_sh * 0.28209479177387814 + 0.5, 0, 1)
                else:
                    cols_logit = tracker.obj_gs.gs_params.colors.detach()
                    cols_rgb = torch.sigmoid(cols_logit).cpu().numpy()
                rr.log("object/gs", rr.Points3D(pts_O, colors=cols_rgb))
            
            # World visualization
            T_WO_est = np.linalg.inv(fd["extrin"]) @ T_CO_est
            rr.log("world/object", rr.Transform3D(mat3x3=T_WO_est[:3, :3], translation=T_WO_est[:3, 3]))

    if ates:
        print(f"\n>>> Final Mean ATE (raw, unaligned): {np.mean(ates):.4f}m")
        valid = [i for i, p in enumerate(traj_gt) if p is not None]
        if len(valid) >= 3:
            est_p = np.stack([traj_est[i][:3, 3] for i in valid])
            gt_p = np.stack([traj_gt[i][:3, 3] for i in valid])
            s, Rot, tr = umeyama(est_p, gt_p)
            aligned = (est_p * s) @ Rot.T + tr
            err = np.linalg.norm(aligned - gt_p, axis=1)
            print(f">>> Raw:          Mean={np.mean(ates):.4f}m, Max={np.max(ates):.4f}m")
            print(f">>> PnP rejections: {tracker.pnp_rejections}/{len(ates)} frames")
            print(f">>> Umeyama-aligned (s={s:.4f}): RMSE={np.sqrt(np.mean(err ** 2)):.4f}m, Max={err.max():.4f}m")
        if cfg.traj_dump:
            dump_path = Path(cfg.traj_dump)
            dump_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                str(dump_path),
                T_CO_est=np.stack(traj_est) if traj_est else np.zeros((0, 4, 4)),
                T_CO_gt=np.array([p if p is not None else np.eye(4) for p in traj_gt]) if traj_gt else np.zeros((0, 4, 4)),
                valid_gt=np.array([p is not None for p in traj_gt], dtype=bool),
                ate=np.array(ates),
            )
            print(f">>> Trajectory dumped to {dump_path}")
    tracker.on_finish()

def main():
    try: mp.set_start_method('spawn', force=True)
    except: pass
    
    cfg = tyro.cli(GlobalConfig)
    ctx = mp.get_context('spawn')
    
    bg_queue = ctx.Queue(maxsize=10)
    static_q = ctx.Queue(maxsize=10)
    dynamic_q = ctx.Queue(maxsize=10)
    
    p_loader = ctx.Process(target=data_loader_worker, args=(cfg, static_q, dynamic_q))
    p_static = ctx.Process(target=static_scene_worker, args=(cfg, bg_queue, static_q))
    p_dynamic = ctx.Process(target=dynamic_worker, args=(cfg, bg_queue, dynamic_q))
    
    p_loader.start()
    if not cfg.disable_bg:
        p_static.start()
    p_dynamic.start()
    
    if not cfg.no_vis:
        rr.init("FullSystem2", recording_id="dyn_gs_unified")
        if not cfg.serve: rr.connect_grpc(cfg.rerun_url)
        rr.send_blueprint(rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="input/image", name="Input"),
                    rrb.Spatial2DView(origin="output/combined", name="Render"),
                ),
                rrb.Spatial3DView(origin="object", name="Object-Centric"),
            ),
            collapse_panels=True
        ))

    p_loader.join()
    if not cfg.disable_bg:
        p_static.join()
    p_dynamic.join()

if __name__ == "__main__":
    main()
