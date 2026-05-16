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
    max_pose_jump: float = 1.0
    num_opt_steps: int = 50
    mask_loss_weight: float = 20.0
    pyr_levels: int = 2
    window_size: int = 10
    grid_spacing: int = 2
    gs_type: Literal["2d", "3d"] = "2d"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    serve: bool = False
    no_vis: bool = False
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
    max_photo_jump_t: float = 0.5
    max_photo_jump_R: float = 20.0
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

# Custom GSMapping that implements median densification and frontal parallel orientation
class MedianGSMapping(GSMapping):
    def densify(self, image_t, depth_np, mask_t, cam, T_CO_t):
        with torch.no_grad():
            render_mode = "3dgs" if self.cfg.gs_type == "3d" else "normal"
            render_image, render_depth, _, render_alpha = self.gs_params.render(T_CO_t, cam.K, cam.width, cam.height, mode=render_mode)
            render_depth = render_depth.squeeze()
            
            # Use current GS points to find median depth
            means_c = torch.einsum('ij,nj->ni', T_CO_t[:3, :3], self.gs_params.means) + T_CO_t[:3, 3]
            z_vals = means_c[:, 2]
            valid_z = z_vals[z_vals > 0.01]
            if len(valid_z) > 0:
                median_z = torch.median(valid_z)
            else:
                median_z = torch.tensor(1.0, device=self.device)
            
            # Alpha-based sampling (Fill holes in the mask) using median depth
            alpha_o = render_alpha.squeeze(0)
            hole_mask = (alpha_o < 0.5) & mask_t
            
            if hole_mask.sum() > 0:
                y, x = torch.where(hole_mask)
                if len(y) > 5000:
                    perm = torch.randperm(len(y), device=self.device)[:5000]
                    y, x = y[perm], x[perm]
                
                z = torch.ones_like(y, dtype=torch.float32) * median_z
                fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
                pts_c = torch.stack([(x.float() - cx) * z / fx, (y.float() - cy) * z / fy, z], dim=-1)
                
                # Orientation: Frontal parallel direction (Normal = [0, 0, -1] in camera space)
                new_normals_c = torch.zeros((len(pts_c), 3), device=self.device)
                new_normals_c[:, 2] = -1.0
                
                T_OC_t = torch.inverse(T_CO_t)
                new_means = torch.einsum('ij,nj->ni', T_OC_t[:3, :3], pts_c) + T_OC_t[:3, 3]
                new_normals_o = F.normalize(torch.einsum('ij,nj->ni', T_OC_t[:3, :3], new_normals_c), dim=1)
                
                from gs_dyn_obj.grouped_gs import RGB2SH
                new_colors = RGB2SH(image_t[:, y, x].permute(1, 0))
                
                from obj_gs_mapping import build_rotation_from_normal
                new_quats = build_rotation_from_normal(new_normals_o)
                
                new_sizes = (z / ((fx+fy)/2)) * 2.0 # Heuristic size
                
                if self.cfg.gs_type == "3d":
                    new_scales = torch.log(new_sizes.view(-1, 1).repeat(1, 3).clamp(1e-6, 1e6))
                else:
                    new_scales = torch.log(new_sizes.view(-1, 1).repeat(1, 2).clamp(1e-6, 1e6))
                    
                new_opacity = torch.logit(torch.ones((len(new_means), 1), device=self.device) * 0.3)
                    
                new_means = new_means.requires_grad_(True)
                new_quats = new_quats.requires_grad_(True)
                new_scales = new_scales.requires_grad_(not self.cfg.fix_scale)
                new_colors = new_colors.requires_grad_(not self.cfg.fix_color)
                new_opacity = new_opacity.requires_grad_(True)

                self.gs_params.means = torch.nn.Parameter(torch.cat([self.gs_params.means.data, new_means], dim=0))
                self.gs_params.quats = torch.nn.Parameter(torch.cat([self.gs_params.quats.data, new_quats], dim=0))
                self.gs_params.scales = torch.nn.Parameter(torch.cat([self.gs_params.scales.data, new_scales], dim=0))
                
                new_colors_raw = image_t[:, y, x].permute(1, 0).clamp(1e-6, 1-1e-6)
                new_colors_logit = torch.logit(new_colors_raw)
                self.gs_params.colors = torch.nn.Parameter(torch.cat([self.gs_params.colors.data, new_colors_logit], dim=0))
                self.gs_params.opacity = torch.nn.Parameter(torch.cat([self.gs_params.opacity.data, new_opacity], dim=0))
                
                if self.gs_params.shs is not None:
                    if new_colors.dim() == 2: new_colors = new_colors.unsqueeze(1)
                    self.gs_params.shs = torch.nn.Parameter(torch.cat([self.gs_params.shs.data, new_colors], dim=0))
                
                if self.gs_params.ray_o is not None:
                    new_ray_o = T_OC_t[:3, 3].view(1, 3).repeat(len(new_means), 1)
                    new_ray_d_c = F.normalize(pts_c, dim=1)
                    new_ray_d_o = F.normalize(torch.einsum('ij,nj->ni', T_OC_t[:3, :3], new_ray_d_c), dim=1)
                    new_ray_dist = torch.norm(pts_c, dim=-1, keepdim=True)
                    
                    self.gs_params.ray_o = torch.cat([self.gs_params.ray_o, new_ray_o], dim=0)
                    self.gs_params.ray_d = torch.cat([self.gs_params.ray_d, new_ray_d_o], dim=0)
                    if self.gs_params.ray_dist is not None:
                        self.gs_params.ray_dist = torch.nn.Parameter(torch.cat([self.gs_params.ray_dist.data, new_ray_dist], dim=0))

                self.setup_optimizer()

def run_median_gs_mapping_process(cfg, initial_gs, kf_queue, p_dict, stop_event):
    """Target function for the Median GS Mapping process."""
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    
    from obj_gs_mapping import GSMapping, MiniCam, gen_virtul_cam, compute_single_view_loss
    import queue
    import torch
    
    print(f"[Mapper] Median Mapping Process started. PID: {os.getpid()}", flush=True)
    
    # Use our custom MedianGSMapping
    mapper = MedianGSMapping(cfg, initial_gs)
    mapper.device = torch.device(cfg.device)
    mapper.gs_params.to(mapper.device)
    
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
            
            mapper.optimize_frame(frame_data, num_steps=cfg.num_steps_per_frame, frame_count=frame_count)
            
            latest_gs = mapper.gs_params.clone().to("cpu")
            for attr in ["means", "quats", "scales", "colors", "opacity"]:
                p = getattr(latest_gs, attr)
                if p is not None:
                    setattr(latest_gs, attr, p.detach().requires_grad_(False))
            p_dict['latest_gs'] = latest_gs
            
            p_dict['last_finished_frame'] = frame_data["frame_idx"]
            frame_count += 1
        except Exception as e:
            print(f"[Mapper] Fatal Error: {e}")
            import traceback
            traceback.print_exc()
            break

class MedianBundleSdfGS(BundleSdfGS):
    def _start_mapping_process(self, gs_params):
        # Start our custom Median Mapper Process
        gs_params_cpu = gs_params.clone().to("cpu")
        for attr in ["means", "quats", "scales", "colors", "opacity", "ray_dist"]:
            p = getattr(gs_params_cpu, attr)
            if p is not None:
                setattr(gs_params_cpu, attr, p.detach().requires_grad_(False))
        
        self.mapping_process = mp.Process(
            target=run_median_gs_mapping_process, 
            args=(self.cfg_mapping, gs_params_cpu, self.kf_queue, self.p_dict, self.stop_event)
        )
        self.mapping_process.start()

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
        informed_thresh=cfg.informed_thresh,
        ransac_thresh=cfg.ransac_thresh,
        grid_spacing=cfg.grid_spacing,
        gs_type=cfg.gs_type,
        use_informed_filtering=cfg.use_informed_filtering,
        use_occlusion_check=cfg.use_occlusion_check,
        align_depth=cfg.align_depth,
        align_with_bias=cfg.align_with_bias,
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
    
    # Fetch first frame
    f0 = data_q.get()
    if f0 is None: return
    
    T_WO_f0 = f0["T_WO_gt"] if f0["T_WO_gt"] is not None else np.eye(4)
    T_CW_f0 = f0["extrin"]
    T_C0O_f0 = T_CW_f0 @ T_WO_f0
    
    # Initialize tracker to get sparse points
    from bundlesdf_gs import GeometricTracker
    temp_tracker = GeometricTracker(tracker_cfg)
    temp_tracker.poses[0] = T_C0O_f0
    temp_tracker.add_new_points_from_depth(0, f0["image"], f0["mask"], f0["depth"], f0["K"], T_C0O_f0, align=True, spacing=4)
    
    # Collect sparse points from tracker
    pts3d_o_sparse = []
    colors_f0_sparse = []
    depths_sparse = []
    for t in temp_tracker.tracks.values():
        pts3d_o_sparse.append(t['pt3d'])
        colors_f0_sparse.append(t['color'])
        # Project back to camera to get depth
        p_c = T_C0O_f0[:3, :3] @ t['pt3d'] + T_C0O_f0[:3, 3]
        depths_sparse.append(p_c[2])
    
    pts3d_o_sparse = np.array(pts3d_o_sparse)
    colors_f0_sparse = np.array(colors_f0_sparse)
    
    # Median depth calculation
    if len(depths_sparse) > 0:
        median_z = np.median(depths_sparse)
    else:
        median_z = 1.0 # Fallback
    
    # Fill rest of mask area with median depth
    mask_f0 = f0["mask"] > 0
    iy, ix = np.where(mask_f0)
    # Randomly sample points to avoid too many
    max_pts = 10000
    if len(iy) > max_pts:
        perm = np.random.choice(len(iy), max_pts, replace=False)
        iy, ix = iy[perm], ix[perm]
    
    fx, fy, cx, cy = f0["K"][0, 0], f0["K"][1, 1], f0["K"][0, 2], f0["K"][1, 2]
    pts3d_c_dense = np.stack([(ix - cx) * median_z / fx, (iy - cy) * median_z / fy, np.ones_like(ix) * median_z], axis=-1)
    
    T_OC0 = np.linalg.inv(T_C0O_f0)
    pts3d_o_dense = (pts3d_c_dense @ T_OC0[:3, :3].T) + T_OC0[:3, 3]
    colors_f0_dense = f0["image"][iy, ix] / 255.0
    
    # Combine sparse and dense points
    pts3d_o = np.concatenate([pts3d_o_sparse, pts3d_o_dense], axis=0)
    colors_f0 = np.concatenate([colors_f0_sparse, colors_f0_dense], axis=0)
    
    # Normal: Frontal parallel (Normal = [0, 0, -1] in camera space)
    normals_c = np.zeros((len(pts3d_o), 3))
    normals_c[:, 2] = -1.0
    normals_o = (normals_c @ T_OC0[:3, :3].T)
    
    # Ray parameters
    ray_o_o = torch.from_numpy(T_OC0[:3, 3]).float().cuda().view(1, 3).repeat(len(pts3d_o), 1)
    # We need camera space points for ray_d and ray_dist
    pts3d_c_combined = (pts3d_o @ T_C0O_f0[:3, :3].T) + T_C0O_f0[:3, 3]
    ray_d_c = pts3d_c_combined / np.linalg.norm(pts3d_c_combined, axis=-1, keepdims=True)
    ray_d_o = torch.from_numpy(ray_d_c @ T_OC0[:3, :3].T).float().cuda()
    ray_dist = torch.norm(torch.from_numpy(pts3d_c_combined).float().cuda(), dim=-1, keepdim=True)
    
    initial_gs = init_gs_from_tracker_points(
        pts3d_o, colors_f0, "cuda", normals=normals_o,
        ray_o=ray_o_o, ray_d=ray_d_o, ray_dist=ray_dist,
        gs_type=cfg.gs_type
    )

    # Initialize MedianBundleSdfGS
    tracker = MedianBundleSdfGS(tracker_cfg, map_cfg, use_multiprocessing=cfg.multiprocess_dyn, initial_gs=initial_gs)
    
    ates = []
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
        
        if not cfg.disable_bg:
            while not bg_queue.empty():
                try: last_static_gs_data = bg_queue.get_nowait()
                except: break
        
        T_WO_gt = fd["T_WO_gt"]
        T_CW_gt = fd["extrin"]
        T_CO_gt = T_CW_gt @ T_WO_gt if T_WO_gt is not None else None
        
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
        
        ate = np.linalg.norm(T_CO_est[:3, 3] - T_CO_gt[:3, 3]) if T_CO_gt is not None else 0.0
        ates.append(ate)
        target_image = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0
        psnr = -10.0 * torch.log10(torch.mean((img_render - target_image)**2) + 1e-10)
        
        mask_obj = fd["mask"]
        if mask_obj is not None:
            mask_obj_t = torch.from_numpy(mask_obj > 0).to(device)
            if mask_obj_t.any() and img_fg_torch is not None:
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
            
            T_WO_est = np.linalg.inv(fd["extrin"]) @ T_CO_est
            rr.log("world/object", rr.Transform3D(mat3x3=T_WO_est[:3, :3], translation=T_WO_est[:3, 3]))

    if ates: print(f"\n>>> Final Mean ATE: {np.mean(ates):.4f}m")
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
