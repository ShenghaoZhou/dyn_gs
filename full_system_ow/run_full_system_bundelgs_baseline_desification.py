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
from obj_gs_mapping import MappingConfig, GSMapping, init_gs_from_tracker_points, unproject_depth, d2n_tblr, MiniCam
from gs_dyn_obj.obj_gs import ObjectGS
from gs_dyn_obj.gs_param import GSParam
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, quaternion_to_matrix
import rerun.blueprint as rrb

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
    gs_type: Literal["2d", "3d"] = "3d" 
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
    prune_every: int = 100 
    densification_interval: int = 100 
    densify_from_iter: int = 500
    densify_until_iter: int = 15000
    densify_grad_threshold: float = 0.0002
    min_kf_rot: float = 5.0
    kf_overlap_thresh: float = 0.8
    min_kf_interval: int = 3
    max_photo_jump_t: float = 0.5
    max_photo_jump_R: float = 20.0
    fix_color: bool = False
    fix_scale: bool = False
    use_ray_dist: bool = False 
    multi_view_ncc_weight: float = 0.0
    kf_every_dyn: int = 5
    do_refine: bool = True
    disable_bg: bool = False 
    densify_error_threshold: float = 10.0 
    
    # Tracker specific
    use_informed_filtering: bool = True
    use_occlusion_check: bool = True
    align_depth: bool = True
    align_with_bias: bool = True
    multiprocess_dyn: bool = False 
    use_pgsr: bool = False

class BaselineGSMapping(GSMapping):
    def __init__(self, cfg, initial_gs, output_queue=None):
        super().__init__(cfg, initial_gs, output_queue)
        self.xyz_gradient_accum = torch.zeros((len(self.gs_params.means), 1), device=self.device)
        self.denom = torch.zeros((len(self.gs_params.means), 1), device=self.device)
        self.max_radii2D = torch.zeros((len(self.gs_params.means)), device=self.device)
        self.cameras_extent = 1.0 

    def setup_optimizer(self):
        params = [
            {"params": [self.gs_params.means], "lr": self.cfg.lr_means, "name": "means"},
            {"params": [self.gs_params.quats], "lr": self.cfg.lr_quats, "name": "quats"},
            {"params": [self.gs_params.opacity], "lr": self.cfg.lr_opacity, "name": "opacity"},
            {"params": [self.gs_params.scales], "lr": self.cfg.lr_scales, "name": "scales"},
        ]
        if self.gs_params.shs is not None:
            params.append({"params": [self.gs_params.shs], "lr": self.cfg.lr_colors, "name": "shs"})
        else:
            params.append({"params": [self.gs_params.colors], "lr": self.cfg.lr_colors, "name": "colors"})
        
        self.optimizer = torch.optim.Adam(params)

    def add_densification_stats(self, viewspace_grad, visibility_filter):
        if viewspace_grad is None: return
        self.xyz_gradient_accum[visibility_filter] += torch.norm(viewspace_grad[visibility_filter, :2], dim=-1, keepdim=True)
        self.denom[visibility_filter] += 1

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(torch.exp(self.gs_params.scales), dim=1).values <= 0.01 * scene_extent)
        
        if not selected_pts_mask.any(): return

        new_means = self.gs_params.means[selected_pts_mask]
        new_quats = self.gs_params.quats[selected_pts_mask]
        new_scales = self.gs_params.scales[selected_pts_mask]
        new_colors = self.gs_params.colors[selected_pts_mask]
        new_opacity = self.gs_params.opacity[selected_pts_mask]
        
        self.gs_params.means = torch.nn.Parameter(torch.cat([self.gs_params.means.data, new_means.data], dim=0))
        self.gs_params.quats = torch.nn.Parameter(torch.cat([self.gs_params.quats.data, new_quats.data], dim=0))
        self.gs_params.scales = torch.nn.Parameter(torch.cat([self.gs_params.scales.data, new_scales.data], dim=0))
        self.gs_params.colors = torch.nn.Parameter(torch.cat([self.gs_params.colors.data, new_colors.data], dim=0))
        self.gs_params.opacity = torch.nn.Parameter(torch.cat([self.gs_params.opacity.data, new_opacity.data], dim=0))
        
        if self.gs_params.shs is not None:
            new_shs = self.gs_params.shs[selected_pts_mask]
            self.gs_params.shs = torch.nn.Parameter(torch.cat([self.gs_params.shs.data, new_shs.data], dim=0))

        num_new = selected_pts_mask.sum()
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((num_new, 1), device=self.device)], dim=0)
        self.denom = torch.cat([self.denom, torch.zeros((num_new, 1), device=self.device)], dim=0)
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros((num_new), device=self.device)], dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.gs_params.means.shape[0]
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(torch.exp(self.gs_params.scales), dim=1).values > 0.01 * scene_extent)

        if not selected_pts_mask.any(): return

        stds = torch.exp(self.gs_params.scales)[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = quaternion_to_matrix(F.normalize(self.gs_params.quats[selected_pts_mask], dim=-1)).repeat(N, 1, 1)
        
        new_means = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.gs_params.means[selected_pts_mask].repeat(N, 1)
        new_scales = torch.log(torch.exp(self.gs_params.scales[selected_pts_mask]).repeat(N, 1) / (0.8 * N))
        new_quats = self.gs_params.quats[selected_pts_mask].repeat(N, 1)
        new_colors = self.gs_params.colors[selected_pts_mask].repeat(N, 1)
        new_opacity = self.gs_params.opacity[selected_pts_mask].repeat(N, 1)

        self.gs_params.means = torch.nn.Parameter(torch.cat([self.gs_params.means.data, new_means.data], dim=0))
        self.gs_params.quats = torch.nn.Parameter(torch.cat([self.gs_params.quats.data, new_quats.data], dim=0))
        self.gs_params.scales = torch.nn.Parameter(torch.cat([self.gs_params.scales.data, new_scales.data], dim=0))
        self.gs_params.colors = torch.nn.Parameter(torch.cat([self.gs_params.colors.data, new_colors.data], dim=0))
        self.gs_params.opacity = torch.nn.Parameter(torch.cat([self.gs_params.opacity.data, new_opacity.data], dim=0))
        
        if self.gs_params.shs is not None:
            new_shs = self.gs_params.shs[selected_pts_mask].repeat(N, 1, 1)
            self.gs_params.shs = torch.nn.Parameter(torch.cat([self.gs_params.shs.data, new_shs.data], dim=0))

        num_new = new_means.shape[0]
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((num_new, 1), device=self.device)], dim=0)
        self.denom = torch.cat([self.denom, torch.zeros((num_new, 1), device=self.device)], dim=0)
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros((num_new), device=self.device)], dim=0)

        prune_filter = torch.zeros(len(self.gs_params.means), device=self.device, dtype=bool)
        prune_filter[:n_init_points] = selected_pts_mask
        self.prune_points(prune_filter)

    def prune_points(self, mask):
        valid_points_mask = ~mask
        self.gs_params.means = torch.nn.Parameter(self.gs_params.means[valid_points_mask].detach().requires_grad_(True))
        self.gs_params.quats = torch.nn.Parameter(self.gs_params.quats[valid_points_mask].detach().requires_grad_(True))
        self.gs_params.scales = torch.nn.Parameter(self.gs_params.scales[valid_points_mask].detach().requires_grad_(True))
        self.gs_params.colors = torch.nn.Parameter(self.gs_params.colors[valid_points_mask].detach().requires_grad_(True))
        self.gs_params.opacity = torch.nn.Parameter(self.gs_params.opacity[valid_points_mask].detach().requires_grad_(True))
        if self.gs_params.shs is not None:
            self.gs_params.shs = torch.nn.Parameter(self.gs_params.shs[valid_points_mask].detach().requires_grad_(True))

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        before = len(self.gs_params.means)
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (torch.sigmoid(self.gs_params.opacity) < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = torch.exp(self.gs_params.scales).max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        self.setup_optimizer()
        torch.cuda.empty_cache()
        after = len(self.gs_params.means)
        print(f"[BaselineMapping] Densified and Pruned: {before} -> {after} points")

    def optimize_frame(self, frame_data, num_steps=None, frame_count=0):
        image, mask = frame_data["image"], frame_data["mask"]
        K = frame_data["K"]
        if "extrin" in frame_data and "T_WO" in frame_data:
            T_CO = frame_data["extrin"] @ frame_data["T_WO"]
        else:
            T_CO = frame_data["T_CiO"]
        
        H, W = image.shape[:2]
        image_t = torch.from_numpy(image).float().to(self.device).permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(mask > 0).bool().to(self.device)
        T_CO_t = torch.from_numpy(T_CO).float().to(self.device)
        K_t = torch.from_numpy(K).float().to(self.device)
        
        if num_steps is None: num_steps = self.cfg.num_steps_per_frame
        if frame_count == 1:
            self.cameras_extent = torch.norm(self.gs_params.means.data, dim=-1).max().item() * 1.1

        total_iters = (frame_count - 1) * num_steps
        
        for step in range(num_steps):
            iter_idx = total_iters + step
            self.optimizer.zero_grad()
            
            render_mode = "3dgs" if self.cfg.gs_type == "3d" else "normal"
            res = self.gs_params.render_full(
                T_CO_t, K_t, W, H,
                mode=render_mode,
                near_plane=self.cfg.near_plane, far_plane=self.cfg.far_plane
            )
            render_image, _, _, render_alpha, _, _, radii, means2d = res
            
            means2d.retain_grad()
            
            loss = F.l1_loss(render_image * mask_t, image_t * mask_t)
            loss.backward()
            
            if iter_idx % 50 == 0:
                print(f"[BaselineMapping] Iter {iter_idx}: Loss {loss.item():.6f}, GS Count {len(self.gs_params.means)}")

            with torch.no_grad():
                visibility_filter = radii > 0
                if visibility_filter.any():
                    self.max_radii2D[visibility_filter] = torch.max(self.max_radii2D[visibility_filter], radii[visibility_filter])
                    self.add_densification_stats(means2d.grad, visibility_filter)

                if iter_idx > 50 and iter_idx < 15000 and iter_idx % 100 == 0:
                    self.densify_and_prune(0.0002, 0.005, self.cameras_extent, 20)
            
            self.optimizer.step()

def load_frame_data_v2(data_dir, frame_idx):
    from scipy.spatial.transform import Rotation as R
    img_path = data_dir / "images" / f"{frame_idx:06d}.png"
    if not img_path.exists():
        img_path = data_dir / "images" / f"{frame_idx:06d}.jpg"
    
    mask_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
    if not mask_path.exists():
        mask_path = data_dir / "dyn_obj_masked_infer" / f"mask_{frame_idx:05d}.png"
    if not mask_path.exists():
        mask_path = data_dir / "obj_masks" / f"{frame_idx:06d}.png"
    
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
        if fd is None: break
        if not cfg.disable_bg:
            static_q.put(fd)
        dynamic_q.put(fd)
        count += 1
    static_q.put(None)
    dynamic_q.put(None)

def dynamic_worker(cfg: GlobalConfig, bg_queue, data_q):
    device = torch.device(cfg.device)
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
    
    f0 = data_q.get()
    if f0 is None: return
    
    T_WO_gt_f0 = f0["T_WO_gt"]
    T_CW_f0 = f0["extrin"]
    T_CO_gt_f0 = T_CW_f0 @ T_WO_gt_f0 if T_WO_gt_f0 is not None else np.eye(4)
    
    color_f0 = f0["image"]
    depth_f0 = f0["depth"]
    mask_f0 = f0.get("mask")
    if mask_f0 is None: mask_f0 = np.ones_like(depth_f0, dtype=bool)
    else: mask_f0 = mask_f0 > 0
    K_f0 = f0["K"]
    
    iy, ix = np.where(mask_f0 & (depth_f0 > 0.01))
    max_pts = 10000
    if len(iy) > max_pts:
        perm = np.random.choice(len(iy), max_pts, replace=False)
        iy, ix = iy[perm], ix[perm]
    
    z = depth_f0[iy, ix]
    fx, fy, cx, cy = K_f0[0, 0], K_f0[1, 1], K_f0[0, 2], K_f0[1, 2]
    pts3d_c = np.stack([(ix - cx) * z / fx, (iy - cy) * z / fy, z], axis=-1)
    
    T_OC0 = np.linalg.inv(T_CO_gt_f0)
    pts3d_o = (pts3d_c @ T_OC0[:3, :3].T) + T_OC0[:3, 3]
    colors_f0 = color_f0[iy, ix] / 255.0
    
    initial_gs = init_gs_from_tracker_points(pts3d_o, colors_f0, "cuda", gs_type=cfg.gs_type)

    tracker = BundleSdfGS(tracker_cfg, map_cfg, use_multiprocessing=cfg.multiprocess_dyn, initial_gs=initial_gs)
    
    ates = []
    traj_obj_est_C, traj_obj_gt_C = [], []
    
    if not cfg.no_vis:
        rr.init("FullSystemBaseline", recording_id="dyn_gs_baseline")
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
        
        # Replace mapper with baseline right before the first run (after init_gs_model)
        if idx == 0 and not cfg.multiprocess_dyn:
            # We need to run frame 0 first so init_gs_model is called
            pass

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
        
        # NOW replace it
        if idx == 0 and not cfg.multiprocess_dyn:
             tracker.mapper = BaselineGSMapping(map_cfg, tracker.obj_gs.gs_params)
             tracker.mapper.setup_optimizer()
             tracker.mapper.mapping_frame_count = 1 
             print("[BaselineMapping] Mapper replaced with BaselineGSMapping after Frame 0")

        T_CO_est, img_fg_pkg = res
        
        img_fg = img_fg_pkg[0] if isinstance(img_fg_pkg, (list, tuple)) else img_fg_pkg
        img_render = img_fg 
        
        ate = np.linalg.norm(T_CO_est[:3, 3] - T_CO_gt[:3, 3]) if T_CO_gt is not None else 0.0
        ates.append(ate)
        
        if idx % 10 == 0:
            print(f"Frame {idx}: ATE={ate:.4f}, GS count={len(tracker.mapper.gs_params.means)}")

        if not cfg.no_vis:
            rr.set_time("frame_idx", sequence=idx)
            img_np = (img_render.permute(1, 2, 0).detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            rr.log("input/image", rr.Image(fd["image"]).compress(jpeg_quality=50))
            rr.log("output/render", rr.Image(img_np).compress(jpeg_quality=50))
            rr.log("output/ate", rr.Scalars(ate))
            
            T_OC_est = np.linalg.inv(T_CO_est)
            rr.log("object/tracker/camera", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
            traj_obj_est_C.append(T_OC_est[:3, 3])
            rr.log("object/tracker/traj_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[[0, 255, 0]], radii=0.003))
            
            if tracker.mapper.gs_params is not None:
                pts_O = tracker.mapper.gs_params.means.detach().cpu().numpy()
                cols_logit = tracker.mapper.gs_params.colors.detach()
                cols_rgb = torch.sigmoid(cols_logit).cpu().numpy()
                rr.log("object/gs", rr.Points3D(pts_O, colors=cols_rgb))

    if ates: print(f"\n>>> Final Mean ATE: {np.mean(ates):.4f}m")

def main():
    try: mp.set_start_method('spawn', force=True)
    except: pass
    
    cfg = tyro.cli(GlobalConfig)
    ctx = mp.get_context('spawn')
    
    static_q = ctx.Queue(maxsize=10)
    dynamic_q = ctx.Queue(maxsize=10)
    
    p_loader = ctx.Process(target=data_loader_worker, args=(cfg, static_q, dynamic_q))
    p_dynamic = ctx.Process(target=dynamic_worker, args=(cfg, None, dynamic_q))
    
    p_loader.start()
    p_dynamic.start()
    
    if not cfg.no_vis:
        rr.init("FullSystemBaseline", recording_id="dyn_gs_baseline")
        if not cfg.serve: rr.connect_grpc(cfg.rerun_url)
        rr.send_blueprint(rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="input/image", name="Input"),
                    rrb.Spatial2DView(origin="output/render", name="Render"),
                ),
                rrb.Spatial3DView(origin="object", name="Object-Centric"),
            ),
            collapse_panels=True
        ))

    p_loader.join()
    p_dynamic.join()

if __name__ == "__main__":
    main()
