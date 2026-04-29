import numpy as np
import cv2
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import rerun.blueprint as rrb

from geometric_tracker import GeometricTracker
from obj_gs_mapping import MappingConfig, MiniCam, get_scaled_cam, compute_single_view_loss, compute_multi_view_loss, build_rotation_from_normal
from gs_dyn_obj.gs_param import GSParam
from gs_dyn_obj.grouped_gs import RGB2SH
from gs_dyn_obj.gs_rendering import render_2dgs
from gs_dyn_obj.utils.init import unproject_depth, d2n_tblr

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    init_frame: int = 0
    n_frames: int = 150
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # Tracker Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    grid_spacing: int = 12
    feature_type: str = "grid" 
    n_features: int = 1200
    ransac_thresh: float = 1.0
    kf_disparity_thresh: float = 20.0
    kf_min_interval: int = 8
    kf_max_interval: int = 15
    kf_overlap_thresh: float = 0.6
    max_keyframes: int = 20
    triangulate: bool = True
    triangulate_thresh: int = 3
    triangulate_parallax_thresh: float = 15.0
    do_refine: bool = True
    use_gt_depth: bool = False
    use_informed_filtering: bool = True
    skip_pnp: bool = False
    informed_thresh: float = 20.0 
    guess_type: str = "GT-Cam+CV-Obj"
    
    # Mapping Parameters
    num_steps_per_frame: int = 50
    lr_means: float = 1e-3
    lr_quats: float = 1e-3
    lr_scales: float = 5e-3
    lr_colors: float = 2.5e-3
    lr_opacity: float = 5e-2
    single_view_weight: float = 0.015
    multi_view_weight: float = 0.1
    near_plane: float = 0.01
    far_plane: float = 10.0
    use_pgsr: bool = True
    no_vis: bool = False
    
    densify_from_tracker: bool = True
    densify_from_depth: bool = False # Whether to also use Laplacian-based densification
    prune_opacity_th: float = 0.01
    prune_screen_size_th: float = 20.0

def load_object_pose_world(data_root, frame_idx):
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists():
        return None
    with open(poses_file, "r") as f:
        lines = f.readlines()
    if frame_idx >= len(lines):
        return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])]) # x y z w
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4)
    T_WO[:3, :3] = r_WO
    T_WO[:3, 3] = t_WO
    return T_WO

def load_frame_data(data_dir, frame_idx):
    stem = f"{frame_idx:06d}"
    img_path = data_dir / "images" / f"{stem}.png"
    if not img_path.exists(): return None
    img = np.array(cv2.imread(str(img_path))[..., ::-1])
    
    mask_est_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
    if not mask_est_path.exists(): mask_est_path = data_dir / "obj_masks" / f"{stem}.png"
    depth_est_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
    
    mask_gt_path = data_dir / "obj_masks" / f"{stem}.png"
    depth_gt_path = data_dir / "depth_dyn" / f"{stem}.npy"

    if not mask_est_path.exists(): return None
    mask = np.array(cv2.imread(str(mask_est_path), cv2.IMREAD_GRAYSCALE))
    
    depth = None
    if depth_est_path.exists():
        depth = np.load(depth_est_path)
        if depth.shape != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    depth_gt = None
    if depth_gt_path.exists():
        depth_gt = np.load(depth_gt_path)
        if depth_gt.shape != img.shape[:2]:
            depth_gt = cv2.resize(depth_gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
    T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
    T_WO_gt = load_object_pose_world(data_dir, frame_idx)
    
    return {
        "image": img, "mask": mask, "depth": depth, "depth_gt": depth_gt,
        "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": frame_idx
    }

def setup_optimizer(gs_params, cfg):
    return torch.optim.Adam([
        {"params": [gs_params.means], "lr": cfg.lr_means},
        {"params": [gs_params.quats], "lr": cfg.lr_quats},
        {"params": [gs_params.scales], "lr": cfg.lr_scales},
        {"params": [gs_params.colors], "lr": cfg.lr_colors},
        {"params": [gs_params.opacity], "lr": cfg.lr_opacity},
    ])

def init_gs_from_tracker_points(points, colors, device, normals=None):
    """
    Initialize GS from sparse points.
    points: [N, 3]
    colors: [N, 3] (0-1)
    """
    num_pts = points.shape[0]
    means = torch.from_numpy(points).float().to(device)
    means.requires_grad = True
    
    # Initialize colors in SH
    colors_sh = RGB2SH(torch.from_numpy(colors).float().to(device))
    colors_sh.requires_grad = True
    
    # Quaternions from normals or identity
    if normals is not None:
        quats = build_rotation_from_normal(torch.from_numpy(normals).float().to(device))
    else:
        quats = torch.zeros((num_pts, 4), device=device)
        quats[:, 0] = 1.0 # identity
    quats.requires_grad = True
    
    # Small scales
    scales = torch.log(torch.ones((num_pts, 2), device=device) * 0.005)
    scales.requires_grad = True
    
    # Mid-range opacity
    opacity = torch.logit(torch.ones((num_pts, 1), device=device) * 0.5)
    opacity.requires_grad = True
    
    return GSParam(means, quats, scales, colors_sh, opacity)

def densify_from_tracker(gs_params, new_points, new_colors, device, new_normals=None):
    """
    Append new points to existing GS.
    """
    if len(new_points) == 0:
        return gs_params
        
    new_gs = init_gs_from_tracker_points(new_points, new_colors, device, new_normals)
    
    gs_params.means = torch.nn.Parameter(torch.cat([gs_params.means.data, new_gs.means.data], dim=0))
    gs_params.quats = torch.nn.Parameter(torch.cat([gs_params.quats.data, new_gs.quats.data], dim=0))
    gs_params.scales = torch.nn.Parameter(torch.cat([gs_params.scales.data, new_gs.scales.data], dim=0))
    gs_params.colors = torch.nn.Parameter(torch.cat([gs_params.colors.data, new_gs.colors.data], dim=0))
    gs_params.opacity = torch.nn.Parameter(torch.cat([gs_params.opacity.data, new_gs.opacity.data], dim=0))
    
    return gs_params

def prune_gs(gs_params, T_CO_t, K_t, width, prune_opacity_th, prune_screen_size_th):
    with torch.no_grad():
        # Correct distance: Z-coordinate in camera space
        means_c = torch.einsum('ij,nj->ni', T_CO_t[:3, :3], gs_params.means) + T_CO_t[:3, 3]
        dist = means_c[:, 2].clamp_min(0.01)
        
        curr_scales = torch.exp(gs_params.scales)
        max_scales = curr_scales.max(dim=-1)[0]
        f = K_t[0, 0]
        screen_size = f * max_scales / dist
        
        valid_mask = (torch.sigmoid(gs_params.opacity.squeeze(-1)) > prune_opacity_th) & (screen_size < prune_screen_size_th * width)
        
        if valid_mask.sum() == 0:
            return gs_params, False
        
        gs_params.means = torch.nn.Parameter(gs_params.means[valid_mask])
        gs_params.quats = torch.nn.Parameter(gs_params.quats[valid_mask])
        gs_params.scales = torch.nn.Parameter(gs_params.scales[valid_mask])
        gs_params.colors = torch.nn.Parameter(gs_params.colors[valid_mask])
        gs_params.opacity = torch.nn.Parameter(gs_params.opacity[valid_mask])
        
        return gs_params, True

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("test_gs_mapping_from_geometry_tracker", spawn=False)
        if cfg.rerun_url:
            rr.connect_grpc(cfg.rerun_url)
        
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Vertical(
                    rrb.Spatial2DView(name="Input Image", origin="/input/image"),
                    rrb.Spatial2DView(name="GS Render", origin="/render/image"),
                    rrb.Spatial2DView(name="Depth Comparison", origin="/render/depth_comp"),
                ),
                rrb.Spatial3DView(name="Object-Centric View", origin="/object"),
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
        rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    data_dir = Path(cfg.data_root) / cfg.clip_id
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    geo_tracker = GeometricTracker(cfg)
    device = torch.device(cfg.device)

    f0 = load_frame_data(data_dir, cfg.init_frame)
    if f0 is None: return

    init_depth = f0["depth_gt"] if cfg.use_gt_depth else f0["depth"]
    T_C0O_gt = f0["T_CW_gt"] @ f0["T_WO_gt"] if f0["T_WO_gt"] is not None else np.eye(4)
    
    geo_tracker.poses[f0["frame_idx"]] = T_C0O_gt
    geo_tracker.K_dict[f0["frame_idx"]] = f0["K"]
    geo_tracker.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], init_depth, f0["K"], T_C0O_gt)
    geo_tracker.keyframes.append(f0["frame_idx"])
    
    # Initialize GS
    active_points = []
    active_colors = []
    active_normals = []
    
    # Precompute normals from depth for densification
    f0_depth_t = torch.from_numpy(init_depth).float().to(device)
    f0_pts_c = unproject_depth(f0_depth_t, torch.from_numpy(f0["K"]).float().to(device), f0["image"].shape[0], f0["image"].shape[1])
    f0_normals_c, _ = d2n_tblr(f0_pts_c.permute(2, 0, 1).unsqueeze(0))
    f0_normals_c = -f0_normals_c[0] # [3, H, W]
    f0_normals_c = F.normalize(f0_normals_c, dim=0)
    T_C0O_inv_t = torch.inverse(torch.from_numpy(T_C0O_gt).float().to(device))
    f0_normals_o = torch.einsum('ij,hwj->hwi', T_C0O_inv_t[:3, :3], f0_normals_c.permute(1, 2, 0))
    f0_normals_o = f0_normals_o.cpu().numpy()

    gs_tid_set = set()
    for tid, t in geo_tracker.tracks.items():
        if f0["frame_idx"] in t['obs']:
            active_points.append(t['pt3d'])
            uv = t['obs'][f0["frame_idx"]]
            ix, iy = int(round(uv[0])), int(round(uv[1]))
            color = f0["image"][iy, ix] / 255.0
            active_colors.append(color)
            active_normals.append(f0_normals_o[iy, ix])
            gs_tid_set.add(tid)
            
    gs_params = init_gs_from_tracker_points(np.array(active_points), np.array(active_colors), device, np.array(active_normals))
    optimizer = setup_optimizer(gs_params, cfg)
    
    keyframes = [] # List of (MiniCam, image_gray)
    
    history_T_WO_gt = [f0["T_WO_gt"]]
    last_kf_idx = cfg.init_frame
    last_kf_gray = cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY)
    
    prev_f = f0
    for i in tqdm(range(1, cfg.n_frames), desc="Tracking + Mapping"):
        idx = cfg.init_frame + i
        fd = load_frame_data(data_dir, idx)
        if fd is None: break
        
        flow = dis.calc(cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY), None)
        
        # Pose Guess
        if cfg.guess_type == "GT-Cam+CV-Obj":
            if len(history_T_WO_gt) >= 2:
                T_prev = history_T_WO_gt[-1]
                T_prev_prev = history_T_WO_gt[-2]
                T_WO_guess = T_prev @ np.linalg.inv(T_prev_prev) @ T_prev
            else:
                T_WO_guess = history_T_WO_gt[-1]
            T_guess = fd["T_CW_gt"] @ T_WO_guess
        else:
            T_guess = geo_tracker.poses[idx-1].copy()
        
        geo_tracker.match_projections(idx, fd["image"], fd["mask"], fd["K"], T_guess)
        success, _ = geo_tracker.step_informed(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], T_guess, flow, skip_pnp=cfg.skip_pnp)
        if not success: break

        # Add new points if needed
        active_tids = [tid for tid, t in geo_tracker.tracks.items() if idx in t['obs']]
        if len(active_tids) < cfg.n_features:
            geo_tracker.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], geo_tracker.poses[idx], align=True)
            
        # Keyframe logic
        curr_gray = cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY)
        flow_kf_curr = dis.calc(last_kf_gray, curr_gray, None)
        mask_curr = fd["mask"] > 0
        avg_disparity = np.median(np.linalg.norm(flow_kf_curr, axis=-1)[mask_curr]) if np.any(mask_curr) else 0
        overlap_ratio = sum(1 for tid in active_tids if last_kf_idx in geo_tracker.tracks[tid]['obs']) / len(active_tids) if active_tids else 0
        
        is_kf = (idx - last_kf_idx >= cfg.kf_min_interval and (avg_disparity > cfg.kf_disparity_thresh or overlap_ratio < cfg.kf_overlap_thresh)) or (idx - last_kf_idx >= cfg.kf_max_interval)
        if is_kf:
            geo_tracker.keyframes.append(idx)
            geo_tracker.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], geo_tracker.poses[idx], align=True)
            geo_tracker.run_ba()
            last_kf_idx = idx
            last_kf_gray = curr_gray

        # --- Online GS Mapping ---
        T_CiO = geo_tracker.poses[idx]
        T_CO_t = torch.from_numpy(T_CiO).float().to(device)
        K_t = torch.from_numpy(fd["K"]).float().to(device)
        image_t = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(fd["mask"] > 0).bool().to(device)
        
        # 1. Densify from Tracker
        if cfg.densify_from_tracker:
            new_pts, new_clrs, new_nmls = [], [], []
            
            # Compute normals for current frame if needed
            if fd["depth"] is not None:
                fd_depth_t = torch.from_numpy(fd["depth"]).float().to(device)
                fd_pts_c = unproject_depth(fd_depth_t, K_t, fd["image"].shape[0], fd["image"].shape[1])
                fd_normals_c, _ = d2n_tblr(fd_pts_c.permute(2, 0, 1).unsqueeze(0))
                fd_normals_c = -fd_normals_c[0]
                fd_normals_c = F.normalize(fd_normals_c, dim=0)
                T_CO_inv_t = torch.inverse(T_CO_t)
                fd_normals_o = torch.einsum('ij,hwj->hwi', T_CO_inv_t[:3, :3], fd_normals_c.permute(1, 2, 0)).cpu().numpy()
            else:
                fd_normals_o = None

            for tid, t in geo_tracker.tracks.items():
                if tid not in gs_tid_set and idx in t['obs']:
                    new_pts.append(t['pt3d'])
                    uv = t['obs'][idx]
                    iy, ix = int(round(uv[1])), int(round(uv[0]))
                    new_clrs.append(fd["image"][iy, ix] / 255.0)
                    if fd_normals_o is not None:
                        new_nmls.append(fd_normals_o[iy, ix])
                    gs_tid_set.add(tid)
            if new_pts:
                new_nmls_np = np.array(new_nmls) if new_nmls else None
                gs_params = densify_from_tracker(gs_params, np.array(new_pts), np.array(new_clrs), device, new_nmls_np)
                optimizer = setup_optimizer(gs_params, cfg)

        # 2. Optimize
        cam = MiniCam(fd["K"], T_CiO, fd["image"].shape[1], fd["image"].shape[0])
        for step in range(cfg.num_steps_per_frame):
            optimizer.zero_grad()
            render_image, render_depth, render_normal, render_alpha = render_2dgs(
                gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales),
                gs_params.colors, torch.sigmoid(gs_params.opacity),
                viewmat=T_CO_t, K=K_t,
                width=cam.width, height=cam.height,
                near_plane=cfg.near_plane, far_plane=cfg.far_plane
            )
            
            loss_photo = F.l1_loss(render_image * mask_t, image_t * mask_t)
            loss_mask = F.l1_loss(render_alpha * (~mask_t).float(), torch.zeros_like(render_alpha))
            
            loss_single = torch.tensor(0.0, device=device)
            if cfg.use_pgsr:
                loss_single = compute_single_view_loss(render_normal, render_depth, image_t, cam, weight=cfg.single_view_weight, mask=mask_t)
                
            loss_multi = torch.tensor(0.0, device=device)
            if cfg.use_pgsr and len(keyframes) > 0:
                kf_idx = np.random.randint(0, len(keyframes))
                kf_cam, kf_image_gray = keyframes[kf_idx]
                grid_y, grid_x = torch.meshgrid(torch.arange(cam.height, device=device), torch.arange(cam.width, device=device), indexing='ij')
                pixels = torch.stack([grid_x, grid_y], dim=-1).float().reshape(-1, 2)
                mask_indices = torch.where(mask_t.reshape(-1))[0]
                if len(mask_indices) > 2000:
                    mask_indices = mask_indices[torch.randperm(len(mask_indices), device=device)[:2000]]
                if len(mask_indices) > 0:
                    ncc, _ = compute_multi_view_loss(image_t.mean(0).unsqueeze(0), kf_image_gray.unsqueeze(0), render_normal, render_depth, cam, kf_cam, pixels=pixels[mask_indices], valid_indices=torch.arange(len(mask_indices), device=device))
                    loss_multi = ncc.mean() * cfg.multi_view_weight
            
            total_loss = loss_photo + loss_mask + loss_single + loss_multi
            total_loss.backward()
            optimizer.step()

        # 3. Update Keyframes for Mapping
        if idx % 5 == 0:
            keyframes.append((cam, image_t.mean(0).detach()))
            if len(keyframes) > 10: keyframes.pop(0)

        # 4. Pruning
        if idx % 20 == 0:
            gs_params, changed = prune_gs(gs_params, T_CO_t, K_t, fd["image"].shape[1], cfg.prune_opacity_th, cfg.prune_screen_size_th)
            if changed: optimizer = setup_optimizer(gs_params, cfg)

        # Visualization
        if not cfg.no_vis:
            rr.set_time("frame", sequence=idx)
            rr.log("input/image", rr.Image(fd["image"]))
            
            # Render final for vis
            with torch.no_grad():
                render_image, render_depth, render_normal, render_alpha = render_2dgs(
                    gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales),
                    gs_params.colors, torch.sigmoid(gs_params.opacity),
                    viewmat=T_CO_t, K=K_t,
                    width=cam.width, height=cam.height
                )
                render_np = render_image.permute(1, 2, 0).cpu().numpy().clip(0, 1)
                rr.log("render/image", rr.Image(render_np))
                
                # Depth comparison
                if fd["depth"] is not None:
                    depth_vis = np.zeros((fd["image"].shape[0], fd["image"].shape[1] * 2))
                    depth_vis[:, :fd["image"].shape[1]] = fd["depth"] / 2.0 # Scale for vis
                    depth_vis[:, fd["image"].shape[1]:] = render_depth[0].cpu().numpy() / 2.0
                    rr.log("render/depth_comp", rr.Image(depth_vis))
    
            T_OC_est = np.linalg.inv(T_CiO)
            rr.log("object/camera", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
            rr.log("object/camera", rr.Pinhole(image_from_camera=fd["K"], width=fd["image"].shape[1], height=fd["image"].shape[0]))
            
            # Log points
            rr.log("object/points/tracker", rr.Points3D(np.array([t['pt3d'] for t in geo_tracker.tracks.values() if idx in t['obs']]), colors=[255, 0, 255], radii=0.002))
            
            with torch.no_grad():
                sh_colors = gs_params.colors.detach()
                # Fix color broadcasting for Rerun
                colors_rgb = (sh_colors[:, 0:1] * 0.28209479177387814 + 0.5).clamp(0, 1).repeat(1, 3).cpu().numpy()
                rr.log("object/points/gs", rr.Points3D(gs_params.means.detach().cpu().numpy(), colors=(colors_rgb * 255).astype(np.uint8), radii=0.001))

        if fd["T_WO_gt"] is not None:
            history_T_WO_gt.append(fd["T_WO_gt"])
            
        prev_f = fd

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
