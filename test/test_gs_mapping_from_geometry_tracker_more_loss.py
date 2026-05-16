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
import random

from geometric_tracker import GeometricTracker
from obj_gs_mapping import MappingConfig, MiniCam, get_scaled_cam, build_rotation_from_normal
from gs_dyn_obj.gs_param import GSParam
from gs_dyn_obj.grouped_gs import RGB2SH
from gs_dyn_obj.gs_rendering import render_2dgs
from gs_dyn_obj.utils.init import unproject_depth, d2n_tblr
from gsplat.exporter import export_splats

# Import PGSR-style loss utilities
import sys
sys.path.append(str(Path(__file__).parent / "third_party" / "PGSR"))
from utils.loss_utils import ssim, lncc, get_img_grad_weight
from utils.graphics_utils import patch_offsets, patch_warp

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    init_frame: int = 0
    n_frames: int = 100
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # Tracker Parameters
    feature_type: str = "grid" 
    grid_spacing: int = 12
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    n_features: int = 2000
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
    informed_thresh: float = 30.0 
    guess_type: str = "GT-Cam+CV-Obj"
    
    # Mapping Parameters
    num_steps_per_frame: int = 200
    lr_means: float = 1e-3
    lr_quats: float = 1e-3
    lr_scales: float = 5e-3
    lr_colors: float = 2.5e-3
    lr_opacity: float = 5e-2
    
    pyr_levels: int = 2
    pyr_interval: int = 30
    
    # PGSR Loss Weights
    lambda_dssim: float = 0.5
    single_view_weight: float = 0.015
    multi_view_ncc_weight: float = 0.5
    multi_view_geo_weight: float = 10.0
    multi_view_photo_weight: float = 10.0
    scale_loss_weight: float = 100.0
    
    # PGSR Multi-view Parameters
    multi_view_min_dis: float = 0.02 # Min distance to be considered a neighbor
    multi_view_max_dis: float = 0.1 # Max distance for virtual cam noise
    multi_view_patch_size: int = 3
    multi_view_sample_num: int = 8000
    multi_view_pixel_noise_th: float = 1.0
    use_virtul_cam: bool = True
    virtul_cam_prob: float = 0.5
    
    # New Loss Options
    use_mask_loss: bool = True
    mask_loss_weight: float = 1.0
    use_depth_loss: bool = True
    depth_loss_weight: float = 10.0
    ref_frame_weight: float = 0.5
    
    near_plane: float = 0.01
    far_plane: float = 100.0
    
    # Densification & Pruning
    densify_from_tracker: bool = True
    densify_from_depth: bool = False 
    prune_opacity_th: float = 0.01
    prune_screen_size_th: float = 100.0
    
    # TSDF Parameters
    run_tsdf: bool = True
    tsdf_voxel_size: float = 0.005
    tsdf_margin: float = 0.02
    
    no_vis: bool = False
    save_ply: bool = True

def get_depth_normal(d, K):
    H, W = d.shape[-2:]
    fy, fx = K[1, 1], K[0, 0]
    cy, cx = K[1, 2], K[0, 2]
    y, x = torch.meshgrid(torch.arange(H, device=d.device), torch.arange(W, device=d.device), indexing='ij')
    pts_c = torch.stack([(x - cx) * d[0] / fx, (y - cy) * d[0] / fy, d[0]], dim=-1)
    dz_dx = torch.gradient(pts_c[..., 0], dim=1)[0]
    dz_dy = torch.gradient(pts_c[..., 1], dim=0)[0]
    v1 = torch.stack([torch.ones_like(dz_dx), torch.zeros_like(dz_dx), dz_dx], dim=-1)
    v2 = torch.stack([torch.zeros_like(dz_dy), torch.ones_like(dz_dy), dz_dy], dim=-1)
    n = torch.cross(v1, v2, dim=-1)
    return F.normalize(n, dim=-1).permute(2, 0, 1)

def sample_patches(img, p):
    H, W = img.shape[-2:]
    p_norm = p.clone()
    p_norm[..., 0] = 2 * p_norm[..., 0] / (W - 1) - 1.0
    p_norm[..., 1] = 2 * p_norm[..., 1] / (H - 1) - 1.0
    val = F.grid_sample(img[None], p_norm.reshape(1, -1, 1, 2), align_corners=True)
    return val.reshape(-1, p.shape[1])

def compute_geo_consistency(cam1, cam2, depth1, depth2, pixel_noise_th=1.0):
    H, W = depth1.shape[-2:]
    device = depth1.device
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixels = torch.stack([grid_x, grid_y], dim=-1).float()
    K1_inv = cam1.get_inv_k().to(device)
    pixels_homo = torch.cat([pixels, torch.ones_like(pixels[..., :1])], dim=-1)
    pts1_c = depth1.reshape(H, W, 1) * (pixels_homo @ K1_inv.t())
    W2C1 = cam1.world_view_transform.t().to(device)
    C2W1 = torch.inverse(W2C1)
    pts1_w = (pts1_c @ C2W1[:3, :3].t()) + C2W1[:3, 3]
    W2C2 = cam2.world_view_transform.t().to(device)
    pts2_c = (pts1_w @ W2C2[:3, :3].t()) + W2C2[:3, 3]
    K2 = cam2.get_k().to(device)
    pts2_p = pts2_c @ K2.t()
    u2 = pts2_p[..., 0] / (pts2_p[..., 2] + 1e-6)
    v2 = pts2_p[..., 1] / (pts2_p[..., 2] + 1e-6)
    z2_proj = pts2_p[..., 2]
    grid2 = torch.stack([2 * u2 / (W - 1) - 1, 2 * v2 / (H - 1) - 1], dim=-1)
    depth2_sampled = F.grid_sample(depth2.reshape(1, 1, H, W), grid2.unsqueeze(0), align_corners=True).squeeze()
    pixels2_homo = torch.stack([u2, v2, torch.ones_like(u2)], dim=-1)
    K2_inv = cam2.get_inv_k().to(device)
    pts2_sampled_c = depth2_sampled.unsqueeze(-1) * (pixels2_homo @ K2_inv.t())
    C2W2 = torch.inverse(W2C2)
    pts2_sampled_w = (pts2_sampled_c @ C2W2[:3, :3].t()) + C2W2[:3, 3]
    pts1_back_c = (pts2_sampled_w @ W2C1[:3, :3].t()) + W2C1[:3, 3]
    pts1_back_p = pts1_back_c @ cam1.K.to(device).t()
    u1_back = pts1_back_p[..., 0] / (pts1_back_p[..., 2] + 1e-6)
    v1_back = pts1_back_p[..., 1] / (pts1_back_p[..., 2] + 1e-6)
    pixel_noise = torch.sqrt((u1_back - pixels[..., 0])**2 + (v1_back - pixels[..., 1])**2)
    d_mask = (z2_proj > 0.1) & (depth2_sampled > 0.1) & (pixel_noise < pixel_noise_th)
    weights = (1.0 / torch.exp(pixel_noise)).detach()
    weights[~d_mask] = 0
    return d_mask, weights, pixel_noise

def get_lapla_norm(img, kernel=None):
    device = img.device
    laplacian_kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    laplacian_kernel = laplacian_kernel.repeat(1, img.shape[0], 1, 1)
    laplacian = F.conv2d(img[None], laplacian_kernel, padding="same")
    laplacian_norm = torch.linalg.vector_norm(laplacian, ord=1, dim=1, keepdim=True)
    laplacian_norm[..., 0, 0] = 0
    if kernel is not None:
        return F.conv2d(laplacian_norm, kernel, padding="same")[0, 0].clamp(0, 1)
    return laplacian_norm[0, 0].clamp(0, 1)

def load_object_pose_world(data_root, frame_idx):
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists(): return None
    with open(poses_file, "r") as f: lines = f.readlines()
    if frame_idx >= len(lines): return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])])
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4)
    T_WO[:3, :3] = r_WO; T_WO[:3, 3] = t_WO
    return T_WO

def load_frame_data(data_dir, frame_idx):
    stem = f"{frame_idx:06d}"
    img_path = data_dir / "images" / f"{stem}.png"
    if not img_path.exists(): return None
    img = np.array(cv2.imread(str(img_path))[..., ::-1])
    mask_est_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
    if not mask_est_path.exists(): mask_est_path = data_dir / "obj_masks" / f"{stem}.png"
    depth_est_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
    if not mask_est_path.exists(): return None
    mask = np.array(cv2.imread(str(mask_est_path), cv2.IMREAD_GRAYSCALE))
    depth = np.load(depth_est_path) if depth_est_path.exists() else None
    if depth is not None and depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
    T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
    T_WO_gt = load_object_pose_world(data_dir, frame_idx)
    return {"image": img, "mask": mask, "depth": depth, "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": frame_idx}

def setup_optimizer(gs_params, cfg):
    return torch.optim.Adam([
        {"params": [gs_params.means], "lr": cfg.lr_means},
        {"params": [gs_params.quats], "lr": cfg.lr_quats},
        {"params": [gs_params.scales], "lr": cfg.lr_scales},
        {"params": [gs_params.colors], "lr": cfg.lr_colors},
        {"params": [gs_params.opacity], "lr": cfg.lr_opacity},
    ])

def init_gs_from_tracker_points(points, colors, device, normals=None):
    num_pts = points.shape[0]
    means = torch.from_numpy(points).float().to(device); means.requires_grad = True
    colors_sh = RGB2SH(torch.from_numpy(colors).float().to(device)); colors_sh.requires_grad = True
    if normals is not None:
        quats = build_rotation_from_normal(torch.from_numpy(normals).float().to(device))
    else:
        quats = torch.zeros((num_pts, 4), device=device); quats[:, 0] = 1.0
    quats.requires_grad = True
    scales = torch.log(torch.ones((num_pts, 2), device=device) * 0.002); scales.requires_grad = True
    opacity = torch.logit(torch.ones((num_pts, 1), device=device) * 0.5); opacity.requires_grad = True
    return GSParam(means, quats, scales, colors_sh, opacity)

def densify_from_tracker(gs_params, new_points, new_colors, device, new_normals=None):
    if len(new_points) == 0: return gs_params
    new_gs = init_gs_from_tracker_points(new_points, new_colors, device, new_normals)
    for attr in ['means', 'quats', 'scales', 'colors', 'opacity']:
        setattr(gs_params, attr, torch.nn.Parameter(torch.cat([getattr(gs_params, attr).data, getattr(new_gs, attr).data], dim=0)))
    return gs_params

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("test_gs_mapping_from_geometry_tracker_more_loss", spawn=False)
        if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
    data_dir = Path(cfg.data_root) / cfg.clip_id
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    geo_tracker = GeometricTracker(cfg)
    device = torch.device(cfg.device)
    f0 = load_frame_data(data_dir, cfg.init_frame)
    T_C0O_gt = f0["T_CW_gt"] @ f0["T_WO_gt"] if f0["T_WO_gt"] is not None else np.eye(4)
    geo_tracker.poses[f0["frame_idx"]] = T_C0O_gt
    geo_tracker.K_dict[f0["frame_idx"]] = f0["K"]
    geo_tracker.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], f0["depth"], f0["K"], T_C0O_gt)
    geo_tracker.keyframes.append(f0["frame_idx"])
    
    active_points, active_colors, active_normals, active_gs_tids = [], [], [], []
    gs_tid_set = set()
    f0_depth_t = torch.from_numpy(f0["depth"]).float().to(device)
    f0_pts_c = unproject_depth(f0_depth_t, torch.from_numpy(f0["K"]).float().to(device), f0["image"].shape[0], f0["image"].shape[1])
    f0_normals_c, _ = d2n_tblr(f0_pts_c.permute(2, 0, 1).unsqueeze(0))
    f0_normals_c = F.normalize(-f0_normals_c[0], dim=0)
    T_C0O_inv_t = torch.inverse(torch.from_numpy(T_C0O_gt).float().to(device))
    f0_normals_o = torch.einsum('ij,hwj->hwi', T_C0O_inv_t[:3, :3], f0_normals_c.permute(1, 2, 0)).cpu().numpy()

    for tid, t in geo_tracker.tracks.items():
        if f0["frame_idx"] in t['obs']:
            active_points.append(t['pt3d'])
            uv = t['obs'][f0["frame_idx"]]; iy, ix = int(round(uv[1])), int(round(uv[0]))
            active_colors.append(f0["image"][iy, ix] / 255.0)
            active_normals.append(f0_normals_o[iy, ix])
            gs_tid_set.add(tid); active_gs_tids.append(tid)
            
    gs_params = init_gs_from_tracker_points(np.array(active_points), np.array(active_colors), device, np.array(active_normals))
    optimizer = setup_optimizer(gs_params, cfg)
    mapping_keyframes = []
    training_psnrs, evaluation_data = [], []
    pbar = tqdm(range(1, cfg.n_frames), desc="Tracking + Mapping")
    prev_f = f0
    disc_kernel = torch.zeros(1, 1, 7, 7, device=device); ky, kx = torch.meshgrid(torch.arange(-3, 4, device=device), torch.arange(-3, 4, device=device), indexing="ij")
    disc_kernel[0, 0, torch.sqrt(kx**2 + ky**2) <= 3.5] = 1; disc_kernel /= disc_kernel.sum()

    for i in pbar:
        idx = cfg.init_frame + i; fd = load_frame_data(data_dir, idx)
        if fd is None: break
        if not cfg.no_vis: rr.set_time("frame_idx", sequence=idx)
        T_guess = fd["T_CW_gt"] @ (prev_f["T_WO_gt"] if prev_f["T_WO_gt"] is not None else np.eye(4))
        geo_tracker.match_projections(idx, fd["image"], fd["mask"], fd["K"], T_guess)
        success, _ = geo_tracker.step_informed(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], T_guess, cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY), None), skip_pnp=cfg.skip_pnp)
        
        # Add new tracker points if count drops below threshold to prevent tracking loss
        n_active = sum(1 for t in geo_tracker.tracks.values() if idx in t['obs'])
        if n_active < 400:
            geo_tracker.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], geo_tracker.poses[idx])
            
        if not success: break
        
        # Mapping
        T_CO_t = torch.from_numpy(geo_tracker.poses[idx]).float().to(device)
        K_t = torch.from_numpy(fd["K"]).float().to(device)
        image_t = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(fd["mask"] > 0).bool().to(device)
        
        # Periodic Densification
        if idx % 5 == 0:
            with torch.no_grad():
                render_image, render_depth, _, render_alpha = render_2dgs(gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales), gs_params.colors, torch.sigmoid(gs_params.opacity), viewmat=T_CO_t, K=K_t, width=fd["image"].shape[1], height=fd["image"].shape[0])
                init_proba = get_lapla_norm(image_t, disc_kernel); penalty = get_lapla_norm(render_image, disc_kernel)
                sample_mask = ((torch.rand_like(init_proba) < (init_proba - penalty) * 10.0) | (torch.from_numpy(fd["depth"]).to(device) < render_depth[0] - 0.01)) & mask_t & (render_alpha[0] < 0.5)
                if sample_mask.any():
                    yy_v, xx_v = torch.where(sample_mask); perm = torch.randperm(len(yy_v), device=device)[:12000]; yy_v, xx_v = yy_v[perm], xx_v[perm]
                    zz_v = torch.from_numpy(fd["depth"]).to(device)[yy_v, xx_v]
                    pts_c = torch.stack([(xx_v.float() - fd["K"][0, 2]) * zz_v / fd["K"][0, 0], (yy_v.float() - fd["K"][1, 2]) * zz_v / fd["K"][1, 1], zz_v], dim=-1)
                    T_OC = torch.inverse(T_CO_t)
                    pts_o = (pts_c @ T_OC[:3, :3].T) + T_OC[:3, 3]
                    new_gs = init_gs_from_tracker_points(pts_o.detach().cpu().numpy(), (image_t[:, yy_v, xx_v].permute(1, 0)).detach().cpu().numpy(), device)
                    for attr in ['means', 'quats', 'scales', 'colors', 'opacity']:
                        setattr(gs_params, attr, torch.nn.Parameter(torch.cat([getattr(gs_params, attr).data, getattr(new_gs, attr).data], dim=0)))
                    active_gs_tids.extend([-1] * len(pts_o)); optimizer = setup_optimizer(gs_params, cfg)

        cam = MiniCam(fd["K"], geo_tracker.poses[idx], fd["image"].shape[1], fd["image"].shape[0])
        image_pyr = [image_t]; mask_pyr = [mask_t.unsqueeze(0).float()]
        for _ in range(cfg.pyr_levels - 1):
            image_pyr.append(F.avg_pool2d(image_pyr[-1], 2)); mask_pyr.append(F.avg_pool2d(mask_pyr[-1], 2))
        mask_pyr = [m[0] > 0.5 for m in mask_pyr]

        # Get sparse depth points from tracker for current frame
        sparse_pts_o = []
        sparse_uvs = []
        for tid, t in geo_tracker.tracks.items():
            if idx in t['obs']:
                sparse_pts_o.append(t['pt3d'])
                sparse_uvs.append(t['obs'][idx])
        
        if len(sparse_pts_o) > 0:
            sparse_pts_o = torch.from_numpy(np.array(sparse_pts_o)).float().to(device)
            sparse_uvs = torch.from_numpy(np.array(sparse_uvs)).float().to(device)
        else:
            sparse_pts_o = None

        for step in range(cfg.num_steps_per_frame):
            lvl = max(0, (cfg.pyr_levels - 1) - (step // cfg.pyr_interval))
            target_image, target_mask = image_pyr[lvl], mask_pyr[lvl]
            curr_K = cam.get_k(1.0 / (2**lvl)).to(device)
            optimizer.zero_grad()
            
            # Current frame rendering and loss
            render_image, render_depth, render_normal, render_alpha = render_2dgs(gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales), gs_params.colors, torch.sigmoid(gs_params.opacity), viewmat=T_CO_t, K=curr_K, width=target_image.shape[2], height=target_image.shape[1])
            
            loss_photo = (1.0 - cfg.lambda_dssim) * F.l1_loss(render_image * target_mask, target_image * target_mask) + cfg.lambda_dssim * (1.0 - ssim((render_image * target_mask).unsqueeze(0), (target_image * target_mask).unsqueeze(0)))
            
            loss_mask = torch.tensor(0.0, device=device)
            if cfg.use_mask_loss:
                # Enforce alpha matches mask (both foreground and background)
                loss_mask = F.l1_loss(render_alpha, target_mask.unsqueeze(0).float()) * cfg.mask_loss_weight
            else:
                loss_mask = F.l1_loss(render_alpha * (~target_mask).float(), torch.zeros_like(render_alpha))
            
            loss_normal = cfg.single_view_weight * (get_img_grad_weight(target_image) * (get_depth_normal(render_depth, curr_K) - render_normal).abs().sum(0)).mean()
            
            # Sparse Depth Loss
            loss_depth = torch.tensor(0.0, device=device)
            if cfg.use_depth_loss and sparse_pts_o is not None and lvl == 0:
                # Project sparse points to current camera frame
                # T_CO_t is world_view_transform (W2C)
                pts_c = (sparse_pts_o @ T_CO_t[:3, :3].T) + T_CO_t[:3, 3]
                z_target = pts_c[:, 2]
                
                # Sample rendered depth at sparse UVs
                # grid_sample expects coordinates in [-1, 1]
                uv_norm = sparse_uvs.clone()
                uv_norm[:, 0] = 2.0 * uv_norm[:, 0] / (fd["image"].shape[1] - 1) - 1.0
                uv_norm[:, 1] = 2.0 * uv_norm[:, 1] / (fd["image"].shape[0] - 1) - 1.0
                
                # Sample render_depth: (1, 1, H, W)
                sampled_depth = F.grid_sample(render_depth.unsqueeze(0), uv_norm.view(1, -1, 1, 2), align_corners=True).view(-1)
                
                # Only penalize if target depth is valid (should be if they were in tracks)
                valid_depth = z_target > 0.01
                if valid_depth.any():
                    loss_depth = F.l1_loss(sampled_depth[valid_depth], z_target[valid_depth]) * cfg.depth_loss_weight

            # Multi-view / Reference Frame Loss
            loss_mv = torch.tensor(0.0, device=device)
            # Always try to use the most recent keyframe as a reference frame if available
            ref_keyframes = []
            if mapping_keyframes:
                # Use the last keyframe as a mandatory reference
                ref_keyframes.append(mapping_keyframes[-1])
                # Plus some random ones if available
                if len(mapping_keyframes) > 1:
                    other_kfs = mapping_keyframes[:-1]
                    ref_keyframes.extend(random.sample(other_kfs, min(2, len(other_kfs))))

            for kf in ref_keyframes:
                m_cam, kf_img, kf_gray, kf_depth, kf_mask = kf
                # For reference frames, we might not need pyramid levels for simplicity, 
                # or we can just use the full resolution.
                r_img, _, _, r_alpha = render_2dgs(gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales), gs_params.colors, torch.sigmoid(gs_params.opacity), viewmat=m_cam.world_view_transform.t(), K=m_cam.K, width=m_cam.width, height=m_cam.height)
                
                # Photo loss on reference frame
                loss_mv += cfg.ref_frame_weight * ((1.0 - cfg.lambda_dssim) * F.l1_loss(r_img * kf_mask, kf_img * kf_mask) + cfg.lambda_dssim * (1.0 - ssim((r_img * kf_mask).unsqueeze(0), (kf_img * kf_mask).unsqueeze(0))))
                
                # Mask loss on reference frame
                if cfg.use_mask_loss:
                    loss_mv += cfg.ref_frame_weight * cfg.mask_loss_weight * F.l1_loss(r_alpha, kf_mask.unsqueeze(0).float())
                else:
                    loss_mv += cfg.ref_frame_weight * F.l1_loss(r_alpha * (~kf_mask).float(), torch.zeros_like(r_alpha))

            total_loss = loss_photo + loss_mask + loss_normal + loss_depth + loss_mv
            total_loss.backward()
            optimizer.step()

        if idx % 5 == 0:
            mapping_keyframes.append((MiniCam(fd["K"], geo_tracker.poses[idx], fd["image"].shape[1], fd["image"].shape[0]), image_t.detach(), image_t.mean(0, keepdim=True).detach(), render_depth.detach(), mask_t.detach()))
            if len(mapping_keyframes) > 20: mapping_keyframes.pop(0)

        with torch.no_grad():
            mse = F.mse_loss(render_image * mask_t, image_t * mask_t, reduction='sum') / (mask_t.sum() * 3)
            psnr = -10 * torch.log10(mse + 1e-10); training_psnrs.append(psnr.item())
            evaluation_data.append((idx, T_CO_t.cpu(), K_t.cpu(), image_t.cpu(), mask_t.cpu()))
            pbar.set_postfix({"PSNR": f"{psnr.item():.2f}", "Pts": f"{len(gs_params.means)}"})
        prev_f = fd

    print(f"\nAverage Training PSNR: {np.mean(training_psnrs):.2f}")
    final_psnrs = []
    for idx, T_CO_eval, K_eval, img_eval, msk_eval in tqdm(evaluation_data, desc="Final Evaluation"):
        with torch.no_grad():
            r_img, _, _, _ = render_2dgs(gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales), gs_params.colors, torch.sigmoid(gs_params.opacity), viewmat=T_CO_eval.to(device), K=K_eval.to(device), width=img_eval.shape[2], height=img_eval.shape[1])
            mse = F.mse_loss(r_img * msk_eval.to(device), img_eval.to(device) * msk_eval.to(device), reduction='sum') / (msk_eval.sum() * 3)
            final_psnrs.append((-10 * torch.log10(mse + 1e-10)).item())
    print(f"Average Final PSNR: {np.mean(final_psnrs):.2f}")

    if cfg.save_ply:
        save_dir = Path("output") / cfg.clip_id
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / "object_gs_more_loss.ply"
        
        with torch.no_grad():
            means = gs_params.means.detach()
            quats = F.normalize(gs_params.quats.detach())
            scales = torch.exp(gs_params.scales.detach())
            if scales.shape[-1] == 2:
                # Pad to 3D for gsplat exporter
                scales = torch.cat([scales, torch.ones_like(scales[:, :1]) * 1e-6], dim=-1)
            
            opacities = torch.sigmoid(gs_params.opacity.detach()).squeeze(-1)
            sh0 = gs_params.colors.detach().unsqueeze(1) # [N, 1, 3]
            shN = torch.zeros((means.shape[0], 0, 3), device=device)
            
            export_splats(
                means, 
                scales, 
                quats, 
                opacities, 
                sh0, 
                shN, 
                format="ply", 
                save_to=str(save_path)
            )
            print(f"Saved object GS to {save_path}")

if __name__ == "__main__":
    main(tyro.cli(Config))
