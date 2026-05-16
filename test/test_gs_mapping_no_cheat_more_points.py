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
import threading
import poselib
import time
import json
import open3d as o3d

from geometric_tracker import GeometricTracker
from gs_dyn_obj.obj_gs import ObjectGS
from obj_gs_mapping import GSMapping, MappingConfig, MiniCam, get_scaled_cam, build_rotation_from_normal
from gs_dyn_obj.gs_param import GSParam
from gs_dyn_obj.grouped_gs import RGB2SH
from gs_dyn_obj.gs_rendering import render_2dgs
from gs_dyn_obj.utils.init import unproject_depth, d2n_tblr

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    init_frame: int = 0
    n_frames_track: int = 30
    n_frames_eval: int = 30
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    no_vis: bool = False
    use_ray_dist: bool = False
    
    # Tracking Options
    n_init_frames: int = 5 # Initialization window
    grid_spacing: int = 2   # Extremely dense tracking
    ransac_thresh: float = 1.0
    max_keyframes: int = 20
    kf_every: int = 2 # More frequent BA

def load_object_pose_world(data_root, frame_idx):
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists(): return None
    with open(poses_file, "r") as f: lines = f.readlines()
    if frame_idx >= len(lines): return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])]) # x y z w
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4)
    T_WO[:3, :3] = r_WO; T_WO[:3, 3] = t_WO
    return T_WO

def load_frame_data(data_dir, frame_idx):
    stem = f"{frame_idx:06d}"
    img_path = data_dir / "images" / f"{frame_idx:05d}.jpg"
    if not img_path.exists():
        img_path = data_dir / "images" / f"{stem}.png"
    if not img_path.exists(): return None
    
    img = np.array(cv2.imread(str(img_path))[..., ::-1])
    mask_est_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
    depth_est_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
    
    if not mask_est_path.exists():
        mask_est_path = data_dir / "obj_masks" / f"{stem}.png"
    
    if not mask_est_path.exists(): return None
    
    mask = np.array(cv2.imread(str(mask_est_path), cv2.IMREAD_GRAYSCALE))
    depth = np.load(depth_est_path) if depth_est_path.exists() else None
    
    if depth is not None and depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    K_path = data_dir / "intrinsics" / f"{stem}.npy"
    if not K_path.exists():
        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                meta = json.load(f)
                K = np.array(meta["K"])
                T_CW_gt = np.eye(4) 
        else:
            K = np.eye(3); T_CW_gt = np.eye(4)
    else:
        K = np.load(K_path)
        T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        
    T_WO_gt = load_object_pose_world(data_dir, frame_idx)
    return {"image": img, "mask": mask, "depth": depth, "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": frame_idx}

def init_gs_from_frame(f, T_CO, device, max_pts=15000):
    img = f["image"]
    mask = f["mask"] > 0
    depth = f["depth"]
    K = f["K"]
    
    H, W = img.shape[:2]
    y, x = np.where(mask & (depth > 0))
    if len(y) == 0: return None
        
    if len(y) > max_pts:
        perm = np.random.permutation(len(y))[:max_pts]
        y, x = y[perm], x[perm]
        
    z = depth[y, x]
    pts_c = np.stack([(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z], axis=-1)
    T_OC = np.linalg.inv(T_CO)
    pts_o = (pts_c @ T_OC[:3, :3].T) + T_OC[:3, 3]
    colors = img[y, x] / 255.0
    
    num_pts = pts_o.shape[0]
    means = torch.from_numpy(pts_o).float().to(device); means.requires_grad = True
    colors_sh = RGB2SH(torch.from_numpy(colors).float().to(device)); colors_sh.requires_grad = True
    
    # Normals from depth
    full_pts_c = unproject_depth(torch.from_numpy(depth).float().to(device), torch.from_numpy(K).float().to(device), H, W)
    normals_c, _ = d2n_tblr(full_pts_c.permute(2, 0, 1).unsqueeze(0))
    normals_c = -F.normalize(normals_c[0][:, y, x].permute(1, 0), dim=1).cpu().numpy()
    normals_o = (normals_c @ T_OC[:3, :3].T)
    
    quats = build_rotation_from_normal(torch.from_numpy(normals_o).float().to(device))
    quats.requires_grad = True
    
    scales = torch.log(torch.ones((num_pts, 2), device=device) * 0.01); scales.requires_grad = True
    opacity = torch.logit(torch.ones((num_pts, 1), device=device) * 0.99); opacity.requires_grad = True
    
    return GSParam(means, quats, scales, colors_sh, opacity)

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("test_gs_mapping_no_cheat_more_points", spawn=False)
        if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
        rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    data_dir = Path(cfg.data_root) / cfg.clip_id
    device = torch.device(cfg.device)
    
    # Mapping Config (Max Stability)
    map_cfg = MappingConfig(
        device=cfg.device, num_steps_per_frame=200, pyr_levels=2, pyr_interval=15,
        fix_color_and_scales=True, use_ray_dist=True, use_pgsr=True, 
        densify_every=5, kf_every=5, lr_means=2e-3, lr_opacity=0.05, prune_opacity_th=0.001
    )
    
    # Phase 0: Photometric Initialization (Frames 0-5)
    print("\n>>> Phase 0: Photometric Initialization (Frames 0-5)")
    f0 = load_frame_data(data_dir, 0)
    T_C0O_gt = f0["T_CW_gt"] @ f0["T_WO_gt"] if f0["T_WO_gt"] is not None else np.eye(4)
    h, w = f0["image"].shape[:2]
    
    # Initialize Mapper with Frame 0 Seed
    gs_params = init_gs_from_frame(f0, T_C0O_gt, device, max_pts=15000)
    mapper = GSMapping(map_cfg, gs_params)
    
    poses = {0: np.eye(4)}; tracks = {}; next_tid = 0; K_dict = {0: f0["K"]}; keyframes = [0]; frames_buffer = [f0]
    global_scale = [1.0, 0.0]
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    
    # Initial tracks at frame 0
    from geometric_tracker import sample_grid_on_mask, interpolate_flow, run_ba
    new_pts2d = sample_grid_on_mask(f0["mask"], cfg.grid_spacing)
    K_inv = np.linalg.inv(f0["K"])
    for uv in new_pts2d:
        z = f0["depth"][int(round(uv[1])), int(round(uv[0]))]
        if z > 0.01:
            pt_C0 = (K_inv @ np.array([uv[0], uv[1], 1.0])) * z
            tracks[next_tid] = {'obs': {0: uv}, 'pt3d': pt_C0}; next_tid += 1

    for idx in range(1, cfg.n_init_frames):
        fd = load_frame_data(data_dir, idx)
        if fd is None: break
        f_prev = frames_buffer[-1]
        
        # Track from previous
        flow = dis.calc(cv2.cvtColor(f_prev["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY), None)
        for tid, t in tracks.items():
            if (idx-1) in t['obs']:
                uv_prev = t['obs'][idx-1]
                delta = interpolate_flow(flow, uv_prev[None])[0]
                uv_curr = uv_prev + delta
                ix, iy = int(round(uv_curr[0])), int(round(uv_curr[1]))
                if 0 <= ix < fd["image"].shape[1] and 0 <= iy < fd["image"].shape[0] and fd["mask"][iy, ix] > 0:
                    t['obs'][idx] = uv_curr

        obs2d, obs3d = [], []
        for t in tracks.values():
            if idx in t['obs']:
                obs2d.append(t['obs'][idx]); obs3d.append(t['pt3d'])
        
        if len(obs2d) >= 10:
            cam_dict = {'model': 'PINHOLE', 'width': fd["image"].shape[1], 'height': fd["image"].shape[0], 'params': [fd["K"][0,0], fd["K"][1,1], fd["K"][0,2], fd["K"][1,2]]}
            res, info = poselib.estimate_absolute_pose(np.array(obs2d), np.array(obs3d), cam_dict, {'max_reproj_error': cfg.ransac_thresh}, None)
            if res and np.linalg.norm(res.pose.q) > 1e-6:
                T = np.eye(4); T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix(); T[:3, 3] = res.pose.t
                poses[idx] = T
            else:
                poses[idx] = poses[idx-1].copy()
        else:
            poses[idx] = poses[idx-1].copy()
            
        # Photometric Snap
        T_CiO = poses[idx] @ T_C0O_gt
        with mapper.lock:
            T_CiO_refined_t, opt_scale = mapper.refine_pose(torch.from_numpy(fd["image"]).to(device), torch.from_numpy(fd["mask"] > 0).to(device), MiniCam(fd["K"], T_CiO, w, h), torch.from_numpy(T_CiO).float().to(device), steps=200)
            T_CiO = T_CiO_refined_t.detach().cpu().numpy()
            poses[idx] = T_CiO @ np.linalg.inv(T_C0O_gt)
            global_scale[0] *= opt_scale
            
        depth_aligned = fd["depth"] * global_scale[0] + global_scale[1]
        mapper.update({"image": fd["image"], "mask": fd["mask"], "depth": depth_aligned, "K": fd["K"], "T_CiO": T_CiO, "frame_idx": idx})
        frames_buffer.append(fd); K_dict[idx] = fd["K"]

    print(f"Seed GS initialized with {len(mapper.gs_params.means)} points and refined startup window.")
    mapping_thread = threading.Thread(target=mapper.run); mapping_thread.daemon = True; mapping_thread.start()
    
    traj_obj_est_C, traj_obj_gt_C = [], []; T_CiO_gt0 = T_C0O_gt; errors_t = []
    prev_f = frames_buffer[-1]
    global_scale = [1.0, 0.0]
    
    pbar = tqdm(range(cfg.n_init_frames, cfg.n_frames_track), desc="Phase 1: Track+Map")
    for i in pbar:
        idx = cfg.init_frame + i
        fd = load_frame_data(data_dir, idx)
        if fd is None: break
        K_dict[idx] = fd["K"]; h, w = fd["image"].shape[:2]; depth = fd["depth"]
        flow = dis.calc(cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY), None)
        
        prev_idx = prev_f["frame_idx"]
        for tid, t in tracks.items():
            if prev_idx in t['obs']:
                uv_prev = t['obs'][prev_idx]
                delta = interpolate_flow(flow, uv_prev[None])[0]
                uv_curr = uv_prev + delta
                ix, iy = int(round(uv_curr[0])), int(round(uv_curr[1]))
                if 0 <= ix < fd["image"].shape[1] and 0 <= iy < fd["image"].shape[0] and fd["mask"][iy, ix] > 0:
                    t['obs'][idx] = uv_curr
        
        obs2d, obs3d = [], []
        for t in tracks.values():
            if idx in t['obs']:
                obs2d.append(t['obs'][idx]); obs3d.append(t['pt3d'])
        
        if len(obs2d) >= 10:
            cam_dict = {'model': 'PINHOLE', 'width': fd["image"].shape[1], 'height': fd["image"].shape[0], 'params': [fd["K"][0,0], f0["K"][1,1], fd["K"][0,2], fd["K"][1,2]]}
            res, info = poselib.estimate_absolute_pose(np.array(obs2d), np.array(obs3d), cam_dict, {'max_reproj_error': cfg.ransac_thresh}, None)
            if res:
                T = np.eye(4); T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix(); T[:3, 3] = res.pose.t
                poses[idx] = T
            else:
                poses[idx] = poses[idx-1].copy()
        else:
            poses[idx] = poses[idx-1].copy()
            
        if i % cfg.kf_every == 0:
            keyframes.append(idx)
            z_est, z_raw = [], []
            for tid, t in tracks.items():
                if idx in t['obs']:
                    uv = t['obs'][idx]
                    pt_Ci = (poses[idx][:3, :3] @ t['pt3d']) + poses[idx][:3, 3]
                    z_est.append(pt_Ci[2])
                    z_raw.append(fd["depth"][int(round(uv[1])), int(round(uv[0]))])
            z_est, z_raw = np.array(z_est), np.array(z_raw)
            valid = (z_raw > 0.01) & (z_est > 0.01)
            if np.sum(valid) > 10:
                A = np.stack([z_raw[valid], np.ones_like(z_raw[valid])], axis=1)
                res_scale = np.linalg.lstsq(A, z_est[valid], rcond=None)[0]
                global_scale = res_scale.tolist()
            
            # Apply current global scale
            depth = fd["depth"] * global_scale[0] + global_scale[1]
                
            K_inv = np.linalg.inv(fd["K"]); T_C0Ci = np.linalg.inv(poses[idx])
            new_pts2d = sample_grid_on_mask(fd["mask"], cfg.grid_spacing)
            for uv in new_pts2d:
                d = depth[int(round(uv[1])), int(round(uv[0]))]
                if d <= 0.01: continue
                pt_Ci = (K_inv @ np.array([uv[0], uv[1], 1.0])) * d
                pt_C0 = (T_C0Ci[:3, :3] @ pt_Ci) + T_C0Ci[:3, 3]
                tracks[next_tid] = {'obs': {idx: uv}, 'pt3d': pt_C0}; next_tid += 1
            run_ba(keyframes[-cfg.max_keyframes:], poses, tracks, K_dict)

        # Always apply best known scale
        depth = fd["depth"] * global_scale[0] + global_scale[1]

        T_CiC0 = poses[idx]
        T_CiO = T_CiC0 @ T_C0O_gt
        
        # Photometric Refinement: "Snap" the pose to the GS model
        with mapper.lock:
            # Only refine if we have some GS points
            if len(mapper.gs_params.means) > 0:
                T_CiO_refined_t, opt_scale = mapper.refine_pose(
                    torch.from_numpy(fd["image"]).to(device), 
                    torch.from_numpy(fd["mask"] > 0).to(device), 
                    MiniCam(fd["K"], T_CiO, w, h),
                    torch.from_numpy(T_CiO).float().to(device),
                    steps=60
                )
                T_CiO = T_CiO_refined_t.detach().cpu().numpy()
                # Update global scale with feedback from photometric alignment
                global_scale[0] *= opt_scale
                # Feedback to tracker: Update poses[idx] so BA and next frame propagate refined pose
                poses[idx] = T_CiO @ np.linalg.inv(T_C0O_gt)
        
        T_CiO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"]; T_OCi_est = np.linalg.inv(T_CiO); C_est = T_OCi_est[:3, 3]
        T_OCi_gt = np.linalg.inv(T_CiO_gt); C_gt = T_OCi_gt[:3, 3]
        err_ate = np.linalg.norm(C_est - C_gt); errors_t.append(err_ate)
        pbar.set_postfix({"ATE": f"{err_ate:.4f}m", "GS": len(mapper.gs_params.means)})
        
        depth_aligned = fd["depth"] * global_scale[0] + global_scale[1]
        mapper.update({"image": fd["image"], "mask": fd["mask"], "depth": depth_aligned, "K": fd["K"], "T_CiO": T_CiO, "frame_idx": idx})
        if not cfg.no_vis:
            rr.set_time("frame_idx", sequence=idx)
            rr.log("input/image", rr.Image(fd["image"]))
            traj_obj_est_C.append(C_est)
            rr.log(f"object/tracker/camera_est", rr.Transform3D(mat3x3=T_OCi_est[:3, :3], translation=T_OCi_est[:3, 3]))
            rr.log(f"object/tracker/traj_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[[0, 255, 0]], radii=0.003))
            traj_obj_gt_C.append(C_gt)
            rr.log(f"object/camera_gt", rr.Transform3D(mat3x3=T_OCi_gt[:3, :3], translation=T_OCi_gt[:3, 3]))
            rr.log(f"object/traj_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[255, 255, 255]], radii=0.003))
            active_tids = [tid for tid, t in tracks.items() if idx in t['obs']]
            if active_tids:
                pts_C0 = np.array([tracks[tid]['pt3d'] for tid in active_tids])
                pts_O = (np.linalg.inv(T_C0O_gt)[:3, :3] @ pts_C0.T).T + np.linalg.inv(T_C0O_gt)[:3, 3]
                rr.log("object/tracker_points", rr.Points3D(pts_O, colors=[[0, 255, 0]], radii=0.005))
            with mapper.lock:
                means_np = mapper.gs_params.means.detach().cpu().numpy(); colors_sh = mapper.gs_params.colors.detach().cpu().numpy()
                colors_rgb = np.clip(colors_sh * 0.28209479177387814 + 0.5, 0, 1)
                rr.log("object/gs_points", rr.Points3D(means_np, colors=colors_rgb, radii=0.004))
            with mapper.lock:
                render_image, _, _, _ = render_2dgs(mapper.gs_params.means, F.normalize(mapper.gs_params.quats), torch.exp(mapper.gs_params.scales), mapper.gs_params.colors, torch.sigmoid(mapper.gs_params.opacity), viewmat=torch.from_numpy(T_CiO).float().to(device), K=torch.from_numpy(fd["K"]).float().to(device), width=w, height=h)
                render_np = (render_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                rr.log("render/image", rr.Image(render_np))
        prev_f = fd
        while mapper.data_queue.full() and not mapper.stop_event.is_set(): time.sleep(0.1)

    print("Finishing Phase 1 mapping... Waiting for mapper to catch up.")
    last_idx = cfg.init_frame + cfg.n_frames_track - 1
    while mapper.last_finished_frame < last_idx and not mapper.stop_event.is_set():
        time.sleep(0.5)
        print(f"Waiting... Mapper at {mapper.last_finished_frame}/{last_idx}, GS count: {len(mapper.gs_params.means)}")
    time.sleep(1.0); mapper.stop(); mapping_thread.join()
    
    print("\n>>> Phase 2: Evaluation (Frames 30-60)")
    eval_psnrs = []
    pbar_eval = tqdm(range(cfg.n_frames_track, cfg.n_frames_track + cfg.n_frames_eval), desc="Phase 2: Eval")
    for i in pbar_eval:
        idx = cfg.init_frame + i; fd = load_frame_data(data_dir, idx)
        if fd is None: break
        T_CiO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"]
        if fd["T_WO_gt"] is None: continue
        with torch.no_grad():
            T_CO_t = torch.from_numpy(T_CiO_gt).float().to(device); K_t = torch.from_numpy(fd["K"]).float().to(device)
            render_image, _, _, _ = render_2dgs(mapper.gs_params.means, F.normalize(mapper.gs_params.quats), torch.exp(mapper.gs_params.scales), mapper.gs_params.colors, torch.sigmoid(mapper.gs_params.opacity), viewmat=T_CO_t, K=K_t, width=w, height=h)
            target_image = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0; mask_t = torch.from_numpy(fd["mask"] > 0).to(device)
            if mask_t.any():
                render_image = render_image.clamp(0, 1); mse = torch.mean((render_image[:, mask_t] - target_image[:, mask_t])**2)
                psnr = -10.0 * torch.log10(mse + 1e-10); eval_psnrs.append(psnr.item()); pbar_eval.set_postfix({"PSNR": f"{psnr.item():.2f}"})
            if not cfg.no_vis:
                rr.set_time("frame_idx", sequence=idx); rr.log("input/image", rr.Image(fd["image"]))
                render_np = (render_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8); rr.log("render/image", rr.Image(render_np))
                T_OCi_gt = np.linalg.inv(T_CiO_gt); C_gt = T_OCi_gt[:3, 3]; traj_obj_gt_C.append(C_gt)
                rr.log(f"object/camera_gt", rr.Transform3D(mat3x3=T_OCi_gt[:3, :3], translation=T_OCi_gt[:3, 3]))
                rr.log(f"object/traj_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[255, 255, 255]], radii=0.003))
    if eval_psnrs: print(f"\nFinal Results: Average PSNR (Frames 30-60): {np.mean(eval_psnrs):.2f} dB")

if __name__ == "__main__":
    main(tyro.cli(Config))
