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
    grid_spacing: int = 8   # Standard grid spacing
    ransac_thresh: float = 1.0
    max_keyframes: int = 20
    kf_every: int = 5

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

def init_gs_from_tracker_points(points, colors, device, normals=None, ray_o=None, ray_d=None, ray_dist=None):
    num_pts = points.shape[0]
    means = torch.from_numpy(points).float().to(device); means.requires_grad = True
    colors_sh = RGB2SH(torch.from_numpy(colors).float().to(device)); colors_sh.requires_grad = True
    if normals is not None:
        quats = build_rotation_from_normal(torch.from_numpy(normals).float().to(device))
    else:
        quats = torch.zeros((num_pts, 4), device=device); quats[:, 0] = 1.0
    quats.requires_grad = True
    scales = torch.log(torch.ones((num_pts, 2), device=device) * 0.01); scales.requires_grad = True # Larger initial scales for sparse pts
    opacity = torch.logit(torch.ones((num_pts, 1), device=device) * 0.7); opacity.requires_grad = True # Higher initial opacity
    
    if ray_o is not None: ray_o = ray_o.to(device)
    if ray_d is not None: ray_d = ray_d.to(device)
    if ray_dist is not None: 
        ray_dist = ray_dist.to(device)
        ray_dist.requires_grad = True
        
    return GSParam(means, quats, scales, colors_sh, opacity, ray_o, ray_d, ray_dist)

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("test_gs_mapping_no_cheat_robust", spawn=False)
        if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
        rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    data_dir = Path(cfg.data_root) / cfg.clip_id
    device = torch.device(cfg.device)
    
    # Mapping Config
    map_cfg = MappingConfig(
        device=cfg.device,
        num_steps_per_frame=100, 
        pyr_levels=2,
        pyr_interval=15,
        fix_color_and_scales=True,
        use_ray_dist=True, 
        use_pgsr=True, 
        densify_every=5, 
        kf_every=5,
        lr_means=1e-3,
        lr_opacity=0.1 
    )
    
    # Tracker Setup (PathProcessor Style)
    from geometric_tracker import run_ba, sample_grid_on_mask, interpolate_flow
    
    poses = {} # idx -> T_CiC0
    tracks = {} # tid -> {'obs': {idx: uv}, 'pt3d': xyz_C0}
    next_tid = 0
    K_dict = {}
    keyframes = []
    
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    print("\n>>> Phase 0: Robust Initialization (Frames 0-5)")
    frames_buffer = []
    for k in range(cfg.n_init_frames):
        fd = load_frame_data(data_dir, cfg.init_frame + k)
        if fd is None: break
        frames_buffer.append(fd)
        K_dict[fd["frame_idx"]] = fd["K"]
    
    f0 = frames_buffer[0]
    h, w = f0["image"].shape[:2]
    T_C0O_gt = f0["T_CW_gt"] @ f0["T_WO_gt"] if f0["T_WO_gt"] is not None else np.eye(4)
    
    # Seed from frame 0
    poses[f0["frame_idx"]] = np.eye(4)
    keyframes.append(f0["frame_idx"])
    
    # Dense seed for GS
    def add_pts(f, T_CiC0, spacing):
        nonlocal next_tid
        pts = sample_grid_on_mask(f["mask"], spacing)
        Ki_inv = np.linalg.inv(f["K"])
        T_C0Ci = np.linalg.inv(T_CiC0)
        for uv in pts:
            d = f["depth"][int(round(uv[1])), int(round(uv[0]))]
            if d <= 0.01: continue
            pt_Ci = (Ki_inv @ np.array([uv[0], uv[1], 1.0])) * d
            pt_C0 = (T_C0Ci[:3, :3] @ pt_Ci) + T_C0Ci[:3, 3]
            tracks[next_tid] = {'obs': {f["frame_idx"]: uv}, 'pt3d': pt_C0}
            next_tid += 1

    add_pts(f0, poses[f0["frame_idx"]], 4) # Dense init
        
    # Track through init window
    for k in range(1, len(frames_buffer)):
        f_prev, f_curr = frames_buffer[k-1], frames_buffer[k]
        idx = f_curr["frame_idx"]
        flow = dis.calc(cv2.cvtColor(f_prev["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
        
        # Propagate tracks
        prev_idx = f_prev["frame_idx"]
        for tid, t in tracks.items():
            if prev_idx in t['obs']:
                uv_prev = t['obs'][prev_idx]
                delta = interpolate_flow(flow, uv_prev[None])[0]
                uv_curr = uv_prev + delta
                ix, iy = int(round(uv_curr[0])), int(round(uv_curr[1]))
                if 0 <= ix < f_curr["image"].shape[1] and 0 <= iy < f_curr["image"].shape[0] and f_curr["mask"][iy, ix] > 0:
                    t['obs'][idx] = uv_curr
        
        # PnP for current frame
        obs2d, obs3d = [], []
        for t in tracks.values():
            if idx in t['obs']:
                obs2d.append(t['obs'][idx]); obs3d.append(t['pt3d'])
        
        if len(obs2d) >= 10:
            cam_dict = {'model': 'PINHOLE', 'width': f_curr["image"].shape[1], 'height': f_curr["image"].shape[0], 'params': [f_curr["K"][0,0], f_curr["K"][1,1], f_curr["K"][0,2], f_curr["K"][1,2]]}
            res, info = poselib.estimate_absolute_pose(np.array(obs2d), np.array(obs3d), cam_dict, {'max_reproj_error': cfg.ransac_thresh}, None)
            if res:
                T = np.eye(4); T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix(); T[:3, 3] = res.pose.t
                poses[idx] = T
            else:
                poses[idx] = poses[idx-1].copy()
        else:
            poses[idx] = poses[idx-1].copy()
            
        # Add more points in init window to ensure density
        add_pts(f_curr, poses[idx], 8)
            
    # Initial BA
    run_ba(list(poses.keys()), poses, tracks, K_dict)
    
    print("\n>>> Phase 1: Tracking + Mapping (Frames 5-30)")
    
    # Initialize GS
    last_init_f = frames_buffer[-1]
    T_CLastO = poses[last_init_f["frame_idx"]] @ T_C0O_gt
    
    active_points, active_colors, active_normals = [], [], []
    depth_t = torch.from_numpy(last_init_f["depth"]).float().to(device)
    full_pts_c = unproject_depth(depth_t, torch.from_numpy(last_init_f["K"]).float().to(device), last_init_f["image"].shape[0], last_init_f["image"].shape[1])
    normals_c, _ = d2n_tblr(full_pts_c.permute(2, 0, 1).unsqueeze(0))
    normals_c = -F.normalize(normals_c[0], dim=0).cpu().numpy()
    T_OC_init = np.linalg.inv(T_CLastO)
    
    # Use keyframe tracks for GS init
    for tid, t in tracks.items():
        if last_init_f["frame_idx"] in t['obs']:
            p_c0 = t['pt3d']
            p_o = (np.linalg.inv(T_C0O_gt)[:3, :3] @ p_c0) + np.linalg.inv(T_C0O_gt)[:3, 3]
            uv = t['obs'][last_init_f["frame_idx"]]
            ix, iy = int(round(uv[0])), int(round(uv[1]))
            active_points.append(p_o)
            active_colors.append(last_init_f["image"][min(iy, last_init_f["image"].shape[0]-1), min(ix, last_init_f["image"].shape[1]-1)] / 255.0)
            n_c = normals_c[:, min(iy, last_init_f["image"].shape[0]-1), min(ix, last_init_f["image"].shape[1]-1)]
            n_o = (n_c @ T_OC_init[:3, :3].T)
            active_normals.append(n_o)
    
    gs_params = init_gs_from_tracker_points(np.array(active_points), np.array(active_colors), device, normals=np.array(active_normals))
    print(f"Initialized GS with {len(active_points)} points.")
    mapper = GSMapping(map_cfg, gs_params)
    mapping_thread = threading.Thread(target=mapper.run); mapping_thread.daemon = True; mapping_thread.start()
    
    # Tracking state
    traj_obj_est_C, traj_obj_gt_C = [], []
    T_CiO_gt0 = T_C0O_gt
    errors_t = []
    
    prev_f = frames_buffer[-1]
    
    pbar = tqdm(range(cfg.n_init_frames, cfg.n_frames_track), desc="Phase 1: Track+Map")
    for i in pbar:
        idx = cfg.init_frame + i
        fd = load_frame_data(data_dir, idx)
        if fd is None: break
        K_dict[idx] = fd["K"]
        h, w = fd["image"].shape[:2]
        depth = fd["depth"]
        
        flow = dis.calc(cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY), None)
        
        # Propagate tracks
        prev_idx = prev_f["frame_idx"]
        for tid, t in tracks.items():
            if prev_idx in t['obs']:
                uv_prev = t['obs'][prev_idx]
                delta = interpolate_flow(flow, uv_prev[None])[0]
                uv_curr = uv_prev + delta
                ix, iy = int(round(uv_curr[0])), int(round(uv_curr[1]))
                if 0 <= ix < fd["image"].shape[1] and 0 <= iy < fd["image"].shape[0] and fd["mask"][iy, ix] > 0:
                    t['obs'][idx] = uv_curr
        
        # PnP
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
            
        # Keyframe Logic
        if i % cfg.kf_every == 0:
            keyframes.append(idx)
            # Re-seed from depth
            z_est, z_raw = [], []
            for tid, t in tracks.items():
                if idx in t['obs']:
                    uv = t['obs'][idx]
                    pt_Ci = (poses[idx][:3, :3] @ t['pt3d']) + poses[idx][:3, 3]
                    z_est.append(pt_Ci[2])
                    z_raw.append(fd["depth"][int(round(uv[1])), int(round(uv[0]))])
            z_est, z_raw = np.array(z_est), np.array(z_raw)
            valid = (z_raw > 0.01) & (z_est > 0.01)
            depth = fd["depth"]
            if np.sum(valid) > 10:
                A = np.stack([z_raw[valid], np.ones_like(z_raw[valid])], axis=1)
                res_scale = np.linalg.lstsq(A, z_est[valid], rcond=None)[0]
                depth = res_scale[0] * depth + res_scale[1]
                
            K_inv = np.linalg.inv(fd["K"])
            T_C0Ci = np.linalg.inv(poses[idx])
            new_pts2d = sample_grid_on_mask(fd["mask"], cfg.grid_spacing)
            for uv in new_pts2d:
                d = depth[int(round(uv[1])), int(round(uv[0]))]
                if d <= 0.01: continue
                pt_Ci = (K_inv @ np.array([uv[0], uv[1], 1.0])) * d
                pt_C0 = (T_C0Ci[:3, :3] @ pt_Ci) + T_C0Ci[:3, 3]
                tracks[next_tid] = {'obs': {idx: uv}, 'pt3d': pt_C0}
                next_tid += 1
            
            run_ba(keyframes[-cfg.max_keyframes:], poses, tracks, K_dict)

        # Final Pose for Mapper and Eval
        T_CiC0 = poses[idx]
        T_CiO = T_CiC0 @ T_C0O_gt
        
        T_CiO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"]
        T_OCi_est = np.linalg.inv(T_CiO)
        C_est = T_OCi_est[:3, 3]
        T_OCi_gt = np.linalg.inv(T_CiO_gt)
        C_gt = T_OCi_gt[:3, 3]
        
        err_ate = np.linalg.norm(C_est - C_gt)
        errors_t.append(err_ate)
        pbar.set_postfix({"ATE": f"{err_ate:.4f}m", "GS": len(mapper.gs_params.means)})
        
        # Use scaled depth for mapper to stay synced with tracker's cloud growth
        mapper.update({
            "image": fd["image"], "mask": fd["mask"], "depth": depth,
            "K": fd["K"], "T_CiO": T_CiO, "frame_idx": idx
        })

        if not cfg.no_vis:
            rr.set_time("frame_idx", sequence=idx)
            rr.log("input/image", rr.Image(fd["image"]))
            
            # --- Object Frame Consistency Check ---
            path_tag = "tracker"
            color = [0, 255, 0]
            
            traj_obj_est_C.append(C_est)
            rr.log(f"object/{path_tag}/camera_est", rr.Transform3D(mat3x3=T_OCi_est[:3, :3], translation=T_OCi_est[:3, 3]))
            rr.log(f"object/{path_tag}/traj_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[color], radii=0.003))
            
            traj_obj_gt_C.append(C_gt)
            rr.log(f"object/camera_gt", rr.Transform3D(mat3x3=T_OCi_gt[:3, :3], translation=T_OCi_gt[:3, 3]))
            rr.log(f"object/traj_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[255, 255, 255]], radii=0.003))

            active_tids = [tid for tid, t in tracks.items() if idx in t['obs']]
            if active_tids:
                pts_C0 = np.array([tracks[tid]['pt3d'] for tid in active_tids])
                pts_O = (np.linalg.inv(T_C0O_gt)[:3, :3] @ pts_C0.T).T + np.linalg.inv(T_C0O_gt)[:3, 3]
                rr.log("object/tracker_points", rr.Points3D(pts_O, colors=[[0, 255, 0]], radii=0.005))
            
            with mapper.lock:
                means_np = mapper.gs_params.means.detach().cpu().numpy()
                colors_sh = mapper.gs_params.colors.detach().cpu().numpy()
                C0 = 0.28209479177387814
                colors_rgb = np.clip(colors_sh * C0 + 0.5, 0, 1)
                rr.log("object/gs_points", rr.Points3D(means_np, colors=colors_rgb, radii=0.004))

            # 2. GS Rendering during Phase 1
            with mapper.lock:
                render_image, _, _, _ = render_2dgs(
                    mapper.gs_params.means, F.normalize(mapper.gs_params.quats), 
                    torch.exp(mapper.gs_params.scales), mapper.gs_params.colors, 
                    torch.sigmoid(mapper.gs_params.opacity),
                    viewmat=torch.from_numpy(T_CiO).float().to(device), 
                    K=torch.from_numpy(fd["K"]).float().to(device), width=w, height=h
                )
                render_np = (render_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                rr.log("render/image", rr.Image(render_np))
        
        prev_f = fd
        
        # Throttle if mapper is falling too far behind (more than 5 frames)
        while mapper.data_queue.full() and not mapper.stop_event.is_set():
            time.sleep(0.1)

    # Finalize Phase 1
    print("Finishing Phase 1 mapping... Waiting for mapper to catch up.")
    last_idx = cfg.init_frame + cfg.n_frames_track - 1
    while mapper.last_finished_frame < last_idx and not mapper.stop_event.is_set():
        time.sleep(0.5)
        print(f"Waiting... Mapper at {mapper.last_finished_frame}/{last_idx}, GS count: {len(mapper.gs_params.means)}")
    
    time.sleep(1.0) # Final polish
    mapper.stop()
    mapping_thread.join()
    
    # Phase 2: Evaluation with GT Poses (next 30 frames)
    print("\n>>> Phase 2: Evaluation (Frames 30-60)")
    eval_psnrs = []
    
    pbar_eval = tqdm(range(cfg.n_frames_track, cfg.n_frames_track + cfg.n_frames_eval), desc="Phase 2: Eval")
    for i in pbar_eval:
        idx = cfg.init_frame + i
        fd = load_frame_data(data_dir, idx)
        if fd is None: break
        
        # Use GT Pose for Evaluation
        T_CW_gt = fd["T_CW_gt"]
        T_WO_gt = fd["T_WO_gt"]
        if T_WO_gt is None: continue
        T_CiO_gt = T_CW_gt @ T_WO_gt
        
        with torch.no_grad():
            T_CO_t = torch.from_numpy(T_CiO_gt).float().to(device)
            K_t = torch.from_numpy(fd["K"]).float().to(device)
            
            render_image, _, _, _ = render_2dgs(
                mapper.gs_params.means, F.normalize(mapper.gs_params.quats), 
                torch.exp(mapper.gs_params.scales), mapper.gs_params.colors, 
                torch.sigmoid(mapper.gs_params.opacity),
                viewmat=T_CO_t, K=K_t, width=w, height=h
            )
            
            target_image = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0
            mask_t = torch.from_numpy(fd["mask"] > 0).to(device)
            
            if mask_t.any():
                render_image = render_image.clamp(0, 1)
                mse = torch.mean((render_image[:, mask_t] - target_image[:, mask_t])**2)
                psnr = -10.0 * torch.log10(mse + 1e-10)
                eval_psnrs.append(psnr.item())
                pbar_eval.set_postfix({"PSNR": f"{psnr.item():.2f}"})
            
            if not cfg.no_vis:
                rr.set_time("frame_idx", sequence=idx)
                rr.log("input/image", rr.Image(fd["image"]))
                render_np = (render_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rr.log("render/image", rr.Image(render_np))
                
                # Update trajectories in Phase 2
                # Use GT for evaluation, but still show the anchored comparison if possible
                # In Phase 2, we just show GT trajectory as we are evaluating rendering
                T_OCi_gt = np.linalg.inv(T_CiO_gt)
                C_gt = T_OCi_gt[:3, 3]
                traj_obj_gt_C.append(C_gt)
                rr.log(f"object/camera_gt", rr.Transform3D(mat3x3=T_OCi_gt[:3, :3], translation=T_OCi_gt[:3, 3]))
                rr.log(f"object/traj_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[255, 255, 255]], radii=0.003))
                
    if eval_psnrs:
        avg_psnr = np.mean(eval_psnrs)
        print(f"\nFinal Results:")
        print(f"Average PSNR (Frames 30-60): {avg_psnr:.2f} dB")
        if avg_psnr < 10:
            print("WARNING: PSNR is below 10 dB. Major bottleneck likely tracking drift or insufficient densification.")
        elif avg_psnr < 20:
            print("INFO: PSNR is between 10 and 20 dB. Reasonable but could be improved.")
        else:
            print("SUCCESS: PSNR is > 20 dB!")
    else:
        print("No evaluation frames processed.")

if __name__ == "__main__":
    main(tyro.cli(Config))
