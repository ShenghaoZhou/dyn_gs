import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import poselib
import pycolmap
import pycolmap.cost_functions
import pyceres
import random
from geometric_tracker import run_ba, sample_grid_on_mask, interpolate_flow

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    window_size: int = 5
    num_samples: int = 20
    device: str = "cuda"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    grid_spacing: int = 4
    
    # Tracking Parameters
    ransac_thresh: float = 1.0
    
    # Depth source
    use_gt_depth: bool = False
    feature_type: str = "flow" # "flow" or "orb"
    n_features: int = 2000

def umeyama(src, dst):
    """Computes Sim(3) transform: dst = s * R * src + t"""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    s_centered = src - mu_s
    d_centered = dst - mu_d
    C = d_centered.T @ s_centered / len(src)
    U, S, Vh = np.linalg.svd(C)
    d = np.linalg.det(U @ Vh)
    S_mat = np.eye(3)
    if d < 0: S_mat[2, 2] = -1
    R_mat = U @ S_mat @ Vh
    var_s = np.var(src, axis=0).sum()
    s = np.trace(np.diag(S) @ S_mat) / (var_s + 1e-8)
    t = mu_d - s * R_mat @ mu_s
    return s, R_mat, t

def load_object_pose_world(data_root, frame_idx):
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists(): return None
    with open(poses_file, "r") as f:
        lines = f.readlines()
    if frame_idx >= len(lines): return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])])
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4)
    T_WO[:3, :3] = r_WO
    T_WO[:3, 3] = t_WO
    return T_WO

def triangulate_linear(P1, P2, pts1, pts2):
    """Linear triangulation for points in 3D."""
    pts3d = []
    for i in range(len(pts1)):
        A = np.zeros((4, 4))
        A[0] = pts1[i, 0] * P1[2, :] - P1[0, :]
        A[1] = pts1[i, 1] * P1[2, :] - P1[1, :]
        A[2] = pts2[i, 0] * P2[2, :] - P2[0, :]
        A[3] = pts2[i, 1] * P2[2, :] - P2[1, :]
        _, _, vh = np.linalg.svd(A)
        p3d = vh[-1, :3] / vh[-1, 3]
        pts3d.append(p3d)
    return np.array(pts3d)

def compute_ate(traj_est, traj_gt):
    """Compute ATE after Sim(3) alignment."""
    s, R_align, t_align = umeyama(traj_est, traj_gt)
    traj_aligned = s * (traj_est @ R_align.T) + t_align
    ate = np.sqrt(np.mean(np.linalg.norm(traj_aligned - traj_gt, axis=1)**2))
    return ate

def get_cam_dict(K, W, H):
    return {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}

def load_frame(data_dir, idx, use_gt_depth):
    stem = f"{idx:06d}"
    img_path = data_dir / "images" / f"{stem}.png"
    if not img_path.exists(): return None
    img = np.array(cv2.imread(str(img_path))[..., ::-1])
    mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
    
    if use_gt_depth:
        depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
    else:
        depth_path = data_dir / "model_infer" / f"depth_{idx:05d}.npy"
        if not depth_path.exists(): depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
    
    if not depth_path.exists(): return None
    depth = np.load(depth_path)
    if depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
    T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
    T_WO_gt = load_object_pose_world(data_dir, idx)
    return img, mask, depth, K, T_CW_gt, T_WO_gt

def compute_scale_ratio(tracks, frames, data_dir):
    """Compute median GT/Est scale ratio using Frame 0 as anchor."""
    f0 = frames[0]
    stem = f"{f0['frame_idx']:06d}"
    gt_depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
    if not gt_depth_path.exists(): return np.nan
    
    depth_gt = np.load(gt_depth_path)
    if depth_gt.shape != f0["image"].shape[:2]:
        depth_gt = cv2.resize(depth_gt, (f0["image"].shape[1], f0["image"].shape[0]), interpolation=cv2.INTER_NEAREST)
    
    ratios = []
    K0_inv = np.linalg.inv(f0["K"])
    for tid, t in tracks.items():
        if f0["frame_idx"] in t["obs"] and "pt3d" in t:
            uv = t["obs"][f0["frame_idx"]]
            d_gt = depth_gt[int(round(uv[1])), int(round(uv[0]))]
            if d_gt > 0.01:
                dist_est = np.linalg.norm(t["pt3d"])
                dist_gt = np.linalg.norm((K0_inv @ np.array([uv[0], uv[1], 1.0])) * d_gt)
                if dist_est > 1e-6:
                    ratios.append(dist_gt / dist_est)
    return np.median(ratios) if ratios else np.nan

def run_experiment(cfg: Config):
    data_dir = Path(cfg.data_root)
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    
    # Get total frames
    img_files = sorted(list((data_dir / "images").glob("*.png")))
    total_frames = len(img_files)
    
    # Randomly sample 20 starting frames
    valid_starts = list(range(0, total_frames - cfg.window_size))
    if len(valid_starts) < cfg.num_samples:
        sampled_starts = valid_starts
    else:
        sampled_starts = random.sample(valid_starts, cfg.num_samples)
    
    results_a, scales_a = [], []
    results_b, scales_b = [], []
    results_c, scales_c = [], []
    init_depth_ratios = []
    
    pbar = tqdm(sampled_starts, desc="Running Benchmark")
    for start_idx in pbar:
        frames = []
        valid_window = True
        for i in range(cfg.window_size):
            fd = load_frame(data_dir, start_idx + i, cfg.use_gt_depth)
            if fd is None:
                valid_window = False
                break
            frames.append({"image": fd[0], "mask": fd[1], "depth": fd[2], "K": fd[3], 
                           "T_CW_gt": fd[4], "T_WO_gt": fd[5], "frame_idx": start_idx + i})
        
        if not valid_window: continue
        H, W = frames[0]["image"].shape[:2]
        
        # --- Initial Depth Scale Ratio (Averaged on sampled window) ---
        window_init_ratios = []
        for f in frames:
            stem = f"{f['frame_idx']:06d}"
            gt_depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
            if gt_depth_path.exists():
                depth_gt = np.load(gt_depth_path)
                if depth_gt.shape != f["image"].shape[:2]:
                    depth_gt = cv2.resize(depth_gt, (W, H), interpolation=cv2.INTER_NEAREST)
                mask = f["mask"] > 0
                valid = (depth_gt > 0.01) & (f["depth"] > 0.01) & mask
                if np.any(valid):
                    window_init_ratios.append(np.median(depth_gt[valid] / f["depth"][valid]))
        if window_init_ratios:
            init_depth_ratios.append(np.mean(window_init_ratios))

        # Ground Truth object-centric trajectory
        traj_gt_obj = []
        for f in frames:
            T_CO_gt = f["T_CW_gt"] @ f["T_WO_gt"]
            T_OC_gt = np.linalg.inv(T_CO_gt)
            traj_gt_obj.append(T_OC_gt[:3, 3])
        traj_gt_obj = np.array(traj_gt_obj)
        
        # Tracking / Correspondence finding
        if cfg.feature_type == "orb":
            orb = cv2.ORB_create(nfeatures=cfg.n_features)
            gray0 = cv2.cvtColor(frames[0]["image"], cv2.COLOR_RGB2GRAY)
            kpts = orb.detect(gray0, mask=frames[0]["mask"])
            current_pts = np.array([kp.pt for kp in kpts], dtype=np.float32)
        else:
            current_pts = sample_grid_on_mask(frames[0]["mask"], cfg.grid_spacing)

        tracks = {i: {"obs": {frames[0]["frame_idx"]: p}, "id": i} for i, p in enumerate(current_pts)}
        for i in range(len(frames) - 1):
            f_prev, f_curr = frames[i], frames[i+1]
            flow = dis.calc(cv2.cvtColor(f_prev["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
            valid_tids = [tid for tid in tracks if f_prev["frame_idx"] in tracks[tid]["obs"]]
            pts_prev = np.array([tracks[tid]["obs"][f_prev["frame_idx"]] for tid in valid_tids])
            pts_next = pts_prev + interpolate_flow(flow, pts_prev)
            ix, iy = np.round(pts_next[:, 0]).astype(int), np.round(pts_next[:, 1]).astype(int)
            valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            for j, tid in enumerate(valid_tids):
                if valid[j] and f_curr["mask"][iy[j], ix[j]] > 0:
                    tracks[tid]["obs"][f_curr["frame_idx"]] = pts_next[j]

        full_tracks = {tid: t for tid, t in tracks.items() if len(t["obs"]) == cfg.window_size}
        if len(full_tracks) < 10: continue
            
        # --- Path A: PnP Initialization ---
        poses_a = {f["frame_idx"]: np.eye(4) for f in frames}
        tracks_a = {tid: {"obs": t["obs"].copy(), "id": tid} for tid, t in full_tracks.items()}
        f0 = frames[0]
        K0_inv = np.linalg.inv(f0["K"])
        for tid, t in tracks_a.items():
            uv = t["obs"][f0["frame_idx"]]
            d = f0["depth"][int(round(uv[1])), int(round(uv[0]))]
            t["pt3d"] = (K0_inv @ np.array([uv[0], uv[1], 1.0])) * d
            
        for i in range(1, cfg.window_size):
            f = frames[i]
            pts2d = np.array([t["obs"][f["frame_idx"]] for t in tracks_a.values()])
            pts3d = np.array([t["pt3d"] for t in tracks_a.values()])
            res, _ = poselib.estimate_absolute_pose(pts2d, pts3d, get_cam_dict(f["K"], W, H), {"max_reproj_error": cfg.ransac_thresh}, None)
            if res is not None:
                T = np.eye(4); T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix(); T[:3, 3] = res.pose.t
                poses_a[f["frame_idx"]] = T
            else:
                poses_a[f["frame_idx"]] = poses_a[frames[i-1]["frame_idx"]]
        run_ba([f["frame_idx"] for f in frames], poses_a, tracks_a, {f["frame_idx"]: f["K"] for f in frames})
        
        traj_a = []
        T_OC0_gt = np.linalg.inv(frames[0]["T_CW_gt"] @ frames[0]["T_WO_gt"])
        for f in frames:
            T_CiC0_est = poses_a[f["frame_idx"]]; T_OCi_est = T_OC0_gt @ np.linalg.inv(T_CiC0_est); traj_a.append(T_OCi_est[:3, 3])
        results_a.append(compute_ate(np.array(traj_a), traj_gt_obj))
        scales_a.append(compute_scale_ratio(tracks_a, frames, data_dir))

        # --- Path B: MonoDepth Relative Initialization ---
        poses_b = {f["frame_idx"]: np.eye(4) for f in frames}
        tracks_b = {tid: {"obs": t["obs"].copy(), "id": tid} for tid, t in full_tracks.items()}
        f0, f4 = frames[0], frames[4]
        pts0 = np.array([t["obs"][f0["frame_idx"]] for t in tracks_b.values()])
        pts4 = np.array([t["obs"][f4["frame_idx"]] for t in tracks_b.values()])
        d0, d4 = np.array([f0["depth"][int(round(p[1])), int(round(p[0]))] for p in pts0]), np.array([f4["depth"][int(round(p[1])), int(round(p[0]))] for p in pts4])
        res_b, _ = poselib.estimate_monodepth_relative_pose(pts0, pts4, d0, d4, get_cam_dict(f0["K"], W, H), get_cam_dict(f4["K"], W, H), {"max_reproj_error": cfg.ransac_thresh})
        if res_b is not None:
            T40 = np.eye(4); T40[:3, :3] = R.from_quat([res_b.pose.q[1], res_b.pose.q[2], res_b.pose.q[3], res_b.pose.q[0]]).as_matrix(); T40[:3, 3] = res_b.pose.t
            poses_b[f4["frame_idx"]] = T40
            pts3d_b = triangulate_linear(f0["K"] @ np.eye(3, 4), f4["K"] @ T40[:3, :], pts0, pts4)
            for i, tid in enumerate(tracks_b.keys()): tracks_b[tid]["pt3d"] = pts3d_b[i]
            for i in range(1, 4):
                f = frames[i]; pts2d = np.array([t["obs"][f["frame_idx"]] for t in tracks_b.values()]); pts3d_in = np.array([t["pt3d"] for t in tracks_b.values()])
                res_abs, _ = poselib.estimate_absolute_pose(pts2d, pts3d_in, get_cam_dict(f["K"], W, H), {"max_reproj_error": cfg.ransac_thresh}, None)
                if res_abs is not None:
                    T = np.eye(4); T[:3, :3] = R.from_quat([res_abs.pose.q[1], res_abs.pose.q[2], res_abs.pose.q[3], res_abs.pose.q[0]]).as_matrix(); T[:3, 3] = res_abs.pose.t; poses_b[f["frame_idx"]] = T
                else: poses_b[f["frame_idx"]] = poses_b[frames[i-1]["frame_idx"]]
            run_ba([f["frame_idx"] for f in frames], poses_b, tracks_b, {f["frame_idx"]: f["K"] for f in frames})
            traj_b = []
            for f in frames:
                T_CiC0_est = poses_b[f["frame_idx"]]; T_OCi_est = T_OC0_gt @ np.linalg.inv(T_CiC0_est); traj_b.append(T_OCi_est[:3, 3])
            results_b.append(compute_ate(np.array(traj_b), traj_gt_obj))
            scales_b.append(compute_scale_ratio(tracks_b, frames, data_dir))
        else: results_b.append(np.nan); scales_b.append(np.nan)

        # --- Path C: Multi-MonoDepth Relative Initialization ---
        poses_c = {f["frame_idx"]: np.eye(4) for f in frames}
        tracks_c = {tid: {"obs": t["obs"].copy(), "id": tid} for tid, t in full_tracks.items()}
        f0 = frames[0]; pts0 = np.array([t["obs"][f0["frame_idx"]] for t in tracks_c.values()]); d0 = np.array([f0["depth"][int(round(p[1])), int(round(p[0]))] for p in pts0])
        valid_c = True
        for i in range(1, cfg.window_size):
            fi = frames[i]; ptsi = np.array([t["obs"][fi["frame_idx"]] for t in tracks_c.values()]); di = np.array([fi["depth"][int(round(p[1])), int(round(p[0]))] for p in ptsi])
            res_c, _ = poselib.estimate_monodepth_relative_pose(pts0, ptsi, d0, di, get_cam_dict(f0["K"], W, H), get_cam_dict(fi["K"], W, H), {"max_reproj_error": cfg.ransac_thresh})
            if res_c is not None:
                Ti0 = np.eye(4); Ti0[:3, :3] = R.from_quat([res_c.pose.q[1], res_c.pose.q[2], res_c.pose.q[3], res_c.pose.q[0]]).as_matrix(); Ti0[:3, 3] = res_c.pose.t; poses_c[fi["frame_idx"]] = Ti0
            else: valid_c = False; break
        if valid_c:
            T40 = poses_c[frames[4]["frame_idx"]]; pts4 = np.array([t["obs"][frames[4]["frame_idx"]] for t in tracks_c.values()])
            pts3d_c = triangulate_linear(f0["K"] @ np.eye(3, 4), frames[4]["K"] @ T40[:3, :], pts0, pts4)
            for i, tid in enumerate(tracks_c.keys()): tracks_c[tid]["pt3d"] = pts3d_c[i]
            run_ba([f["frame_idx"] for f in frames], poses_c, tracks_c, {f["frame_idx"]: f["K"] for f in frames})
            traj_c = []
            for f in frames:
                T_CiC0_est = poses_c[f["frame_idx"]]; T_OCi_est = T_OC0_gt @ np.linalg.inv(T_CiC0_est); traj_c.append(T_OCi_est[:3, 3])
            results_c.append(compute_ate(np.array(traj_c), traj_gt_obj))
            scales_c.append(compute_scale_ratio(tracks_c, frames, data_dir))
        else: results_c.append(np.nan); scales_c.append(np.nan)

        pbar.set_postfix({"A": f"{np.nanmean(results_a):.4f}", "B": f"{np.nanmean(results_b):.4f}", "C": f"{np.nanmean(results_c):.4f}"})

    print(f"\n--- Initial Condition (Reference) ---")
    print(f"Avg Initial Depth Scale Ratio (GT/Inferred): {np.nanmean(init_depth_ratios):.6f}")

    print("\n--- Final Results (ATE & Scale Ratio) ---")
    results_a, results_b, results_c = np.array(results_a), np.array(results_b), np.array(results_c)
    scales_a, scales_b, scales_c = np.array(scales_a), np.array(scales_b), np.array(scales_c)
    
    print(f"Path A (PnP):        ATE: {np.nanmean(results_a):.6f} ± {np.nanstd(results_a):.6f} | Scale Ratio: {np.nanmean(scales_a):.6f}")
    print(f"Path B (Mono):       ATE: {np.nanmean(results_b):.6f} ± {np.nanstd(results_b):.6f} | Scale Ratio: {np.nanmean(scales_b):.6f}")
    print(f"Path C (Multi-Mono): ATE: {np.nanmean(results_c):.6f} ± {np.nanstd(results_c):.6f} | Scale Ratio: {np.nanmean(scales_c):.6f}")
    
    means = [np.nanmean(results_a), np.nanmean(results_b), np.nanmean(results_c)]
    better = ["Path A", "Path B", "Path C"][np.nanargmin(means)]
    print(f"\nConclusion: {better} is best on average.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    run_experiment(cfg)
