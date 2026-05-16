import numpy as np
import cv2
import rerun as rr
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
from gs_dyn_obj.utils.init import unproject_depth

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 0
    window_size: int = 5
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    grid_spacing: int = 4
    
    # Tracking Parameters
    ransac_thresh: float = 1.0
    
    # Comparison Flag
    compare_gt: bool = True
    
    # Depth source
    use_gt_depth: bool = False

    # Visualization
    point_radii: float = 0.01
    traj_radii: float = 0.002

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

def interpolate_flow(flow, pts):
    x, y = pts[:, 0], pts[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    h, w = flow.shape[:2]
    x0, x1 = np.clip(x0, 0, w-1), np.clip(x1, 0, w-1)
    y0, y1 = np.clip(y0, 0, h-1), np.clip(y1, 0, h-1)
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    return (wa[:, None] * flow[y0, x0] + wb[:, None] * flow[y1, x0] + 
            wc[:, None] * flow[y0, x1] + wd[:, None] * flow[y1, x1])

def sample_grid_on_mask(mask, spacing):
    h, w = mask.shape
    yy, xx = np.mgrid[spacing//2:h:spacing, spacing//2:w:spacing]
    pts = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
    ix, iy = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy, pts = ix[valid], iy[valid], pts[valid]
    return pts[mask[iy, ix] > 0]

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

def run_window_ba(frame_indices, poses, tracks, K_dict, fix_first=True):
    if len(frame_indices) < 2: return
    prob = pyceres.Problem()
    loss = pyceres.HuberLoss(1.0)
    pose_params = {}
    for idx in frame_indices:
        T = poses[idx]
        q = R.from_matrix(T[:3, :3]).as_quat()
        q_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
        t = T[:3, 3].copy().astype(np.float64)
        pose_params[idx] = (q_wxyz, t)
    
    track_params = {}
    relevant_tracks = []
    for tid, track in tracks.items():
        win_obs = {f_idx: uv for f_idx, uv in track['obs'].items() if f_idx in frame_indices}
        if len(win_obs) >= 2:
            track_params[tid] = track['pt3d'].copy().astype(np.float64)
            relevant_tracks.append((tid, win_obs))
    
    if not relevant_tracks: return
    K0 = K_dict[frame_indices[0]]
    cam_params = np.array([K0[0,0], K0[1,1], K0[0,2], K0[1,2]], dtype=np.float64)
    
    for tid, win_obs in relevant_tracks:
        pt3d = track_params[tid]
        for f_idx, uv in win_obs.items():
            q_wxyz, t = pose_params[f_idx]
            cost = pycolmap.cost_functions.ReprojErrorCost('PINHOLE', uv.astype(np.float64))
            prob.add_residual_block(cost, loss, [q_wxyz, t, pt3d, cam_params])
    
    prob.set_parameter_block_constant(cam_params)
    if fix_first:
        ref_idx = frame_indices[0]
        prob.set_parameter_block_constant(pose_params[ref_idx][0])
        prob.set_parameter_block_constant(pose_params[ref_idx][1])
    
    quat_manifold = pyceres.EigenQuaternionManifold()
    for idx in frame_indices:
        q_wxyz, t = pose_params[idx]
        if not prob.is_parameter_block_constant(q_wxyz):
            prob.set_manifold(q_wxyz, quat_manifold)
            
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
    options.max_num_iterations = 50
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    
    for idx, (q_wxyz, t) in pose_params.items():
        T = np.eye(4)
        T[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
        T[:3, 3] = t
        poses[idx] = T
    for tid, _ in relevant_tracks:
        tracks[tid]['pt3d'] = track_params[tid]

def align_and_evaluate(stage_name, frames, poses_est, tracks_est, compare_gt, point_radii=0.01, traj_radii=0.002):
    T_WO0 = frames[0]["T_WO_gt"]
    T_WC0 = np.linalg.inv(frames[0]["T_CW_gt"])
    T_OC0_gt = np.linalg.inv(T_WO0) @ T_WC0
    
    traj_est_obj = []
    traj_est_world = []
    for f in frames:
        idx = f["frame_idx"]
        T_CiC0_est = poses_est[idx]
        T_C0Ci_est = np.linalg.inv(T_CiC0_est)
        T_OCi_est = T_OC0_gt @ T_C0Ci_est
        T_WCi_est = T_WO0 @ T_OCi_est # Trajectory in Frame 0's object-centric world
        traj_est_obj.append(T_OCi_est[:3, 3])
        traj_est_world.append(T_WCi_est[:3, 3])

    # Ground Truth object-centric trajectory: T_OC = inv(T_CW @ T_WO)
    traj_gt_obj = []
    for f in frames:
        T_CO_gt = f["T_CW_gt"] @ f["T_WO_gt"]
        T_OC_gt = np.linalg.inv(T_CO_gt)
        traj_gt_obj.append(T_OC_gt[:3, 3])
    
    # Trajectory in Frame 0's world: T_WC_viz = T_WO0 @ T_OC
    traj_gt_world = [(T_WO0[:3, :3] @ p.T).T + T_WO0[:3, 3] for p in traj_gt_obj]
    
    s, R_align, t_align = 1.0, np.eye(3), np.zeros(3)
    if compare_gt:
        s, R_align, t_align = umeyama(np.array(traj_est_obj), np.array(traj_gt_obj))
    
    # Apply alignment to object-centric trajectory
    traj_obj_aligned = s * (np.array(traj_est_obj) @ R_align.T) + t_align
    ate = np.sqrt(np.mean(np.linalg.norm(traj_obj_aligned - np.array(traj_gt_obj), axis=1)**2))
    
    # Map aligned object-centric trajectory back to Frame 0's world for visualization
    # T_WC_viz = T_WO0 @ T_OC_aligned
    traj_world_aligned = (T_WO0[:3, :3] @ traj_obj_aligned.T).T + T_WO0[:3, 3]
    
    # Sanitize stage name for Rerun path
    path_name = stage_name.replace(" ", "_").replace("(", "").replace(")", "")
    
    # Log Trajectory
    rr.log(f"world/trajectories/{path_name}", rr.LineStrips3D([np.array(traj_world_aligned)], colors=[[0, 0, 255]], radii=traj_radii), static=True)
    
    # Log Aligned Points
    pts3d = np.array([t['pt3d'] for t in tracks_est.values()])
    # Apply same Sim(3) to points: p_obj = s * R * p_C0 + t
    # Wait, our traj_est_obj was [T_OC0_gt @ inv(T_CiC0_est)][:3, 3]
    # So the points in Object frame are: pts_obj = T_OC0_gt @ pts_C0
    pts_obj = (T_OC0_gt[:3, :3] @ pts3d.T).T + T_OC0_gt[:3, 3]
    pts_obj_aligned = s * (pts_obj @ R_align.T) + t_align
    pts_world_aligned = (T_WO0[:3, :3] @ pts_obj_aligned.T).T + T_WO0[:3, 3]
    
    colors = np.array([frames[0]["image"][int(round(t['obs'][frames[0]["frame_idx"]][1])), int(round(t['obs'][frames[0]["frame_idx"]][0]))] for t in tracks_est.values()])
    rr.log(f"world/points/{path_name}", rr.Points3D(pts_world_aligned, colors=colors, radii=point_radii), static=True)

    print(f"  {stage_name} RMS ATE (Obj-centric): {ate:.6f}")
    return ate

def main(cfg: Config):
    rr.init("test_ba_vs_pnp", spawn=False)
    if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    data_dir = Path(cfg.data_root)
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    
    def load_frame(idx):
        stem = f"{idx:06d}"
        img = np.array(cv2.imread(str(data_dir / "images" / f"{stem}.png"))[..., ::-1])
        mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        if cfg.use_gt_depth:
            depth = np.load(data_dir / "depth_dyn" / f"{stem}.npy")
        else:
            d_path = data_dir / "model_infer" / f"depth_{idx:05d}.npy"
            if not d_path.exists(): d_path = data_dir / "depth_dyn" / f"{stem}.npy"
            depth = np.load(d_path)
        if depth.shape != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        T_WO_gt = load_object_pose_world(cfg.data_root, idx)
        return img, mask, depth, K, T_CW_gt, T_WO_gt

    # Use depth as-is (model or GT)
    depth_scale = 1.0
    
    frames = []
    for i in range(cfg.window_size):
        fd = load_frame(cfg.init_frame + i)
        frames.append({"image": fd[0], "mask": fd[1], "depth": fd[2], "K": fd[3], 
                       "T_CW_gt": fd[4], "T_WO_gt": fd[5], "frame_idx": cfg.init_frame + i})

    H, W = frames[0]["image"].shape[:2]
    T_WO0 = frames[0]["T_WO_gt"]
    
    # Log GT once
    traj_gt_obj = [np.linalg.inv(f["T_CW_gt"] @ f["T_WO_gt"])[:3, 3] for f in frames]
    traj_gt_world = [(T_WO0[:3, :3] @ p.T).T + T_WO0[:3, 3] for p in traj_gt_obj]
    rr.log("world/trajectories/gt", rr.LineStrips3D([np.array(traj_gt_world)], colors=[[0, 255, 0]], radii=cfg.traj_radii), static=True)
    
    # Log Frame 0 GT points as static reference
    f0 = frames[0]
    depth_gt0 = np.load(Path(cfg.data_root) / "depth_dyn" / f"{cfg.init_frame:06d}.npy")
    if depth_gt0.shape != f0["image"].shape[:2]:
        depth_gt0 = cv2.resize(depth_gt0, (f0["image"].shape[1], f0["image"].shape[0]), interpolation=cv2.INTER_NEAREST)
    yy0, xx0 = np.where(f0["mask"] > 0)
    d0 = depth_gt0[yy0, xx0]
    valid0 = d0 > 0.01
    yy0, xx0, d0 = yy0[valid0], xx0[valid0], d0[valid0]
    K_inv0 = np.linalg.inv(f0["K"])
    pts_c0 = (K_inv0 @ np.stack([xx0, yy0, np.ones_like(xx0)], axis=0)) * d0
    T_OC0_gt = np.linalg.inv(f0["T_CW_gt"] @ f0["T_WO_gt"])
    T_WC0_viz = T_WO0 @ T_OC0_gt
    pts_w_viz0 = (T_WC0_viz[:3, :3] @ pts_c0).T + T_WC0_viz[:3, 3]
    rr.log("world/gt_points_static", rr.Points3D(pts_w_viz0, colors=f0["image"][yy0, xx0], radii=cfg.point_radii), static=True)

    for f in frames:
        rr.set_time("frame", sequence=f["frame_idx"])
        T_OC_gt = np.linalg.inv(f["T_CW_gt"] @ f["T_WO_gt"])
        T_WC_gt_viz = T_WO0 @ T_OC_gt
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=f["K"], width=W, height=H))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WC_gt_viz[:3, :3], translation=T_WC_gt_viz[:3, 3]))
        rr.log("world/camera_gt", rr.ViewCoordinates.RDF, static=True) # Ensure standard RDF
        rr.log("world/camera_gt/image", rr.Image(f["image"]))
        
        # Log time-varying GT points from actual GT depth
        depth_gt = np.load(Path(cfg.data_root) / "depth_dyn" / f"{f['frame_idx']:06d}.npy")
        if depth_gt.shape != f["image"].shape[:2]:
            depth_gt = cv2.resize(depth_gt, (f["image"].shape[1], f["image"].shape[0]), interpolation=cv2.INTER_NEAREST)
        yy, xx = np.where(f["mask"] > 0)
        d = depth_gt[yy, xx]
        valid = d > 0.01
        yy, xx, d = yy[valid], xx[valid], d[valid]
        K_inv = np.linalg.inv(f["K"])
        pts_c = (K_inv @ np.stack([xx, yy, np.ones_like(xx)], axis=0)) * d
        pts_w_viz = (T_WC_gt_viz[:3, :3] @ pts_c).T + T_WC_gt_viz[:3, 3]
        rr.log("world/gt_points", rr.Points3D(pts_w_viz, colors=f["image"][yy, xx], radii=cfg.point_radii))

    print("Tracking with DIS flow advection...")
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
    print(f"Full tracks: {len(full_tracks)}")
    def get_cam_dict(f): return {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f["K"][0, 0], f["K"][1, 1], f["K"][0, 2], f["K"][1, 2]]}

    # --- Path A: PnP Initialization (Frame 0 Depth) ---
    print("\n--- Path A: PnP Initialization ---")
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
        res, _ = poselib.estimate_absolute_pose(pts2d, pts3d, get_cam_dict(f), {"max_reproj_error": cfg.ransac_thresh}, None)
        T = np.eye(4)
        T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix()
        T[:3, 3] = res.pose.t
        poses_a[f["frame_idx"]] = T
    align_and_evaluate("Path A (Init)", frames, poses_a, tracks_a, cfg.compare_gt, cfg.point_radii, cfg.traj_radii)
    run_window_ba([f["frame_idx"] for f in frames], poses_a, tracks_a, {f["frame_idx"]: f["K"] for f in frames})
    align_and_evaluate("Path A (BA)", frames, poses_a, tracks_a, cfg.compare_gt, cfg.point_radii, cfg.traj_radii)

    # --- Path B: MonoDepth Relative Initialization ---
    print("\n--- Path B: MonoDepth Relative Initialization ---")
    poses_b = {f["frame_idx"]: np.eye(4) for f in frames}
    tracks_b = {tid: {"obs": t["obs"].copy(), "id": tid} for tid, t in full_tracks.items()}
    f0, f4 = frames[0], frames[4]
    pts0 = np.array([t["obs"][f0["frame_idx"]] for t in tracks_b.values()])
    pts4 = np.array([t["obs"][f4["frame_idx"]] for t in tracks_b.values()])
    d0 = np.array([f0["depth"][int(round(p[1])), int(round(p[0]))] for p in pts0])
    d4 = np.array([f4["depth"][int(round(p[1])), int(round(p[0]))] for p in pts4])
    res, info = poselib.estimate_monodepth_relative_pose(pts0, pts4, d0, d4, get_cam_dict(f0), get_cam_dict(f4), {"max_reproj_error": cfg.ransac_thresh})
    T40 = np.eye(4)
    T40[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix()
    T40[:3, 3] = res.pose.t
    poses_b[f4["frame_idx"]] = T40
    # Triangulate using estimated pose
    pts3d = triangulate_linear(f0["K"] @ np.eye(3, 4), f4["K"] @ T40[:3, :], pts0, pts4)
    for i, tid in enumerate(tracks_b.keys()): tracks_b[tid]["pt3d"] = pts3d[i]
    for i in range(1, 4):
        f = frames[i]
        pts2d = np.array([t["obs"][f["frame_idx"]] for t in tracks_b.values()])
        pts3d_in = np.array([t["pt3d"] for t in tracks_b.values()])
        res_abs, _ = poselib.estimate_absolute_pose(pts2d, pts3d_in, get_cam_dict(f), {"max_reproj_error": cfg.ransac_thresh}, None)
        T = np.eye(4)
        T[:3, :3] = R.from_quat([res_abs.pose.q[1], res_abs.pose.q[2], res_abs.pose.q[3], res_abs.pose.q[0]]).as_matrix()
        T[:3, 3] = res_abs.pose.t
        poses_b[f["frame_idx"]] = T
    align_and_evaluate("Path B (Init)", frames, poses_b, tracks_b, cfg.compare_gt, cfg.point_radii, cfg.traj_radii)
    run_window_ba([f["frame_idx"] for f in frames], poses_b, tracks_b, {f["frame_idx"]: f["K"] for f in frames})
    align_and_evaluate("Path B (BA)", frames, poses_b, tracks_b, cfg.compare_gt, cfg.point_radii, cfg.traj_radii)

    # --- Path C: Standard Relative Initialization ---
    print("\n--- Path C: Standard Relative Initialization ---")
    poses_c = {f["frame_idx"]: np.eye(4) for f in frames}
    tracks_c = {tid: {"obs": t["obs"].copy(), "id": tid} for tid, t in full_tracks.items()}
    res, info = poselib.estimate_relative_pose(pts0, pts4, get_cam_dict(f0), get_cam_dict(f4), {"max_reproj_error": cfg.ransac_thresh})
    T40 = np.eye(4)
    T40[:3, :3] = R.from_quat([res.q[1], res.q[2], res.q[3], res.q[0]]).as_matrix()
    T40[:3, 3] = res.t
    poses_c[f4["frame_idx"]] = T40
    pts3d = triangulate_linear(f0["K"] @ np.eye(3, 4), f4["K"] @ T40[:3, :], pts0, pts4)
    for i, tid in enumerate(tracks_c.keys()): tracks_c[tid]["pt3d"] = pts3d[i]
    for i in range(1, 4):
        f = frames[i]
        pts2d = np.array([t["obs"][f["frame_idx"]] for t in tracks_c.values()])
        pts3d_in = np.array([t["pt3d"] for t in tracks_c.values()])
        res_abs, _ = poselib.estimate_absolute_pose(pts2d, pts3d_in, get_cam_dict(f), {"max_reproj_error": cfg.ransac_thresh}, None)
        T = np.eye(4)
        T[:3, :3] = R.from_quat([res_abs.pose.q[1], res_abs.pose.q[2], res_abs.pose.q[3], res_abs.pose.q[0]]).as_matrix()
        T[:3, 3] = res_abs.pose.t
        poses_c[f["frame_idx"]] = T
    align_and_evaluate("Path C (Init)", frames, poses_c, tracks_c, cfg.compare_gt, cfg.point_radii, cfg.traj_radii)
    run_window_ba([f["frame_idx"] for f in frames], poses_c, tracks_c, {f["frame_idx"]: f["K"] for f in frames})
    align_and_evaluate("Path C (BA)", frames, poses_c, tracks_c, cfg.compare_gt, cfg.point_radii, cfg.traj_radii)

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
