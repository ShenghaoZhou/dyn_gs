import numpy as np
import cv2
import rerun as rr
import torch
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R
import poselib
import pycolmap
import pycolmap.cost_functions
import pyceres
import shutil
import tempfile
from gs_dyn_obj.utils.init import unproject_depth

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 0
    n_frames: int = 100
    kf_every: int = 2
    max_keyframes: int = 60
    window_size: int = 5 # Used for initial window
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    grid_spacing: int = 8
    
    # Tracking Parameters
    ransac_thresh: float = 1.0
    min_tracks: int = 100
    
    # Comparison Flag
    compare_gt: bool = True
    
    # Depth source
    use_gt_depth: bool = False

    # Visualization
    point_radii: float = 0.01
    traj_radii: float = 0.002

    # New Parameters
    g_window: int = 3 # Window size for generalized absolute pose

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

def run_ba(frame_indices, poses, tracks, K_dict, fix_first=True, outlier_thresh=2.0):
    if len(frame_indices) < 2: return
    
    # Iterate to filter outliers
    for iteration in range(2):
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
                # If we have an existing pt3d, use it, otherwise triangulate
                track_params[tid] = track['pt3d'].copy().astype(np.float64)
                relevant_tracks.append((tid, win_obs))
        
        if not relevant_tracks: return
        
        residual_blocks = []
        for tid, win_obs in relevant_tracks:
            pt3d = track_params[tid]
            for f_idx, uv in win_obs.items():
                K = K_dict[f_idx]
                cam_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float64)
                q_wxyz, t = pose_params[f_idx]
                cost = pycolmap.cost_functions.ReprojErrorCost('PINHOLE', uv.astype(np.float64))
                b_id = prob.add_residual_block(cost, loss, [q_wxyz, t, pt3d, cam_params])
                prob.set_parameter_block_constant(cam_params)
                residual_blocks.append((b_id, tid, f_idx))
        
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
        options.max_num_iterations = 25
        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        
        # Update results
        for idx, (q_wxyz, t) in pose_params.items():
            T = np.eye(4)
            T[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
            T[:3, 3] = t
            poses[idx] = T
        for tid, _ in relevant_tracks:
            tracks[tid]['pt3d'] = track_params[tid]
            
        # Filter outliers
        if iteration == 0:
            # We would need to evaluate residuals. For now just rely on Huber.
            pass

class PathProcessor:
    def __init__(self, name, color, cfg):
        self.name = name
        self.color = color
        self.cfg = cfg
        self.poses = {} # f_idx -> T_CiC0
        self.tracks = {} # tid -> {'obs': {f_idx: uv}, 'pt3d': xyz}
        self.keyframes = [] # list of frame_idx
        self.next_tid = 0
        self.ate_history = []
        self.path_name = name.replace(" ", "_").replace("(", "").replace(")", "")

    def add_new_points_from_depth(self, frame_idx, image, mask, depth, K, T_CiC0, align=False):
        # Sample points on mask
        pts2d = sample_grid_on_mask(mask, self.cfg.grid_spacing)
        
        if align and self.tracks:
            # Use current alignment if available, otherwise solve
            z_est, z_raw = [], []
            for tid, t in self.tracks.items():
                if frame_idx in t['obs']:
                    uv = t['obs'][frame_idx]
                    pt_Ci = (T_CiC0[:3, :3] @ t['pt3d']) + T_CiC0[:3, 3]
                    z_est.append(pt_Ci[2])
                    z_raw.append(depth[int(round(uv[1])), int(round(uv[0]))])
            
            z_est, z_raw = np.array(z_est), np.array(z_raw)
            valid = (z_raw > 0.01) & (z_est > 0.01)
            if np.sum(valid) > 10:
                A = np.stack([z_raw[valid], np.ones_like(z_raw[valid])], axis=1)
                res = np.linalg.lstsq(A, z_est[valid], rcond=None)[0]
                s, b = res[0], res[1]
                depth = s * depth + b
        elif hasattr(self, 'current_s'):
            depth = self.current_s * depth + self.current_b

        # Add new points
        K_inv = np.linalg.inv(K)
        T_C0Ci = np.linalg.inv(T_CiC0)
        added_count = 0
        for uv in pts2d:
            # Avoid adding points too close to existing ones
            # (Simple grid sampling already handles this mostly)
            d = depth[int(round(uv[1])), int(round(uv[0]))]
            if d <= 0.01: continue
            pt_Ci = (K_inv @ np.array([uv[0], uv[1], 1.0])) * d
            pt_C0 = (T_C0Ci[:3, :3] @ pt_Ci) + T_C0Ci[:3, 3]
            self.tracks[self.next_tid] = {'obs': {frame_idx: uv}, 'pt3d': pt_C0}
            self.next_tid += 1
            added_count += 1
        return added_count

    def refine_pose(self, frame_idx, K):
        """Motion-only BA (Pose Refinement)"""
        if frame_idx not in self.poses: return
        pts2d, pts3d = [], []
        for tid, t in self.tracks.items():
            if frame_idx in t['obs']:
                pts2d.append(t['obs'][frame_idx])
                pts3d.append(t['pt3d'])
        if len(pts2d) < 10: return

        prob = pyceres.Problem()
        loss = pyceres.HuberLoss(1.0)
        T = self.poses[frame_idx]
        q = R.from_matrix(T[:3, :3]).as_quat()
        q_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
        t = T[:3, 3].copy().astype(np.float64)
        cam_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float64)
        
        # We need to copy the 3d points because pyceres needs double pointers
        pts3d_copy = {i: p.copy().astype(np.float64) for i, p in enumerate(pts3d)}

        for i in range(len(pts2d)):
            cost = pycolmap.cost_functions.ReprojErrorCost('PINHOLE', pts2d[i].astype(np.float64))
            prob.add_residual_block(cost, loss, [q_wxyz, t, pts3d_copy[i], cam_params])
            prob.set_parameter_block_constant(pts3d_copy[i])
        
        prob.set_parameter_block_constant(cam_params)
        prob.set_manifold(q_wxyz, pyceres.EigenQuaternionManifold())
        
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        options.max_num_iterations = 10
        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        
        T_refined = np.eye(4)
        T_refined[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
        T_refined[:3, 3] = t
        self.poses[frame_idx] = T_refined

    def step(self, frame_idx, image, mask, depth, K, flow_prev_curr=None, use_gpnp=False, prev_data=None):
        if flow_prev_curr is not None:
            # Advect tracks
            prev_idx = frame_idx - 1
            for tid, t in self.tracks.items():
                if prev_idx in t['obs']:
                    uv_prev = t['obs'][prev_idx]
                    delta = interpolate_flow(flow_prev_curr, uv_prev[None])[0]
                    uv_curr = uv_prev + delta
                    ix, iy = int(round(uv_curr[0])), int(round(uv_curr[1]))
                    if 0 <= ix < image.shape[1] and 0 <= iy < image.shape[0] and mask[iy, ix] > 0:
                        t['obs'][frame_idx] = uv_curr
        
        # Estimate Pose
        pts2d, pts3d = [], []
        active_tids = []
        for tid, t in self.tracks.items():
            if frame_idx in t['obs']:
                pts2d.append(t['obs'][frame_idx])
                pts3d.append(t['pt3d'])
                active_tids.append(tid)
        
        if len(pts2d) < 10:
            print(f"[{self.name}] Lost tracking at frame {frame_idx}")
            self.poses[frame_idx] = self.poses.get(frame_idx-1, np.eye(4)).copy()
            return False
        
        # Initial guess from Motion Model (Constant Velocity or just Previous)
        T_prev = self.poses.get(frame_idx-1, np.eye(4))
        T_prev_prev = self.poses.get(frame_idx-2, T_prev)
        T_guess = T_prev @ np.linalg.inv(T_prev_prev) @ T_prev
        
        cam_dict = {'model': 'PINHOLE', 'width': image.shape[1], 'height': image.shape[0], 'params': [K[0,0], K[1,1], K[0,2], K[1,2]]}
        
        if not use_gpnp:
            # PnP with Robust RANSAC
            res, info = poselib.estimate_absolute_pose(
                np.array(pts2d), np.array(pts3d), cam_dict, 
                {'max_reproj_error': self.cfg.ransac_thresh}, None
            )
            T = np.eye(4)
            if res:
                T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix()
                T[:3, 3] = res.pose.t
                self.poses[frame_idx] = T
                # Per-frame refinement
                self.refine_pose(frame_idx, K)
            else:
                print(f"[{self.name}] PnP failed at frame {frame_idx}")
                self.poses[frame_idx] = self.poses.get(frame_idx-1, np.eye(4)).copy()
                return False
        else:
            # Generalized PnP (Restore old logic)
            camera_ext, camera_dicts, p2ds, p3ds = [], [], [], []
            window_frames = [f for f in range(frame_idx - self.cfg.g_window + 1, frame_idx + 1) if f in self.poses or f == frame_idx]
            T_CW_ref = self.poses.get(frame_idx-1, np.eye(4))
            for f in window_frames:
                f_p2d, f_p3d = [], []
                for tid, t in self.tracks.items():
                    if f in t['obs']:
                        f_p2d.append(t['obs'][f]); f_p3d.append(t['pt3d'])
                if len(f_p2d) > 0:
                    p2ds.append(np.array(f_p2d)); p3ds.append(np.array(f_p3d))
                    T_CfCi = self.poses[f] @ np.linalg.inv(T_CW_ref) if f != frame_idx else np.eye(4)
                    cp = poselib.CameraPose()
                    q = R.from_matrix(T_CfCi[:3, :3]).as_quat()
                    cp.q = [q[3], q[0], q[1], q[2]]; cp.t = T_CfCi[:3, 3]
                    camera_ext.append(cp)
                    camera_dicts.append(cam_dict)
            if len(p2ds) >= 1:
                res, info = poselib.estimate_generalized_absolute_pose(p2ds, p3ds, camera_ext, camera_dicts, {'max_reproj_error': self.cfg.ransac_thresh}, None)
                T = np.eye(4); T[:3, :3] = R.from_quat([res.q[1], res.q[2], res.q[3], res.q[0]]).as_matrix(); T[:3, 3] = res.t
                self.poses[frame_idx] = T
        return True

    def run_kf_ba(self, K_dict, indices=None):
        if indices is None:
            kf_indices = self.keyframes[-self.cfg.max_keyframes:]
        else:
            kf_indices = indices
        run_ba(kf_indices, self.poses, self.tracks, K_dict)

def main(cfg: Config):
    rr.init("test_ba_vs_pnp_seq", spawn=False)
    if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    data_dir = Path(cfg.data_root)
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    
    def load_frame(idx):
        stem = f"{idx:06d}"
        img_path = data_dir / "images" / f"{stem}.png"
        if not img_path.exists(): return None
        img = np.array(cv2.imread(str(img_path))[..., ::-1])
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
        return {"image": img, "mask": mask, "depth": depth, "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": idx}

    # Frame Cache
    frame_cache = {}
    def get_frame(idx):
        if idx not in frame_cache:
            frame_cache[idx] = load_frame(idx)
        return frame_cache[idx]

    # Initial Scale Estimation (Removed)
    f0_data = get_frame(cfg.init_frame)
    depth_scale = 1.0

    path_a = PathProcessor("Path A (Depth)", [0, 0, 255], cfg)
    path_b = PathProcessor("Path B (RelPose)", [255, 165, 0], cfg)
    path_c = PathProcessor("Path C (COLMAP)", [255, 0, 255], cfg)
    path_d = PathProcessor("Path D (gPnP)", [0, 255, 255], cfg)
    path_e = PathProcessor("Path E (e5p1)", [255, 0, 0], cfg)
    
    K_dict = {}
    frames_buffer = []
    
    T_WO0 = f0_data["T_WO_gt"]
    
    # Initialization Path A & B
    print("Initializing Path A & B...")
    for i in range(cfg.window_size):
        fd = get_frame(cfg.init_frame + i)
        frames_buffer.append(fd)
        K_dict[fd["frame_idx"]] = fd["K"]

    # --- Path A Init ---
    f0 = frames_buffer[0]
    path_a.poses[f0["frame_idx"]] = np.eye(4)
    path_a.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], f0["depth"], f0["K"], path_a.poses[f0["frame_idx"]])
    path_a.keyframes.append(f0["frame_idx"])
    for i in range(1, cfg.window_size):
        f_prev, f_curr = frames_buffer[i-1], frames_buffer[i]
        flow = dis.calc(cv2.cvtColor(f_prev["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
        path_a.step(f_curr["frame_idx"], f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow)
    path_a.run_kf_ba(K_dict)

    # --- Path B Init ---
    f0, f_last = frames_buffer[0], frames_buffer[cfg.window_size-1]
    path_b.poses[f0["frame_idx"]] = np.eye(4)
    # Relative pose to init points
    pts0 = sample_grid_on_mask(f0["mask"], cfg.grid_spacing)
    # Find these in f_last via simple flow concatenation (or just DIS flow between them)
    flow0_last = dis.calc(cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_last["image"], cv2.COLOR_RGB2GRAY), None)
    pts_last = pts0 + interpolate_flow(flow0_last, pts0)
    # Mask check
    ix, iy = np.round(pts_last[:, 0]).astype(int), np.round(pts_last[:, 1]).astype(int)
    valid = (ix >= 0) & (ix < f0["image"].shape[1]) & (iy >= 0) & (iy < f0["image"].shape[0])
    valid[valid] &= (f_last["mask"][iy[valid], ix[valid]] > 0)
    pts0, pts_last = pts0[valid], pts_last[valid]
    d0 = np.array([f0["depth"][int(round(p[1])), int(round(p[0]))] for p in pts0])
    d_last = np.array([f_last["depth"][int(round(p[1])), int(round(p[0]))] for p in pts_last])
    
    cam0 = {'model': 'PINHOLE', 'width': f0["image"].shape[1], 'height': f0["image"].shape[0], 'params': [f0["K"][0,0], f0["K"][1,1], f0["K"][0,2], f0["K"][1,2]]}
    cam_last = {'model': 'PINHOLE', 'width': f_last["image"].shape[1], 'height': f_last["image"].shape[0], 'params': [f_last["K"][0,0], f_last["K"][1,1], f_last["K"][0,2], f_last["K"][1,2]]}
    res_rel, _ = poselib.estimate_monodepth_relative_pose(pts0, pts_last, d0, d_last, cam0, cam_last, {'max_reproj_error': cfg.ransac_thresh})
    T_L0 = np.eye(4)
    T_L0[:3, :3] = R.from_quat([res_rel.pose.q[1], res_rel.pose.q[2], res_rel.pose.q[3], res_rel.pose.q[0]]).as_matrix()
    T_L0[:3, 3] = res_rel.pose.t
    path_b.poses[f_last["frame_idx"]] = T_L0
    # Triangulate
    pts3d = triangulate_linear(f0["K"] @ np.eye(3, 4), f_last["K"] @ T_L0[:3, :], pts0, pts_last)
    for i, p in enumerate(pts3d):
        path_b.tracks[i] = {'obs': {f0["frame_idx"]: pts0[i], f_last["frame_idx"]: pts_last[i]}, 'pt3d': p}
    path_b.next_tid = len(pts3d)
    path_b.keyframes.append(f0["frame_idx"])
    # PnP for intermediate
    for i in range(1, cfg.window_size - 1):
        f_curr = frames_buffer[i]
        flow_0_curr = dis.calc(cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
        pts_curr = pts0 + interpolate_flow(flow_0_curr, pts0)
        for j, p in enumerate(pts_curr):
            if i < cfg.window_size - 1: path_b.tracks[j]['obs'][f_curr["frame_idx"]] = p
        path_b.step(f_curr["frame_idx"], f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], None)
    path_b.run_kf_ba(K_dict)
    # --- Path D & E Init ---
    for p in [path_d, path_e]:
        p.poses[f0["frame_idx"]] = np.eye(4)
        p.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], f0["depth"], f0["K"], p.poses[f0["frame_idx"]])
        p.keyframes.append(f0["frame_idx"])
        for i in range(1, cfg.window_size):
            f_prev, f_curr = frames_buffer[i-1], frames_buffer[i]
            flow = dis.calc(cv2.cvtColor(f_prev["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
            p.step(f_curr["frame_idx"], f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow)
        p.run_kf_ba(K_dict)

    # --- Main Sequential Loop ---
    prev_f = frames_buffer[-1]
    for i in tqdm(range(cfg.window_size, cfg.n_frames)):
        curr_idx = cfg.init_frame + i
        f_curr = get_frame(curr_idx)
        if f_curr is None: break
        K_dict[curr_idx] = f_curr["K"]
        
        flow = dis.calc(cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
        
        # Step Path A
        path_a.step(curr_idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow, prev_data=prev_f)
        # Step Path B
        path_b.step(curr_idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow)
        # Step Path D (gPnP)
        path_d.step(curr_idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow, use_gpnp=True)
        # Step Path E (e5p1)
        path_e.step(curr_idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow)
        
        # Keyframe Logic
        if i % cfg.kf_every == 0:
            path_a.keyframes.append(curr_idx)
            path_b.keyframes.append(curr_idx)
            path_d.keyframes.append(curr_idx)
            path_e.keyframes.append(curr_idx)
            
            # Path A: Re-seed with alignment
            path_a.add_new_points_from_depth(curr_idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], path_a.poses[curr_idx], align=True)
            
            # Path B: RelPose
            prev_kf_idx = path_b.keyframes[-2]
            f_prev_kf = get_frame(prev_kf_idx) 
            # Find correspondences between prev_kf and curr_kf
            # Using current tracks that exist in both
            pts_prev, pts_curr = [], []
            d_prev, d_curr = [], []
            for tid, t in path_b.tracks.items():
                if prev_kf_idx in t['obs'] and curr_idx in t['obs']:
                    uv_p = t['obs'][prev_kf_idx]
                    uv_c = t['obs'][curr_idx]
                    pts_prev.append(uv_p)
                    pts_curr.append(uv_c)
                    d_prev.append(f_prev_kf["depth"][int(round(uv_p[1])), int(round(uv_p[0]))])
                    d_curr.append(f_curr["depth"][int(round(uv_c[1])), int(round(uv_c[0]))])
            
            if len(pts_prev) > 10:
                cam_prev = {'model': 'PINHOLE', 'width': f_curr["image"].shape[1], 'height': f_curr["image"].shape[0], 'params': [f_prev_kf["K"][0,0], f_prev_kf["K"][1,1], f_prev_kf["K"][0,2], f_prev_kf["K"][1,2]]}
                cam_curr = {'model': 'PINHOLE', 'width': f_curr["image"].shape[1], 'height': f_curr["image"].shape[0], 'params': [f_curr["K"][0,0], f_curr["K"][1,1], f_curr["K"][0,2], f_curr["K"][1,2]]}
                res_rel, info_rel = poselib.estimate_monodepth_relative_pose(
                    np.array(pts_prev, dtype=np.float64), 
                    np.array(pts_curr, dtype=np.float64), 
                    np.array(d_prev, dtype=np.float64), 
                    np.array(d_curr, dtype=np.float64), 
                    cam_prev, cam_curr, 
                    {'max_reproj_error': cfg.ransac_thresh}
                )
                T_curr_prev = np.eye(4)
                T_curr_prev[:3, :3] = R.from_quat([res_rel.pose.q[1], res_rel.pose.q[2], res_rel.pose.q[3], res_rel.pose.q[0]]).as_matrix()
                T_curr_prev[:3, 3] = res_rel.pose.t
                
                # Triangulate new points using this relpose
                # We need to sample new points on prev_kf
                pts0_new = sample_grid_on_mask(f_prev_kf["mask"], cfg.grid_spacing)
                # Flow to curr
                flow_kf = dis.calc(cv2.cvtColor(f_prev_kf["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
                pts1_new = pts0_new + interpolate_flow(flow_kf, pts0_new)
                # Filter
                ix, iy = np.round(pts1_new[:, 0]).astype(int), np.round(pts1_new[:, 1]).astype(int)
                valid = (ix >= 0) & (ix < f_curr["image"].shape[1]) & (iy >= 0) & (iy < f_curr["image"].shape[0])
                valid[valid] &= (f_curr["mask"][iy[valid], ix[valid]] > 0)
                pts0_new, pts1_new = pts0_new[valid], pts1_new[valid]
                
                if len(pts0_new) > 0:
                    T_prev_W = path_b.poses[prev_kf_idx]
                    T_curr_W = T_curr_prev @ T_prev_W
                    P0 = f_prev_kf["K"] @ T_prev_W[:3, :]
                    P1 = f_curr["K"] @ T_curr_W[:3, :]
                    pts3d_new = triangulate_linear(P0, P1, pts0_new, pts1_new)
                    for j, p in enumerate(pts3d_new):
                        path_b.tracks[path_b.next_tid] = {'obs': {prev_kf_idx: pts0_new[j], curr_idx: pts1_new[j]}, 'pt3d': p}
                        path_b.next_tid += 1
                
                print(f"[{path_b.name}] RelPose + Triangulation at KF {curr_idx}, added {len(pts0_new)} points")

            # Re-seed Path B by triangulation or just add from depth if needed? 
            # User said "run estimate_monodepth_relative_pose again".
            # I'll add new points via triangulation between kf-5 and kf.
            # But simpler is to add from depth for now if tracks are low.
            if len(path_b.tracks) < cfg.min_tracks:
                 path_b.add_new_points_from_depth(curr_idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], path_b.poses[curr_idx], align=False)

            # Run BA
            path_a.run_kf_ba(K_dict)
            path_b.run_kf_ba(K_dict)
            path_d.run_kf_ba(K_dict)
            
            # Path E: Generalized Relative Pose (e5p1 variant)
            # Use a sliding window of frames as generalized cameras
            # Rig 1: [prev_kf-1, prev_kf], Rig 2: [curr_kf-1, curr_kf]
            if len(path_e.keyframes) >= 2:
                prev_kf_idx = path_e.keyframes[-2]
                prev_kf = get_frame(prev_kf_idx)
                curr_kf = get_frame(curr_idx)
            
            # Find matches between prev and curr
            pts_p, pts_c = [], []
            for tid, t in path_e.tracks.items():
                if path_e.keyframes[-2] in t['obs'] and curr_idx in t['obs']:
                    pts_p.append(t['obs'][path_e.keyframes[-2]])
                    pts_c.append(t['obs'][curr_idx])
            
            if len(pts_p) > 15:
                # We can use generalized relative pose if we treat windows as rigs.
                # For e5p1 specifically, it's a minimal solver. 
                # Let's use the robust generalized relative pose estimator.
                # matches: list of PairwiseMatches
                # PairwiseMatches: {cam_id1, cam_id2, x1, x2}
                pm = poselib.PairwiseMatches()
                pm.cam_id1 = 0
                pm.cam_id2 = 0
                pm.x1 = np.array(pts_p)
                pm.x2 = np.array(pts_c)
                
                # Rigid transforms for rigs (identity since it's a single camera in each "rig")
                # Wait, if it's a single camera, it's just standard relative pose.
                # To make it "Generalized", we could use [prev_kf, curr_idx-1] as Rig 1 and [curr_idx-1, curr_idx] as Rig 2.
                # But that requires knowing the internal relative pose.
                
                # Let's use Path E as a path that uses estimate_generalized_relative_pose 
                # between two windows of size 2.
                # Rig 1: frames [path_e.keyframes[-2]-1, path_e.keyframes[-2]]
                # Rig 2: frames [curr_idx-1, curr_idx]
                
                rig1_indices = [max(cfg.init_frame, path_e.keyframes[-2]-1), path_e.keyframes[-2]]
                rig2_indices = [max(cfg.init_frame, curr_idx-1), curr_idx]
                
                # Ensure they are distinct if possible, or just unique
                rig1_indices = sorted(list(set(rig1_indices)))
                rig2_indices = sorted(list(set(rig2_indices)))
                
                matches = []
                for r1 in rig1_indices:
                    for r2 in rig2_indices:
                        m_p, m_c = [], []
                        for tid, t in path_e.tracks.items():
                            if r1 in t['obs'] and r2 in t['obs']:
                                m_p.append(t['obs'][r1])
                                m_c.append(t['obs'][r2])
                        if len(m_p) > 5:
                            pm_sub = poselib.PairwiseMatches()
                            pm_sub.cam_id1 = rig1_indices.index(r1)
                            pm_sub.cam_id2 = rig2_indices.index(r2)
                            pm_sub.x1 = np.array(m_p)
                            pm_sub.x2 = np.array(m_c)
                            matches.append(pm_sub)
                
                if len(matches) > 0:
                    # Define Rig 1: frame r1 wrt path_e.keyframes[-2]
                    camera1_ext = []
                    for r1 in rig1_indices:
                        T_r1_ref = path_e.poses[r1] @ np.linalg.inv(path_e.poses[path_e.keyframes[-2]])
                        cp = poselib.CameraPose()
                        q = R.from_matrix(T_r1_ref[:3, :3]).as_quat()
                        cp.q = [q[3], q[0], q[1], q[2]]
                        cp.t = T_r1_ref[:3, 3]
                        camera1_ext.append(cp)
                    
                    # Define Rig 2: frame r2 wrt curr_idx
                    camera2_ext = []
                    # Guess T_curr_prev_kf from path_e.poses[curr_idx]
                    T_curr_W = path_e.poses[curr_idx]
                    for r2 in rig2_indices:
                        T_r2_ref = path_e.poses[r2] @ np.linalg.inv(T_curr_W)
                        cp = poselib.CameraPose()
                        q = R.from_matrix(T_r2_ref[:3, :3]).as_quat()
                        cp.q = [q[3], q[0], q[1], q[2]]
                        cp.t = T_r2_ref[:3, 3]
                        camera2_ext.append(cp)
                    
                    H, W = f_curr["image"].shape[:2]
                    cams1 = [{'model': 'PINHOLE', 'width': W, 'height': H, 'params': [K_dict[r][0,0], K_dict[r][1,1], K_dict[r][0,2], K_dict[r][1,2]]} for r in rig1_indices]
                    cams2 = [{'model': 'PINHOLE', 'width': W, 'height': H, 'params': [K_dict[r][0,0], K_dict[r][1,1], K_dict[r][0,2], K_dict[r][1,2]]} for r in rig2_indices]
                    
                    try:
                        res_grel, info_grel = poselib.estimate_generalized_relative_pose(matches, camera1_ext, cams1, camera2_ext, cams2, {'max_epipolar_error': cfg.ransac_thresh})
                        T_curr_prev_kf = np.eye(4)
                        T_curr_prev_kf[:3, :3] = R.from_quat([res_grel.q[1], res_grel.q[2], res_grel.q[3], res_grel.q[0]]).as_matrix()
                        T_curr_prev_kf[:3, 3] = res_grel.t
                        
                        # Update current pose
                        path_e.poses[curr_idx] = T_curr_prev_kf @ path_e.poses[path_e.keyframes[-2]]
                        print(f"[{path_e.name}] Generalized RelPose at KF {curr_idx}")
                    except Exception as e:
                        print(f"[{path_e.name}] Generalized RelPose failed: {e}")

            path_e.run_kf_ba(K_dict)

        # Log Visualization
        rr.set_time("frame", sequence=curr_idx)
        # GT
        T_OC_gt = np.linalg.inv(f_curr["T_CW_gt"] @ f_curr["T_WO_gt"])
        T_WC_gt_viz = T_WO0 @ T_OC_gt
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=f_curr["K"], width=f_curr["image"].shape[1], height=f_curr["image"].shape[0]))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WC_gt_viz[:3, :3], translation=T_WC_gt_viz[:3, 3]))
        rr.log("world/camera_gt/image", rr.Image(f_curr["image"]))
        
        # GT trajectory for current sequence
        gt_traj_obj = []
        for idx in range(cfg.init_frame, curr_idx + 1):
            f_gt = get_frame(idx)
            gt_traj_obj.append(np.linalg.inv(f_gt["T_CW_gt"] @ f_gt["T_WO_gt"])[:3, 3])

        # Log full GT trajectory
        gt_traj_world = [(T_WO0[:3, :3] @ p.T).T + T_WO0[:3, 3] for p in gt_traj_obj]
        rr.log("world/trajectories/gt", rr.LineStrips3D([np.array(gt_traj_world)], colors=[[0, 255, 0]], radii=cfg.traj_radii))

        def log_path(path, gt_traj_obj):
            T_CiC0 = path.poses[curr_idx]
            T_C0Ci = np.linalg.inv(T_CiC0)
            T_OC0_gt = np.linalg.inv(f0_data["T_CW_gt"] @ f0_data["T_WO_gt"])
            T_OCi_est = T_OC0_gt @ T_C0Ci
            T_WCi_est = T_WO0 @ T_OCi_est
            
            rr.log(f"world/camera_{path.path_name}", rr.Pinhole(image_from_camera=f_curr["K"], width=f_curr["image"].shape[1], height=f_curr["image"].shape[0]))
            rr.log(f"world/camera_{path.path_name}", rr.Transform3D(mat3x3=T_WCi_est[:3, :3], translation=T_WCi_est[:3, 3]))
            
            # Trajectory
            traj_est_obj = []
            for idx in range(cfg.init_frame, curr_idx + 1):
                if idx in path.poses:
                    T_idx = path.poses[idx]
                    T_OCidx = T_OC0_gt @ np.linalg.inv(T_idx)
                    traj_est_obj.append(T_OCidx[:3, 3])

            # Points
            pts3d = np.array([t['pt3d'] for t in path.tracks.values()])
            
            s, R_align, t_align = 1.0, np.eye(3), np.zeros(3)
            if cfg.compare_gt and len(traj_est_obj) >= 3:
                # Need enough points for Sim(3)
                # Ensure we have matches with gt_traj_obj
                # Actually, traj_est_obj and gt_traj_obj correspond 1-to-1 for existing frames
                # Filter to common indices
                est_arr = np.array(traj_est_obj)
                gt_arr = np.array(gt_traj_obj[:len(est_arr)])
                s, R_align, t_align = umeyama(est_arr, gt_arr)

            # Apply alignment
            traj_est_aligned = s * (np.array(traj_est_obj) @ R_align.T) + t_align
            traj_world_aligned = [(T_WO0[:3, :3] @ p.T).T + T_WO0[:3, 3] for p in traj_est_aligned]
            
            rr.log(f"world/trajectories/{path.path_name}", rr.LineStrips3D([np.array(traj_world_aligned)], colors=[path.color], radii=cfg.traj_radii))
            
            if len(pts3d) > 0:
                T_WC0_viz = T_WO0 @ T_OC0_gt
                # Apply same Sim(3) to points
                pts_obj_aligned = s * (pts3d @ R_align.T) + t_align
                pts_viz = (T_WO0[:3, :3] @ pts_obj_aligned.T).T + T_WO0[:3, 3]
                rr.log(f"world/points/{path.path_name}", rr.Points3D(pts_viz, colors=[path.color] * len(pts_viz), radii=cfg.point_radii))
            
            # Calculate ATE on aligned trajectory
            err = np.linalg.norm(np.array(traj_est_aligned) - np.array(gt_traj_obj[:len(traj_est_aligned)]), axis=1)
            ate = np.sqrt(np.mean(err**2))
            path.ate_history.append(ate)

        # GT trajectory for current sequence
        gt_traj_obj = []
        for idx in range(cfg.init_frame, curr_idx + 1):
            f_gt = get_frame(idx)
            gt_traj_obj.append(np.linalg.inv(f_gt["T_CW_gt"] @ f_gt["T_WO_gt"])[:3, 3])
        
        log_path(path_a, gt_traj_obj)
        log_path(path_b, gt_traj_obj)
        log_path(path_d, gt_traj_obj)
        log_path(path_e, gt_traj_obj)
        
        prev_f = f_curr

    print("Sequence processed.")
    
    print("Sequence processed.")

    # --- Path C: COLMAP Incremental Mapping ---
    print("Running Path C (COLMAP)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "colmap.db"
        img_tmp_dir = tmp_path / "images"
        img_tmp_dir.mkdir()
        
        db = pycolmap.Database.open(str(db_path))
        K0 = get_frame(cfg.init_frame)["K"]
        H, W = get_frame(cfg.init_frame)["image"].shape[:2]
        camera = pycolmap.Camera(model="PINHOLE", width=W, height=H, 
                                 params=[K0[0,0], K0[1,1], K0[0,2], K0[1,2]])
        cam_id = db.write_camera(camera)
        
        indices = sorted(K_dict.keys())
        image_ids = {}
        for idx in indices:
            fd = get_frame(idx)
            img_name = f"image_{idx:06d}.png"
            cv2.imwrite(str(img_tmp_dir / img_name), cv2.cvtColor(fd["image"], cv2.COLOR_RGB2BGR))
            image_id = db.write_image(pycolmap.Image(name=img_name, camera_id=cam_id))
            image_ids[idx] = image_id
            
        # Write Keypoints and Matches from Path A tracks
        # Path A tracks are tidal -> {obs: {f_idx: uv}}
        for idx in indices:
            frame_tids = sorted([tid for tid, t in path_a.tracks.items() if idx in t['obs']])
            keypoints = np.array([path_a.tracks[tid]['obs'][idx] for tid in frame_tids], dtype=np.float32)
            kpts_full = np.zeros((len(keypoints), 4), dtype=np.float32)
            kpts_full[:, :2] = keypoints
            kpts_full[:, 2] = 1.0
            kpts_full[:, 3] = 0.0 # dummy scale/orientation
            db.write_keypoints(image_ids[idx], kpts_full)
            
        pairs = []
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                # Find common tracks
                tids1 = set([tid for tid, t in path_a.tracks.items() if idx1 in t['obs']])
                tids2 = set([tid for tid, t in path_a.tracks.items() if idx2 in t['obs']])
                common_tids = sorted(list(tids1 & tids2))
                
                if len(common_tids) >= 15:
                    # Map tids to local indices in write_keypoints
                    frame_tids1 = sorted([tid for tid, t in path_a.tracks.items() if idx1 in t['obs']])
                    frame_tids2 = sorted([tid for tid, t in path_a.tracks.items() if idx2 in t['obs']])
                    tid_to_idx1 = {tid: k for k, tid in enumerate(frame_tids1)}
                    tid_to_idx2 = {tid: k for k, tid in enumerate(frame_tids2)}
                    
                    matches = np.array([[tid_to_idx1[tid], tid_to_idx2[tid]] for tid in common_tids], dtype=np.uint32)
                    db.write_matches(image_ids[idx1], image_ids[idx2], matches)
                    pairs.append((f"image_{idx1:06d}.png", f"image_{idx2:06d}.png"))
        
        db.close()
        
        pairs_path = tmp_path / "pairs.txt"
        with open(pairs_path, "w") as f_pairs:
            for p1, p2 in pairs: f_pairs.write(f"{p1} {p2}\n")
            
        pycolmap.verify_matches(str(db_path), str(pairs_path))
        colmap_out = tmp_path / "reconstruction"
        colmap_out.mkdir()
        reconstructions = pycolmap.incremental_mapping(str(db_path), str(img_tmp_dir), str(colmap_out))
        
        if reconstructions:
            rec = sorted(reconstructions.values(), key=lambda x: x.num_points3D(), reverse=True)[0]
            # Extract Poses
            traj_c_obj = []
            gt_traj_c_obj = []
            valid_indices = []
            
            # T_OC0_gt for Sim(3) alignment reference
            T_OC0_gt = np.linalg.inv(get_frame(cfg.init_frame)["T_CW_gt"] @ get_frame(cfg.init_frame)["T_WO_gt"])
            
            # Create a mapping for easy lookup
            name_to_img = {img.name: img for img in rec.images.values()}

            # Get registered image IDs
            reg_ids = rec.reg_image_ids()

            for idx in indices:
                img_name = f"image_{idx:06d}.png"
                if img_name in name_to_img:
                    img = name_to_img[img_name]
                    if img.image_id in reg_ids:
                        # Extract 3x4 or 4x4 matrix
                        m = img.cam_from_world().matrix()
                        T_CW = np.eye(4)
                        T_CW[:3, :4] = m[:3, :4]
                        T_WC = np.linalg.inv(T_CW)
                        # COLMAP's world is arbitrary. Let's just collect positions.
                        traj_c_obj.append(T_WC[:3, 3])
                        gt_traj_c_obj.append(np.linalg.inv(get_frame(idx)["T_CW_gt"] @ get_frame(idx)["T_WO_gt"])[:3, 3])
                        valid_indices.append(idx)
            
            if len(traj_c_obj) >= 3:
                s, R_align, t_align = umeyama(np.array(traj_c_obj), np.array(gt_traj_c_obj))
                traj_c_aligned = s * (np.array(traj_c_obj) @ R_align.T) + t_align
                
                # Log Result
                ate_c = np.sqrt(np.mean(np.linalg.norm(traj_c_aligned - np.array(gt_traj_c_obj), axis=1)**2))
                path_c.ate_history = [ate_c] # Store for summary
                
                # Log full trajectory as a LineStrip (static for easy visualization)
                traj_world_full = (np.array(traj_c_aligned) @ T_WO0[:3, :3].T) + T_WO0[:3, 3]
                rr.log(f"world/trajectories/{path_c.path_name}", rr.LineStrips3D([traj_world_full], colors=[path_c.color], radii=cfg.traj_radii * 2), static=True)

                # Log to Rerun: Camera frustums per frame
                for i, idx in enumerate(valid_indices):
                    rr.set_time("frame", sequence=idx)
                    # For Path C, we don't have the original CiC0, but we have the aligned traj_c_aligned
                    # To log the frustum, we need the rotation too.
                    # R_align was found such that: gt = s * R_align * est + t
                    # So R_est_aligned = R_align @ R_est
                    # T_WC_aligned rotation = T_WO0_rot @ R_est_aligned.T ? No.
                    # Let's re-extract T_WC for each frame from the alignment.
                    img_name = f"image_{idx:06d}.png"
                    img = name_to_img[img_name]
                    T_CW = np.eye(4)
                    T_CW[:3, :4] = img.cam_from_world().matrix()
                    T_WC = np.linalg.inv(T_CW)
                    
                    # Align T_WC
                    R_WC_aligned = R_align @ T_WC[:3, :3]
                    t_WC_aligned = traj_c_aligned[i]
                    
                    T_WC_viz = np.eye(4)
                    T_WC_viz[:3, :3] = T_WO0[:3, :3] @ R_WC_aligned
                    T_WC_viz[:3, 3] = (T_WO0[:3, :3] @ t_WC_aligned) + T_WO0[:3, 3]
                    
                    rr.log(f"world/camera_{path_c.path_name}", rr.Pinhole(image_from_camera=get_frame(idx)["K"], width=W, height=H))
                    rr.log(f"world/camera_{path_c.path_name}", rr.Transform3D(mat3x3=T_WC_viz[:3, :3], translation=T_WC_viz[:3, 3]))
                
                # Also points
                colmap_pts = np.array([p.xyz for p in rec.points3D.values()])
                if len(colmap_pts) > 0:
                    pts_aligned = s * (colmap_pts @ R_align.T) + t_align
                    pts_world = (T_WO0[:3, :3] @ pts_aligned.T).T + T_WO0[:3, 3]
                    rr.log(f"world/points/{path_c.path_name}", rr.Points3D(pts_world, colors=[path_c.color] * len(pts_world), radii=cfg.point_radii), static=True)
        else:
            print("Path C (COLMAP) failed to produce a reconstruction.")

    print("\n" + "="*30)
    print("      Final ATE Summary")
    print("="*30)
    for p in [path_a, path_b, path_d, path_e, path_c]:
        if p.ate_history:
            print(f"  {p.name:25s}: {p.ate_history[-1]:.6f}")
        else:
            print(f"  {p.name:25s}: FAILED")
    print("="*30)
    
    print("All paths processed.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
