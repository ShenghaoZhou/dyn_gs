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
    kf_every: int = 5
    max_keyframes: int = 5
    window_size: int = 5 # Used for initial window
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    grid_spacing: int = 8
    
    # ORB Parameters
    max_orb_features: int = 1000
    use_klt: bool = True
    
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

def run_ba(frame_indices, poses, tracks, K_dict, fix_first=True):
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
    
    for tid, win_obs in relevant_tracks:
        pt3d = track_params[tid]
        for f_idx, uv in win_obs.items():
            K = K_dict[f_idx]
            cam_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float64)
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
        
        # ORB and KLT
        self.orb = cv2.ORB_create(nfeatures=cfg.max_orb_features)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def add_new_points_from_depth(self, frame_idx, image, mask, depth, K, T_CiC0, align=False):
        # Detect ORB features on mask
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kps = self.orb.detect(gray, mask=mask)
        pts2d = np.array([kp.pt for kp in kps], dtype=np.float32)
        
        if align and self.tracks:
            # Align depth to existing points
            z_est = []
            z_raw = []
            for tid, t in self.tracks.items():
                if frame_idx in t['obs']:
                    uv = t['obs'][frame_idx]
                    pt_Ci = (T_CiC0[:3, :3] @ t['pt3d']) + T_CiC0[:3, 3]
                    z_est.append(pt_Ci[2])
                    z_raw.append(depth[int(round(uv[1])), int(round(uv[0]))])
            
            z_est = np.array(z_est)
            z_raw = np.array(z_raw)
            valid = (z_raw > 0.01) & (z_est > 0.01)
            if np.sum(valid) > 10:
                A = np.stack([z_raw[valid], np.ones_like(z_raw[valid])], axis=1)
                res = np.linalg.lstsq(A, z_est[valid], rcond=None)[0]
                s, b = res[0], res[1]
                print(f"[{self.name}] Depth Alignment: s={s:.4f}, b={b:.4f}")
                depth = s * depth + b

        # Add new points
        K_inv = np.linalg.inv(K)
        T_C0Ci = np.linalg.inv(T_CiC0)
        for uv in pts2d:
            # Check if this point is already tracked (simplified)
            # In a real system, we'd check distance to existing points
            d = depth[int(round(uv[1])), int(round(uv[0]))]
            if d <= 0.01: continue
            pt_Ci = (K_inv @ np.array([uv[0], uv[1], 1.0])) * d
            pt_C0 = (T_C0Ci[:3, :3] @ pt_Ci) + T_C0Ci[:3, 3]
            self.tracks[self.next_tid] = {'obs': {frame_idx: uv}, 'pt3d': pt_C0}
            self.next_tid += 1

    def step(self, frame_idx, image, mask, depth, K, flow_prev_curr=None, gray_prev=None, gray_curr=None):
        if flow_prev_curr is not None:
            prev_idx = frame_idx - 1
            valid_tids = [tid for tid, t in self.tracks.items() if prev_idx in t['obs']]
            if valid_tids:
                pts_prev = np.array([self.tracks[tid]['obs'][prev_idx] for tid in valid_tids], dtype=np.float32)
                pts_next_dis = (pts_prev + interpolate_flow(flow_prev_curr, pts_prev)).astype(np.float32)
                
                if self.cfg.use_klt and gray_prev is not None and gray_curr is not None:
                    pts_prev_klt = pts_prev.reshape(-1, 1, 2)
                    pts_next_init = pts_next_dis.reshape(-1, 1, 2)
                    pts_next_klt, status, _ = cv2.calcOpticalFlowPyrLK(
                        gray_prev, gray_curr, pts_prev_klt, pts_next_init, 
                        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk_params
                    )
                    pts_next = pts_next_klt.reshape(-1, 2)
                    status = status.flatten().astype(bool)
                else:
                    pts_next = pts_next_dis
                    status = np.ones(len(pts_next), dtype=bool)

                ix, iy = np.round(pts_next[:, 0]).astype(int), np.round(pts_next[:, 1]).astype(int)
                mask_valid = (ix >= 0) & (ix < image.shape[1]) & (iy >= 0) & (iy < image.shape[0])
                for j, tid in enumerate(valid_tids):
                    if status[j] and mask_valid[j] and mask[iy[j], ix[j]] > 0:
                        self.tracks[tid]['obs'][frame_idx] = pts_next[j]
        
        # Estimate Pose with PnP
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
        
        cam_dict = {'model': 'PINHOLE', 'width': image.shape[1], 'height': image.shape[0], 'params': [K[0,0], K[1,1], K[0,2], K[1,2]]}
        res, info = poselib.estimate_absolute_pose(np.array(pts2d), np.array(pts3d), cam_dict, {'max_reproj_error': self.cfg.ransac_thresh}, None)
        T = np.eye(4)
        if res:
            T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix()
            T[:3, 3] = res.pose.t
        self.poses[frame_idx] = T
        return True

    def run_kf_ba(self, K_dict):
        kf_indices = self.keyframes[-self.cfg.max_keyframes:]
        run_ba(kf_indices, self.poses, self.tracks, K_dict)

def main(cfg: Config):
    rr.init("test_ba_vs_pnp_seq_orb", spawn=False)
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

    frame_cache = {}
    def get_frame(idx):
        if idx not in frame_cache:
            frame_cache[idx] = load_frame(idx)
        return frame_cache[idx]

    f0_data = get_frame(cfg.init_frame)
    path_a = PathProcessor("Path A (Depth)", [0, 0, 255], cfg)
    path_b = PathProcessor("Path B (RelPose)", [255, 165, 0], cfg)
    path_c = PathProcessor("Path C (COLMAP)", [255, 0, 255], cfg)
    
    K_dict = {}
    frames_buffer = []
    T_WO0 = f0_data["T_WO_gt"]
    H, W = f0_data["image"].shape[:2]
    
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
        gray_prev = cv2.cvtColor(f_prev["image"], cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY)
        flow = dis.calc(gray_prev, gray_curr, None)
        path_a.step(f_curr["frame_idx"], f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow, gray_prev, gray_curr)
    path_a.run_kf_ba(K_dict)

    # --- Path B Init ---
    f0, f_last = frames_buffer[0], frames_buffer[cfg.window_size-1]
    path_b.poses[f0["frame_idx"]] = np.eye(4)
    # Detect ORB in f0
    gray0 = cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY)
    kps0 = path_b.orb.detect(gray0, mask=f0["mask"])
    pts0 = np.array([kp.pt for kp in kps0], dtype=np.float32)
    # Track to f_last
    flow0_last = dis.calc(gray0, cv2.cvtColor(f_last["image"], cv2.COLOR_RGB2GRAY), None)
    pts_last_dis = (pts0 + interpolate_flow(flow0_last, pts0)).astype(np.float32)
    # KLT refine
    pts_last_klt, status, _ = cv2.calcOpticalFlowPyrLK(gray0, cv2.cvtColor(f_last["image"], cv2.COLOR_RGB2GRAY), pts0.reshape(-1, 1, 2).astype(np.float32), pts_last_dis.reshape(-1, 1, 2).astype(np.float32), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **path_b.lk_params)
    pts_last = pts_last_klt.reshape(-1, 2)
    status = status.flatten().astype(bool)
    
    ix, iy = np.round(pts_last[:, 0]).astype(int), np.round(pts_last[:, 1]).astype(int)
    valid = status & (ix >= 0) & (ix < f0["image"].shape[1]) & (iy >= 0) & (iy < f0["image"].shape[0])
    valid[valid] &= (f_last["mask"][iy[valid], ix[valid]] > 0)
    pts0, pts_last = pts0[valid], pts_last[valid]
    d0 = np.array([f0["depth"][int(round(p[1])), int(round(p[0]))] for p in pts0])
    d_last = np.array([f_last["depth"][int(round(p[1])), int(round(p[0]))] for p in pts_last])
    
    cam0 = {'model': 'PINHOLE', 'width': f0["image"].shape[1], 'height': f0["image"].shape[0], 'params': [f0["K"][0,0], f0["K"][1,1], f0["K"][0,2], f0["K"][1,2]]}
    cam_last = {'model': 'PINHOLE', 'width': f_last["image"].shape[1], 'height': f_last["image"].shape[0], 'params': [f_last["K"][0,0], f_last["K"][1,1], f_last["K"][0,2], f_last["K"][1,2]]}
    res_rel, _ = poselib.estimate_monodepth_relative_pose(pts0, pts_last, d0, d_last, cam0, cam_last, {'max_reproj_error': cfg.ransac_thresh})
    T_L0 = np.eye(4)
    if res_rel:
        T_L0[:3, :3] = R.from_quat([res_rel.pose.q[1], res_rel.pose.q[2], res_rel.pose.q[3], res_rel.pose.q[0]]).as_matrix()
        T_L0[:3, 3] = res_rel.pose.t
    path_b.poses[f_last["frame_idx"]] = T_L0
    pts3d = triangulate_linear(f0["K"] @ np.eye(3, 4), f_last["K"] @ T_L0[:3, :], pts0, pts_last)
    for i, p in enumerate(pts3d):
        path_b.tracks[i] = {'obs': {f0["frame_idx"]: pts0[i], f_last["frame_idx"]: pts_last[i]}, 'pt3d': p}
    path_b.next_tid = len(pts3d)
    path_b.keyframes.append(f0["frame_idx"])
    for i in range(1, cfg.window_size - 1):
        f_curr = frames_buffer[i]
        # Just use DIS flow for intermediate obs
        flow_0_curr = dis.calc(gray0, cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
        pts_curr = pts0 + interpolate_flow(flow_0_curr, pts0)
        for j, p in enumerate(pts_curr):
             path_b.tracks[j]['obs'][f_curr["frame_idx"]] = p
        path_b.step(f_curr["frame_idx"], f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], None)
    path_b.run_kf_ba(K_dict)

    # Dense Cloud Tracking
    dense_tracks = {i: {"obs": {f0["frame_idx"]: p}} for i, p in enumerate(sample_grid_on_mask(f0["mask"], cfg.grid_spacing))}

    # --- Main Sequential Loop ---
    prev_f = frames_buffer[-1]
    for i in tqdm(range(cfg.window_size, cfg.n_frames)):
        curr_idx = cfg.init_frame + i
        f_curr = get_frame(curr_idx)
        if f_curr is None: break
        K_dict[curr_idx] = f_curr["K"]
        
        gray_prev = cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY)
        flow = dis.calc(gray_prev, gray_curr, None)
        
        path_a.step(curr_idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow, gray_prev, gray_curr)
        path_b.step(curr_idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow, gray_prev, gray_curr)
        
        # Update dense tracks
        valid_dense_tids = [tid for tid in dense_tracks if (curr_idx - 1) in dense_tracks[tid]["obs"]]
        if valid_dense_tids:
            pts_prev = np.array([dense_tracks[tid]["obs"][curr_idx - 1] for tid in valid_dense_tids])
            pts_next = pts_prev + interpolate_flow(flow, pts_prev)
            ix, iy = np.round(pts_next[:, 0]).astype(int), np.round(pts_next[:, 1]).astype(int)
            mask_valid = (ix >= 0) & (ix < f_curr["image"].shape[1]) & (iy >= 0) & (iy < f_curr["image"].shape[0])
            for j, tid in enumerate(valid_dense_tids):
                if mask_valid[j] and f_curr["mask"][iy[j], ix[j]] > 0:
                    dense_tracks[tid]["obs"][curr_idx] = pts_next[j]

        if i % cfg.kf_every == 0:
            path_a.keyframes.append(curr_idx)
            path_b.keyframes.append(curr_idx)
            path_a.add_new_points_from_depth(curr_idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], path_a.poses[curr_idx], align=True)
            
            prev_kf_idx = path_b.keyframes[-2]
            f_prev_kf = get_frame(prev_kf_idx) 
            pts_prev, pts_curr, d_prev, d_curr = [], [], [], []
            for tid, t in path_b.tracks.items():
                if prev_kf_idx in t['obs'] and curr_idx in t['obs']:
                    uv_p, uv_c = t['obs'][prev_kf_idx], t['obs'][curr_idx]
                    pts_prev.append(uv_p)
                    pts_curr.append(uv_c)
                    d_prev.append(f_prev_kf["depth"][int(round(uv_p[1])), int(round(uv_p[0]))])
                    d_curr.append(f_curr["depth"][int(round(uv_c[1])), int(round(uv_c[0]))])
            
            if len(pts_prev) > 10:
                cam_prev = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f_prev_kf["K"][0,0], f_prev_kf["K"][1,1], f_prev_kf["K"][0,2], f_prev_kf["K"][1,2]]}
                cam_curr = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f_curr["K"][0,0], f_curr["K"][1,1], f_curr["K"][0,2], f_curr["K"][1,2]]}
                res_rel, _ = poselib.estimate_monodepth_relative_pose(np.array(pts_prev, dtype=np.float64), np.array(pts_curr, dtype=np.float64), np.array(d_prev, dtype=np.float64), np.array(d_curr, dtype=np.float64), cam_prev, cam_curr, {'max_reproj_error': cfg.ransac_thresh})
                if res_rel:
                    T_curr_prev = np.eye(4)
                    T_curr_prev[:3, :3] = R.from_quat([res_rel.pose.q[1], res_rel.pose.q[2], res_rel.pose.q[3], res_rel.pose.q[0]]).as_matrix()
                    T_curr_prev[:3, 3] = res_rel.pose.t
                    # Triangulate new ORB points
                    kps_kf = path_b.orb.detect(cv2.cvtColor(f_prev_kf["image"], cv2.COLOR_RGB2GRAY), mask=f_prev_kf["mask"])
                    pts0_new = np.array([kp.pt for kp in kps_kf], dtype=np.float32)
                    flow_kf = dis.calc(cv2.cvtColor(f_prev_kf["image"], cv2.COLOR_RGB2GRAY), gray_curr, None)
                    pts1_new_dis = (pts0_new + interpolate_flow(flow_kf, pts0_new)).astype(np.float32)
                    pts1_new_klt, status, _ = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(f_prev_kf["image"], cv2.COLOR_RGB2GRAY), gray_curr, pts0_new.reshape(-1, 1, 2).astype(np.float32), pts1_new_dis.reshape(-1, 1, 2).astype(np.float32), flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **path_b.lk_params)
                    pts1_new = pts1_new_klt.reshape(-1, 2)
                    status = status.flatten().astype(bool)
                    ix, iy = np.round(pts1_new[:, 0]).astype(int), np.round(pts1_new[:, 1]).astype(int)
                    valid = status & (ix >= 0) & (ix < f_curr["image"].shape[1]) & (iy >= 0) & (iy < f_curr["image"].shape[0])
                    valid[valid] &= (f_curr["mask"][iy[valid], ix[valid]] > 0)
                    pts0_new, pts1_new = pts0_new[valid], pts1_new[valid]
                    if len(pts0_new) > 0:
                        T_prev_W, T_curr_W = path_b.poses[prev_kf_idx], T_curr_prev @ path_b.poses[prev_kf_idx]
                        P0, P1 = f_prev_kf["K"] @ T_prev_W[:3, :], f_curr["K"] @ T_curr_W[:3, :]
                        pts3d_new = triangulate_linear(P0, P1, pts0_new, pts1_new)
                        for j, p in enumerate(pts3d_new):
                            path_b.tracks[path_b.next_tid] = {'obs': {prev_kf_idx: pts0_new[j], curr_idx: pts1_new[j]}, 'pt3d': p}
                            path_b.next_tid += 1
            if len(path_b.tracks) < cfg.min_tracks:
                 path_b.add_new_points_from_depth(curr_idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], path_b.poses[curr_idx], align=False)
            path_a.run_kf_ba(K_dict)
            path_b.run_kf_ba(K_dict)

        # Log Visualization
        rr.set_time("frame", sequence=curr_idx)
        T_OC_gt = np.linalg.inv(f_curr["T_CW_gt"] @ f_curr["T_WO_gt"])
        T_WC_gt_viz = T_WO0 @ T_OC_gt
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=f_curr["K"], width=f_curr["image"].shape[1], height=f_curr["image"].shape[0]))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WC_gt_viz[:3, :3], translation=T_WC_gt_viz[:3, 3]))
        rr.log("world/camera_gt/image", rr.Image(f_curr["image"]))
        
        gt_traj_obj = []
        for idx in range(cfg.init_frame, curr_idx + 1):
            f_gt = get_frame(idx)
            gt_traj_obj.append(np.linalg.inv(f_gt["T_CW_gt"] @ f_gt["T_WO_gt"])[:3, 3])
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
            traj_est_obj = []
            for idx in range(cfg.init_frame, curr_idx + 1):
                if idx in path.poses:
                    T_OCidx = T_OC0_gt @ np.linalg.inv(path.poses[idx])
                    traj_est_obj.append(T_OCidx[:3, 3])
            pts3d = np.array([t['pt3d'] for t in path.tracks.values()])
            s, R_align, t_align = 1.0, np.eye(3), np.zeros(3)
            if cfg.compare_gt and len(traj_est_obj) >= 3:
                est_arr, gt_arr = np.array(traj_est_obj), np.array(gt_traj_obj[:len(traj_est_obj)])
                s, R_align, t_align = umeyama(est_arr, gt_arr)
            traj_est_aligned = s * (np.array(traj_est_obj) @ R_align.T) + t_align
            traj_world_aligned = [(T_WO0[:3, :3] @ p.T).T + T_WO0[:3, 3] for p in traj_est_aligned]
            rr.log(f"world/trajectories/{path.path_name}", rr.LineStrips3D([np.array(traj_world_aligned)], colors=[path.color], radii=cfg.traj_radii))
            if len(pts3d) > 0:
                pts_obj_aligned = s * (pts3d @ R_align.T) + t_align
                pts_viz = (T_WO0[:3, :3] @ pts_obj_aligned.T).T + T_WO0[:3, 3]
                rr.log(f"world/points/{path.path_name}", rr.Points3D(pts_viz, colors=[path.color] * len(pts_viz), radii=cfg.point_radii))
            err = np.linalg.norm(np.array(traj_est_aligned) - np.array(gt_traj_obj[:len(traj_est_aligned)]), axis=1)
            path.ate_history.append(np.sqrt(np.mean(err**2)))

        log_path(path_a, gt_traj_obj)
        log_path(path_b, gt_traj_obj)
        
        # Log Dense Cloud
        valid_dense_tids = [tid for tid in dense_tracks if curr_idx in dense_tracks[tid]["obs"]]
        if valid_dense_tids:
            pts_uv = np.array([dense_tracks[tid]["obs"][curr_idx] for tid in valid_dense_tids])
            d = np.array([f_curr["depth"][int(round(uv[1])), int(round(uv[0]))] for uv in pts_uv])
            valid_d = d > 0.01
            pts_uv, d = pts_uv[valid_d], d[valid_d]
            K_inv = np.linalg.inv(f_curr["K"])
            pts_ci = (K_inv @ np.vstack([pts_uv.T, np.ones(len(pts_uv))])) * d
            # Use Path A (Depth) pose for visualization reference
            T_CiC0 = path_a.poses[curr_idx]
            T_OC0_gt = np.linalg.inv(f0_data["T_CW_gt"] @ f0_data["T_WO_gt"])
            T_WCi_viz = T_WO0 @ T_OC0_gt @ np.linalg.inv(T_CiC0)
            pts_w_viz = (T_WCi_viz[:3, :3] @ pts_ci).T + T_WCi_viz[:3, 3]
            colors = f_curr["image"][np.round(pts_uv[:, 1]).astype(int), np.round(pts_uv[:, 0]).astype(int)]
            rr.log("world/dense_cloud", rr.Points3D(pts_w_viz, colors=colors, radii=cfg.point_radii))
        
        prev_f = f_curr

    # --- Path C: COLMAP ---
    print("Running Path C (COLMAP)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path, db_path, img_tmp_dir = Path(tmp_dir), Path(tmp_dir) / "colmap.db", Path(tmp_dir) / "images"
        img_tmp_dir.mkdir()
        db = pycolmap.Database.open(str(db_path))
        H, W = f0_data["image"].shape[:2]
        cam_id = db.write_camera(pycolmap.Camera(model="PINHOLE", width=W, height=H, params=[f0_data["K"][0,0], f0_data["K"][1,1], f0_data["K"][0,2], f0_data["K"][1,2]]))
        indices, image_ids = sorted(K_dict.keys()), {}
        for idx in indices:
            fd = get_frame(idx)
            img_name = f"image_{idx:06d}.png"
            cv2.imwrite(str(img_tmp_dir / img_name), cv2.cvtColor(fd["image"], cv2.COLOR_RGB2BGR))
            image_ids[idx] = db.write_image(pycolmap.Image(name=img_name, camera_id=cam_id))
        for idx in indices:
            frame_tids = sorted([tid for tid, t in path_a.tracks.items() if idx in t['obs']])
            keypoints = np.array([path_a.tracks[tid]['obs'][idx] for tid in frame_tids], dtype=np.float32)
            kpts_full = np.zeros((len(keypoints), 4), dtype=np.float32)
            kpts_full[:, :2], kpts_full[:, 2] = keypoints, 1.0
            db.write_keypoints(image_ids[idx], kpts_full)
        pairs = []
        for i in range(len(indices)):
            for j in range(i + 1, min(i + 15, len(indices))): # Limit matching window
                idx1, idx2 = indices[i], indices[j]
                tids1 = set([tid for tid, t in path_a.tracks.items() if idx1 in t['obs']])
                tids2 = set([tid for tid, t in path_a.tracks.items() if idx2 in t['obs']])
                common_tids = sorted(list(tids1 & tids2))
                if len(common_tids) >= 15:
                    tid_to_idx1 = {tid: k for k, tid in enumerate(sorted(list(tids1)))}
                    tid_to_idx2 = {tid: k for k, tid in enumerate(sorted(list(tids2)))}
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
            traj_c_obj, gt_traj_c_obj, valid_indices = [], [], []
            name_to_img = {img.name: img for img in rec.images.values()}
            reg_ids = rec.reg_image_ids()
            for idx in indices:
                img_name = f"image_{idx:06d}.png"
                if img_name in name_to_img and name_to_img[img_name].image_id in reg_ids:
                    img = name_to_img[img_name]
                    T_CW = np.eye(4)
                    m = img.cam_from_world().matrix()
                    T_CW[:3, :4] = m[:3, :4]
                    T_WC = np.linalg.inv(T_CW)
                    traj_c_obj.append(T_WC[:3, 3]); gt_traj_c_obj.append(np.linalg.inv(get_frame(idx)["T_CW_gt"] @ get_frame(idx)["T_WO_gt"])[:3, 3]); valid_indices.append(idx)
            if len(traj_c_obj) >= 3:
                s, R_align, t_align = umeyama(np.array(traj_c_obj), np.array(gt_traj_c_obj))
                traj_c_aligned = s * (np.array(traj_c_obj) @ R_align.T) + t_align
                path_c.ate_history = [np.sqrt(np.mean(np.linalg.norm(traj_c_aligned - np.array(gt_traj_c_obj), axis=1)**2))]
                traj_world_full = (np.array(traj_c_aligned) @ T_WO0[:3, :3].T) + T_WO0[:3, 3]
                rr.log(f"world/trajectories/{path_c.path_name}", rr.LineStrips3D([traj_world_full], colors=[path_c.color], radii=cfg.traj_radii * 2), static=True)
                for i, idx in enumerate(valid_indices):
                    rr.set_time("frame", sequence=idx)
                    img = name_to_img[f"image_{idx:06d}.png"]
                    T_CW = np.eye(4); T_CW[:3, :4] = img.cam_from_world().matrix()
                    T_WC = np.linalg.inv(T_CW)
                    R_WC_aligned = R_align @ T_WC[:3, :3]
                    t_WC_aligned = traj_c_aligned[i]
                    T_WC_viz = np.eye(4); T_WC_viz[:3, :3] = T_WO0[:3, :3] @ R_WC_aligned; T_WC_viz[:3, 3] = (T_WO0[:3, :3] @ t_WC_aligned) + T_WO0[:3, 3]
                    rr.log(f"world/camera_{path_c.path_name}", rr.Pinhole(image_from_camera=get_frame(idx)["K"], width=W, height=H))
                    rr.log(f"world/camera_{path_c.path_name}", rr.Transform3D(mat3x3=T_WC_viz[:3, :3], translation=T_WC_viz[:3, 3]))
                colmap_pts = np.array([p.xyz for p in rec.points3D.values()])
                if len(colmap_pts) > 0:
                    pts_aligned = s * (colmap_pts @ R_align.T) + t_align
                    pts_world = (T_WO0[:3, :3] @ pts_aligned.T).T + T_WO0[:3, 3]
                    rr.log(f"world/points/{path_c.path_name}", rr.Points3D(pts_world, colors=[path_c.color] * len(pts_world), radii=cfg.point_radii), static=True)
    print("\n" + "="*30 + "\n      Final ATE Summary\n" + "="*30)
    for p in [path_a, path_b, path_c]:
        print(f"  {p.name:25s}: {p.ate_history[-1]:.6f}" if p.ate_history else f"  {p.name:25s}: FAILED")
    print("="*30)

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
