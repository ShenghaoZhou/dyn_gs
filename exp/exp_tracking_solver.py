import numpy as np
import cv2
import torch
import rerun as rr
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R
import poselib
import pyceres
import pycolmap
import pycolmap.cost_functions
import rerun.blueprint as rrb

# --- Helper Functions ---

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
        if prob.has_parameter_block(pose_params[ref_idx][0]):
            prob.set_parameter_block_constant(pose_params[ref_idx][0])
        if prob.has_parameter_block(pose_params[ref_idx][1]):
            prob.set_parameter_block_constant(pose_params[ref_idx][1])
    
    quat_manifold = pyceres.EigenQuaternionManifold()
    for idx in frame_indices:
        q_wxyz, t = pose_params[idx]
        if prob.has_parameter_block(q_wxyz) and not prob.is_parameter_block_constant(q_wxyz):
            prob.set_manifold(q_wxyz, quat_manifold)
            
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
    options.max_num_iterations = 25
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    
    for idx, (q_wxyz, t) in pose_params.items():
        T = np.eye(4)
        T[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
        T[:3, 3] = t
        poses[idx] = T
    for tid, _ in relevant_tracks:
        tracks[tid]['pt3d'] = track_params[tid]

# --- Path Processor ---

class PathProcessor:
    def __init__(self, name, cfg, color):
        self.name = name
        self.cfg = cfg
        self.color = color
        self.path_tag = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        self.poses = {} # f_idx -> T_CiC0 (Camera-from-C0)
        self.tracks = {} # tid -> {'obs': {f_idx: uv}, 'pt3d': xyz}
        self.keyframes = []
        self.next_tid = 0
        self.errors_t = []
        self.errors_R = []
        self.traj_obj_est_C = [] # Camera position in Object frame
        self.traj_world_est_O = [] # Object position in World frame
        self.traj_world_est_C = [] # Camera position in World frame
        self.last_pnp_data = None # Store {pts2d, inliers, tids} for viz
        self.K = None

    def reset_trajectories(self, T_C0O_gt, T_WC0_gt):
        self.traj_obj_est_C = []
        self.traj_world_est_O = []
        self.traj_world_est_C = []
        
        # Anchor at frame 0
        T_OC0_gt = np.linalg.inv(T_C0O_gt)
        self.traj_obj_est_C.append(T_OC0_gt[:3, 3])
        
        T_WO0_gt = T_WC0_gt @ T_C0O_gt
        self.traj_world_est_O.append(T_WO0_gt[:3, 3])

        # Initial Camera in World frame
        self.traj_world_est_C.append(T_WC0_gt[:3, 3])

    def add_new_points_from_depth(self, frame_idx, image, mask, depth, K, T_CiC0, align=False):
        if self.K is None: self.K = K
        pts2d = sample_grid_on_mask(mask, self.cfg.grid_spacing)
        
        if align and self.tracks:
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

        K_inv = np.linalg.inv(K)
        T_C0Ci = np.linalg.inv(T_CiC0)
        for uv in pts2d:
            d = depth[int(round(uv[1])), int(round(uv[0]))]
            if d <= 0.01: continue
            pt_Ci = (K_inv @ np.array([uv[0], uv[1], 1.0])) * d
            pt_C0 = (T_C0Ci[:3, :3] @ pt_Ci) + T_C0Ci[:3, 3]
            self.tracks[self.next_tid] = {'obs': {frame_idx: uv}, 'pt3d': pt_C0}
            self.next_tid += 1

    def step(self, frame_idx, image, mask, depth, K, flow_prev_curr, prev_idx, use_gpnp=False):
        if self.K is None: self.K = K
        # 1. Propagate points from prev_idx to frame_idx using flow
        if flow_prev_curr is not None:
            for tid, t in self.tracks.items():
                if prev_idx in t['obs']:
                    uv_prev = t['obs'][prev_idx]
                    delta = interpolate_flow(flow_prev_curr, uv_prev[None])[0]
                    uv_curr = uv_prev + delta
                    ix, iy = int(round(uv_curr[0])), int(round(uv_curr[1]))
                    if 0 <= ix < image.shape[1] and 0 <= iy < image.shape[0] and mask[iy, ix] > 0:
                        t['obs'][frame_idx] = uv_curr
        
        pts2d, pts3d, tids = [], [], []
        for tid, t in self.tracks.items():
            if frame_idx in t['obs']:
                pts2d.append(t['obs'][frame_idx])
                pts3d.append(t['pt3d'])
                tids.append(tid)
        
        if len(pts2d) < 10:
            self.poses[frame_idx] = self.poses.get(frame_idx-1, np.eye(4)).copy()
            self.last_pnp_data = None
            return False
        
        cam_dict = {'model': 'PINHOLE', 'width': image.shape[1], 'height': image.shape[0], 'params': [K[0,0], K[1,1], K[0,2], K[1,2]]}
        
        if not use_gpnp:
            res, info = poselib.estimate_absolute_pose(
                np.array(pts2d), np.array(pts3d), cam_dict, 
                {'max_reproj_error': self.cfg.ransac_thresh}, None
            )
            self.last_pnp_data = {'pts2d': np.array(pts2d), 'inliers': np.array(info['inliers']), 'tids': tids}
            T = np.eye(4)
            if res:
                T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix()
                T[:3, 3] = res.pose.t
                self.poses[frame_idx] = T
            else:
                self.poses[frame_idx] = self.poses.get(frame_idx-1, np.eye(4)).copy()
                return False
        else:
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
                # For gPnP, inliers are per-camera. Just take the last camera for viz
                self.last_pnp_data = {'pts2d': p2ds[-1], 'inliers': np.array(info['inliers'][-1]), 'tids': tids} # approximation
                if res:
                    T = np.eye(4); T[:3, :3] = R.from_quat([res.q[1], res.q[2], res.q[3], res.q[0]]).as_matrix(); T[:3, 3] = res.t
                    self.poses[frame_idx] = T
                else:
                    self.poses[frame_idx] = self.poses.get(frame_idx-1, np.eye(4)).copy()
                    return False
        return True

    def run_ba(self, K_dict):
        kf_indices = self.keyframes[-self.cfg.max_keyframes:]
        run_ba(kf_indices, self.poses, self.tracks, K_dict)

    def evaluate(self, idx, fd, T_C0O_gt):
        # T_C0O_gt: Object in Camera 0
        # self.poses[idx]: Camera i in Camera 0
        # Anchor to frame 0 GT to isolate drift (No Umeyama alignment used)
        T_CiO_est = self.poses[idx] @ T_C0O_gt
        
        # Camera center in object frame: -T_CiO_est[:3, :3].T @ T_CiO_est[:3, 3]
        T_OCi_est = np.linalg.inv(T_CiO_est)
        C_est = T_OCi_est[:3, 3]
        self.traj_obj_est_C.append(C_est)

        if fd["T_WO_gt"] is not None:
            T_CiO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"]
            T_OCi_gt = np.linalg.inv(T_CiO_gt)
            C_gt = T_OCi_gt[:3, 3]
            
            err_t = np.linalg.norm(C_est - C_gt)
            R_rel = T_CiO_est[:3, :3].T @ T_CiO_gt[:3, :3]
            tr = np.trace(R_rel)
            err_R = np.rad2deg(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))
            
            self.errors_t.append(err_t)
            self.errors_R.append(err_R)
            return err_t, err_R
        return None, None

    def log_rerun(self, idx, fd, T_C0O_gt, T_WC0_gt):
        # --- Object-Centric View ---
        # Fixed object at origin, show estimated camera moving
        T_CiO_est = self.poses[idx] @ T_C0O_gt
        T_OCi_est = np.linalg.inv(T_CiO_est)
        
        path_tag = self.path_tag
        rr.log(f"object/{path_tag}/camera_est", rr.Transform3D(mat3x3=T_OCi_est[:3, :3], translation=T_OCi_est[:3, 3]))
        rr.log(f"object/{path_tag}/traj", rr.LineStrips3D([np.array(self.traj_obj_est_C)], colors=[self.color], radii=0.001))
        
        # Log keyframes as full poses
        for kf_idx in self.keyframes:
            if kf_idx in self.poses:
                T_CkO_est = self.poses[kf_idx] @ T_C0O_gt
                T_OCk_est = np.linalg.inv(T_CkO_est)
                rr.log(f"object/{path_tag}/keyframes/{kf_idx}", rr.Transform3D(mat3x3=T_OCk_est[:3, :3], translation=T_OCk_est[:3, 3]))
                if self.K is not None:
                    rr.log(f"object/{path_tag}/keyframes/{kf_idx}", rr.Pinhole(image_from_camera=self.K, width=fd["image"].shape[1], height=fd["image"].shape[0]))
        
        # Log active points in Object frame
        active_pts3d_C0 = [t['pt3d'] for t in self.tracks.values() if idx in t['obs']]
        if active_pts3d_C0:
            T_OC0_gt = np.linalg.inv(T_C0O_gt)
            active_pts3d_O = [(T_OC0_gt[:3, :3] @ p + T_OC0_gt[:3, 3]) for p in active_pts3d_C0]
            rr.log(f"object/{path_tag}/points", rr.Points3D(active_pts3d_O, colors=[self.color], radii=0.002))
            
            # --- 2D Projections on Image ---
            # pt_Ci = T_CiO_est @ pt_O
            # uv = K @ pt_Ci
            K = fd["K"]
            pts3d_O = np.array(active_pts3d_O)
            pts3d_Ci = (T_CiO_est[:3, :3] @ pts3d_O.T).T + T_CiO_est[:3, 3]
            valid = pts3d_Ci[:, 2] > 0.01
            pts2d_proj_hom = (K @ pts3d_Ci[valid].T).T
            pts2d_proj = pts2d_proj_hom[:, :2] / pts2d_proj_hom[:, 2:3]
            rr.log(f"world/camera_gt/image/projections/{path_tag}", rr.Points2D(pts2d_proj, colors=[self.color], radii=1.5))

        # --- Inliers Visualization ---
        if self.last_pnp_data is not None:
            pts2d = self.last_pnp_data['pts2d']
            inliers = self.last_pnp_data['inliers']
            if len(pts2d) > 0:
                rr.log(f"world/camera_gt/image/inliers/{path_tag}", rr.Points2D(pts2d[inliers], colors=[self.color], radii=2.5))

        # --- World-Centric View ---
        # Show estimated object moving relative to GT camera
        T_WCi_gt = np.linalg.inv(fd["T_CW_gt"])
        T_WO_est = T_WCi_gt @ T_CiO_est
        self.traj_world_est_O.append(T_WO_est[:3, 3])
        
        rr.log(f"world/{path_tag}/object_est", rr.Transform3D(mat3x3=T_WO_est[:3, :3], translation=T_WO_est[:3, 3]))
        rr.log(f"world/{path_tag}/traj", rr.LineStrips3D([np.array(self.traj_world_est_O)], colors=[self.color], radii=0.001))
        
        # Show estimated camera in World frame (using T_WC0_gt as anchor)
        T_C0Ci_est = np.linalg.inv(self.poses[idx])
        T_WCi_est = T_WC0_gt @ T_C0Ci_est
        self.traj_world_est_C.append(T_WCi_est[:3, 3])
        
        rr.log(f"world/{path_tag}/camera_est", rr.Transform3D(mat3x3=T_WCi_est[:3, :3], translation=T_WCi_est[:3, 3]))
        rr.log(f"world/{path_tag}/camera_traj", rr.LineStrips3D([np.array(self.traj_world_est_C)], colors=[self.color], radii=0.001))
        if self.K is not None:
            rr.log(f"world/{path_tag}/camera_est", rr.Pinhole(image_from_camera=self.K, width=fd["image"].shape[1], height=fd["image"].shape[0]))

        if active_pts3d_C0:
            # Points attached to the estimated object
            rr.log(f"world/{path_tag}/object_est/points", rr.Points3D(active_pts3d_O, colors=[self.color], radii=0.002))

# --- Main Script ---

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 0
    n_frames: int = 100
    window_size: int = 5
    kf_every: int = 10
    max_keyframes: int = 20
    grid_spacing: int = 12
    ransac_thresh: float = 1.0
    g_window: int = 3
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    no_vis: bool = False

def load_frame(clip_path: Path, idx):
    stem = f"{idx:06d}"
    img_path = clip_path / "images" / f"{stem}.png"
    if not img_path.exists(): return None
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask_path = clip_path / "obj_masks" / f"{stem}.png"
    if not mask_path.exists(): return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    d_path = clip_path / "model_infer" / f"depth_{idx:05d}.npy"
    if not d_path.exists(): d_path = clip_path / "depth_dyn" / f"{stem}.npy"
    if not d_path.exists(): return None
    depth = np.load(d_path)
    
    if depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    K_path = clip_path / "intrinsics" / f"{stem}.npy"
    E_path = clip_path / "extrinsics" / f"{stem}.npy"
    if not K_path.exists() or not E_path.exists(): return None
    
    K = np.load(K_path)
    T_CW_gt = np.load(E_path)
    T_WO_gt = load_object_pose_world(clip_path, idx)
    return {"image": img, "mask": mask, "depth": depth, "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": idx}

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("exp_tracking_solver", spawn=False)
        if cfg.rerun_url:
            rr.connect_grpc(cfg.rerun_url)
        
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(name="World-Centric View", origin="/world"),
                    rrb.Spatial3DView(name="Object-Centric View", origin="/object"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Input Image", origin="/world/camera_gt/image"),
                    rrb.Spatial2DView(name="Motion Field", origin="/input/flow"),
                ),
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    clip_path = Path(cfg.data_root)
    f0_data = load_frame(clip_path, cfg.init_frame)
    if f0_data is None or f0_data["T_WO_gt"] is None:
        print("Failed to load initial frame or GT object pose")
        return

    T_C0O_gt = f0_data["T_CW_gt"] @ f0_data["T_WO_gt"]
    T_WC0_gt = np.linalg.inv(f0_data["T_CW_gt"])
    
    paths = [
        PathProcessor("Path A (Depth)", cfg, [255, 0, 0]),
        PathProcessor("Path B (RelPose)", cfg, [0, 255, 0]),
        PathProcessor("Path D (gPnP)", cfg, [0, 0, 255]),
        PathProcessor("Path E (e5p1)", cfg, [255, 255, 0])
    ]
    for p in paths:
        p.reset_trajectories(T_C0O_gt, T_WC0_gt)
    
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    K_dict = {}
    
    # --- Initialization Window ---
    frames_buffer = []
    for i in range(cfg.window_size):
        fd = load_frame(clip_path, cfg.init_frame + i)
        if fd is None: break
        frames_buffer.append(fd)
        K_dict[fd["frame_idx"]] = fd["K"]
    
    if len(frames_buffer) < cfg.window_size: return

    # Init Path A, D, E (Absolute PnP variants)
    for p in [paths[0], paths[2], paths[3]]:
        f0 = frames_buffer[0]
        p.poses[f0["frame_idx"]] = np.eye(4)
        p.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], f0["depth"], f0["K"], p.poses[f0["frame_idx"]])
        p.keyframes.append(f0["frame_idx"])
        for i in range(1, cfg.window_size):
            f_prev, f_curr = frames_buffer[i-1], frames_buffer[i]
            flow = dis.calc(cv2.cvtColor(f_prev["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
            p.step(f_curr["frame_idx"], f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow, f_prev["frame_idx"], use_gpnp=(p.name == "Path D (gPnP)"))
        p.run_ba(K_dict)

    # Init Path B (Relative Pose)
    pb = paths[1]
    f0, f_last = frames_buffer[0], frames_buffer[cfg.window_size-1]
    pb.poses[f0["frame_idx"]] = np.eye(4)
    pts0 = sample_grid_on_mask(f0["mask"], cfg.grid_spacing)
    flow0_last = dis.calc(cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_last["image"], cv2.COLOR_RGB2GRAY), None)
    pts_last = pts0 + interpolate_flow(flow0_last, pts0)
    d0 = np.array([f0["depth"][int(round(p[1])), int(round(p[0]))] for p in pts0])
    d_last = np.array([f_last["depth"][int(round(p[1])), int(round(p[0]))] for p in pts_last])
    cam0 = {'model': 'PINHOLE', 'width': f0["image"].shape[1], 'height': f0["image"].shape[0], 'params': [f0["K"][0,0], f0["K"][1,1], f0["K"][0,2], f0["K"][1,2]]}
    cam_last = {'model': 'PINHOLE', 'width': f_last["image"].shape[1], 'height': f_last["image"].shape[0], 'params': [f_last["K"][0,0], f_last["K"][1,1], f_last["K"][0,2], f_last["K"][1,2]]}
    res_rel, _ = poselib.estimate_monodepth_relative_pose(pts0, pts_last, d0, d_last, cam0, cam_last, {'max_reproj_error': cfg.ransac_thresh})
    if res_rel:
        T_L0 = np.eye(4); T_L0[:3, :3] = R.from_quat([res_rel.pose.q[1], res_rel.pose.q[2], res_rel.pose.q[3], res_rel.pose.q[0]]).as_matrix(); T_L0[:3, 3] = res_rel.pose.t
        pb.poses[f_last["frame_idx"]] = T_L0
        pts3d = triangulate_linear(f0["K"] @ np.eye(3, 4), f_last["K"] @ T_L0[:3, :], pts0, pts_last)
        for i, p in enumerate(pts3d): pb.tracks[i] = {'obs': {f0["frame_idx"]: pts0[i], f_last["frame_idx"]: pts_last[i]}, 'pt3d': p}
        pb.next_tid = len(pts3d); pb.keyframes.append(f0["frame_idx"])
        for i in range(1, cfg.window_size - 1):
            f_curr = frames_buffer[i]
            flow0_i = dis.calc(cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
            pts_i = pts0 + interpolate_flow(flow0_i, pts0)
            for j, p in enumerate(pts_i): pb.tracks[j]['obs'][f_curr["frame_idx"]] = p
            pb.step(f_curr["frame_idx"], f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], None, f0["frame_idx"])
        pb.run_ba(K_dict)
    else:
        pb.poses[f0["frame_idx"]] = np.eye(4)
        pb.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], f0["depth"], f0["K"], pb.poses[f0["frame_idx"]])
        pb.keyframes.append(f0["frame_idx"])
        for i in range(1, cfg.window_size):
            f_prev, f_curr = frames_buffer[i-1], frames_buffer[i]
            flow = dis.calc(cv2.cvtColor(f_prev["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
            pb.step(f_curr["frame_idx"], f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow, f_prev["frame_idx"])
        pb.run_ba(K_dict)

    # --- Main Loop ---
    prev_f = frames_buffer[-1]
    
    # Initialize GT trajectories with frame 0
    T_OC0_gt = np.linalg.inv(T_C0O_gt)
    traj_obj_gt_C = [T_OC0_gt[:3, 3]]
    traj_world_gt_O = [f0_data["T_WO_gt"][:3, 3]]
    
    for i in tqdm(range(cfg.window_size, cfg.n_frames), desc="Processing"):
        fd = load_frame(clip_path, cfg.init_frame + i)
        if fd is None: break
        idx = fd["frame_idx"]
        K_dict[idx] = fd["K"]
        
        # DIS Flow for visualization and tracking
        gray_prev = cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY)
        flow = dis.calc(gray_prev, gray_curr, None)
        
        # Log all DIS flow matches for visibility
        if not cfg.no_vis:
            rr.set_time("frame", sequence=idx)
            # Sample a dense grid of flow matches in the object area
            h, w = fd["image"].shape[:2]
            yy, xx = np.mgrid[5:h:8, 5:w:8] # Denser grid
            pts_grid = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
            
            # Filter by mask
            mask_prev = prev_f["mask"]
            ix, iy = np.round(pts_grid[:, 0]).astype(int), np.round(pts_grid[:, 1]).astype(int)
            valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            ix, iy, pts_grid = ix[valid], iy[valid], pts_grid[valid]
            pts_prev = pts_grid[mask_prev[iy, ix] > 0]
            
            if len(pts_prev) > 0:
                delta = interpolate_flow(flow, pts_prev)
                pts_curr = pts_prev + delta
                
                # Color encoding based on flow direction and magnitude
                mag = np.linalg.norm(delta, axis=1)
                ang = np.arctan2(delta[:, 1], delta[:, 0])
                hsv = np.zeros((1, len(delta), 3), dtype=np.uint8)
                hsv[0, :, 0] = (((ang + np.pi) / (2 * np.pi) * 179)).astype(np.uint8)
                hsv[0, :, 1] = np.clip(mag * 20, 100, 255).astype(np.uint8) # Saturation for speed
                hsv[0, :, 2] = 255 # Max brightness
                colors = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0]
                
                # Log to dedicated flow view
                matches = np.stack([pts_prev, pts_curr], axis=1) # [N, 2, 2]
                rr.log("input/flow/image", rr.Image(fd["image"]))
                rr.log("input/flow/matches", rr.LineStrips2D(matches, colors=colors, radii=0.6))
            else:
                rr.log("input/flow/image", rr.Image(fd["image"]))

        # GT Trajectory for comparison (moved down to avoid double logging)
        # if fd["T_WO_gt"] is not None:
        #     T_CiO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"]
        #     T_OCi_gt = np.linalg.inv(T_CiO_gt)
        #     traj_obj_gt_C.append(T_OCi_gt[:3, 3])

        if not cfg.no_vis:
            rr.set_time("frame", sequence=idx)
            
            # Log GT
            T_WCi_gt = np.linalg.inv(fd["T_CW_gt"])
            rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WCi_gt[:3, :3], translation=T_WCi_gt[:3, 3]))
            rr.log("world/camera_gt", rr.Pinhole(image_from_camera=fd["K"], width=fd["image"].shape[1], height=fd["image"].shape[0]))
            rr.log("world/camera_gt/image", rr.Image(fd["image"]))

            if fd["T_WO_gt"] is not None:
                traj_world_gt_O.append(fd["T_WO_gt"][:3, 3])
                rr.log("world/object_gt", rr.Transform3D(mat3x3=fd["T_WO_gt"][:3, :3], translation=fd["T_WO_gt"][:3, 3]))
                rr.log("world/traj_gt", rr.LineStrips3D([np.array(traj_world_gt_O)], colors=[[0, 255, 255]], radii=0.001))
                
                T_OCi_gt = np.linalg.inv(fd["T_CW_gt"] @ fd["T_WO_gt"])
                traj_obj_gt_C.append(T_OCi_gt[:3, 3])
                rr.log("object/camera_gt", rr.Transform3D(mat3x3=T_OCi_gt[:3, :3], translation=T_OCi_gt[:3, 3]))
                rr.log("object/traj_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[255, 255, 255]], radii=0.001))

        for p in paths:
            use_gpnp = (p.name == "Path D (gPnP)")
            success = p.step(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], flow, prev_f["frame_idx"], use_gpnp=use_gpnp)
            if success:
                err_t, err_R = p.evaluate(idx, fd, T_C0O_gt)
                if not cfg.no_vis:
                    p.log_rerun(idx, fd, T_C0O_gt, T_WC0_gt)
                    if err_t is not None:
                        rr.log(f"metrics/{p.path_tag}/err_t", rr.Scalars(err_t))
                        rr.log(f"metrics/{p.path_tag}/err_R", rr.Scalars(err_R))

        # Keyframe Logic
        if i % cfg.kf_every == 0:
            for p in paths:
                p.keyframes.append(idx)
                if p.name == "Path A (Depth)":
                    p.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], p.poses[idx], align=True)
                else:
                    p.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], p.poses[idx], align=False)
                p.run_ba(K_dict)
            
        prev_f = fd

    print("\nFinal Results:")
    for p in paths:
        if p.errors_t:
            ate = np.sqrt(np.mean(np.square(p.errors_t)))
            mean_R = np.mean(p.errors_R)
            print(f"  {p.name}: ATE={ate:.4f}m, Mean R Err={mean_R:.2f} deg")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
