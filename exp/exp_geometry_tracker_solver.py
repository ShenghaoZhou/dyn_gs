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
from typing import List, Dict, Optional

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

def detect_features_on_mask(image, mask, nfeatures=2000, feature_type="orb"):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    if feature_type == "orb":
        detector = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=5)
        kpts = detector.detect(gray, mask=mask)
    elif feature_type == "gftt":
        detector = cv2.GFTTDetector_create(maxCorners=nfeatures, minDistance=1, qualityLevel=0.01)
        kpts = detector.detect(gray, mask=mask)
    else: return np.array([], dtype=np.float32).reshape(0, 2)
    if not kpts: return np.array([], dtype=np.float32).reshape(0, 2)
    pts = np.array([kp.pt for kp in kpts], dtype=np.float32)
    return pts

# --- Geometric Tracker ---

class GeometricTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.poses = {} # f_idx -> T_CiO
        self.tracks = {} # tid -> {'obs': {f_idx: uv}, 'pt3d': xyz_O}
        self.keyframes = []
        self.next_tid = 0
        self.K_dict = {}
        self.triangulated_tids = set()

    def add_new_points_from_depth(self, frame_idx, image, mask, depth, K, T_CiO, align=False):
        self.K_dict[frame_idx] = K
        mask_work = mask.copy()
        active_uvs = [t['obs'][frame_idx] for t in self.tracks.values() if frame_idx in t['obs']]
        for uv in active_uvs:
            cv2.circle(mask_work, (int(round(uv[0])), int(round(uv[1]))), 5, 0, -1)

        feature_type = getattr(self.cfg, "feature_type", "grid")
        n_to_detect = getattr(self.cfg, "n_features", 500)
        
        if feature_type in ["orb", "gftt"]:
            pts2d = detect_features_on_mask(image, mask_work, nfeatures=n_to_detect, feature_type=feature_type)
        else:
            pts2d = sample_grid_on_mask(mask_work, self.cfg.grid_spacing)
            
        K_inv = np.linalg.inv(K)
        T_OCi = np.linalg.inv(T_CiO)
        for uv in pts2d:
            d = depth[int(round(uv[1])), int(round(uv[0]))]
            if d <= 0.01: continue
            pt_Ci = (K_inv @ np.array([uv[0], uv[1], 1.0])) * d
            pt_O = (T_OCi[:3, :3] @ pt_Ci) + T_OCi[:3, 3]
            self.tracks[self.next_tid] = {'obs': {frame_idx: uv}, 'pt3d': pt_O}
            self.next_tid += 1

    def match_projections(self, frame_idx, image, mask, K, T_CiO):
        h, w = image.shape[:2]
        P = K @ T_CiO[:3, :]
        untracked_tids = [tid for tid, t in self.tracks.items() if frame_idx not in t['obs']]
        matched_count = 0
        for tid in untracked_tids:
            t = self.tracks[tid]
            p_hom = P @ np.append(t['pt3d'], 1.0)
            if p_hom[2] <= 0.01: continue
            uv_proj = p_hom[:2] / p_hom[2]
            ix, iy = int(round(uv_proj[0])), int(round(uv_proj[1]))
            if 0 <= ix < w and 0 <= iy < h and mask[iy, ix] > 0:
                t['obs'][frame_idx] = uv_proj
                matched_count += 1
        return matched_count

    def refine_pose(self, frame_idx, K, inlier_mask=None, tids_pnp=None):
        pts2d, pts3d = [], []
        if inlier_mask is not None and tids_pnp is not None:
            for i, ok in enumerate(inlier_mask):
                if ok:
                    tid = tids_pnp[i]
                    if tid in self.tracks and frame_idx in self.tracks[tid]['obs']:
                        pts2d.append(self.tracks[tid]['obs'][frame_idx])
                        pts3d.append(self.tracks[tid]['pt3d'])
        else:
            for tid, t in self.tracks.items():
                if frame_idx in t['obs']:
                    pts2d.append(t['obs'][frame_idx]); pts3d.append(t['pt3d'])
        
        if len(pts2d) < 10: return
        
        prob = pyceres.Problem()
        loss = pyceres.HuberLoss(1.0)
        T = self.poses[frame_idx].copy()
        q = R.from_matrix(T[:3, :3]).as_quat()
        q_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
        t_vec = T[:3, 3].copy().astype(np.float64)
        cam_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float64)
        
        for i in range(len(pts2d)):
            cost = pycolmap.cost_functions.ReprojErrorCost('PINHOLE', pts2d[i].astype(np.float64))
            # We fix point and intrinsics, only optimize pose
            pt3d = pts3d[i].copy().astype(np.float64)
            prob.add_residual_block(cost, loss, [q_wxyz, t_vec, pt3d, cam_params])
            prob.set_parameter_block_constant(pt3d)
            
        prob.set_parameter_block_constant(cam_params)
        prob.set_manifold(q_wxyz, pyceres.EigenQuaternionManifold())
        
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        options.max_num_iterations = 10
        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        
        if summary.final_cost < summary.initial_cost:
            T_ref = np.eye(4)
            T_ref[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
            T_ref[:3, 3] = t_vec
            self.poses[frame_idx] = T_ref

    def triangulate_tracks(self, frame_idx, thresh=5):
        for tid, t in self.tracks.items():
            if len(t['obs']) >= thresh and tid not in self.triangulated_tids:
                obs = t['obs']
                f_indices = sorted(list(obs.keys()))
                if len(f_indices) < 2: continue
                # Check baseline
                T1, T2 = self.poses.get(f_indices[0]), self.poses.get(f_indices[-1])
                if T1 is None or T2 is None: continue
                if np.linalg.norm(T1[:3, 3] - T2[:3, 3]) < 0.005: continue
                
                prob = pyceres.Problem()
                loss = pyceres.HuberLoss(1.0)
                pt3d = t['pt3d'].copy().astype(np.float64)
                for f_idx, uv in obs.items():
                    if f_idx not in self.poses: continue
                    T = self.poses[f_idx]
                    q = R.from_matrix(T[:3, :3]).as_quat()
                    qwxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
                    tvec = T[:3, 3].astype(np.float64)
                    K = self.K_dict[f_idx]
                    cam_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float64)
                    cost = pycolmap.cost_functions.ReprojErrorCost('PINHOLE', uv.astype(np.float64))
                    prob.add_residual_block(cost, loss, [qwxyz, tvec, pt3d, cam_params])
                    prob.set_parameter_block_constant(qwxyz); prob.set_parameter_block_constant(tvec); prob.set_parameter_block_constant(cam_params)
                
                options = pyceres.SolverOptions()
                options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
                pyceres.solve(options, prob, pyceres.SolverSummary())
                t['pt3d'] = pt3d
                self.triangulated_tids.add(tid)

    def step_informed(self, frame_idx, image, mask, depth, K, T_guess, flow_prev_curr=None, pose_method='depth+pnp', prev_data=None):
        self.K_dict[frame_idx] = K
        if flow_prev_curr is not None:
            prev_idx = frame_idx - 1
            for tid, t in self.tracks.items():
                if prev_idx in t['obs']:
                    uv_prev = t['obs'][prev_idx]
                    delta = interpolate_flow(flow_prev_curr, uv_prev[None])[0]
                    uv_curr = uv_prev + delta
                    ix, iy = int(round(uv_curr[0])), int(round(uv_curr[1]))
                    if 0 <= ix < image.shape[1] and 0 <= iy < image.shape[0] and mask[iy, ix] > 0:
                        if T_guess is not None and getattr(self.cfg, "use_informed_filtering", True):
                            p_Ci = T_guess[:3, :3] @ t['pt3d'] + T_guess[:3, 3]
                            if p_Ci[2] > 0.01:
                                uv_proj = K @ (p_Ci / p_Ci[2])
                                if np.linalg.norm(uv_curr - uv_proj[:2]) > getattr(self.cfg, "informed_thresh", 10.0):
                                    continue
                        t['obs'][frame_idx] = uv_curr
        
        pts2d, pts3d, tids = [], [], []
        for tid, t in self.tracks.items():
            if frame_idx in t['obs']:
                pts2d.append(t['obs'][frame_idx]); pts3d.append(t['pt3d']); tids.append(tid)
        
        if len(pts2d) < 10:
            self.poses[frame_idx] = T_guess if T_guess is not None else self.poses.get(frame_idx-1, np.eye(4))
            return False
        
        cam_dict = {'model': 'PINHOLE', 'width': image.shape[1], 'height': image.shape[0], 'params': [K[0,0], K[1,1], K[0,2], K[1,2]]}
        tids_pnp = np.array(tids)
        T = None
        info = {'inliers': []}

        if pose_method == 'depth+pnp' or (pose_method == 'relpose' and frame_idx == self.keyframes[0]):
            res, info = poselib.estimate_absolute_pose(np.array(pts2d), np.array(pts3d), cam_dict, {'max_reproj_error': self.cfg.ransac_thresh}, None)
            if res:
                T = np.eye(4); T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix(); T[:3, 3] = res.pose.t
        
        elif pose_method == 'gpnp':
            camera_ext, camera_dicts, p2ds, p3ds = [], [], [], []
            g_window = getattr(self.cfg, "g_window", 3)
            window_frames = [f for f in range(frame_idx - g_window + 1, frame_idx + 1) if f in self.poses or f == frame_idx]
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
                if res:
                    T = np.eye(4); T[:3, :3] = R.from_quat([res.q[1], res.q[2], res.q[3], res.q[0]]).as_matrix(); T[:3, 3] = res.t

        elif (pose_method == 'relpose' or pose_method == 'e5p1') and prev_data is not None:
            # Relative Pose between prev and curr
            prev_idx = frame_idx - 1
            pts_prev, pts_curr = [], []
            for tid, t in self.tracks.items():
                if prev_idx in t['obs'] and frame_idx in t['obs']:
                    pts_prev.append(t['obs'][prev_idx]); pts_curr.append(t['obs'][frame_idx])
            
            if len(pts_prev) >= 10:
                pts_prev, pts_curr = np.array(pts_prev), np.array(pts_curr)
                d_prev = np.array([prev_data['depth'][int(round(p[1])), int(round(p[0]))] for p in pts_prev])
                d_curr = np.array([depth[int(round(p[1])), int(round(p[0]))] for p in pts_curr])
                cam_prev = {'model': 'PINHOLE', 'width': image.shape[1], 'height': image.shape[0], 'params': [self.K_dict[prev_idx][0,0], self.K_dict[prev_idx][1,1], self.K_dict[prev_idx][0,2], self.K_dict[prev_idx][1,2]]}
                res, info = poselib.estimate_monodepth_relative_pose(pts_prev, pts_curr, d_prev, d_curr, cam_prev, cam_dict, {'max_reproj_error': self.cfg.ransac_thresh})
                if res:
                    T_curr_prev = np.eye(4); T_curr_prev[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix(); T_curr_prev[:3, 3] = res.pose.t
                    T = T_curr_prev @ self.poses[prev_idx]

        if T is not None:
            if T_guess is not None:
                diff_t = np.linalg.norm(T[:3, 3] - T_guess[:3, 3])
                if diff_t > getattr(self.cfg, "max_pose_jump", 0.2):
                    T = T_guess.copy()
            self.poses[frame_idx] = T
            # Refine
            if getattr(self.cfg, "do_refine", False):
                self.refine_pose(frame_idx, K, inlier_mask=info.get('inliers'), tids_pnp=tids_pnp)
        else:
            self.poses[frame_idx] = T_guess if T_guess is not None else self.poses.get(frame_idx-1, np.eye(4))
            return False
        return True

    def run_ba(self, K_dict):
        kf_indices = self.keyframes[-self.cfg.max_keyframes:]
        if getattr(self.cfg, "triangulate", True):
            self.triangulate_tracks(kf_indices[-1])
        run_ba(kf_indices, self.poses, self.tracks, K_dict)

# --- Path Processors ---

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

    def step(self, frame_idx, image, mask, depth, K, flow_prev_curr, prev_idx, T_guess=None, pose_method='depth+pnp', prev_data=None):
        if self.K is None: self.K = K
        success = self.tracker.step_informed(frame_idx, image, mask, depth, K, T_guess, flow_prev_curr=flow_prev_curr, pose_method=pose_method, prev_data=prev_data)
        
        if success:
            self.poses[frame_idx] = self.tracker.poses[frame_idx]
            
            # Geo-specific recovery: projections
            matched = self.tracker.match_projections(frame_idx, image, mask, K, self.poses[frame_idx])
            
            # Hybrid Keyframe trigger
            is_kf = False
            if frame_idx % self.cfg.kf_every == 0:
                is_kf = True
            else:
                # Flow-based drift trigger
                pass
            
            if is_kf:
                self.keyframes.append(frame_idx)
                self.tracker.keyframes.append(frame_idx)
                # Re-seed
                self.tracker.add_new_points_from_depth(frame_idx, image, mask, depth, K, self.poses[frame_idx], align=True)
                # Global optimization
                self.tracker.run_ba(self.tracker.K_dict)
        else:
            self.poses[frame_idx] = self.tracker.poses[frame_idx]
        return success

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

        # Show estimated object moving relative to GT camera
        T_WCi_gt = np.linalg.inv(fd["T_CW_gt"])
        T_WO_est = T_WCi_gt @ T_CiO_est
        self.traj_world_est_O.append(T_WO_est[:3, 3])
        
        rr.log(f"world/{path_tag}/object_est", rr.Transform3D(mat3x3=T_WO_est[:3, :3], translation=T_WO_est[:3, 3]))
        rr.log(f"world/{path_tag}/traj", rr.LineStrips3D([np.array(self.traj_world_est_O)], colors=[self.color], radii=0.001))

        if active_pts3d_C0:
            # Points attached to the estimated object
            rr.log(f"world/{path_tag}/object_est/points", rr.Points3D(active_pts3d_O, colors=[self.color], radii=0.002))

class GeoPathProcessor(PathProcessor):
    def __init__(self, name, cfg, color):
        super().__init__(name, cfg, color)
        self.tracker = GeometricTracker(cfg)
        self.history_T_WO_est = []

    def reset_trajectories(self, T_C0O_gt, T_WC0_gt):
        super().reset_trajectories(T_C0O_gt, T_WC0_gt)
        self.tracker.poses[0] = T_C0O_gt # In GeoTracker, f0 pose is T_C0O
        # history_T_WO_est initialized in main

    def add_new_points_from_depth(self, frame_idx, image, mask, depth, K, T_CiC0, align=False):
        # For Path G, T_CiC0 is not used directly, we use tracker.poses[frame_idx] which is T_CiO
        T_CiO = self.tracker.poses[frame_idx]
        self.tracker.add_new_points_from_depth(frame_idx, image, mask, depth, K, T_CiO, align=align)
        # Sync tracks for visualization
        self.tracks = self.tracker.tracks

    def step(self, frame_idx, image, mask, depth, K, flow_prev_curr, prev_idx, T_guess=None, pose_method='depth+pnp', prev_data=None):
        self.K = K
        self.tracker.K_dict[frame_idx] = K
        
        # Occlusion Recovery
        if T_guess is not None:
            self.tracker.match_projections(frame_idx, image, mask, K, T_guess)
            
        success = self.tracker.step_informed(frame_idx, image, mask, depth, K, T_guess, flow_prev_curr=flow_prev_curr, pose_method=pose_method, prev_data=prev_data)
        
        # Update inherited state for evaluation/logging
        # We need self.poses[frame_idx] to be T_CiC0
        # T_CiO = T_CiC0 @ T_C0O  =>  T_CiC0 = T_CiO @ inv(T_C0O)
        T_CiO = self.tracker.poses[frame_idx]
        T_C0O_inv = np.linalg.inv(self.tracker.poses[self.keyframes[0]])
        self.poses[frame_idx] = T_CiO @ T_C0O_inv
        
        # Sync tracks
        self.tracks = self.tracker.tracks
        
        # Update inliers for viz
        pts2d, tids = [], []
        for tid, t in self.tracks.items():
            if frame_idx in t['obs']:
                pts2d.append(t['obs'][frame_idx]); tids.append(tid)
        if pts2d:
            self.last_pnp_data = {'pts2d': np.array(pts2d), 'inliers': np.arange(len(pts2d)), 'tids': tids}
        else:
            self.last_pnp_data = None
            
        return success

    def run_ba(self, K_dict):
        self.tracker.keyframes = self.keyframes
        self.tracker.run_ba(K_dict)
        # Sync back poses
        T_C0O_inv = np.linalg.inv(self.tracker.poses[self.keyframes[0]])
        for idx in self.keyframes:
            if idx in self.tracker.poses:
                self.poses[idx] = self.tracker.poses[idx] @ T_C0O_inv

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
    
    # GeoTracker Specific
    feature_type: str = "grid"
    n_features: int = 800
    use_informed_filtering: bool = True
    informed_thresh: float = 30.0 
    max_pose_jump: float = 2.0 # More permissive
    kf_disparity_thresh: float = 20.0
    kf_overlap_thresh: float = 0.6
    kf_min_features: int = 400
    kf_min_interval: int = 5
    
    # Advanced GeoTracker options
    do_refine: bool = False
    triangulate: bool = False
    report_interval: int = 30

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
        rr.init("GeoTrackerBenchmark", spawn=False)
        if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    all_results = []
    data_root = Path(cfg.data_root)
    if (data_root / "images").exists():
        res = run_clip(cfg, data_root)
        if res: all_results.append(res)
    else:
        clips = sorted(list(data_root.glob("clip-*")))
        if not clips:
            print(f"No clips found in {data_root}")
            return
        for clip in clips:
            res = run_clip(cfg, clip)
            if res: all_results.append(res)

    if all_results:
        print("\n" + "="*40)
        print("GLOBAL BENCHMARK SUMMARY")
        print("="*40)
        path_names = list(all_results[0].keys())
        for name in path_names:
            ates = [r[name]['ate'] for r in all_results if name in r]
            errs_R = [r[name]['mean_R'] for r in all_results if name in r]
            if ates:
                avg_ate = np.mean(ates)
                avg_R = np.mean(errs_R)
                print(f"{name}:")
                print(f"  Avg ATE: {avg_ate:.4f}m")
                print(f"  Avg R Err: {avg_R:.2f} deg")
                print(f"  (Processed {len(ates)} clips)")
        print("="*40)

def run_clip(cfg: Config, clip_path: Path) -> dict:
    print(f"\n>>> Running Benchmark on: {clip_path.name}")
    f0_data = load_frame(clip_path, cfg.init_frame)
    if f0_data is None or f0_data["T_WO_gt"] is None:
        print(f"Failed to load initial frame or GT object pose for {clip_path.name}")
        return

    T_C0O_gt = f0_data["T_CW_gt"] @ f0_data["T_WO_gt"]
    T_WC0_gt = np.linalg.inv(f0_data["T_CW_gt"])
    
    paths = [
        GeoPathProcessor("Path A (Geo+PnP)", cfg, [0, 0, 255]),
        GeoPathProcessor("Path B (Geo+Rel)", cfg, [255, 165, 0]),
        GeoPathProcessor("Path D (Geo+gPnP)", cfg, [0, 255, 255]),
        GeoPathProcessor("Path E (Geo+e5p1)", cfg, [255, 0, 0]),
        GeoPathProcessor("Path G (Geo+PnP+Inf)", cfg, [0, 255, 0])
    ]
    for p in paths:
        p.reset_trajectories(T_C0O_gt, T_WC0_gt)
    
    # Map method names
    method_map = {
        "Path A (Geo+PnP)": "depth+pnp",
        "Path B (Geo+Rel)": "relpose",
        "Path D (Geo+gPnP)": "gpnp",
        "Path E (Geo+e5p1)": "relpose",
        "Path G (Geo+PnP+Inf)": "depth+pnp"
    }
    
    for p in paths:
        if isinstance(p, GeoPathProcessor):
            p.history_T_WO_est = [f0_data["T_WO_gt"]]
    
    last_kf_grays = {p.name: cv2.cvtColor(f0_data["image"], cv2.COLOR_RGB2GRAY) for p in paths if isinstance(p, GeoPathProcessor)}
    
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

    # Init all Geo-paths
    for p in paths:
        f0 = frames_buffer[0]
        p.tracker.poses[f0["frame_idx"]] = T_C0O_gt
        p.tracker.keyframes.append(f0["frame_idx"])
        p.keyframes.append(f0["frame_idx"])
        p.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], f0["depth"], f0["K"], None)
            
        for i in range(1, cfg.window_size):
            f_prev, f_curr = frames_buffer[i-1], frames_buffer[i]
            flow = dis.calc(cv2.cvtColor(f_prev["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
            pose_method = method_map[p.name]
            p.step(f_curr["frame_idx"], f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow, f_prev["frame_idx"], pose_method=pose_method, prev_data=f_prev)
        p.run_ba(K_dict)

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
            pose_method = method_map[p.name]
            
            # Informed Motion Model Guess for Path G (and others if requested)
            T_guess = None
            if p.name == "Path G (Geo+PnP+Inf)":
                if len(p.history_T_WO_est) >= 2:
                    T_prev = p.history_T_WO_est[-1]
                    T_prev_prev = p.history_T_WO_est[-2]
                    T_WO_guess = T_prev @ np.linalg.inv(T_prev_prev) @ T_prev
                else:
                    T_WO_guess = p.history_T_WO_est[-1]
                T_guess = fd["T_CW_gt"] @ T_WO_guess
            else:
                # Relative guess (simpler previous pose guess for better stability)
                T_guess = p.poses.get(idx-1, np.eye(4)).copy()

            # Only use informed filtering for Path G which has a reliable GT-camera based guess
            p.tracker.cfg.use_informed_filtering = (p.name == "Path G (Geo+PnP+Inf)")
            success = p.step(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], flow, prev_f["frame_idx"], T_guess=T_guess, pose_method=pose_method, prev_data=prev_f)
            
            if success and p.name == "Path G (Geo+PnP+Inf)":
                # Update history for informed tracking
                T_WO_est = np.linalg.inv(fd["T_CW_gt"]) @ p.tracker.poses[idx]
                p.history_T_WO_est.append(T_WO_est)
            
            if success:
                err_t, err_R = p.evaluate(idx, fd, T_C0O_gt)
                if not cfg.no_vis:
                    p.log_rerun(idx, fd, T_C0O_gt, T_WC0_gt)
                    if err_t is not None:
                        rr.log(f"metrics/{p.path_tag}/err_t", rr.Scalars(err_t))
                        rr.log(f"metrics/{p.path_tag}/err_R", rr.Scalars(err_R))
            else:
                print(f"  {p.name} failed at frame {idx}")

        # Keyframe Logic
        for p in paths:
            is_kf = False
            if isinstance(p, GeoPathProcessor):
                # Hybrid Keyframe Selection for Path G
                last_kf_idx = p.keyframes[-1]
                frames_since_kf = idx - last_kf_idx
                
                # Geometric Disparity
                gray_kf = last_kf_grays[p.name]
                flow_kf_curr = dis.calc(gray_kf, gray_curr, None)
                mask_curr = fd["mask"] > 0
                if np.any(mask_curr):
                    disp_map = np.linalg.norm(flow_kf_curr, axis=-1)
                    avg_disparity = np.median(disp_map[mask_curr])
                else: avg_disparity = 0
                
                # Overlap
                active_tids = [tid for tid, t in p.tracks.items() if idx in t['obs']]
                if active_tids:
                    overlap_count = sum(1 for tid in active_tids if last_kf_idx in p.tracks[tid]['obs'])
                    overlap_ratio = overlap_count / len(active_tids)
                else: overlap_ratio = 0
                
                is_kf = (frames_since_kf >= cfg.kf_min_interval and (avg_disparity > cfg.kf_disparity_thresh or overlap_ratio < cfg.kf_overlap_thresh or len(active_tids) < cfg.kf_min_features)) or (frames_since_kf >= cfg.max_keyframes)
                if is_kf:
                    last_kf_grays[p.name] = gray_curr
            else:
                # Fixed Interval for Baselines
                is_kf = (i % cfg.kf_every == 0)

        pass # Keyframe logic is handled inside GeoPathProcessor.step
            
        if idx > cfg.init_frame and idx % cfg.report_interval == 0:
            print(f"  [Frame {idx}] Intermediate ATE:")
            for p in paths:
                if p.errors_t:
                    ate = np.sqrt(np.mean(np.square(p.errors_t)))
                    print(f"    {p.name}: {ate:.4f}m")
            
        prev_f = fd

    results = {}
    print("\nFinal Results:")
    for p in paths:
        if p.errors_t:
            ate = np.sqrt(np.mean(np.square(p.errors_t)))
            mean_R = np.mean(p.errors_R)
            print(f"  {p.name}: ATE={ate:.4f}m, Mean R Err={mean_R:.2f} deg")
            results[p.name] = {'ate': ate, 'mean_R': mean_R}
    return results

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
