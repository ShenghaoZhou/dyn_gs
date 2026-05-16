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
import matplotlib.pyplot as plt

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    n_frames: int = 30
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    
    # Tracking Parameters
    num_pts: int = 2000
    ransac_thresh: float = 1.0 # pixel threshold for RANSAC
    ba_window: int = 15 # Number of frames in BA window
    
    # Depth source
    depth_source: str = "DA3METRIC-LARGE"
    use_gt_depth: bool = False

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

def interpolate_flow(flow, pts):
    """Bilinear interpolation of flow at sub-pixel point positions."""
    x = pts[:, 0]
    y = pts[:, 1]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    h, w = flow.shape[:2]
    x0, x1 = np.clip(x0, 0, w-1), np.clip(x1, 0, w-1)
    y0, y1 = np.clip(y0, 0, h-1), np.clip(y1, 0, h-1)
    
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    
    f_p = (wa[:, None] * flow[y0, x0] + 
           wb[:, None] * flow[y1, x0] + 
           wc[:, None] * flow[y0, x1] + 
           wd[:, None] * flow[y1, x1])
    return f_p

def sample_mask_points(mask, num_pts):
    yy, xx = np.where(mask > 0)
    if len(xx) == 0:
        return np.array([])
    indices = np.random.choice(len(xx), min(num_pts, len(xx)), replace=False)
    return np.stack([xx[indices], yy[indices]], axis=-1).astype(np.float32)

def triangulate_linear(P1, P2, p1, p2):
    """Linear triangulation for N points."""
    N = p1.shape[0]
    pts3d = np.zeros((N, 3))
    for i in range(N):
        A = np.zeros((4, 4))
        A[0] = p1[i, 0] * P1[2] - P1[0]
        A[1] = p1[i, 1] * P1[2] - P1[1]
        A[2] = p2[i, 0] * P2[2] - P2[0]
        A[3] = p2[i, 1] * P2[2] - P2[1]
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        pts3d[i] = X[:3] / X[3]
    return pts3d

def main(cfg: Config):
    rr.init("exp_poselib_depth_seq", spawn=False)
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    data_dir = Path(cfg.data_root)
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)

    def load_frame_data(frame_idx):
        stem = f"{frame_idx:06d}"
        img_path = data_dir / "images" / f"{stem}.png"
        if not img_path.exists(): return None
        img = np.array(cv2.imread(str(img_path))[..., ::-1])
        mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        
        if cfg.use_gt_depth:
            depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
            if not depth_path.exists():
                depth_path = data_dir / "depth_cache" / cfg.depth_source / f"{stem}.npy"
        else:
            depth_path = data_dir / "depth_cache" / cfg.depth_source / f"{stem}.npy"
            
        if not depth_path.exists():
            print(f"Depth not found: {depth_path}")
            return None
        depth = np.load(depth_path)
        if depth.shape != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Always load true GT depth for visualization
        gt_depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
        if gt_depth_path.exists():
            depth_gt = np.load(gt_depth_path)
        else:
            depth_gt = depth

        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        T_WO_gt = load_object_pose_world(cfg.data_root, frame_idx)
        return {
            "image": img,
            "mask": mask,
            "depth": depth,
            "depth_gt": depth_gt,
            "K": K,
            "T_CW_gt": T_CW_gt,
            "T_WO_gt": T_WO_gt,
            "frame_idx": frame_idx
        }

    traj_est = []
    traj_gt = []
    poses_est = {} # frame_idx -> T_CiC0 (World-to-Camera)
    
    f0 = load_frame_data(cfg.init_frame)
    if f0 is None:
        print("Initial frame not found.")
        return

    H, W = f0["image"].shape[:2]
    T_C0W = np.eye(4)
    poses_est[f0["frame_idx"]] = T_C0W

    def get_gt_pose_WC(frame_data):
        """Returns Camera-to-World (T_WC) in the visualization world."""
        T_WO_curr = frame_data["T_WO_gt"]
        T_CW_gt = frame_data["T_CW_gt"]
        if T_WO_curr is None or f0["T_WO_gt"] is None:
            return np.linalg.inv(T_CW_gt)
        # T_CO = T_CW_gt @ T_WO_curr
        T_C_O = T_CW_gt @ T_WO_curr
        # T_WC_viz = T_WO_f0 @ T_OC
        T_WC = f0["T_WO_gt"] @ np.linalg.inv(T_C_O)
        return T_WC

    T_WC0_viz = get_gt_pose_WC(f0)
    T_C0W_viz = np.linalg.inv(T_WC0_viz)
    
    def log_frame(fd, T_CiW, point_cloud_xyz=None, point_cloud_rgb=None):
        rr.set_time("frame", sequence=fd["frame_idx"])
        
        # GT Pose (Visualization World)
        T_WCi_gt = get_gt_pose_WC(fd)
        T_CiW_gt = np.linalg.inv(T_WCi_gt)
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WCi_gt[:3, :3], translation=T_WCi_gt[:3, 3]))
        rr.log("world/camera_gt/image", rr.Image(fd["image"]))
        
        # GT Points
        sample_step = 4
        grid_y, grid_x = np.meshgrid(np.arange(0, H, sample_step), np.arange(0, W, sample_step), indexing='ij')
        u_gt, v_gt = grid_x.flatten(), grid_y.flatten()
        z_gt = fd["depth_gt"][v_gt, u_gt].flatten()
        mask_gt = fd["mask"][v_gt, u_gt].flatten() > 0
        valid_gt = mask_gt & (z_gt > 0.01)
        u_gt, v_gt, z_gt = u_gt[valid_gt], v_gt[valid_gt], z_gt[valid_gt]
        K_inv = np.linalg.inv(fd["K"])
        pts2d_homog = np.stack([u_gt, v_gt, np.ones_like(u_gt)], axis=1)
        pts_Ci = (K_inv @ pts2d_homog.T).T * z_gt[:, None]
        pts_viz_gt = (T_WCi_gt[:3, :3] @ pts_Ci.T).T + T_WCi_gt[:3, 3]
        rr.log("world/points_gt", rr.Points3D(pts_viz_gt, colors=fd["image"][v_gt, u_gt], radii=0.001))

        # Estimated Pose
        if T_CiW is not None:
            # T_CiW is World-to-Camera (relative to f0)
            # T_WCi_est = T_WC0_viz @ T_C0Ci
            T_CiW_viz_est = T_CiW @ T_C0W_viz
            T_WCi_viz_est = np.linalg.inv(T_CiW_viz_est)
            
            rr.log("world/camera_est", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
            rr.log("world/camera_est", rr.Transform3D(mat3x3=T_WCi_viz_est[:3, :3], translation=T_WCi_viz_est[:3, 3]))
            
            traj_gt.append(T_WCi_gt[:3, 3])
            traj_est.append(T_WCi_viz_est[:3, 3])
            
            rr.log("world/trajectories/gt", rr.LineStrips3D([np.array(traj_gt)], colors=[[0, 255, 0]], radii=0.002))
            rr.log("world/trajectories/est", rr.LineStrips3D([np.array(traj_est)], colors=[[0, 0, 255]], radii=0.002))

        if point_cloud_xyz is not None:
            pts_viz_est = (T_WC0_viz[:3, :3] @ point_cloud_xyz.T).T + T_WC0_viz[:3, 3]
            rr.log("world/points_est", rr.Points3D(pts_viz_est, colors=point_cloud_rgb, radii=0.001))

    log_frame(f0, T_C0W)
    
    prev_f = f0
    point_cloud_xyz = None
    point_cloud_rgb = None
    
    current_tracks = [] # list of dicts { 'pt3d': xyz, 'color': rgb, 'obs': {frame_idx: uv}, 'id': track_id }
    next_track_id = 0

    def run_window_ba(frame_indices, poses, tracks):
        if len(frame_indices) < 2: return
        prob = pyceres.Problem()
        loss = pyceres.HuberLoss(1.0)
        pose_params = {}
        for idx in frame_indices:
            T = poses[idx]
            q = R.from_matrix(T[:3, :3]).as_quat()
            q_wxyz = np.array([q[3], q[0], q[1], q[2]])
            t = T[:3, 3].copy()
            pose_params[idx] = (q_wxyz, t)
        track_params = {}
        relevant_tracks = []
        for track in tracks:
            win_obs = {f_idx: uv for f_idx, uv in track['obs'].items() if f_idx in frame_indices}
            if len(win_obs) >= 2:
                track_params[track['id']] = track['pt3d'].copy()
                relevant_tracks.append((track, win_obs))
        if len(relevant_tracks) == 0: return
        f_ref = load_frame_data(frame_indices[0])
        K = f_ref["K"]
        cam_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
        added_poses = set()
        for track, win_obs in relevant_tracks:
            pt3d = track_params[track['id']]
            for f_idx, uv in win_obs.items():
                q_wxyz, t = pose_params[f_idx]
                cost = pycolmap.cost_functions.ReprojErrorCost('PINHOLE', uv)
                prob.add_residual_block(cost, loss, [q_wxyz, t, pt3d, cam_params])
                added_poses.add(f_idx)
        ref_idx = frame_indices[0]
        if ref_idx in added_poses:
            prob.set_parameter_block_constant(pose_params[ref_idx][0])
            prob.set_parameter_block_constant(pose_params[ref_idx][1])
        prob.set_parameter_block_constant(cam_params)
        quat_manifold = pyceres.EigenQuaternionManifold()
        for idx in added_poses:
            q_wxyz, t = pose_params[idx]
            if not prob.is_parameter_block_constant(q_wxyz):
                prob.set_manifold(q_wxyz, quat_manifold)
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
        options.max_num_iterations = 10
        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        for idx, (q_wxyz, t) in pose_params.items():
            T = np.eye(4)
            T[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
            T[:3, 3] = t
            poses[idx] = T
        for track, win_obs in relevant_tracks:
            track['pt3d'] = track_params[track['id']]
    
    for i in tqdm(range(1, cfg.n_frames)):
        curr_frame_idx = cfg.init_frame + i
        f_curr = load_frame_data(curr_frame_idx)
        if f_curr is None: break
        
        # 1. Compute Relative Pose (prev -> curr)
        gray_prev = cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY)
        flow = dis.calc(gray_prev, gray_curr, None)
        
        pts_prev = sample_mask_points(prev_f["mask"], cfg.num_pts)
        delta = interpolate_flow(flow, pts_prev)
        pts_curr = pts_prev + delta
        
        # Filter by mask and depth
        u_c, v_c = np.round(pts_curr[:, 0]).astype(int), np.round(pts_curr[:, 1]).astype(int)
        valid = (u_c >= 0) & (u_c < W) & (v_c >= 0) & (v_c < H)
        if np.any(valid):
            valid[valid] &= (f_curr["mask"][v_c[valid], u_c[valid]] > 0)
        
        pts_prev, pts_curr = pts_prev[valid], pts_curr[valid]
        u_p, v_p = np.round(pts_prev[:, 0]).astype(int), np.round(pts_prev[:, 1]).astype(int)
        u_c, v_c = np.round(pts_curr[:, 0]).astype(int), np.round(pts_curr[:, 1]).astype(int)
        
        d_p = prev_f["depth"][v_p, u_p].astype(np.float64)
        d_c = f_curr["depth"][v_c, u_c].astype(np.float64)
        valid_d = (d_p > 0.01) & (d_c > 0.01)
        pts_prev, pts_curr, d_p, d_c = pts_prev[valid_d], pts_curr[valid_d], d_p[valid_d], d_c[valid_d]
        
        if len(pts_prev) < 10:
            print(f"Lost tracking at frame {curr_frame_idx}")
            break

        cam_prev = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [prev_f["K"][0, 0], prev_f["K"][1, 1], prev_f["K"][0, 2], prev_f["K"][1, 2]]}
        cam_curr = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f_curr["K"][0, 0], f_curr["K"][1, 1], f_curr["K"][0, 2], f_curr["K"][1, 2]]}
        ransac_opt = {'max_reproj_error': cfg.ransac_thresh}
        
        res, info = poselib.estimate_monodepth_relative_pose(pts_prev, pts_curr, d_p, d_c, cam_prev, cam_curr, ransac_opt)
        
        pose_rel = res.pose
        T_curr_prev = np.eye(4)
        T_curr_prev[:3, :3] = R.from_quat([pose_rel.q[1], pose_rel.q[2], pose_rel.q[3], pose_rel.q[0]]).as_matrix()
        T_curr_prev[:3, 3] = pose_rel.t
        
        # Chain Pose: T_curr_W = T_curr_prev @ T_prev_W
        T_CiW = T_curr_prev @ poses_est[prev_f["frame_idx"]]
        poses_est[curr_frame_idx] = T_CiW
        
        # 2. Triangulate points for visualization
        inliers = np.array(info['inliers'])
        pts_p_in, pts_c_in = pts_prev[inliers], pts_curr[inliers]
        P_prev = prev_f["K"] @ poses_est[prev_f["frame_idx"]][:3, :]
        P_curr = f_curr["K"] @ T_CiW[:3, :]
        new_pts3d = triangulate_linear(P_prev, P_curr, pts_p_in, pts_c_in)
        
        # Update/Create tracks
        # For simplicity in this frame-to-frame script, we just create new tracks for all inliers
        # In a real system, we would maintain tracks across frames
        for k in range(len(new_pts3d)):
            current_tracks.append({
                'pt3d': new_pts3d[k],
                'color': prev_f["image"][np.round(pts_p_in[k, 1]).astype(int), np.round(pts_p_in[k, 0]).astype(int)],
                'obs': {prev_f["frame_idx"]: pts_p_in[k], curr_frame_idx: pts_c_in[k]},
                'id': next_track_id
            })
            next_track_id += 1

        # Run BA every 5 frames
        if i % 5 == 0:
            ba_indices = [curr_frame_idx - j for j in range(min(i + 1, cfg.ba_window))]
            ba_indices = sorted(ba_indices)
            run_window_ba(ba_indices, poses_est, current_tracks)

        # Update point cloud for viz
        point_cloud_xyz = np.stack([t['pt3d'] for t in current_tracks[-2000:]])
        point_cloud_rgb = np.stack([t['color'] for t in current_tracks[-2000:]])
            
        # Log state
        log_frame(f_curr, T_CiW, point_cloud_xyz, point_cloud_rgb)
        prev_f = f_curr

    print("Tracking complete. Check Rerun.")
    
    # Compute ATE
    traj_est_arr = np.array(traj_est)
    traj_gt_arr = np.array(traj_gt)
    if len(traj_est_arr) > 0:
        ate = np.sqrt(np.mean(np.linalg.norm(traj_est_arr - traj_gt_arr, axis=1)**2))
        print(f"Final Trajectory Metric (RMS ATE): {ate:.6f}")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
