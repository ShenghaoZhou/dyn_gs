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
    init_frame: int = 0
    n_frames: int = 100
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    
    # Tracking Parameters
    num_pts: int = 2000
    ransac_thresh: float = 1.0 # pixel threshold for RANSAC
    ba_window: int = 15 # Number of frames in BA window
    
    # Depth source
    use_gt_depth: bool = False # Use kvtracker depth

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

# Removed triangulate_linear as it is no longer used.

def main(cfg: Config):
    rr.init("exp_kvtracker_pnp_seq", spawn=False)
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
        mask_gt = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        if cfg.use_gt_depth:
            mask_path = data_dir / "obj_masks" / f"{stem}.png"
            depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
        else:
            mask_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
            if not mask_path.exists():
                mask_path = data_dir / "obj_masks" / f"{stem}.png"
            depth_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
            
        mask = np.array(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE))
            
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
            "mask_gt": mask_gt,
            "depth": depth,
            "depth_gt": depth_gt,
            "K": K,
            "T_CW_gt": T_CW_gt,
            "T_WO_gt": T_WO_gt,
            "frame_idx": frame_idx
        }

    # Tracking State
    poses_est = {} # frame_idx -> T_CiC0
    traj_est = []
    traj_gt = []
    traj_abs_est = []
    traj_abs_gt = []
    
    # Load first frame (properly scaled)
    f0 = load_frame_data(cfg.init_frame)
    if f0 is None:
        print("Initial frame not found.")
        return

    H, W = f0["image"].shape[:2]
    T_C0C0 = np.eye(4)
    poses_est[f0["frame_idx"]] = T_C0C0

    # Helper for visualization (relative to f0)
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
    
    # Object Frame Mapping
    T_WO0_gt = f0["T_WO_gt"]
    T_CW0_abs_gt = f0["T_CW_gt"]
    T_OC0_gt = np.linalg.inv(T_CW0_abs_gt @ T_WO0_gt)
    def log_frame(fd, T_CiC0, pts3d=None, colors=None, info=None):
        rr.set_time("frame", sequence=fd["frame_idx"])
        
        # GT Pose (Visualization World)
        T_WCi_gt = get_gt_pose_WC(fd)
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WCi_gt[:3, :3], translation=T_WCi_gt[:3, 3]))
        rr.log("world/camera_gt/image", rr.Image(fd["image"]))
        
        # Log Object Frame Transform relative to World
        T_WO_curr = fd["T_WO_gt"]
        if T_WO_curr is not None:
            rr.log("world/object", rr.Transform3D(mat3x3=T_WO_curr[:3, :3], translation=T_WO_curr[:3, 3]))
        
        # GT Points in Object Frame (STATIC)
        if fd["frame_idx"] == f0["frame_idx"] and T_WO_curr is not None:
            sample_step = 4
            grid_y, grid_x = np.meshgrid(np.arange(0, H, sample_step), np.arange(0, W, sample_step), indexing='ij')
            u_gt, v_gt = grid_x.flatten(), grid_y.flatten()
            z_gt = fd["depth_gt"][v_gt, u_gt].flatten()
            mask_gt = fd["mask_gt"][v_gt, u_gt].flatten() > 0
            valid_gt = mask_gt & (z_gt > 0.01)
            u_gt, v_gt, z_gt = u_gt[valid_gt], v_gt[valid_gt], z_gt[valid_gt]
            K_inv = np.linalg.inv(fd["K"])
            pts2d_homog = np.stack([u_gt, v_gt, np.ones_like(u_gt)], axis=1)
            pts_Ci = (K_inv @ pts2d_homog.T).T * z_gt[:, None]
            pts_O_gt = (T_OC0_gt[:3, :3] @ pts_Ci.T).T + T_OC0_gt[:3, 3]
            rr.log("world/object/points_gt", rr.Points3D(pts_O_gt, colors=fd["image"][v_gt, u_gt], radii=0.001), static=True)

        # Absolute GT Camera (Original World)
        T_CW_abs_gt = fd["T_CW_gt"]
        T_WC_abs_gt = np.linalg.inv(T_CW_abs_gt)
        rr.log("world/camera_abs_gt", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
        rr.log("world/camera_abs_gt", rr.Transform3D(mat3x3=T_WC_abs_gt[:3, :3], translation=T_WC_abs_gt[:3, 3]))

        # Estimated Pose
        if T_CiC0 is not None:
            # T_CiW = T_CiC0 @ T_C0W
            T_CiW_est = T_CiC0 @ T_C0W_viz
            T_WCi_est = np.linalg.inv(T_CiW_est)
            
            rr.log("world/camera_est", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
            rr.log("world/camera_est", rr.Transform3D(mat3x3=T_WCi_est[:3, :3], translation=T_WCi_est[:3, 3]))
            
            traj_gt.append(T_WCi_gt[:3, 3])
            traj_est.append(T_WCi_est[:3, 3])
            
            # Absolute trajectory (Original World)
            T_CW0_abs_gt = f0["T_CW_gt"]
            T_WC0_abs_gt = np.linalg.inv(T_CW0_abs_gt)
            T_CiW_abs_est = T_CiC0 @ T_CW0_abs_gt
            T_WCi_abs_est = np.linalg.inv(T_CiW_abs_est)
            
            traj_abs_gt.append(T_WC_abs_gt[:3, 3])
            traj_abs_est.append(T_WCi_abs_est[:3, 3])
            
            rr.log("world/trajectories/gt", rr.LineStrips3D([np.array(traj_gt)], colors=[[0, 255, 0]], radii=0.002))
            rr.log("world/trajectories/est", rr.LineStrips3D([np.array(traj_est)], colors=[[0, 0, 255]], radii=0.002))
            rr.log("world/trajectories/abs_gt", rr.LineStrips3D([np.array(traj_abs_gt)], colors=[[100, 255, 100]], radii=0.001))
            rr.log("world/trajectories/abs_est", rr.LineStrips3D([np.array(traj_abs_est)], colors=[[100, 100, 255]], radii=0.001))

        if pts3d is not None and T_WO_curr is not None:
            # Transform to Object Frame
            pts_O_est = (T_OC0_gt[:3, :3] @ pts3d.T).T + T_OC0_gt[:3, 3]
            rr.log("world/object/points_est", rr.Points3D(pts_O_est, colors=colors, radii=0.001))
            
            # Log Camera in Object Frame
            T_WC_est = np.linalg.inv(T_CiW_est)
            T_OC_est = np.linalg.inv(T_WO_curr) @ T_WC_est
            rr.log("world/object/camera_est", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
        
        if info is not None:
            inliers = info.get('inliers', [])
            num_inliers = len(inliers)
            rr.log("info/num_inliers", rr.Scalars(num_inliers))
            
            # Estimate median reprojection error if points are provided
            # This is a bit complex as we need the 3D and 2D points used for PnP
            pass

    # 1. First Window: Initialize with Sequential Tracking (0 to 4)
    f_start = f0
    frames_init = [f_start]
    for k in range(1, 5):
        fd = load_frame_data(cfg.init_frame + k)
        if fd is None:
            print(f"Could not load frame {cfg.init_frame + k} for initialization.")
            return
        frames_init.append(fd)
    
    pts_start = sample_mask_points(f_start["mask"], cfg.num_pts)
    obs_list = [pts_start]
    for k in range(4):
        f_i, f_next = frames_init[k], frames_init[k+1]
        gray_i = cv2.cvtColor(f_i["image"], cv2.COLOR_RGB2GRAY)
        gray_next = cv2.cvtColor(f_next["image"], cv2.COLOR_RGB2GRAY)
        flow = dis.calc(gray_i, gray_next, None)
        delta = interpolate_flow(flow, obs_list[-1])
        obs_list.append(obs_list[-1] + delta)
    
    f_end = frames_init[-1]
    pts_end = obs_list[-1]
    
    # Filter by mask and depth
    u_e, v_e = np.round(pts_end[:, 0]).astype(int), np.round(pts_end[:, 1]).astype(int)
    valid = (u_e >= 0) & (u_e < W) & (v_e >= 0) & (v_e < H)
    if np.any(valid):
        valid[valid] &= (f_end["mask"][v_e[valid], u_e[valid]] > 0)
    
    u_s, v_s = np.round(pts_start[:, 0]).astype(int), np.round(pts_start[:, 1]).astype(int)
    d_s = f_start["depth"][v_s, u_s].astype(np.float64)
    d_e = np.zeros_like(d_s)
    d_e[valid] = f_end["depth"][v_e[valid], u_e[valid]].astype(np.float64)
    
    valid &= (d_s > 0.01) & (d_e > 0.01)
    obs_list = [o[valid] for o in obs_list]
    d_s, d_e = d_s[valid], d_e[valid]
    pts_start, pts_end = obs_list[0], obs_list[-1]
    
    cam_start = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f_start["K"][0, 0], f_start["K"][1, 1], f_start["K"][0, 2], f_start["K"][1, 2]]}
    cam_end = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f_end["K"][0, 0], f_end["K"][1, 1], f_end["K"][0, 2], f_end["K"][1, 2]]}
    ransac_opt = {'max_reproj_error': cfg.ransac_thresh}
    
    res, info = poselib.estimate_monodepth_relative_pose(pts_start, pts_end, d_s, d_e, cam_start, cam_end, ransac_opt)
    pose_rel = res.pose
    T_end_start = np.eye(4)
    T_end_start[:3, :3] = R.from_quat([pose_rel.q[1], pose_rel.q[2], pose_rel.q[3], pose_rel.q[0]]).as_matrix()
    T_end_start[:3, 3] = pose_rel.t
    
    poses_est[f_end["frame_idx"]] = T_end_start @ poses_est[f_start["frame_idx"]]
    
    # Triangulate initial 3D points
    inliers = np.array(info['inliers'])
    obs_list_in = [o[inliers] for o in obs_list]
    pts_s_in, pts_e_in = obs_list_in[0], obs_list_in[-1]
    K_inv_start = np.linalg.inv(f_start["K"])
    pts2d_homog_start = np.concatenate([pts_s_in, np.ones((len(pts_s_in), 1))], axis=-1)
    d_s_in = d_s[inliers]
    initial_pts3d = (K_inv_start @ pts2d_homog_start.T).T * d_s_in[:, None]
    initial_colors = f_start["image"][np.round(pts_s_in[:, 1]).astype(int), np.round(pts_s_in[:, 0]).astype(int)]
    
    # Store observations for the initialization window to build long feature tracks
    tracks_obs_init = [{f_start["frame_idx"]: pts_s_in[k], f_end["frame_idx"]: pts_e_in[k]} for k in range(len(pts_s_in))]
    
    # Unified Chronological Loop
    prev_f = f0
    current_tracks = [] # list of dicts { 'pt3d': xyz, 'color': rgb, 'last_2d': uv, 'obs': {frame_idx: uv}, 'id': track_id }
    next_track_id = 0
    
    def run_window_ba(frame_indices, poses, tracks):
        if len(frame_indices) < 2: return
        
        prob = pyceres.Problem()
        loss = pyceres.HuberLoss(1.0)
        
        # 1. Poses in window
        pose_params = {} # idx -> [q, t]
        for idx in frame_indices:
            T = poses[idx]
            q = R.from_matrix(T[:3, :3]).as_quat() # x,y,z,w
            q_wxyz = np.array([q[3], q[0], q[1], q[2]])
            t = T[:3, 3].copy()
            pose_params[idx] = (q_wxyz, t)
            
        # 2. Tracks in window
        track_params = {} # track_id -> pt3d
        relevant_tracks = []
        for track in tracks:
            win_obs = {f_idx: uv for f_idx, uv in track['obs'].items() if f_idx in frame_indices}
            if len(win_obs) >= 2:
                track_params[track['id']] = track['pt3d'].copy()
                relevant_tracks.append((track, win_obs))
        
        if len(relevant_tracks) == 0: return

        # 3. Camera Params
        f_ref = load_frame_data(frame_indices[0])
        K = f_ref["K"]
        cam_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]])
        
        # 4. Add Residuals
        added_poses = set()
        added_points = set()
        for track, win_obs in relevant_tracks:
            pt3d = track_params[track['id']]
            for f_idx, uv in win_obs.items():
                q_wxyz, t = pose_params[f_idx]
                cost = pycolmap.cost_functions.ReprojErrorCost('PINHOLE', uv)
                prob.add_residual_block(cost, loss, [q_wxyz, t, pt3d, cam_params])
                added_poses.add(f_idx)
                added_points.add(track['id'])
                
        # 5. Constrain Gauge
        # Fix first frame pose if it was added
        ref_idx = frame_indices[0]
        if ref_idx in added_poses:
            prob.set_parameter_block_constant(pose_params[ref_idx][0])
            prob.set_parameter_block_constant(pose_params[ref_idx][1])
        # Fix camera params
        if prob.num_residual_blocks() > 0:
            prob.set_parameter_block_constant(cam_params)
        
        # 6. Set Manifolds
        quat_manifold = pyceres.EigenQuaternionManifold()
        for idx in added_poses:
            q_wxyz, t = pose_params[idx]
            if not prob.is_parameter_block_constant(q_wxyz):
                prob.set_manifold(q_wxyz, quat_manifold)
                
        # 7. Solve
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
        options.max_num_iterations = 20
        options.num_threads = 8
        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        
        # 8. Update State
        for idx, (q_wxyz, t) in pose_params.items():
            T = np.eye(4)
            T[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
            T[:3, 3] = t
            poses[idx] = T
            
        for track, win_obs in relevant_tracks:
            track['pt3d'] = track_params[track['id']]
    
    for i in tqdm(range(cfg.n_frames)):
        idx = cfg.init_frame + i
        f_curr = load_frame_data(idx)
        if f_curr is None: break
        
        T_CiC0 = None
        info_log = None
        log_pts3d = None
        log_colors = None

        if i == 0:
            T_CiC0 = poses_est[idx]
        elif i < 4:
            # Localize via sequential tracked observations
            pts_2d = obs_list_in[i]
            
            u, v = np.round(pts_2d[:, 0]).astype(int), np.round(pts_2d[:, 1]).astype(int)
            valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if np.any(valid):
                valid[valid] &= (f_curr["mask"][v[valid], u[valid]] > 0)
            
            pts_2d_valid = pts_2d[valid].astype(np.float64)
            pts_3d_valid = initial_pts3d[valid].astype(np.float64)
            
            cam = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f_curr["K"][0, 0], f_curr["K"][1, 1], f_curr["K"][0, 2], f_curr["K"][1, 2]]}
            res_abs, info_abs = poselib.estimate_absolute_pose(pts_2d_valid, pts_3d_valid, cam, ransac_opt, None)
            
            pose_abs = res_abs.pose.q, res_abs.pose.t
            T_CiC0 = np.eye(4)
            T_CiC0[:3, :3] = R.from_quat([pose_abs[0][1], pose_abs[0][2], pose_abs[0][3], pose_abs[0][0]]).as_matrix()
            T_CiC0[:3, 3] = pose_abs[1]
            poses_est[idx] = T_CiC0
            
            # Store inlier observations for intermediate frames
            inliers_abs, valid_indices = info_abs['inliers'], np.where(valid)[0]
            for idx_inlier in range(len(inliers_abs)):
                if inliers_abs[idx_inlier]:
                    tracks_obs_init[valid_indices[idx_inlier]][idx] = pts_2d_valid[idx_inlier]

            info_log = info_abs
            log_pts3d = initial_pts3d
            log_colors = initial_colors
        elif i == 4:
            T_CiC0 = poses_est[idx]
            # Initialize sequential tracks from the 0-4 pair inliers, including intermediate observations
            for k in range(len(initial_pts3d)):
                current_tracks.append({
                    'pt3d': initial_pts3d[k],
                    'color': initial_colors[k],
                    'last_2d': pts_e_in[k],
                    'obs': tracks_obs_init[k],
                    'id': next_track_id,
                    'frame_idx': idx
                })
                next_track_id += 1
            log_pts3d = initial_pts3d
            log_colors = initial_colors
        else:
            # Sequential Tracking
            gray_prev = cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY)
            gray_curr = cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY)
            flow = dis.calc(gray_prev, gray_curr, None)
            
            active_pts_2d = np.stack([t['last_2d'] for t in current_tracks])
            active_pts_3d = np.stack([t['pt3d'] for t in current_tracks])
            
            delta = interpolate_flow(flow, active_pts_2d)
            new_2d = active_pts_2d + delta
            
            # Update last_2d for next frame
            for k, t in enumerate(current_tracks):
                t['last_2d'] = new_2d[k]
                
            u, v = np.round(new_2d[:, 0]).astype(int), np.round(new_2d[:, 1]).astype(int)
            valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            if np.any(valid):
                valid[valid] &= (f_curr["mask"][v[valid], u[valid]] > 0)
            
            pts_2d_v = new_2d[valid].astype(np.float64)
            pts_3d_v = active_pts_3d[valid].astype(np.float64)
            
            if len(pts_2d_v) < 10:
                print(f"Lost tracking at frame {idx}")
                break
                
            cam = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f_curr["K"][0, 0], f_curr["K"][1, 1], f_curr["K"][0, 2], f_curr["K"][1, 2]]}
            res_abs, info_abs = poselib.estimate_absolute_pose(pts_2d_v, pts_3d_v, cam, ransac_opt, None)
            
            pose_abs = res_abs.pose.q, res_abs.pose.t
            T_CiC0 = np.eye(4)
            T_CiC0[:3, :3] = R.from_quat([pose_abs[0][1], pose_abs[0][2], pose_abs[0][3], pose_abs[0][0]]).as_matrix()
            T_CiC0[:3, 3] = pose_abs[1]
            poses_est[idx] = T_CiC0
            
            # Update tracks
            new_tracks = []
            inliers_abs = info_abs['inliers']
            inlier_indices = np.where(inliers_abs)[0] if np.array(inliers_abs).dtype == bool else inliers_abs
            valid_indices = np.where(valid)[0]
            
            for j in inlier_indices:
                orig_idx = int(valid_indices[j])
                track = current_tracks[orig_idx].copy()
                track['last_2d'] = pts_2d_v[j]
                track['obs'] = track['obs'].copy()
                track['obs'][idx] = pts_2d_v[j]
                track['frame_idx'] = idx
                new_tracks.append(track)
            
            current_tracks = new_tracks
            
            # Keyframe logic every 5th frame
            if i % 5 == 0:
                idx_prev_kf = idx - 5
                f_prev_kf = load_frame_data(idx_prev_kf)
                gray_prev_kf = cv2.cvtColor(f_prev_kf["image"], cv2.COLOR_RGB2GRAY)
                flow_kf = dis.calc(gray_prev_kf, gray_curr, None)
                
                pts_pkf = sample_mask_points(f_prev_kf["mask"], cfg.num_pts)
                delta_kf = interpolate_flow(flow_kf, pts_pkf)
                pts_ckf = pts_pkf + delta_kf
                
                u_ckf, v_ckf = np.round(pts_ckf[:, 0]).astype(int), np.round(pts_ckf[:, 1]).astype(int)
                valid_kf = (u_ckf >= 0) & (u_ckf < W) & (v_ckf >= 0) & (v_ckf < H)
                if np.any(valid_kf):
                    valid_kf[valid_kf] &= (f_curr["mask"][v_ckf[valid_kf], u_ckf[valid_kf]] > 0)
                
                pts_pkf, pts_ckf = pts_pkf[valid_kf], pts_ckf[valid_kf]
                u_pkf, v_pkf = np.round(pts_pkf[:, 0]).astype(int), np.round(pts_pkf[:, 1]).astype(int)
                u_ckf, v_ckf = np.round(pts_ckf[:, 0]).astype(int), np.round(pts_ckf[:, 1]).astype(int)
                d_pkf = f_prev_kf["depth"][v_pkf, u_pkf].astype(np.float64)
                d_ckf = f_curr["depth"][v_ckf, u_ckf].astype(np.float64)
                valid_dkf = (d_pkf > 0.01) & (d_ckf > 0.01)
                pts_pkf, pts_ckf, d_pkf, d_ckf = pts_pkf[valid_dkf], pts_ckf[valid_dkf], d_pkf[valid_dkf], d_ckf[valid_dkf]
                
                cam_pkf = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f_prev_kf["K"][0, 0], f_prev_kf["K"][1, 1], f_prev_kf["K"][0, 2], f_prev_kf["K"][1, 2]]}
                cam_ckf = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f_curr["K"][0, 0], f_curr["K"][1, 1], f_curr["K"][0, 2], f_curr["K"][1, 2]]}
                
                res_kf, info_kf = poselib.estimate_monodepth_relative_pose(pts_pkf.astype(np.float64), pts_ckf.astype(np.float64), d_pkf, d_ckf, cam_pkf, cam_ckf, ransac_opt)
                
                inliers_kf = np.array(info_kf['inliers'])
                pts_p_in_kf, pts_c_in_kf = pts_pkf[inliers_kf], pts_ckf[inliers_kf]
                d_p_in_kf = d_pkf[inliers_kf]
                
                K_inv_p = np.linalg.inv(f_prev_kf["K"])
                pts2d_homog_p = np.concatenate([pts_p_in_kf, np.ones((len(pts_p_in_kf), 1))], axis=-1)
                pts3d_p = (K_inv_p @ pts2d_homog_p.T).T * d_p_in_kf[:, None]
                
                T_C0Cp = np.linalg.inv(poses_est[idx_prev_kf])
                new_pts3d_kf = (T_C0Cp[:3, :3] @ pts3d_p.T).T + T_C0Cp[:3, 3]
                
                new_colors_kf = f_prev_kf["image"][np.round(pts_p_in_kf[:, 1]).astype(int), np.round(pts_p_in_kf[:, 0]).astype(int)]
                
                z_p = (poses_est[idx_prev_kf][:3, :3] @ new_pts3d_kf.T).T + poses_est[idx_prev_kf][:3, 3]
                z_c = (poses_est[idx][:3, :3] @ new_pts3d_kf.T).T + poses_est[idx][:3, 3]
                valid_tri = (z_p[:, 2] > 0.01) & (z_c[:, 2] > 0.01)
                
                for k in np.where(valid_tri)[0]:
                    current_tracks.append({
                        'pt3d': new_pts3d_kf[k],
                        'color': new_colors_kf[k],
                        'last_2d': pts_c_in_kf[k],
                        'obs': {idx_prev_kf: pts_p_in_kf[k], idx: pts_c_in_kf[k]},
                        'id': next_track_id,
                        'frame_idx': idx
                    })
                    next_track_id += 1
                
                # Windowed BA
                ba_indices = [idx - j for j in range(min(idx - cfg.init_frame + 1, cfg.ba_window))]
                ba_indices = sorted(ba_indices)
                print(f"Running Window BA for frames {ba_indices[0]}-{ba_indices[-1]}...")
                run_window_ba(ba_indices, poses_est, current_tracks)
            
            info_log = info_abs
            log_pts3d = np.stack([t['pt3d'] for t in current_tracks])
            log_colors = np.stack([t['color'] for t in current_tracks])

        log_frame(f_curr, T_CiC0, log_pts3d, log_colors, info_log)
        prev_f = f_curr

    print("Tracking complete. Check Rerun.")
    
    # Compute and report ATE
    traj_est_arr = np.array(traj_est)
    traj_gt_arr = np.array(traj_gt)
    if len(traj_est_arr) == len(traj_gt_arr) and len(traj_est_arr) > 0:
        ate = np.sqrt(np.mean(np.linalg.norm(traj_est_arr - traj_gt_arr, axis=1)**2))
        print(f"Final Trajectory Metric (RMS ATE): {ate:.6f}")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
