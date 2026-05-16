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
    
    # Depth source (only used for initialization)
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
    rr.init("exp_poselib_seq", spawn=False)
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
            
        depth = np.load(depth_path)

        # Always load true GT depth for visualization if possible
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

    # Storage for tracking
    active_3d_pts = [] # List of (xyz_world, color)
    active_2d_pts = [] # List of (uv)
    
    traj_est = []
    traj_gt = []
    
    # Store poses: T_CiW (World is the object space at first frame)
    poses_est = {} 
    
    # Initialization with first two frames
    f0 = load_frame_data(cfg.init_frame)
    f1 = load_frame_data(cfg.init_frame + 1)
    
    if f0 is None or f1 is None:
        print("Initial frames not found.")
        return

    H, W = f0["image"].shape[:2]
    
    # Initial Poses (World = C0)
    T_C0W = np.eye(4)
    poses_est[f0["frame_idx"]] = T_C0W
    
    # GT Poses for comparison (relative to initial object pose)
    def get_viz_pose(frame_data):
        # T_WC = T_WO_init * T_OW_curr = T_WO_init * inv(T_CW_curr * T_WO_curr)
        T_WO_curr = frame_data["T_WO_gt"]
        T_CW_gt = frame_data["T_CW_gt"]
        if T_WO_curr is None:
            # Fallback if no object pose
            return np.linalg.inv(T_CW_gt)
        T_C_O = T_CW_gt @ T_WO_curr
        # T_WC_viz: Camera pose in object-centric world (at first frame)
        # We use T_WO_init to define the "Viz World" coordinate system
        T_WC = f0["T_WO_gt"] @ np.linalg.inv(T_C_O)
        return T_WC

    T_WC0_viz = get_viz_pose(f0)
    
    def log_frame(fd, T_CiC0, point_cloud_xyz=None, point_cloud_rgb=None, is_gt=False):
        rr.set_time("frame", sequence=fd["frame_idx"])
        
        # GT Pose
        T_WCi_gt = get_viz_pose(fd)
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WCi_gt[:3, :3], translation=T_WCi_gt[:3, 3]))
        rr.log("world/camera_gt/image", rr.Image(fd["image"]))
        
        # GT Points
        grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        u = grid_x.flatten()
        v = grid_y.flatten()
        z = fd["depth_gt"].flatten()
        mask = fd["mask"].flatten() > 0
        valid = mask & (z > 0.01)
        
        u, v, z = u[valid], v[valid], z[valid]
        K_inv = np.linalg.inv(fd["K"])
        pts2d_homog = np.stack([u, v, np.ones_like(u)], axis=1)
        pts_Ci = (K_inv @ pts2d_homog.T).T * z[:, None]
        pts_viz_gt = (T_WCi_gt[:3, :3] @ pts_Ci.T).T + T_WCi_gt[:3, 3]
        colors_gt = fd["image"].reshape(-1, 3)[valid]
        rr.log("world/points_gt", rr.Points3D(pts_viz_gt, colors=colors_gt, radii=0.001))

        # Estimated Pose
        if T_CiC0 is not None:
            T_WCi_est = T_WC0_viz @ np.linalg.inv(T_CiC0)
            rr.log("world/camera_est", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
            rr.log("world/camera_est", rr.Transform3D(mat3x3=T_WCi_est[:3, :3], translation=T_WCi_est[:3, 3]))
            
            traj_gt.append(T_WCi_gt[:3, 3])
            traj_est.append(T_WCi_est[:3, 3])
            
            rr.log("world/trajectories/gt", rr.LineStrips3D([np.array(traj_gt)], colors=[[0, 255, 0]], radii=0.002))
            rr.log("world/trajectories/est", rr.LineStrips3D([np.array(traj_est)], colors=[[0, 0, 255]], radii=0.002))

        # Accumulated Points
        if point_cloud_xyz is not None:
            pts_viz_est = (T_WC0_viz[:3, :3] @ point_cloud_xyz.T).T + T_WC0_viz[:3, 3]
            rr.log("world/points_est", rr.Points3D(pts_viz_est, colors=point_cloud_rgb, radii=0.001))

    # Log initial frames
    log_frame(f0, T_C0W)

    # 1. First Pair Relative Pose
    gray0 = cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY)
    gray1 = cv2.cvtColor(f1["image"], cv2.COLOR_RGB2GRAY)
    flow01 = dis.calc(gray0, gray1, None)
    
    pts0 = sample_mask_points(f0["mask"], cfg.num_pts)
    delta = interpolate_flow(flow01, pts0)
    pts1 = pts0 + delta
    
    # Filter by mask in f1
    u1, v1 = np.round(pts1[:, 0]).astype(int), np.round(pts1[:, 1]).astype(int)
    valid = (u1 >= 0) & (u1 < W) & (v1 >= 0) & (v1 < H)
    if np.any(valid):
        valid[valid] &= (f1["mask"][v1[valid], u1[valid]] > 0)
    pts0, pts1 = pts0[valid], pts1[valid]
    
    # Filter by depth
    u0, v0 = np.round(pts0[:, 0]).astype(int), np.round(pts0[:, 1]).astype(int)
    u1, v1 = np.round(pts1[:, 0]).astype(int), np.round(pts1[:, 1]).astype(int)
    d0_vals = f0["depth"][v0, u0].astype(np.float64)
    d1_vals = f1["depth"][v1, u1].astype(np.float64)
    valid_d = (d0_vals > 0.01) & (d1_vals > 0.01)
    pts0, pts1, d0_vals, d1_vals = pts0[valid_d], pts1[valid_d], d0_vals[valid_d], d1_vals[valid_d]

    camera0_dict = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f0["K"][0, 0], f0["K"][1, 1], f0["K"][0, 2], f0["K"][1, 2]]}
    camera1_dict = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f1["K"][0, 0], f1["K"][1, 1], f1["K"][0, 2], f1["K"][1, 2]]}
    ransac_opt = {'max_reproj_error': cfg.ransac_thresh}
    
    res, info = poselib.estimate_monodepth_relative_pose(pts0, pts1, d0_vals, d1_vals, camera0_dict, camera1_dict, ransac_opt)
    
    pose = res.pose
    T_C1C0 = np.eye(4)
    T_C1C0[:3, :3] = R.from_quat([pose.q[1], pose.q[2], pose.q[3], pose.q[0]]).as_matrix()
    T_C1C0[:3, 3] = pose.t
    poses_est[f1["frame_idx"]] = T_C1C0 # T_C1W since C0 is W
    
    # Triangulate
    inliers = np.array(info['inliers'])
    pts0_in, pts1_in = pts0[inliers], pts1[inliers]
    P0 = f0["K"] @ np.eye(3, 4)
    P1 = f1["K"] @ T_C1C0[:3, :]
    pts3d = triangulate_linear(P0, P1, pts0_in, pts1_in)
    
    # Filter points by depth in both
    z0 = pts3d[:, 2]
    pts3d_C1 = (T_C1C0[:3, :3] @ pts3d.T).T + T_C1C0[:3, 3]
    z1 = pts3d_C1[:, 2]
    valid_tri = (z0 > 0.01) & (z1 > 0.01)
    
    pts3d = pts3d[valid_tri]
    pts1_in = pts1_in[valid_tri]
    colors = f0["image"][np.round(pts0_in[valid_tri][:, 1]).astype(int), np.round(pts0_in[valid_tri][:, 0]).astype(int)]
    
    active_3d_pts = pts3d
    active_2d_pts = pts1_in
    point_cloud_xyz = pts3d
    point_cloud_rgb = colors
    
    # Log Frame 1
    log_frame(f1, T_C1C0, point_cloud_xyz, point_cloud_rgb)
    
    # Sequential Tracking
    prev_f = f1
    prev_gray = gray1
    
    for i in tqdm(range(2, cfg.n_frames)):
        curr_frame_idx = cfg.init_frame + i
        f_curr = load_frame_data(curr_frame_idx)
        if f_curr is None: break
        
        curr_gray = cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY)
        flow_prev_curr = dis.calc(prev_gray, curr_gray, None)
        
        # 1. Advect existing points
        delta = interpolate_flow(flow_prev_curr, active_2d_pts)
        pts_curr = active_2d_pts + delta
        
        # Filter by bounds and mask
        u_c, v_c = np.round(pts_curr[:, 0]).astype(int), np.round(pts_curr[:, 1]).astype(int)
        valid = (u_c >= 0) & (u_c < W) & (v_c >= 0) & (v_c < H)
        if np.any(valid):
            valid[valid] &= (f_curr["mask"][v_c[valid], u_c[valid]] > 0)
        
        pts_curr_v = pts_curr[valid]
        pts3d_v = active_3d_pts[valid]
        
        if len(pts_curr_v) < 10:
            print(f"Lost tracking at frame {curr_frame_idx}")
            break
            
        # 2. Solve PnP (Absolute Pose)
        camera_dict = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f_curr["K"][0, 0], f_curr["K"][1, 1], f_curr["K"][0, 2], f_curr["K"][1, 2]]}
        
        # Ensure float64 for PoseLib
        pts_curr_v = pts_curr_v.astype(np.float64)
        pts3d_v = pts3d_v.astype(np.float64)

        # PoseLib estimate_absolute_pose returns (Image, info)
        res_pnp, info_pnp = poselib.estimate_absolute_pose(pts_curr_v, pts3d_v, camera_dict, ransac_opt)
        
        pose_pnp = res_pnp.pose
        T_CiW = np.eye(4)
        T_CiW[:3, :3] = R.from_quat([pose_pnp.q[1], pose_pnp.q[2], pose_pnp.q[3], pose_pnp.q[0]]).as_matrix()
        T_CiW[:3, 3] = pose_pnp.t
        poses_est[curr_frame_idx] = T_CiW
        
        # 3. Triangulate more points from (prev, curr)
        # Sample new points on prev frame mask
        new_pts_prev = sample_mask_points(prev_f["mask"], 500)
        delta_new = interpolate_flow(flow_prev_curr, new_pts_prev)
        new_pts_curr = new_pts_prev + delta_new
        
        # Filter new points
        u_n, v_n = np.round(new_pts_curr[:, 0]).astype(int), np.round(new_pts_curr[:, 1]).astype(int)
        valid_n = (u_n >= 0) & (u_n < W) & (v_n >= 0) & (v_n < H)
        if np.any(valid_n):
            valid_n[valid_n] &= (f_curr["mask"][v_n[valid_n], u_n[valid_n]] > 0)
        
        new_pts_prev = new_pts_prev[valid_n]
        new_pts_curr = new_pts_curr[valid_n]
        
        P_prev = prev_f["K"] @ poses_est[prev_f["frame_idx"]][:3, :]
        P_curr = f_curr["K"] @ T_CiW[:3, :]
        new_pts3d = triangulate_linear(P_prev, P_curr, new_pts_prev, new_pts_curr)
        
        # Filter by depth
        z_p = (poses_est[prev_f["frame_idx"]][:3, :3] @ new_pts3d.T).T + poses_est[prev_f["frame_idx"]][:3, 3]
        z_c = (T_CiW[:3, :3] @ new_pts3d.T).T + T_CiW[:3, 3]
        valid_tri_n = (z_p[:, 2] > 0.01) & (z_c[:, 2] > 0.01)
        
        new_pts3d = new_pts3d[valid_tri_n]
        new_pts_curr = new_pts_curr[valid_tri_n]
        new_colors = prev_f["image"][np.round(new_pts_prev[valid_tri_n][:, 1]).astype(int), np.round(new_pts_prev[valid_tri_n][:, 0]).astype(int)]
        
        # Update active points for next frame
        # Keep inliers from PnP + newly triangulated points
        inliers_pnp = np.array(info_pnp['inliers'])
        active_3d_pts = np.vstack([pts3d_v[inliers_pnp], new_pts3d])
        active_2d_pts = np.vstack([pts_curr_v[inliers_pnp], new_pts_curr])
        
        # Append to global cloud
        point_cloud_xyz = np.vstack([point_cloud_xyz, new_pts3d])
        point_cloud_rgb = np.vstack([point_cloud_rgb, new_colors])

        # Log current state
        log_frame(f_curr, T_CiW, point_cloud_xyz, point_cloud_rgb)

        # Prep for next
        prev_f = f_curr
        prev_gray = curr_gray

    print("Tracking sequence complete. Check Rerun.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
