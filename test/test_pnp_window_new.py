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
    init_frame: int = 40
    window_size: int = 5 # Frame 0 to 4
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # Tracking Parameters
    num_pts: int = 2000
    ransac_thresh: float = 1.0 # pixel threshold for RANSAC
    
    # Debug Parameters
    stop_at: str = "init" # Options: "init", "pnp", "ba"
    
    # Depth source
    use_gt_depth: bool = False # Use model-inferred depth by default

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

def sample_mask_points(mask, num_pts):
    yy, xx = np.where(mask > 0)
    if len(xx) == 0:
        return np.array([])
    indices = np.random.choice(len(xx), min(num_pts, len(xx)), replace=False)
    return np.stack([xx[indices], yy[indices]], axis=-1).astype(np.float32)

def log_results(stage_name, frames, poses_est, tracks, pts_O_gt, color_gt, T_OC0_gt, W, H):
    print(f"\n--- Results after {stage_name} ---")
    traj_gt = [f["T_WC_gt"][:3, 3] for f in frames]
    traj_est = []
    traj_O_est = []
    
    for f in frames:
        idx = f["frame_idx"]
        rr.set_time("frame", sequence=idx)
        T_WO_curr = f["T_WO_gt"]
        T_WC_gt = f["T_WC_gt"]
        
        # Log GT Camera for comparison
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=f["K"], width=W, height=H))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WC_gt[:3, :3], translation=T_WC_gt[:3, 3]))
        rr.log("world/camera_gt/image", rr.Image(f["image"]))

        # Log Object Frame Transform relative to World
        rr.log("world/object", rr.Transform3D(mat3x3=T_WO_curr[:3, :3], translation=T_WO_curr[:3, 3]))
        
        # Log GT points in Object Frame (STATIC)
        if idx == frames[0]["frame_idx"]:
            rr.log("world/object/points_gt", rr.Points3D(pts_O_gt, colors=color_gt, radii=0.001), static=True)

        # Calculate Est Camera Pose in World via Object Frame
        T_CiC0_est = poses_est[idx]
        T_C0Ci_est = np.linalg.inv(T_CiC0_est)
        T_WC_est = T_WO_curr @ T_OC0_gt @ T_C0Ci_est
        traj_est.append(T_WC_est[:3, 3])
        
        # Log Camera in World and Object Frame
        rr.log(f"world/{stage_name}/camera", rr.Pinhole(image_from_camera=f["K"], width=W, height=H))
        rr.log(f"world/{stage_name}/camera", rr.Transform3D(mat3x3=T_WC_est[:3, :3], translation=T_WC_est[:3, 3]))
        rr.log(f"world/{stage_name}/camera/image", rr.Image(f["image"]))
        
        # Camera in Object Frame: T_OC = T_OW * T_WC
        T_OC_est = np.linalg.inv(T_WO_curr) @ T_WC_est
        traj_O_est.append(T_OC_est[:3, 3])
        rr.log(f"world/object/{stage_name}/camera", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))

        # Log Estimated Points (STATIC in Object Frame)
        if idx == frames[0]["frame_idx"]:
            pts3d_C0 = np.stack([t['pt3d'] for t in tracks])
            pts_O_est = (T_OC0_gt[:3, :3] @ pts3d_C0.T).T + T_OC0_gt[:3, 3]
            rr.log(f"world/object/{stage_name}/points_est", rr.Points3D(pts_O_est, colors=np.stack([t['color'] for t in tracks]), radii=0.001), static=True)

        # Reprojection Error and 2D Visualization
        K = f["K"]
        errs = []
        projections = []
        observations = []
        for t in tracks:
            if idx in t['obs']:
                p_c = T_CiC0_est[:3, :3] @ t['pt3d'] + T_CiC0_est[:3, 3]
                uv_p = (K @ p_c)
                uv_p = uv_p[:2] / uv_p[2]
                errs.append(np.linalg.norm(uv_p - t['obs'][idx]))
                projections.append(uv_p)
                observations.append(t['obs'][idx])
        
        if len(projections) > 0:
            rr.log(f"world/{stage_name}/camera/image/projections", rr.Points2D(projections, colors=[0, 255, 255], radii=1.0))
            rr.log(f"world/{stage_name}/camera/image/observations", rr.Points2D(observations, colors=[255, 0, 0], radii=1.0))
            
        if len(errs) > 0:
            print(f"  Frame {idx}: {np.mean(errs):.4f} px")

    # Log Final Trajectories (STATIC to show all the time)
    rr.log(f"world/trajectories/{stage_name}_est", rr.LineStrips3D([np.array(traj_est)], colors=[[0, 0, 255]], radii=0.002), static=True)
    rr.log(f"world/object/trajectories/{stage_name}_est", rr.LineStrips3D([np.array(traj_O_est)], colors=[[0, 0, 255]], radii=0.001), static=True)

    ate = np.sqrt(np.mean(np.linalg.norm(np.array(traj_est) - np.array(traj_gt), axis=1)**2))
    print(f"  {stage_name} RMS ATE: {ate:.6f}")
    return ate

def main(cfg: Config):
    rr.init("test_pnp_window", spawn=False)
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    data_dir = Path(cfg.data_root)

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
        if not depth_path.exists(): return None
        depth = np.load(depth_path)
        if depth.shape != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        T_WC_gt = np.linalg.inv(T_CW_gt)
        T_WO_gt = load_object_pose_world(cfg.data_root, frame_idx)
        
        # Ground Truth Depth for scale check
        gt_depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
        depth_gt = np.load(gt_depth_path) if gt_depth_path.exists() else depth
        if depth_gt.shape != img.shape[:2]:
            depth_gt = cv2.resize(depth_gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        return {
            "image": img, "mask": mask, "mask_gt": mask_gt,
            "depth": depth, "depth_gt": depth_gt, "K": K,
            "T_WC_gt": T_WC_gt, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, 
            "frame_idx": frame_idx
        }

    # Load frames
    f0 = load_frame_data(cfg.init_frame)
    if f0 is None: return
    depth_scale = 1.0
    if not cfg.use_gt_depth:
        m = (f0["mask"] > 0) & (f0["depth"] > 0) & (f0["depth_gt"] > 0)
        if np.any(m):
            depth_scale = np.median(f0["depth_gt"][m] / f0["depth"][m])
            print(f"Computed depth scale factor: {depth_scale:.4f}")

    frames = []
    for i in range(cfg.window_size):
        fd = load_frame_data(cfg.init_frame + i)
        if fd:
            fd["depth"] *= depth_scale
            frames.append(fd)
    if len(frames) < cfg.window_size: return

    H, W = frames[0]["image"].shape[:2]
    poses_est = {f["frame_idx"]: np.eye(4) for f in frames}
    K_dict = {f["frame_idx"]: f["K"] for f in frames}

    # Initialize Object Space Mapping
    T_C0W_gt = frames[0]["T_CW_gt"]
    T_WO_gt0 = frames[0]["T_WO_gt"]
    T_OC0_gt = np.linalg.inv(T_C0W_gt @ T_WO_gt0)
    
    # Initialize GT points in Object Frame once
    v_gt0, u_gt0 = np.where((frames[0]["mask_gt"] > 0) & (frames[0]["depth_gt"] > 0.01))
    pts_C0_gt = (np.linalg.inv(frames[0]["K"]) @ np.stack([u_gt0, v_gt0, np.ones_like(u_gt0)], axis=1).T).T * frames[0]["depth_gt"][v_gt0, u_gt0][:, None]
    pts_O_gt = (T_OC0_gt[:3, :3] @ pts_C0_gt.T).T + T_OC0_gt[:3, 3]
    color_gt = frames[0]["image"][v_gt0, u_gt0]
    
    # Log GT Trajectories once (STATIC)
    traj_gt = [f["T_WC_gt"][:3, 3] for f in frames]
    traj_O_gt = [(np.linalg.inv(f["T_WO_gt"]) @ f["T_WC_gt"])[:3, 3] for f in frames]
    rr.log("world/trajectories/gt", rr.LineStrips3D([np.array(traj_gt)], colors=[[0, 255, 0]], radii=0.002), static=True)
    rr.log("world/object/trajectories/gt", rr.LineStrips3D([np.array(traj_O_gt)], colors=[[0, 255, 0]], radii=0.001), static=True)

    # Two-view initialization (Sequential Tracking)
    f_start, f_end = frames[0], frames[-1]
    pts_start = sample_mask_points(f_start["mask"], cfg.num_pts)
    obs_list = [pts_start]
    
    # Use Robust LK Tracker instead of global DIS
    lk_params = dict(winSize=(31, 31), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    
    curr_pts = pts_start
    valid_mask = np.ones(len(pts_start), dtype=bool)
    
    for i in range(len(frames) - 1):
        f_i, f_next = frames[i], frames[i+1]
        gray_i = cv2.cvtColor(f_i["image"], cv2.COLOR_RGB2GRAY)
        gray_next = cv2.cvtColor(f_next["image"], cv2.COLOR_RGB2GRAY)
        
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_i, gray_next, curr_pts, None, **lk_params)
        curr_pts = p1.reshape(-1, 2)
        valid_mask &= (st.flatten() == 1)
        obs_list.append(curr_pts.copy())
    
    # Final filtering of points that stayed valid across the whole window
    obs_list = [o[valid_mask] for o in obs_list]
    pts_start, pts_end = obs_list[0], obs_list[-1]
    
    u_e, v_e = np.round(pts_end[:, 0]).astype(int), np.round(pts_end[:, 1]).astype(int)
    valid = (u_e >= 0) & (u_e < W) & (v_e >= 0) & (v_e < H)
    if np.any(valid): valid[valid] &= (f_end["mask"][v_e[valid], u_e[valid]] > 0)
    
    u_s, v_s = np.round(pts_start[:, 0]).astype(int), np.round(pts_start[:, 1]).astype(int)
    d_s = f_start["depth"][v_s, u_s].astype(np.float64)
    d_e = np.zeros_like(d_s)
    d_e[valid] = f_end["depth"][v_e[valid], u_e[valid]].astype(np.float64)
    
    valid &= (d_s > 0.01) & (d_e > 0.01)
    obs_list = [o[valid] for o in obs_list]
    d_s, d_e = d_s[valid], d_e[valid]
    pts_start, pts_end = obs_list[0], obs_list[-1]
    
    # Diagnostics
    T_C1O1_gt = np.linalg.inv(f_end["T_WC_gt"]) @ f_end["T_WO_gt"]
    T_C0O0_gt = np.linalg.inv(f_start["T_WC_gt"]) @ f_start["T_WO_gt"]
    T_C1C0_gt = T_C1O1_gt @ np.linalg.inv(T_C0O0_gt)
    X_C0_gt = (np.linalg.inv(f_start["K"]) @ np.concatenate([pts_start, np.ones((len(pts_start), 1))], axis=-1).T).T * d_s[:, None]
    X_C1_gt = (T_C1C0_gt[:3, :3] @ X_C0_gt.T).T + T_C1C0_gt[:3, 3]
    pts_end_gt_h = (f_end["K"] @ X_C1_gt.T).T
    pts_end_gt = pts_end_gt_h[:, :2] / pts_end_gt_h[:, 2:]
    print(f"Mean Tracked Displacement: {np.mean(np.linalg.norm(pts_end - pts_start, axis=1)):.4f} px")
    print(f"Mean GT Relative Displacement: {np.mean(np.linalg.norm(pts_end_gt - pts_start, axis=1)):.4f} px")
    print(f"Mean Flow Error: {np.mean(np.linalg.norm(pts_end - pts_end_gt, axis=1)):.4f} px")

    cam_start = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f_start["K"][0, 0], f_start["K"][1, 1], f_start["K"][0, 2], f_start["K"][1, 2]]}
    cam_end = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f_end["K"][0, 0], f_end["K"][1, 1], f_end["K"][0, 2], f_end["K"][1, 2]]}
    ransac_opt = {'max_reproj_error': cfg.ransac_thresh}
    res, info = poselib.estimate_monodepth_relative_pose(pts_start, pts_end, d_s, d_e, cam_start, cam_end, ransac_opt)
    
    print(f"Init RelPose Inliers: {info['num_inliers']}/{len(pts_start)}")

    T_end_start = np.eye(4)
    T_end_start[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix()
    T_end_start[:3, 3] = res.pose.t
    poses_est[f_end["frame_idx"]] = T_end_start
    inliers = np.array(info['inliers'])
    obs_list_in = [o[inliers] for o in obs_list]
    pts_s_in, d_s_in = obs_list_in[0], d_s[inliers]
    pts_e_in = obs_list_in[-1]
    initial_pts3d_C0 = (np.linalg.inv(f_start["K"]) @ np.concatenate([pts_s_in, np.ones((len(pts_s_in), 1))], axis=-1).T).T * d_s_in[:, None]
    initial_colors = f_start["image"][np.round(pts_s_in[:, 1]).astype(int), np.round(pts_s_in[:, 0]).astype(int)]
    
    tracks = []
    for k in range(len(pts_s_in)):
        track_obs = {f_start["frame_idx"]: pts_s_in[k], f_end["frame_idx"]: pts_e_in[k]}
        tracks.append({'pt3d': initial_pts3d_C0[k], 'color': initial_colors[k], 'obs': track_obs, 'id': k})

    if cfg.stop_at == "init":
        log_results("init", frames, poses_est, tracks, pts_O_gt, color_gt, T_OC0_gt, W, H)
        return

    # Intermediate PnP (not implemented here for simplicity as we focus on init)
    # ...
    log_results("pnp", frames, poses_est, tracks, pts_O_gt, color_gt, T_OC0_gt, W, H)

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
