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
import pickle
import matplotlib.pyplot as plt

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    tracks_path: str = "outputs/dynBA_tracks_dis/clip-003312/feature_tracks.pkl"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # Tracking Parameters
    ransac_thresh: float = 1.0 # pixel threshold for RANSAC
    min_inliers: int = 10
    
    # Initialization
    init_frame_offset: int = 0
    n_frames: int = 60
    
    # Depth source (only used for initialization)
    depth_source: str = "DA3METRIC-LARGE"
    use_gt_depth: bool = False

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

def get_viz_pose(frame_data, f0_data):
    """Compute camera pose in object-centric world (at first frame)."""
    T_WO_curr = frame_data["T_w_o_gt"]
    T_CW_gt = frame_data["T_c_w"]
    if T_WO_curr is None or f0_data["T_w_o_gt"] is None:
        return np.linalg.inv(T_CW_gt)
    
    # T_C_O = T_C_W * T_W_O
    T_C_O = T_CW_gt @ T_WO_curr
    # T_WC_viz: Camera pose in object-centric world defined by f0's object pose
    T_WC = f0_data["T_w_o_gt"] @ np.linalg.inv(T_C_O)
    return T_WC

def main(cfg: Config):
    rr.init("test_ba_window_generalized", spawn=False)
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # 1. Load tracks
    with open(cfg.tracks_path, "rb") as f:
        data = pickle.load(f)
    
    tracks = data["tracks"]
    frame_info = data["frame_info"]
    indices = data["indices"]
    
    if len(indices) < 3:
        print("Not enough frames for windowing.")
        return

    # Use subset of frames
    indices = indices[cfg.init_frame_offset : cfg.init_frame_offset + cfg.n_frames]
    
    # 2. Load tracks data dir
    seq_name = Path(cfg.tracks_path).parent.name
    data_dir = Path(cfg.data_root) / seq_name

    def load_extra_data(idx):
        f = frame_info[idx]
        stem = f"{idx:06d}"
        img_path = data_dir / "images" / f"{stem}.png"
        img = cv2.imread(str(img_path))[..., ::-1]
        
        mask_path = data_dir / "obj_masks" / f"{stem}.png"
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if cfg.use_gt_depth:
            depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
        else:
            depth_path = data_dir / "depth_cache" / cfg.depth_source / f"{stem}.npy"
        
        depth = np.load(depth_path)
        return {**f, "image": img, "mask": mask, "depth": depth, "idx": idx}

    f0 = load_extra_data(indices[0])
    f1 = load_extra_data(indices[1])
    
    H, W = f0["image"].shape[:2]
    T_WC0_viz = get_viz_pose(f0, f0)
    
    traj_est = []
    traj_gt = []
    
    def log_frame(fd, T_CiW_est, point_cloud_xyz=None, point_cloud_rgb=None):
        rr.set_time("frame", sequence=fd["idx"])
        
        # GT Pose
        T_WCi_gt = get_viz_pose(fd, f0)
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WCi_gt[:3, :3], translation=T_WCi_gt[:3, 3]))
        rr.log("world/camera_gt/image", rr.Image(fd["image"]))
        
        # Estimated Pose
        if T_CiW_est is not None:
            # T_WCi_est = T_WC0_viz * T_C0W * inv(T_CiW) = T_WC0_viz * inv(T_CiW) (since C0 is world)
            T_WCi_est = T_WC0_viz @ np.linalg.inv(T_CiW_est)
            rr.log("world/camera_est", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
            rr.log("world/camera_est", rr.Transform3D(mat3x3=T_WCi_est[:3, :3], translation=T_WCi_est[:3, 3]))
            
            traj_gt.append(T_WCi_gt[:3, 3])
            traj_est.append(T_WCi_est[:3, 3])
            
            rr.log("world/trajectories/gt", rr.LineStrips3D([np.array(traj_gt)], colors=[[0, 255, 0]], radii=0.002))
            rr.log("world/trajectories/est", rr.LineStrips3D([np.array(traj_est)], colors=[[0, 0, 255]], radii=0.002))

        # Accumulated Points
        if point_cloud_xyz is not None:
            # pts_viz = T_WC0_viz * pts_world
            pts_viz_est = (T_WC0_viz[:3, :3] @ point_cloud_xyz.T).T + T_WC0_viz[:3, 3]
            rr.log("world/points_est", rr.Points3D(pts_viz_est, colors=point_cloud_rgb, radii=0.001))

    # 3. Initialization (Frame 0 & 1)
    # Get common tracks
    common_ids = [tid for tid, track in tracks.items() if indices[0] in track and indices[1] in track]
    pts0 = np.array([tracks[tid][indices[0]] for tid in common_ids])
    pts1 = np.array([tracks[tid][indices[1]] for tid in common_ids])
    
    # Get depth values
    u0, v0 = np.round(pts0[:, 0]).astype(int), np.round(pts0[:, 1]).astype(int)
    u1, v1 = np.round(pts1[:, 0]).astype(int), np.round(pts1[:, 1]).astype(int)
    d0_vals = f0["depth"][v0, u0].astype(np.float64)
    d1_vals = f1["depth"][v1, u1].astype(np.float64)
    
    valid_d = (d0_vals > 0.01) & (d1_vals > 0.01)
    pts0, pts1, d0_vals, d1_vals = pts0[valid_d], pts1[valid_d], d0_vals[valid_d], d1_vals[valid_d]
    common_ids = [common_ids[i] for i in range(len(valid_d)) if valid_d[i]]

    camera0_dict = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f0["K"][0, 0], f0["K"][1, 1], f0["K"][0, 2], f0["K"][1, 2]]}
    camera1_dict = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [f1["K"][0, 0], f1["K"][1, 1], f1["K"][0, 2], f1["K"][1, 2]]}
    ransac_opt = {'max_reproj_error': cfg.ransac_thresh}
    
    res, info = poselib.estimate_monodepth_relative_pose(pts0, pts1, d0_vals, d1_vals, camera0_dict, camera1_dict, ransac_opt)
    
    pose = res.pose
    T_C1C0 = np.eye(4)
    T_C1C0[:3, :3] = R.from_quat([pose.q[1], pose.q[2], pose.q[3], pose.q[0]]).as_matrix()
    T_C1C0[:3, 3] = pose.t
    
    # Store poses (T_CiW where W=C0)
    poses_est = {indices[0]: np.eye(4), indices[1]: T_C1C0}
    
    # Triangulate all common tracks
    inliers = np.array(info['inliers'])
    pts0_in, pts1_in = pts0[inliers], pts1[inliers]
    common_ids_in = [common_ids[j] for j in range(len(inliers)) if inliers[j]]
    
    P0 = f0["K"] @ np.eye(3, 4)
    P1 = f1["K"] @ T_C1C0[:3, :]
    pts3d = triangulate_linear(P0, P1, pts0_in, pts1_in)
    
    # Store 3D positions for tracks
    track_3d = {tid: pt for tid, pt in zip(common_ids_in, pts3d)}
    
    # Initial point cloud for viz
    point_cloud_xyz = pts3d
    u0_in, v0_in = np.round(pts0_in[:, 0]).astype(int), np.round(pts0_in[:, 1]).astype(int)
    point_cloud_rgb = f0["image"][v0_in, u0_in]
    
    log_frame(f0, np.eye(4))
    log_frame(f1, T_C1C0, point_cloud_xyz, point_cloud_rgb)
    
    # 4. Windowed Processing
    for i in tqdm(range(2, len(indices))):
        idx_i = indices[i]
        idx_im1 = indices[i-1]
        idx_im2 = indices[i-2]
        
        fi = load_extra_data(idx_i)
        
        # Find tracks seen in i that have 3D positions
        tids_seen = [tid for tid, track in tracks.items() if idx_i in track]
        tids_with_3d = [tid for tid in tids_seen if tid in track_3d]
        
        if len(tids_with_3d) < cfg.min_inliers:
            print(f"Frame {idx_i}: Not enough 3D points for resection ({len(tids_with_3d)})")
            # Try to triangulate more from (i-2, i-1)
            common_prev = [tid for tid, track in tracks.items() if idx_im2 in track and idx_im1 in track and tid not in track_3d]
            if len(common_prev) > 0:
                p_im2 = np.array([tracks[tid][idx_im2] for tid in common_prev])
                p_im1 = np.array([tracks[tid][idx_im1] for tid in common_prev])
                P_im2 = frame_info[idx_im2]["K"] @ poses_est[idx_im2][:3, :]
                P_im1 = frame_info[idx_im1]["K"] @ poses_est[idx_im1][:3, :]
                pts3d_new = triangulate_linear(P_im2, P_im1, p_im2, p_im1)
                for tid, pt in zip(common_prev, pts3d_new):
                    track_3d[tid] = pt
                tids_with_3d = [tid for tid in tids_seen if tid in track_3d]

        if len(tids_with_3d) < cfg.min_inliers:
             print(f"Frame {idx_i}: Still not enough 3D points. Stopping.")
             break

        pts_i = np.array([tracks[tid][idx_i] for tid in tids_with_3d])
        pts3d_i = np.array([track_3d[tid] for tid in tids_with_3d])
        
        # Resect with estimate_generalized_absolute_pose
        # Since it's a single camera, we wrap it in lists
        points2D_rig = [pts_i.astype(np.float64)]
        points3D_rig = [pts3d_i.astype(np.float64)]
        camera_ext = [poselib.CameraPose()] # Identity
        camera_dicts = [{'model': 'PINHOLE', 'width': W, 'height': H, 'params': [fi["K"][0, 0], fi["K"][1, 1], fi["K"][0, 2], fi["K"][1, 2]]}]
        
        try:
            res_pnp, info_pnp = poselib.estimate_generalized_absolute_pose(
                points2D_rig, points3D_rig, camera_ext, camera_dicts, ransac_opt
            )
            
            pose_pnp = res_pnp
            T_CiW = np.eye(4)
            T_CiW[:3, :3] = R.from_quat([pose_pnp.q[1], pose_pnp.q[2], pose_pnp.q[3], pose_pnp.q[0]]).as_matrix()
            T_CiW[:3, 3] = pose_pnp.t
            poses_est[idx_i] = T_CiW
            # Update track_3d inliers with the refined positions if desired, 
            # or just keep track of which points are robust.
            inliers_pnp = info_pnp['inliers'][0]
            inlier_tids = [tids_with_3d[j] for j in range(len(inliers_pnp)) if inliers_pnp[j]]
            
            # 3. Triangulate NEW points between i-1 and i to maintain track density
            # We look for tracks that are seen in the current and previous frame but don't have 3D yet
            new_common = [tid for tid, track in tracks.items() if idx_im1 in track and idx_i in track and tid not in track_3d]
            if len(new_common) > 0:
                p_im1 = np.array([tracks[tid][idx_im1] for tid in new_common])
                p_i = np.array([tracks[tid][idx_i] for tid in new_common])
                
                # Projection matrices in world (C0) frame
                P_im1 = frame_info[idx_im1]["K"] @ poses_est[idx_im1][:3, :]
                P_i = fi["K"] @ T_CiW[:3, :]
                
                pts3d_new = triangulate_linear(P_im1, P_i, p_im1, p_i)
                
                # Filter by depth in both cameras to avoid degenerate triangulations
                pts3d_Ci = (T_CiW[:3, :3] @ pts3d_new.T).T + T_CiW[:3, 3]
                pts3d_Cim1 = (poses_est[idx_im1][:3, :3] @ pts3d_new.T).T + poses_est[idx_im1][:3, 3]
                valid_tri = (pts3d_Ci[:, 2] > 0.01) & (pts3d_Cim1[:, 2] > 0.01)
                
                # Add valid points to our 3D track database
                added_count = 0
                for j in range(len(valid_tri)):
                    if valid_tri[j]:
                        tid = new_common[j]
                        track_3d[tid] = pts3d_new[j]
                        added_count += 1
                
                # Update global visualization cloud
                if added_count > 0:
                    point_cloud_xyz = np.vstack([point_cloud_xyz, pts3d_new[valid_tri]])
                    colors_new = fi["image"][np.round(p_i[valid_tri][:, 1]).astype(int), np.round(p_i[valid_tri][:, 0]).astype(int)]
                    point_cloud_rgb = np.vstack([point_cloud_rgb, colors_new])

            # Log current frame state to Rerun
            log_frame(fi, T_CiW, point_cloud_xyz, point_cloud_rgb)
            
        except Exception as e:
            print(f"Frame {idx_i}: Pose estimation failed: {e}")
            break

    print("Processing complete.")

if __name__ == "__main__":
    main(tyro.cli(Config))
