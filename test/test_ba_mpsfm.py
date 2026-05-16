import numpy as np
import cv2
import rerun as rr
import torch
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import pycolmap
import shutil

from mpsfm_vo import estimate_vo_relative_pose

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    n_frames: int = 30
    device: str = "cuda"
    max_features: int = 1500 # Threshold to trigger re-seeding
    min_features: int = 800  # Minimum features to maintain
    grid_spacing: int = 4    # Density of grid points
    min_track_len: int = 10
    output_dir: str = "outputs/test_ba_mpsfm"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM

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

def sample_grid_on_mask(mask, spacing):
    """Samples points on a regular grid constrained by the object mask."""
    h, w = mask.shape
    yy, xx = np.mgrid[spacing//2:h:spacing, spacing//2:w:spacing]
    pts = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
    ix, iy = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    
    valid_bounds = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy = ix[valid_bounds], iy[valid_bounds]
    pts = pts[valid_bounds]
    
    valid_mask = mask[iy, ix] > 0
    return pts[valid_mask]

def main(cfg: Config):
    # Initialize Rerun
    rr.init("test_ba_mpsfm", spawn=False)
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    data_dir = Path(cfg.data_root)
    device = torch.device(cfg.device)

    # 1. Load constants
    T_WO_init = load_object_pose_world(cfg.data_root, cfg.init_frame)
    if T_WO_init is None:
        print("Could not load object poses.")
        return

    # DIS Optical Flow setup
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    
    prev_gray = None
    prev_image = None
    prev_mask = None
    prev_depth = None
    prev_K = None
    prev_pts = None
    
    current_T_CW_rec = None
    T_C0O_gt = None
    
    traj_rec = []
    traj_gt = []
    
    all_refined_points_W = []
    all_refined_colors = []

    print(f"Tracking with MPSfM VO over {cfg.n_frames} frames starting from {cfg.init_frame}...")
    
    for i in tqdm(range(cfg.n_frames)):
        frame_idx = cfg.init_frame + i
        rr.set_time("frame", sequence=frame_idx)
        stem = f"{frame_idx:06d}"
        
        img_path = data_dir / "images" / f"{stem}.png"
        if not img_path.exists(): break
            
        image = np.array(cv2.imread(str(img_path))[..., ::-1])
        mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        H, W = image.shape[:2]
        
        depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
        if not depth_path.exists():
            depth_path = data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{stem}.npy"
        depth = np.load(depth_path)

        T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        T_WO_curr = load_object_pose_world(cfg.data_root, frame_idx)
        T_CiO_gt = T_CW_gt @ T_WO_curr

        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        if i == 0:
            T_C0O_gt = T_CiO_gt
            current_T_CW_rec = np.eye(4)
            # Initial seed points
            prev_pts = sample_grid_on_mask(mask, cfg.grid_spacing)
        else:
            # 1. Compute DIS Flow
            flow = dis.calc(prev_gray, gray, None)
            
            # 2. Track points
            delta = interpolate_flow(flow, prev_pts)
            curr_pts = prev_pts + delta
            
            # 3. Filter tracked points
            ix, iy = np.round(curr_pts[:, 0]).astype(int), np.round(curr_pts[:, 1]).astype(int)
            valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
            mask_at_pts = np.zeros(len(curr_pts), dtype=bool)
            mask_at_pts[valid] = (mask[iy[valid], ix[valid]] > 0)
            
            m1 = prev_pts[mask_at_pts]
            m2 = curr_pts[mask_at_pts]
            matches = np.column_stack([np.arange(len(m1)), np.arange(len(m1))])
            
            if len(m1) < 15:
                print(f"Tracking failed at frame {frame_idx}: too few matches.")
                break
            
            # 4. MPSfM VO Relative Pose
            camera1 = pycolmap.Camera(model="PINHOLE", width=W, height=H, 
                                     params=[prev_K[0,0], prev_K[1,1], prev_K[0,2], prev_K[1,2]])
            camera2 = pycolmap.Camera(model="PINHOLE", width=W, height=H, 
                                     params=[K[0,0], K[1,1], K[0,2], K[1,2]])
            
            result = estimate_vo_relative_pose(m1, m2, matches, camera1, camera2, prev_depth)
            
            if result is None:
                print(f"VO failed at frame {frame_idx}.")
                break
            
            # 5. Update Pose
            # result['cam2_from_cam1'] is pycolmap.Rigid3d (T_Ci_Ci-1)
            T_rel = np.eye(4)
            T_rel[:3, :3] = result['cam2_from_cam1'].rotation.matrix()
            T_rel[:3, 3] = result['cam2_from_cam1'].translation
            
            # New pose: T_Ci_C0 = T_rel @ T_Ci-1_C0
            current_T_CW_rec = T_rel @ current_T_CW_rec
            
            # 6. Refined Points
            # points3D are in camera 1 (prev camera) frame: P_Ci-1
            pts3D_Ci_minus_1 = result['points3D']
            # Transform to world viz frame
            T_Ci_minus_1_C0 = current_T_CW_rec @ np.linalg.inv(T_rel)
            T_C0_Ci_minus_1 = np.linalg.inv(T_Ci_minus_1_C0)
            
            T_WC0 = T_WO_init @ np.linalg.inv(T_C0O_gt)
            T_WCi_minus_1 = T_WC0 @ T_C0_Ci_minus_1
            
            pts3D_W = (T_WCi_minus_1[:3, :3] @ pts3D_Ci_minus_1.T).T + T_WCi_minus_1[:3, 3]
            
            # Get colors for these points
            m1_inliers = result['matches'][:, 0]
            u_in, v_in = np.round(m1[m1_inliers, 0]).astype(int), np.round(m1[m1_inliers, 1]).astype(int)
            colors = prev_image[v_in, u_in]
            
            all_refined_points_W.append(pts3D_W)
            all_refined_colors.append(colors)
            
            # 7. Update seed points for next frame
            prev_pts = m2
            if len(prev_pts) < cfg.min_features:
                occupancy = mask.copy()
                for pt in prev_pts: 
                    cv2.circle(occupancy, (int(round(pt[0])), int(round(pt[1]))), cfg.grid_spacing // 2, 0, -1)
                new_seeds = sample_grid_on_mask(occupancy, cfg.grid_spacing)
                if len(new_seeds) > 0:
                    prev_pts = np.vstack([prev_pts, new_seeds])

        # Visualization
        T_WC0 = T_WO_init @ np.linalg.inv(T_C0O_gt)
        T_WCi_rec = T_WC0 @ np.linalg.inv(current_T_CW_rec)
        T_WCi_gt = T_WO_init @ np.linalg.inv(T_CiO_gt)
        
        traj_rec.append(T_WCi_rec[:3, 3])
        traj_gt.append(T_WCi_gt[:3, 3])
        
        rr.log("world/camera_rec", rr.Pinhole(image_from_camera=K, width=W, height=H))
        rr.log("world/camera_rec", rr.Transform3D(mat3x3=T_WCi_rec[:3, :3], translation=T_WCi_rec[:3, 3]))
        rr.log("world/camera_rec/image", rr.Image(image))
        
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=K, width=W, height=H))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WCi_gt[:3, :3], translation=T_WCi_gt[:3, 3]))
        
        if len(traj_rec) > 1:
            rr.log("world/trajectories/rec", rr.LineStrips3D([np.array(traj_rec)], colors=[[0, 0, 255]], radii=0.002))
            rr.log("world/trajectories/gt", rr.LineStrips3D([np.array(traj_gt)], colors=[[0, 255, 0]], radii=0.002))
            
        if all_refined_points_W:
            flat_pts = np.concatenate(all_refined_points_W, axis=0)
            flat_colors = np.concatenate(all_refined_colors, axis=0)
            rr.log("world/refined_pc", rr.Points3D(flat_pts, colors=flat_colors, radii=0.001))

        # Update previous frame state
        prev_gray = gray.copy()
        prev_image = image.copy()
        prev_mask = mask.copy()
        prev_depth = depth.copy()
        prev_K = K.copy()

    print("Done.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
