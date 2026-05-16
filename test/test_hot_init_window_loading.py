import numpy as np
import cv2
import rerun as rr
import torch
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from gs_dyn_obj.utils.init import unproject_depth
from scipy.spatial.transform import Rotation as R

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    n_frames: int = 30
    device: str = "cuda"

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

def main(cfg: Config):
    # Initialize Rerun
    rr.init("test_hot_init_window_loading", spawn=False)
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    
    data_dir = Path(cfg.data_root)
    device = torch.device(cfg.device)

    # 1. Load init frame data to get constants
    init_stem = f"{cfg.init_frame:06d}"
    T_CW_init = np.load(data_dir / "extrinsics" / f"{init_stem}.npy")
    T_WO_init = load_object_pose_world(cfg.data_root, cfg.init_frame)
    if T_WO_init is None: return

    print(f"Loading window of {cfg.n_frames} frames starting from {cfg.init_frame}...")

    trajectory_gt = []
    trajectory_origin = []

    # 2. Loop through the window
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
        
        # Load GT Depth for current frame
        depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
        if not depth_path.exists():
            depth_path = data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{stem}.npy"
        depth = np.load(depth_path)

        # World to Camera
        T_CW_actual = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        # Camera to World
        T_WC_actual = np.linalg.inv(T_CW_actual)
        T_WO_curr = load_object_pose_world(cfg.data_root, frame_idx)
        if T_WO_curr is None: break

        # Relative pose for camera_gt view
        T_C_O_curr = T_CW_actual @ T_WO_curr
        # T_WC_viz = T_WO_init @ inv(T_C_O_curr)
        T_WC_viz_gt = T_WO_init @ np.linalg.inv(T_C_O_curr)
        
        # camera_origin: Camera in the world pose directly.
        T_WC_viz_origin = T_WC_actual 

        # Unproject current depth to camera frame
        xyz_C = unproject_depth(
            torch.from_numpy(depth).float().to(device), 
            torch.from_numpy(K).float().to(device), 
            H, W
        ).reshape(-1, 3).cpu().numpy()
        
        mask_flat = mask.reshape(-1) > 0
        valid_depth = (depth.reshape(-1)[mask_flat] > 0.01)
        pts_C_masked = xyz_C[mask_flat][valid_depth]
        colors_masked = image.reshape(-1, 3)[mask_flat][valid_depth]
        
        # Transform points to the World (viz) frame using camera_gt pose
        # X_W_viz = T_WC_viz_gt @ X_C
        pts_W_viz = (T_WC_viz_gt[:3, :3] @ pts_C_masked.T).T + T_WC_viz_gt[:3, 3]
        
        # Log to top-level world frame
        rr.log("world/gt_pc_frame", rr.Points3D(pts_W_viz, colors=colors_masked, radii=0.001))

        # Log trajectories and camera poses
        trajectory_gt.append(T_WC_viz_gt[:3, 3])
        trajectory_origin.append(T_WC_actual[:3, 3])
        
        if len(trajectory_gt) > 1:
            rr.log("world/trajectories/gt", rr.LineStrips3D([np.array(trajectory_gt)], colors=[[0, 255, 0]], radii=0.002))
            rr.log("world/trajectories/origin", rr.LineStrips3D([np.array(trajectory_origin)], colors=[[255, 0, 0]], radii=0.002))
        
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=K, width=W, height=H))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WC_viz_gt[:3, :3], translation=T_WC_viz_gt[:3, 3]))
        rr.log("world/camera_gt/image", rr.Image(image))
        
        rr.log("world/camera_origin", rr.Pinhole(image_from_camera=K, width=W, height=H))
        rr.log("world/camera_origin", rr.Transform3D(mat3x3=T_WC_actual[:3, :3], translation=T_WC_actual[:3, 3]))
        rr.log("world/camera_origin/image", rr.Image(image))

    print("Visualization complete. Check Rerun.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
