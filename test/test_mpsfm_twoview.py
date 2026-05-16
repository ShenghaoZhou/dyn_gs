import numpy as np
import cv2
import rerun as rr
import torch
from pathlib import Path
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import pycolmap
import shutil

from mpsfm_vo import estimate_vo_relative_pose
from gs_dyn_obj.utils.init import unproject_depth

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    target_frame: int = 40
    device: str = "cuda"
    grid_spacing: int = 4
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    use_gt_depth: bool = False # If False, uses predicted depth from depth_cache

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

def main(cfg: Config):
    # Initialize Rerun
    rr.init("test_mpsfm_twoview", spawn=False)
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    data_dir = Path(cfg.data_root)
    device = torch.device(cfg.device)

    # 1. Load Data for Frame 1
    stem1 = f"{cfg.init_frame:06d}"
    image1 = np.array(cv2.imread(str(data_dir / "images" / f"{stem1}.png"))[..., ::-1])
    mask1 = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem1}.png"), cv2.IMREAD_GRAYSCALE))
    K1 = np.load(data_dir / "intrinsics" / f"{stem1}.npy")
    
    # Depth loading logic
    gt_depth1_path = data_dir / "depth_dyn" / f"{stem1}.npy"
    pred_depth1_path = data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{stem1}.npy"
    
    if cfg.use_gt_depth and gt_depth1_path.exists():
        print("Using Ground Truth Depth...")
        depth1 = np.load(gt_depth1_path)
    else:
        print("Using Predicted Depth (DA3)...")
        depth1 = np.load(pred_depth1_path)
    
    # Always load GT depth for reference visualization
    depth1_gt = np.load(gt_depth1_path) if gt_depth1_path.exists() else depth1

    T_CW1_gt = np.load(data_dir / "extrinsics" / f"{stem1}.npy")
    T_WO1_gt = load_object_pose_world(cfg.data_root, cfg.init_frame)
    T_C1O_gt = T_CW1_gt @ T_WO1_gt

    # 2. Load Data for Frame 2
    stem2 = f"{cfg.target_frame:06d}"
    image2 = np.array(cv2.imread(str(data_dir / "images" / f"{stem2}.png"))[..., ::-1])
    mask2 = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem2}.png"), cv2.IMREAD_GRAYSCALE))
    K2 = np.load(data_dir / "intrinsics" / f"{stem2}.npy")
    T_CW2_gt = np.load(data_dir / "extrinsics" / f"{stem2}.npy")
    T_WO2_gt = load_object_pose_world(cfg.data_root, cfg.target_frame)
    T_C2O_gt = T_CW2_gt @ T_WO2_gt

    # 3. Compute Relative Pose (GT)
    T_C2C1_gt = T_C2O_gt @ np.linalg.inv(T_C1O_gt)

    # 4. DIS Flow & Matching
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    gray1 = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    flow = dis.calc(gray1, gray2, None)

    pts1 = sample_grid_on_mask(mask1, cfg.grid_spacing)
    delta = interpolate_flow(flow, pts1)
    pts2 = pts1 + delta

    H, W = image1.shape[:2]
    ix, iy = np.round(pts2[:, 0]).astype(int), np.round(pts2[:, 1]).astype(int)
    valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    mask_at_pts = np.zeros(len(pts2), dtype=bool)
    mask_at_pts[valid] = (mask2[iy[valid], ix[valid]] > 0)
    
    m1 = pts1[mask_at_pts]
    m2 = pts2[mask_at_pts]
    matches = np.column_stack([np.arange(len(m1)), np.arange(len(m1))])

    # 5. MPSfM VO Estimation
    camera1 = pycolmap.Camera(model="PINHOLE", width=W, height=H, 
                             params=[K1[0,0], K1[1,1], K1[0,2], K1[1,2]])
    camera2 = pycolmap.Camera(model="PINHOLE", width=W, height=H, 
                             params=[K2[0,0], K2[1,1], K2[0,2], K2[1,2]])
    
    result = estimate_vo_relative_pose(m1, m2, matches, camera1, camera2, depth1)

    if result is None:
        print("VO Estimation Failed.")
        return

    # 6. Comparison
    T_C2C1_est = np.eye(4)
    T_C2C1_est[:3, :3] = result['cam2_from_cam1'].rotation.matrix()
    T_C2C1_est[:3, 3] = result['cam2_from_cam1'].translation

    print("\nRelative Pose Comparison (T_C2C1):")
    print("GT Translation:", T_C2C1_gt[:3, 3])
    print("Est Translation:", T_C2C1_est[:3, 3])

    # 7. Visualization
    T_WC1_viz = T_WO1_gt @ np.linalg.inv(T_C1O_gt)
    T_WC2_viz_gt = T_WC1_viz @ np.linalg.inv(T_C2C1_gt)
    T_WC2_viz_est = T_WC1_viz @ np.linalg.inv(T_C2C1_est)

    # Initial Point Cloud (Unprojected from depth1)
    pts1_C = unproject_depth(torch.from_numpy(depth1).float().to(device), 
                             torch.from_numpy(K1).float().to(device), H, W).reshape(-1, 3).cpu().numpy()
    mask1_flat = mask1.reshape(-1) > 0
    valid_d1 = (depth1.reshape(-1) > 0.01) & mask1_flat
    pts1_C_valid = pts1_C[valid_d1]
    colors1_valid = image1.reshape(-1, 3)[valid_d1]
    pts1_W_viz = (T_WC1_viz[:3, :3] @ pts1_C_valid.T).T + T_WC1_viz[:3, 3]

    # Refined Point Cloud (from VO result)
    pts_refined_C1 = result['points3D']
    m1_inliers = result['matches'][:, 0]
    u_in, v_in = np.round(m1[m1_inliers, 0]).astype(int), np.round(m1[m1_inliers, 1]).astype(int)
    colors_refined = image1[v_in, u_in]
    pts_refined_W_viz = (T_WC1_viz[:3, :3] @ pts_refined_C1.T).T + T_WC1_viz[:3, 3]

    # GT Point Cloud
    pts1_C_gt = unproject_depth(torch.from_numpy(depth1_gt).float().to(device), 
                                torch.from_numpy(K1).float().to(device), H, W).reshape(-1, 3).cpu().numpy()
    valid_d1_gt = (depth1_gt.reshape(-1) > 0.01) & mask1_flat
    pts1_W_viz_gt = (T_WC1_viz[:3, :3] @ pts1_C_gt[valid_d1_gt].T).T + T_WC1_viz[:3, 3]
    colors1_gt = image1.reshape(-1, 3)[valid_d1_gt]

    # Log to Rerun
    rr.log("world/camera1", rr.Pinhole(image_from_camera=K1, width=W, height=H))
    rr.log("world/camera1", rr.Transform3D(mat3x3=T_WC1_viz[:3, :3], translation=T_WC1_viz[:3, 3]))
    rr.log("world/camera1/image", rr.Image(image1))

    rr.log("world/camera2_gt", rr.Pinhole(image_from_camera=K2, width=W, height=H))
    rr.log("world/camera2_gt", rr.Transform3D(mat3x3=T_WC2_viz_gt[:3, :3], translation=T_WC2_viz_gt[:3, 3]))
    
    rr.log("world/camera2_est", rr.Pinhole(image_from_camera=K2, width=W, height=H))
    rr.log("world/camera2_est", rr.Transform3D(mat3x3=T_WC2_viz_est[:3, :3], translation=T_WC2_viz_est[:3, 3]))
    rr.log("world/camera2_est/image", rr.Image(image2))

    rr.log("world/pc_initial", rr.Points3D(pts1_W_viz, colors=colors1_valid, radii=0.001))
    rr.log("world/pc_refined", rr.Points3D(pts_refined_W_viz, colors=colors_refined, radii=0.002))
    rr.log("world/pc_gt", rr.Points3D(pts1_W_viz_gt, colors=colors1_gt, radii=0.001))

    print("\nVisualization complete. Check Rerun.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
