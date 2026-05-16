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
from uniflowmatch.models.ufm import UniFlowMatchConfidence

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    target_frame: int = 40
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    
    # Tracking Parameters
    num_pts: int = 5000
    ransac_thresh: float = 1.0 # pixel threshold for RANSAC
    use_gt_pose: bool = False
    use_ufm: bool = False
    ufm_model: str = "infinity1096/UFM-Base"
    
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

def main(cfg: Config):
    rr.init("exp_poselib_depth_pose", spawn=False)
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    data_dir = Path(cfg.data_root)
    
    # 1. Load data
    def load_frame_data(frame_idx):
        stem = f"{frame_idx:06d}"
        img = np.array(cv2.imread(str(data_dir / "images" / f"{stem}.png"))[..., ::-1])
        mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        
        if cfg.use_gt_depth:
            depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
            if not depth_path.exists():
                print(f"Warning: GT depth not found at {depth_path}, falling back to cache.")
                depth_path = data_dir / "depth_cache" / cfg.depth_source / f"{stem}.npy"
        else:
            depth_path = data_dir / "depth_cache" / cfg.depth_source / f"{stem}.npy"
            
        depth = np.load(depth_path)
        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        T_WO_gt = load_object_pose_world(cfg.data_root, frame_idx)
        return {
            "image": img,
            "mask": mask,
            "depth": depth,
            "K": K,
            "T_CW_gt": T_CW_gt,
            "T_WO_gt": T_WO_gt
        }

    f1 = load_frame_data(cfg.init_frame)
    f2 = load_frame_data(cfg.target_frame)
    
    H, W = f1["image"].shape[:2]
    
    # 2. Compute Correspondences
    if cfg.use_ufm:
        print(f"Loading UFM model {cfg.ufm_model}...")
        device = torch.device(cfg.device)
        model = UniFlowMatchConfidence.from_pretrained(cfg.ufm_model).to(device).eval()
        
        print("Computing UFM flow...")
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                result = model.predict_correspondences_batched(
                    source_image=torch.from_numpy(f1["image"]).to(device),
                    target_image=torch.from_numpy(f2["image"]).to(device),
                )
            flow_ufm = result.flow.flow_output[0].cpu().float().permute(1, 2, 0).numpy()
            covis_ufm = result.covisibility.mask[0].cpu().float().numpy()
        
        flow = flow_ufm
        mask_ufm = (covis_ufm > 0.5)
        # Combine with f1 mask
        combined_mask = f1["mask"].copy()
        combined_mask[~mask_ufm] = 0
        pts1 = sample_mask_points(combined_mask, cfg.num_pts)
        delta = interpolate_flow(flow, pts1)
        pts2 = pts1 + delta
    else:
        print("Computing DIS flow...")
        dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
        gray1 = cv2.cvtColor(f1["image"], cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(f2["image"], cv2.COLOR_RGB2GRAY)
        flow = dis.calc(gray1, gray2, None)
        
        # 3. Sample matches
        pts1 = sample_mask_points(f1["mask"], cfg.num_pts)
        delta = interpolate_flow(flow, pts1)
        pts2 = pts1 + delta
    
    # Filter by mask in target frame
    u2 = np.round(pts2[:, 0]).astype(int)
    v2 = np.round(pts2[:, 1]).astype(int)
    valid = (u2 >= 0) & (u2 < W) & (v2 >= 0) & (v2 < H)
    if np.any(valid):
        valid[valid] &= (f2["mask"][v2[valid], u2[valid]] > 0)
    
    pts1 = pts1[valid]
    pts2 = pts2[valid]
    
    # 4. Prepare depth values
    u1 = np.round(pts1[:, 0]).astype(int)
    v1 = np.round(pts1[:, 1]).astype(int)
    u2 = np.round(pts2[:, 0]).astype(int)
    v2 = np.round(pts2[:, 1]).astype(int)
    
    depth1_vals = f1["depth"][v1, u1].astype(np.float64)
    depth2_vals = f2["depth"][v2, u2].astype(np.float64)
    
    # Filter valid depth
    valid_depth = (depth1_vals > 0.01) & (depth2_vals > 0.01)
    pts1 = pts1[valid_depth]
    pts2 = pts2[valid_depth]
    depth1_vals = depth1_vals[valid_depth]
    depth2_vals = depth2_vals[valid_depth]
    
    print(f"Number of matches: {len(pts1)}")
    
    # 5. PoseLib Estimation
    print("Estimating relative pose with PoseLib...")
    
    # GT Relative Pose for reference and potential fallback
    T_C1O_gt = f1["T_CW_gt"] @ f1["T_WO_gt"]
    T_C2O_gt = f2["T_CW_gt"] @ f2["T_WO_gt"]
    T_C2C1_gt = T_C2O_gt @ np.linalg.inv(T_C1O_gt)
    
    camera1_dict = {
        'model': 'PINHOLE',
        'width': W,
        'height': H,
        'params': [f1["K"][0, 0], f1["K"][1, 1], f1["K"][0, 2], f1["K"][1, 2]]
    }
    camera2_dict = {
        'model': 'PINHOLE',
        'width': W,
        'height': H,
        'params': [f2["K"][0, 0], f2["K"][1, 1], f2["K"][0, 2], f2["K"][1, 2]]
    }
    
    ransac_opt = {'max_reproj_error': cfg.ransac_thresh}
    
    if cfg.use_gt_pose:
        print("Using Ground Truth Pose...")
        T_C2C1 = T_C2C1_gt
        # Dummy values for visualization
        scale, shift1, shift2 = 1.0, 0.0, 0.0
    else:
        if not hasattr(poselib, "estimate_monodepth_relative_pose"):
            msg = (
                "ERROR: poselib.estimate_monodepth_relative_pose NOT FOUND! "
                "The current PoseLib installation does not include monodepth solvers. "
                "Please build PoseLib from source in /home/shzhou/project/PoseLib with pybind enabled."
            )
            raise RuntimeError(msg)

        res, info = poselib.estimate_monodepth_relative_pose(
            pts1, pts2, depth1_vals, depth2_vals, camera1_dict, camera2_dict, ransac_opt
        )
        pose = res.pose
        scale = res.scale
        shift1 = res.shift1
        shift2 = res.shift2
        T_C2C1 = np.eye(4)
        T_C2C1[:3, :3] = R.from_quat([pose.q[1], pose.q[2], pose.q[3], pose.q[0]]).as_matrix()
        T_C2C1[:3, 3] = pose.t
        
        print(f"RANSAC Inliers: {info['num_inliers']}/{len(pts1)}")
        print(f"Recovered Scale: {scale:.4f}, Shift1: {shift1:.4f}, Shift2: {shift2:.4f}")

    # 6. Triangulate Inlier Matches
    inliers = np.array(info['inliers'])
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    
    def triangulate_linear(P1, P2, p1, p2):
        """Linear triangulation for N points."""
        # p1, p2: (N, 2) pixels
        # P1, P2: (3, 4) projection matrices
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

    # P matrices in Frame 1 (World) frame
    P1 = f1["K"] @ np.eye(3, 4)
    # T_C2C1 transforms points from C1 to C2
    # x_C2 = R*x_C1 + t
    # So P2 = K2 @ [R | t]
    P2 = f2["K"] @ T_C2C1[:3, :]
    
    pts3d_tri = triangulate_linear(P1, P2, pts1_in, pts2_in)
    
    # Filter points by depth
    # Depth in camera 1 (world)
    z1 = pts3d_tri[:, 2]
    # Depth in camera 2
    pts3d_C2 = (T_C2C1[:3, :3] @ pts3d_tri.T).T + T_C2C1[:3, 3]
    z2 = pts3d_C2[:, 2]
    
    valid_tri = (z1 > 0) & (z2 > 0)
    pts3d_tri = pts3d_tri[valid_tri]
    
    # Get colors for triangulated points
    u1_in = np.round(pts1_in[valid_tri][:, 0]).astype(int)
    v1_in = np.round(pts1_in[valid_tri][:, 1]).astype(int)
    colors_tri = f1["image"][v1_in, u1_in]

    # 7. Visualization
    rr.set_time("frame", sequence=cfg.target_frame)
    
    print("\nRelative Pose Comparison (T_C2C1):")
    print("GT Translation:\n", T_C2C1_gt[:3, 3])
    print("Est Translation:\n", T_C2C1[:3, 3])
    
    r_gt = R.from_matrix(T_C2C1_gt[:3, :3]).as_euler('xyz', degrees=True)
    r_est = R.from_matrix(T_C2C1[:3, :3]).as_euler('xyz', degrees=True)
    print("GT Rotation (deg):\n", r_gt)
    print("Est Rotation (deg):\n", r_est)

    # --- Log Cameras ---
    # We'll put C1 at origin in world for visualization
    T_WC1 = np.eye(4)
    T_WC2 = T_WC1 @ np.linalg.inv(T_C2C1)
    T_WC2_gt = T_WC1 @ np.linalg.inv(T_C2C1_gt)

    rr.log("world/camera1", rr.Pinhole(image_from_camera=f1["K"], width=W, height=H))
    rr.log("world/camera1", rr.Transform3D(mat3x3=T_WC1[:3, :3], translation=T_WC1[:3, 3]))
    rr.log("world/camera1/image", rr.Image(f1["image"]))

    rr.log("world/camera2_est", rr.Pinhole(image_from_camera=f2["K"], width=W, height=H))
    rr.log("world/camera2_est", rr.Transform3D(mat3x3=T_WC2[:3, :3], translation=T_WC2[:3, 3]))
    rr.log("world/camera2_est/image", rr.Image(f2["image"]))

    rr.log("world/camera2_gt", rr.Pinhole(image_from_camera=f2["K"], width=W, height=H))
    rr.log("world/camera2_gt", rr.Transform3D(mat3x3=T_WC2_gt[:3, :3], translation=T_WC2_gt[:3, 3]))

    # --- Point Cloud Recovery ---
    # Corrected Depth: d_corr = d_orig * scale + shift
    depth1_corr = f1["depth"] * scale + shift1
    depth2_corr = f2["depth"] * scale + shift2
    
    def get_pc(depth, K, T_WC, mask, image):
        H, W = depth.shape
        grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        u = grid_x.reshape(-1)
        v = grid_y.reshape(-1)
        z = depth.reshape(-1)
        m = mask.reshape(-1) > 0
        
        valid = m & (z > 0.01)
        u, v, z = u[valid], v[valid], z[valid]
        
        K_inv = np.linalg.inv(K)
        pts2d_homog = np.stack([u, v, np.ones_like(u)], axis=1)
        pts_C = (K_inv @ pts2d_homog.T).T * z[:, None]
        pts_W = (T_WC[:3, :3] @ pts_C.T).T + T_WC[:3, 3]
        colors = image.reshape(-1, 3)[valid]
        return pts_W, colors

    pts1_W, colors1 = get_pc(depth1_corr, f1["K"], T_WC1, f1["mask"], f1["image"])
    rr.log("world/pc1_corrected", rr.Points3D(pts1_W, colors=colors1, radii=0.001))

    pts2_W, colors2 = get_pc(depth2_corr, f2["K"], T_WC2, f2["mask"], f2["image"])
    rr.log("world/pc2_corrected", rr.Points3D(pts2_W, colors=colors2, radii=0.001))

    # Log triangulated point cloud
    rr.log("world/triangulated_pc", rr.Points3D(pts3d_tri, colors=colors_tri, radii=0.002))

    # Log matches
    if len(pts1) > 0:
        num_vis = min(100, len(pts1))
        vis_idx = np.random.choice(len(pts1), num_vis, replace=False)
        p1 = pts1[vis_idx]
        p2 = pts2[vis_idx]
        
        # Reprojection of corrected depth 1 points into camera 2
        z1 = depth1_vals[vis_idx] * scale + shift1
        K1_inv = np.linalg.inv(f1["K"])
        p1_homog = np.concatenate([p1, np.ones((len(p1), 1))], axis=1)
        pts1_C1 = (K1_inv @ p1_homog.T).T * z1[:, None]
        pts1_C2 = (T_C2C1[:3, :3] @ pts1_C1.T).T + T_C2C1[:3, 3]
        p2_reproj = (f2["K"] @ pts1_C2.T).T
        p2_reproj = p2_reproj[:, :2] / p2_reproj[:, 2:3]
        
        rr.log("world/camera2_est/reproj_matches", rr.Points2D(p2_reproj, colors=[255, 0, 0], radii=3))
        rr.log("world/camera2_est/target_matches", rr.Points2D(p2, colors=[0, 255, 0], radii=2))

    print("Visualization complete. Check Rerun.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
