import numpy as np
import cv2
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import poselib
import time

from batch_normal_integration import normal_integration_batch
from gs_dyn_obj.utils.init import unproject_depth
from run_single_view_loss import normal_from_depth_image

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    n_frames: int = 30
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    
    # Normal Integration Parameters
    cg_max_iter: int = 5000
    cg_tol: float = 1e-3
    
    # Tracking Parameters
    num_pts: int = 2000
    ransac_thresh: float = 1.0 # pixel threshold for RANSAC
    use_gt_depth: bool = False # use ground truth depth for first frame

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
    # Initialize Rerun
    rr.init("exp_depth_p3p_track", spawn=False)
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    data_dir = Path(cfg.data_root)
    device = torch.device(cfg.device)

    # Pre-loading and constants
    T_WO_init = load_object_pose_world(cfg.data_root, cfg.init_frame)
    if T_WO_init is None:
        print("Could not load object poses.")
        return

    # Tracking state
    pts3d = None # Fixed 3D points in Camera 0 frame
    pts2d_curr = None
    recovered_poses = {} # frame_idx -> T_CiC0

    print(f"Tracking over {cfg.n_frames} frames starting from {cfg.init_frame}...")
    
    prev_gray = None
    prev_img_rgb = None
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    
    traj_rec = []
    traj_gt = []
    
    T_C0O_gt = None

    for i in tqdm(range(cfg.n_frames)):
        frame_idx = cfg.init_frame + i
        rr.set_time("frame", sequence=frame_idx)
        stem = f"{frame_idx:06d}"
        
        # 1. Load frame data
        img_path = data_dir / "images" / f"{stem}.png"
        if not img_path.exists(): break
        image = np.array(cv2.imread(str(img_path))[..., ::-1])
        mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        H, W = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # GT for reference
        T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        T_WO_curr = load_object_pose_world(cfg.data_root, frame_idx)
        T_CiO_gt = T_CW_gt @ T_WO_curr 
        
        if i == 0:
            T_C0O_gt = T_CiO_gt
            recovered_poses[frame_idx] = np.eye(4)
        else:
            T_CiC0_gt = T_CiO_gt @ np.linalg.inv(T_C0O_gt)

        # 2. Frame 0 Initialization (Depth)
        if i == 0:
            if cfg.use_gt_depth:
                print("Using ground truth depth for Frame 0...")
                depth_gt_path = data_dir / "depth_dyn" / f"{stem}.npy"
                if not depth_gt_path.exists():
                    print(f"GT Depth not found: {depth_gt_path}")
                    return
                depth_refined = np.load(depth_gt_path)
            else:
                # Load DA3 Depth
                depth_da3_path = data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{stem}.npy"
                if not depth_da3_path.exists():
                    print(f"Depth not found: {depth_da3_path}")
                    return
                depth_da3 = np.load(depth_da3_path)
                
                # Load MoGE Normals
                normal_moge_path = data_dir / "moge_normal" / f"{stem}.npy"
                normal_moge = np.load(normal_moge_path)
                if normal_moge.shape[0] == 3:
                    normal_moge = normal_moge.transpose(1, 2, 0)
                
                # Normal Integration Refinement
                print("Refining depth via normal integration...")
                K_torch = torch.from_numpy(K).float().to(device)
                mask_t = torch.from_numpy(mask > 0).to(device)
                
                # Align normals (flip check)
                depth_ref_normal = normal_from_depth_image(
                    torch.from_numpy(depth_da3).float().to(device), K_torch
                ).permute(2, 0, 1)
                nm_t = torch.from_numpy(normal_moge).float().to(device).permute(2, 0, 1)
                
                best_dot = -1.0
                best_flip = (1, 1, 1)
                ref_norm = F.normalize(depth_ref_normal, dim=0)
                for fx in [1, -1]:
                    for fy in [1, -1]:
                        for fz in [1, -1]:
                            fvec = torch.tensor([fx, fy, fz], device=device).view(3, 1, 1)
                            nm_f = F.normalize(nm_t * fvec, dim=0)
                            dot = (ref_norm * nm_f).sum(dim=0)[mask_t].mean().item()
                            if dot > best_dot:
                                best_dot = dot
                                best_flip = (fx, fy, fz)
                
                normal_moge_corr = normal_moge * np.array(best_flip).reshape(1, 1, 3)
                normal_moge_corr_t = torch.from_numpy(normal_moge_corr).float().to(device)
                
                # Integrate with data term
                log_depth_prior = torch.from_numpy(np.log(depth_da3 + 1e-6)).float().to(device)
                depth_integrated_flat = normal_integration_batch(
                    normal_moge_corr_t, mask_t.unsqueeze(0), K=K_torch,
                    log_depth_prior=log_depth_prior, lambda_data=0.05,
                    cg_max_iter=cfg.cg_max_iter, cg_tol=cfg.cg_tol, verbose=False
                )
                depth_integrated = torch.zeros((H, W), device=device)
                depth_integrated[mask_t] = depth_integrated_flat
                
                # Scale alignment
                depth_integrated_np = depth_integrated.cpu().numpy()
                mask_valid = (mask > 0) & (depth_da3 > 0.01) & (depth_integrated_np > 0.01)
                if np.any(mask_valid):
                    scale_factor = np.median(depth_da3[mask_valid] / depth_integrated_np[mask_valid])
                    print(f"Median scale factor for depth alignment: {scale_factor:.4f}")
                    depth_refined = depth_integrated_np * scale_factor
                else:
                    depth_refined = depth_integrated_np
            
            # Sample points and lift to 3D
            pts2d_curr = sample_mask_points(mask, cfg.num_pts)
            u = np.round(pts2d_curr[:, 0]).astype(int)
            v = np.round(pts2d_curr[:, 1]).astype(int)
            z = depth_refined[v, u]
            
            # Filter valid depth
            valid_z = z > 0.01
            pts2d_curr = pts2d_curr[valid_z]
            z = z[valid_z]
            u, v = u[valid_z], v[valid_z]
            
            # Unproject to 3D in Camera 0 frame
            K_inv = np.linalg.inv(K)
            pts2d_homog = np.concatenate([pts2d_curr, np.ones((len(pts2d_curr), 1))], axis=1)
            pts3d = (K_inv @ pts2d_homog.T).T * z[:, None]
            
            # Log initial refined point cloud in World Frame
            T_WC0 = T_WO_init @ np.linalg.inv(T_C0O_gt)
            pts3d_W = (T_WC0[:3, :3] @ pts3d.T).T + T_WC0[:3, 3]
            colors = image[v, u] / 255.0
            rr.log("world/refined_pc", rr.Points3D(pts3d_W, colors=colors, radii=0.002))
            
        else:
            # 3. Subsequent Frames: Track and Solve PnP
            flow = dis.calc(prev_gray, gray, None)
            delta = interpolate_flow(flow, pts2d_curr)
            pts2d_next = pts2d_curr + delta
            
            # Filter by mask and boundaries
            u_next = np.round(pts2d_next[:, 0]).astype(int)
            v_next = np.round(pts2d_next[:, 1]).astype(int)
            valid_bound = (u_next >= 0) & (u_next < W) & (v_next >= 0) & (v_next < H)
            valid_mask = valid_bound.copy()
            if np.any(valid_bound):
                valid_mask[valid_bound] &= (mask[v_next[valid_bound], u_next[valid_bound]] > 0)
            
            pts2d_prev_tracked = pts2d_curr[valid_mask]
            pts2d_next = pts2d_next[valid_mask]
            pts3d_tracked = pts3d[valid_mask]
            
            if len(pts2d_next) < 10:
                print(f"Tracking failed at frame {frame_idx}: too few points.")
                break
                
            # --- DIS Match Visualization ---
            if i > 0 and len(pts2d_next) > 0:
                num_vis = min(20, len(pts2d_next))
                vis_indices = np.random.choice(len(pts2d_next), num_vis, replace=False)
                p_prev = pts2d_prev_tracked[vis_indices]
                p_next = pts2d_next[vis_indices]
                
                # Create side-by-side image
                img_combined = np.hstack([prev_img_rgb, image])
                rr.log("2d/dis_matches", rr.Image(img_combined))
                
                # Draw lines
                line_strips = []
                for p0, p1 in zip(p_prev, p_next):
                    p1_offset = p1 + np.array([W, 0])
                    line_strips.append([p0, p1_offset])
                rr.log("2d/dis_matches/lines", rr.LineStrips2D(line_strips, colors=[0, 255, 0], radii=2))

            # Solve PnP using poselib
            camera_dict = {
                'model': 'PINHOLE',
                'width': W,
                'height': H,
                'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
            }
            ransac_opt = {'max_reproj_error': cfg.ransac_thresh}
            
            pose, info = poselib.estimate_absolute_pose(pts2d_next, pts3d_tracked, camera_dict, ransac_opt)
            
            R_mat = R.from_quat([pose.q[1], pose.q[2], pose.q[3], pose.q[0]]).as_matrix()
            T_CiC0 = np.eye(4)
            T_CiC0[:3, :3] = R_mat
            T_CiC0[:3, 3] = pose.t
            recovered_poses[frame_idx] = T_CiC0
            
            pts2d_curr = pts2d_next
            pts3d = pts3d_tracked
            
            # print(f"Frame {frame_idx}: Inliers {info['num_inliers']}/{len(pts2d_next)}")

        # 4. Visualization
        prev_gray = gray.copy()
        prev_img_rgb = image.copy()
        
        T_WC0 = T_WO_init @ np.linalg.inv(T_C0O_gt)
        T_WCi_rec = T_WC0 @ np.linalg.inv(recovered_poses[frame_idx])
        T_WCi_gt = T_WO_init @ np.linalg.inv(T_CiO_gt)
        
        # Log GT Point Cloud if depth is available
        depth_gt_path = data_dir / "depth_dyn" / f"{stem}.npy"
        if depth_gt_path.exists():
            depth_gt = np.load(depth_gt_path)
            K_torch = torch.from_numpy(K).float().to(device)
            # Unproject GT
            pts_C_gt = unproject_depth(torch.from_numpy(depth_gt).float().to(device), K_torch, H, W).reshape(-1, 3).cpu().numpy()
            mask_flat = mask.reshape(-1) > 0
            valid_depth = (depth_gt.reshape(-1) > 0.01) & mask_flat
            pts_C_gt_masked = pts_C_gt[valid_depth]
            colors_gt = image.reshape(-1, 3)[valid_depth] / 255.0
            
            # Transform to World
            pts_W_gt = (T_WCi_gt[:3, :3] @ pts_C_gt_masked.T).T + T_WCi_gt[:3, 3]
            rr.log("world/gt_pc_frame", rr.Points3D(pts_W_gt, colors=colors_gt, radii=0.001))

        # Log Recovered Camera
        rr.log("world/camera_recovered", rr.Pinhole(image_from_camera=K, width=W, height=H))
        rr.log("world/camera_recovered", rr.Transform3D(mat3x3=T_WCi_rec[:3, :3], translation=T_WCi_rec[:3, 3]))
        rr.log("world/camera_recovered/image", rr.Image(image))
        
        # Project 3D points back to image using estimated pose
        if pts3d is not None:
            T_CiC0 = recovered_poses[frame_idx]
            pts_Ci = (T_CiC0[:3, :3] @ pts3d.T).T + T_CiC0[:3, 3]
            pts_2d_reproj = (K @ pts_Ci.T).T
            pts_2d_reproj = pts_2d_reproj[:, :2] / (pts_2d_reproj[:, 2:3] + 1e-12)
            rr.log("world/camera_recovered/reproj", rr.Points2D(pts_2d_reproj, colors=[255, 0, 0], radii=2))

        # Log GT Camera
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=K, width=W, height=H))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WCi_gt[:3, :3], translation=T_WCi_gt[:3, 3]))
        rr.log("world/camera_gt/image", rr.Image(image))

        # Log trajectories
        traj_rec.append(T_WCi_rec[:3, 3])
        traj_gt.append(T_WCi_gt[:3, 3])
        
        if len(traj_rec) > 1:
            rr.log("world/trajectories/recovered", rr.LineStrips3D([np.array(traj_rec)], colors=[[0, 0, 255]], radii=0.002))
            rr.log("world/trajectories/gt", rr.LineStrips3D([np.array(traj_gt)], colors=[[0, 255, 0]], radii=0.002))

    print("Done.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
