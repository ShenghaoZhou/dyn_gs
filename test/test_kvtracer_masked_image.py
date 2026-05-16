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
import open3d as o3d
from data import HOT3DDataLoader

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    data_seq: str = "clip-003312"
    init_frame: int = 0
    n_frames: int = 50
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    
    # Tracking Parameters
    num_pts: int = 2000
    ransac_thresh: float = 1.0 # pixel threshold for RANSAC
    
    # TSDF Parameters
    voxel_length: float = 0.005 # 5mm voxels
    sdf_trunc: float = 0.02    # 2cm truncation
    
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
    rr.init("test_kvtracer_masked_image", spawn=False)
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)
    
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    # Load Data
    data_path = Path(cfg.data_root) / cfg.data_seq
    data_loader = HOT3DDataLoader(data_path)
    
    # Initialize TSDF Volume
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.voxel_length,
        sdf_trunc=cfg.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    
    # Tracking State
    current_pts = None
    prev_gray = None
    
    # We will use the first frame to define the object coordinate system
    # and use the depth map to initialize 3D positions of the points.
    f0 = data_loader[cfg.init_frame]
    H, W = f0["image"].shape[:2]
    K = f0["K"]
    
    # Load depth from model_infer as suggested by exp_kvtracker_pnp_seq.py
    depth0_path = data_path / "model_infer" / f"depth_{cfg.init_frame:05d}.npy"
    if not depth0_path.exists():
        # Fallback to depth_dyn if model_infer doesn't exist
        depth0_path = data_path / "depth_dyn" / f"{cfg.init_frame:06d}.npy"
    
    if not depth0_path.exists():
        print(f"Error: Could not find depth map at {depth0_path}")
        return
        
    depth0 = np.load(depth0_path)
    if depth0.shape != (H, W):
        depth0 = cv2.resize(depth0, (W, H), interpolation=cv2.INTER_NEAREST)
    
    # Initialize points on object mask
    mask0 = f0["obj_mask"]
    current_pts = sample_mask_points(mask0, cfg.num_pts)
    
    # Unproject initial points to 3D in camera frame
    K_inv = np.linalg.inv(K)
    u, v = current_pts[:, 0], current_pts[:, 1]
    z = depth0[np.round(v).astype(int), np.round(u).astype(int)]
    valid = z > 0.01
    current_pts = current_pts[valid]
    z = z[valid]
    
    pts_homog = np.stack([current_pts[:, 0], current_pts[:, 1], np.ones_like(z)], axis=1)
    pts3d_c0 = (K_inv @ pts_homog.T).T * z[:, None]
    
    # We set T_world_c0 = Identity, so the reconstruction is in the first camera's frame
    T_c0_w = np.eye(4)
    
    # List to store estimated poses (T_ci_w)
    poses_est = [T_c0_w]
    
    # Log initial frame
    masked_img0 = f0["image"].copy()
    masked_img0[mask0 == 0] = 0
    rr.set_time("frame_idx", sequence=cfg.init_frame)
    rr.log("world/camera", rr.Pinhole(image_from_camera=K, width=W, height=H))
    rr.log("world/camera", rr.Transform3D(mat3x3=np.eye(3), translation=np.zeros(3)))
    rr.log("world/camera/image", rr.Image(masked_img0))
    
    # Integrate first frame into TSDF
    color_o3d = o3d.geometry.Image(masked_img0)
    depth_o3d = o3d.geometry.Image((depth0).astype(np.float32)) 
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False
    )
    intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(W, H, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    volume.integrate(rgbd, intrinsic_o3d, np.eye(4)) # T_cam_world = Identity
    
    prev_gray = cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY)
    
    print(f"Starting tracking and fusion for {cfg.n_frames} frames...")
    
    for i in tqdm(range(1, cfg.n_frames)):
        frame_idx = cfg.init_frame + i
        frame = data_loader[frame_idx]
        if frame is None: break
        
        mask = frame["obj_mask"]
        gray = cv2.cvtColor(frame["image"], cv2.COLOR_RGB2GRAY)
        
        # 1. Track points via DIS Flow
        flow = dis.calc(prev_gray, gray, None)
        delta = interpolate_flow(flow, current_pts)
        next_pts = current_pts + delta
        
        # 2. Filter points by mask
        u, v = np.round(next_pts[:, 0]).astype(int), np.round(next_pts[:, 1]).astype(int)
        valid = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if np.any(valid):
            valid[valid] &= (mask[v[valid], u[valid]] > 0)
        
        # Keep only valid tracked points
        tracked_pts = next_pts[valid]
        tracked_pts3d = pts3d_c0[valid]
        
        # Re-seeding if point count is low
        if len(tracked_pts) < cfg.num_pts // 2:
            # Create an occupancy mask to avoid seeding too close to existing points
            occupancy = mask.copy()
            for pt in tracked_pts:
                cv2.circle(occupancy, (int(round(pt[0])), int(round(pt[1]))), 10, 0, -1)
            
            new_seeds = sample_mask_points(occupancy, cfg.num_pts - len(tracked_pts))
            if len(new_seeds) > 0:
                # We need to unproject new seeds to 3D. 
                # Since we want to stay in the first camera's frame, we need the CURRENT camera's pose.
                # But we haven't estimated it yet! 
                # So we use the PREVIOUS frame's pose as an estimate or wait until after PnP.
                pass 
            
        if len(tracked_pts) < 10:
            print(f"Tracking lost at frame {frame_idx}")
            break
            
        pts_2d = tracked_pts.astype(np.float64)
        pts_3d = tracked_pts3d.astype(np.float64)
        
        # 3. Estimate Camera Pose (PnP)
        cam_info = {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}
        ransac_opt = {'max_reproj_error': cfg.ransac_thresh}
        res, info = poselib.estimate_absolute_pose(pts_2d, pts_3d, cam_info, ransac_opt, None)
        
        T_ci_w = np.eye(4)
        T_ci_w[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix()
        T_ci_w[:3, 3] = res.pose.t
        
        # 4. Integrate into TSDF
        # Load depth map for current frame
        depth_path = data_path / "model_infer" / f"depth_{frame_idx:05d}.npy"
        if not depth_path.exists():
            depth_path = data_path / "depth_dyn" / f"{frame_idx:06d}.npy"
        
        if depth_path.exists():
            depth = np.load(depth_path)
            if depth.shape != (H, W):
                depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_NEAREST)
            
            # Re-seeding logic (now that we have a pose and depth)
            if len(tracked_pts) < cfg.num_pts // 2:
                T_w_ci = np.linalg.inv(T_ci_w)
                occupancy = mask.copy()
                for pt in tracked_pts:
                    cv2.circle(occupancy, (int(round(pt[0])), int(round(pt[1]))), 10, 0, -1)
                
                new_seeds = sample_mask_points(occupancy, cfg.num_pts - len(tracked_pts))
                if len(new_seeds) > 0:
                    u_s, v_s = new_seeds[:, 0], new_seeds[:, 1]
                    z_s = depth[np.round(v_s).astype(int), np.round(u_s).astype(int)]
                    valid_s = z_s > 0.01
                    new_seeds = new_seeds[valid_s]
                    z_s = z_s[valid_s]
                    
                    if len(new_seeds) > 0:
                        pts_homog_s = np.stack([new_seeds[:, 0], new_seeds[:, 1], np.ones_like(z_s)], axis=1)
                        pts3d_ci_s = (K_inv @ pts_homog_s.T).T * z_s[:, None]
                        pts3d_w_s = (T_w_ci[:3, :3] @ pts3d_ci_s.T).T + T_w_ci[:3, 3]
                        
                        tracked_pts = np.vstack([tracked_pts, new_seeds])
                        tracked_pts3d = np.vstack([tracked_pts3d, pts3d_w_s])
            
            # Mask current image
            masked_img = frame["image"].copy()
            masked_img[mask == 0] = 0
            
            # Rerun Logging
            rr.set_time("frame_idx", sequence=frame_idx)
            T_w_ci = np.linalg.inv(T_ci_w)
            rr.log("world/camera", rr.Transform3D(mat3x3=T_w_ci[:3, :3], translation=T_w_ci[:3, 3]))
            rr.log("world/camera/image", rr.Image(masked_img))
            
            # Unproject points for visualization
            u_grid, v_grid = np.meshgrid(np.arange(0, W, 4), np.arange(0, H, 4))
            u_flat, v_flat = u_grid.flatten(), v_grid.flatten()
            z_flat = depth[v_flat, u_flat]
            mask_flat = mask[v_flat, u_flat] > 0
            valid_viz = mask_flat & (z_flat > 0.01)
            u_v, v_v, z_v = u_flat[valid_viz], v_flat[valid_viz], z_flat[valid_viz]
            
            pts2d_v = np.stack([u_v, v_v, np.ones_like(z_v)], axis=1)
            pts3d_ci = (K_inv @ pts2d_v.T).T * z_v[:, None]
            pts3d_w = (T_w_ci[:3, :3] @ pts3d_ci.T).T + T_w_ci[:3, 3]
            
            rr.log("world/unprojected_points", rr.Points3D(pts3d_w, colors=masked_img[v_v, u_v], radii=0.001))
            
            # TSDF Integration
            color_o3d = o3d.geometry.Image(masked_img)
            depth_o3d = o3d.geometry.Image((depth).astype(np.float32))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color_o3d, depth_o3d, depth_scale=1.0, depth_trunc=4.0, convert_rgb_to_intensity=False
            )
            volume.integrate(rgbd, intrinsic_o3d, T_ci_w)
            
        prev_gray = gray
        current_pts = tracked_pts
        pts3d_c0 = tracked_pts3d
        
    # Extract Final Mesh
    print("Extracting final mesh...")
    mesh_o3d = volume.extract_triangle_mesh()
    mesh_o3d.compute_vertex_normals()
    
    # Log Mesh to Rerun
    vertices = np.asarray(mesh_o3d.vertices)
    triangles = np.asarray(mesh_o3d.triangles)
    normals = np.asarray(mesh_o3d.vertex_normals)
    colors = np.asarray(mesh_o3d.vertex_colors)
    
    rr.log("world/final_mesh", rr.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=triangles,
        vertex_normals=normals,
        vertex_colors=colors
    ))
    
    # Show mesh in Open3D if requested (optional)
    # o3d.visualization.draw_geometries([mesh_o3d])
    
    print("Pipeline finished. Results logged to Rerun.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
