import numpy as np
import cv2
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from gs_dyn_obj.utils.init import d2n_tblr, unproject_depth
import rerun.blueprint as rrb

def setup_blueprint():
    blueprint = rrb.Blueprint(
        rrb.Tabs(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="DIS Flow", contents=["dis/flow_masked"]),
                    rrb.Spatial2DView(name="GT Flow", contents=["gt/flow"]),
                    rrb.Spatial2DView(name="Flow Error", contents=["error/flow"]),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Normal Comparison", contents=["gt/normal", "estimated/normal"]),
                    rrb.Spatial2DView(name="Distance Comparison", contents=["gt/distance_d", "estimated/distance_d"]),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="V Field", contents=["v_field/rgb"]),
                    rrb.Spatial2DView(name="Error Dist", contents=["error/distance_d"]),
                ),
                name="2D Analysis"
            ),
            rrb.Spatial3DView(
                name="3D Analysis",
                contents=["world/**"]
            )
        )
    )
    return blueprint

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    target_frame: int = 40
    device: str = "cuda"
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    dis_coarsest_scale: int = 8
    dis_variational_refinement_iter: int = 5
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"

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

def load_pose_T_CO(data_root, frame_idx):
    T_WO = load_object_pose_world(data_root, frame_idx)
    if T_WO is None: return None
    ext_file = Path(data_root) / "extrinsics" / f"{frame_idx:06d}.npy"
    if not ext_file.exists(): return T_WO 
    T_WC = np.load(ext_file) 
    T_CW = np.linalg.inv(T_WC)
    return T_CW @ T_WO

def depth_to_rgb(depth, min_val=0.0, max_val=2.0):
    depth_norm = np.clip((depth - min_val) / (max_val - min_val), 0, 1)
    colormap = plt.get_cmap("magma")
    rgb = (colormap(depth_norm)[..., :3] * 255).astype(np.uint8)
    return rgb

def main(cfg: Config):
    rr.init("exp_solve_n_d_dis_twoview", spawn=False)
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)
    
    rr.send_blueprint(setup_blueprint())
    
    data_dir = Path(cfg.data_root)
    device = torch.device(cfg.device)
    
    # 1. Load data
    def load_frame_data(frame_idx):
        stem = f"{frame_idx:06d}"
        img = np.array(cv2.imread(str(data_dir / "images" / f"{stem}.png"))[..., ::-1])
        mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        T_CO = load_pose_T_CO(cfg.data_root, frame_idx)
        return img, mask, K, T_CO

    img1, mask1, K1, T_CO1 = load_frame_data(cfg.init_frame)
    img2, mask2, K2, T_CO2 = load_frame_data(cfg.target_frame)
    
    H, W = img1.shape[:2]
    
    # 2. Compute DIS flow
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    dis.setVariationalRefinementIterations(cfg.dis_variational_refinement_iter)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    print("Computing DIS flow...")
    flow = dis.calc(gray1, gray2, None) # [H, W, 2]
    
    # 3. Compute relative pose T_21 = T_CO2 * inv(T_CO1)
    # This is the pose of camera 2 relative to camera 1 in object frame.
    # Wait, the homography formula p2 = (R - t v^T) p1 assumes p1 and p2 are in respective camera frames.
    # So we need the transformation from camera 1 to camera 2.
    # T_C2O = T_CO2, T_C1O = T_CO1
    # p_O = T_OC1 * p_C1
    # p_C2 = T_C2O * p_O = T_C2O * T_OC1 * p_C1
    T_21 = T_CO2 @ np.linalg.inv(T_CO1)
    R21 = T_21[:3, :3]
    t21 = T_21[:3, 3]
    
    # 4. Compute s = -1/z1 for each pixel
    K1_inv = np.linalg.inv(K1)
    
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u1 = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1).reshape(-1, 3) # [HW, 3]
    x1 = (K1_inv @ u1.T).T # [HW, 3]
    
    u2 = u1[:, :2] + flow.reshape(-1, 2)
    u2_homog = np.concatenate([u2, np.ones((len(u2), 1))], axis=1)
    x2 = (np.linalg.inv(K2) @ u2_homog.T).T # [HW, 3]
    
    # x2 ~ R x1 - s t
    # x2 x R x1 = s (x2 x t)
    Rx1 = (R21 @ x1.T).T
    x2_cross_Rx1 = np.cross(x2, Rx1)
    x2_cross_t = np.cross(x2, t21)
    
    # Solve for s: s = (a . b) / (b . b)
    s_num = np.sum(x2_cross_Rx1 * x2_cross_t, axis=1)
    s_den = np.sum(x2_cross_t * x2_cross_t, axis=1)
    s = s_num / (s_den + 1e-12)
    s_map = s.reshape(H, W)
    
    # 5. Robust Patch-based solver for v = n/d
    # Each pixel provides: (x2 x t) (x1^T v) = x2 x R x1
    # Let b = x2 x t, a = x2 x R x1
    # We solve min sum || b_i (x1_i^T v) - a_i ||^2 in a local window.
    # Normal equations: (sum ||b_i||^2 x1_i x1_i^T) v = sum (b_i^T a_i) x1_i
    
    a = x2_cross_Rx1 # [HW, 3]
    b = x2_cross_t   # [HW, 3]
    
    # Reshape to [H, W, 3]
    a_map = a.reshape(H, W, 3)
    b_map = b.reshape(H, W, 3)
    x1_map = x1.reshape(H, W, 3)
    
    b_sq_norm = np.sum(b_map * b_map, axis=-1, keepdims=True) # [H, W, 1]
    b_dot_a = np.sum(b_map * a_map, axis=-1, keepdims=True)    # [H, W, 1]
    
    # Outer product x1_i x1_i^T -> [H, W, 3, 3]
    x1_x1T = x1_map[:, :, :, None] * x1_map[:, :, None, :]
    
    # Matrix field C = ||b||^2 * x1 * x1^T
    C = b_sq_norm[:, :, :, None] * x1_x1T # [H, W, 3, 3]
    # Vector field d = (b^T a) * x1
    d = b_dot_a * x1_map # [H, W, 3]
    
    # Aggregate over a window
    ksize = 7
    C_sum = cv2.boxFilter(C.reshape(H, W, 9), -1, (ksize, ksize), normalize=False).reshape(H, W, 3, 3)
    d_sum = cv2.boxFilter(d, -1, (ksize, ksize), normalize=False)
    
    # Solve C_sum v = d_sum
    # Add a small identity for regularization
    reg = 1e-6 * np.eye(3)
    v_map = np.linalg.solve(C_sum + reg, d_sum) # [H, W, 3]
    
    # 7. Recover n and d
    v_norm = np.linalg.norm(v_map, axis=-1, keepdims=True)
    n_map = v_map / (v_norm + 1e-12)
    d_map = 1.0 / (v_norm + 1e-12)
    
    # 8. Visualization
    rr.log("init/image", rr.Image(img1))
    rr.log("target/image", rr.Image(img2))
    
    flow_rgb = cv2.cvtColor(cv2.applyColorMap((np.linalg.norm(flow, axis=-1)*10).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    flow_rgb_masked = flow_rgb.copy()
    flow_rgb_masked[mask1 == 0] = 0
    rr.log("dis/flow_masked", rr.Image(flow_rgb_masked))
    
    # Masking
    mask_bool = (mask1 > 0)
    v_map_vis = v_map.copy()
    v_map_vis[~mask_bool] = 0
    
    # v_map can have large values, let's normalize for visualization
    v_map_norm = (v_map_vis - v_map_vis.min()) / (v_map_vis.max() - v_map_vis.min() + 1e-8)
    rr.log("v_field/rgb", rr.Image((v_map_norm * 255).astype(np.uint8)))
    
    n_vis = (n_map + 1.0) / 2.0
    n_vis[~mask_bool] = 0
    rr.log("estimated/normal", rr.Image((n_vis.clip(0, 1) * 255).astype(np.uint8)))
    
    d_vis = d_map[..., 0]
    d_vis[~mask_bool] = 0
    rr.log("estimated/distance_d", rr.Image(depth_to_rgb(d_vis)))
    
    # Compare with GT depth if available
    depth1_path = data_dir / "depth_dyn" / f"{cfg.init_frame:06d}.npy"
    if depth1_path.exists():
        depth1_gt = np.load(depth1_path)
        rr.log("gt/depth", rr.Image(depth_to_rgb(depth1_gt)))
        
        # Error map (depth-based, keeping for comparison)
        error_depth = np.abs(d_vis - depth1_gt)
        error_depth[~mask_bool] = 0
        rr.log("error/depth", rr.Image(depth_to_rgb(error_depth, max_val=0.2)))

        # 9. GT Normal and Distance d
        depth1_gt_torch = torch.from_numpy(depth1_gt).float().to(device)
        K1_torch = torch.from_numpy(K1).float().to(device)
        pts1_C_gt_torch = unproject_depth(depth1_gt_torch, K1_torch, H, W) # [H, W, 3]
        
        # Compute GT normal
        # d2n_tblr expects (B, 3, H, W)
        normal1_gt_torch, _ = d2n_tblr(pts1_C_gt_torch.permute(2, 0, 1).unsqueeze(0))
        normal1_gt = -normal1_gt_torch[0].permute(1, 2, 0).cpu().numpy() # [H, W, 3]
        
        # Compute GT distance d: d = -n^T p
        pts1_C_gt_np = pts1_C_gt_torch.cpu().numpy()
        d1_gt = -np.sum(normal1_gt * pts1_C_gt_np, axis=-1)
        
        # Visualize GT
        n1_gt_vis = (normal1_gt + 1.0) / 2.0
        n1_gt_vis[~mask_bool] = 0
        rr.log("gt/normal", rr.Image((n1_gt_vis.clip(0, 1) * 255).astype(np.uint8)))
        
        d1_gt_vis = d1_gt.copy()
        d1_gt_vis[~mask_bool] = 0
        rr.log("gt/distance_d", rr.Image(depth_to_rgb(d1_gt_vis)))

        # Error in distance d
        error_dist = np.abs(d_vis - d1_gt)
        error_dist[~mask_bool] = 0
        rr.log("error/distance_d", rr.Image(depth_to_rgb(error_dist, max_val=0.2)))

        # 10. Derive GT Optical Flow from GT n/d
        # v_gt = n_gt / d_gt
        # homography per pixel: H = R - t * v_gt^T
        v_gt = normal1_gt / (d1_gt[..., None] + 1e-12) # [H, W, 3]
        v_gt_flat = v_gt.reshape(-1, 3)
        
        # H_batch = R - t * v^T
        tnT_d_gt = np.matmul(t21.reshape(3, 1), v_gt_flat.reshape(-1, 1, 3)) # [HW, 3, 3]
        H_batch_gt = R21.reshape(1, 3, 3) - tnT_d_gt
        H_batch_gt_pix = K2 @ H_batch_gt @ K1_inv
        
        # Project x1 through H
        u1_homog_vec = u1.reshape(-1, 3, 1)
        u2_homog_gt = np.matmul(H_batch_gt_pix, u1_homog_vec).squeeze(-1) # [HW, 3]
        u2_gt = u2_homog_gt[:, :2] / (u2_homog_gt[:, 2:3] + 1e-12)
        flow_gt = (u2_gt - u1[:, :2]).reshape(H, W, 2)
        
        # Visualize GT Flow
        flow_gt_rgb = cv2.cvtColor(cv2.applyColorMap((np.linalg.norm(flow_gt, axis=-1)*10).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        flow_gt_rgb[mask1 == 0] = 0
        rr.log("gt/flow", rr.Image(flow_gt_rgb))
        
        # Flow Error
        flow_error = np.linalg.norm(flow - flow_gt, axis=-1)
        flow_error[mask1 == 0] = 0
        rr.log("error/flow", rr.Image(depth_to_rgb(flow_error, max_val=10.0)))
        
        mean_flow_err = np.median(flow_error[mask1 > 0])
        print(f"Median Flow Error: {mean_flow_err:.4f} pixels")

    # Log 3D point cloud from recovered n, d
    # p = z1 * x1, z1 = d / (n^T x1)? No, z1 = -d / (n^T x1).
    # Wait, s = -1/z1. So z1 = -1/s.
    z1_rec = -1.0 / (s_map + 1e-12)
    z1_rec[~mask_bool] = 0
    pts1_C = (z1_rec.reshape(-1, 1) * x1).reshape(H, W, 3)
    
    # Transform to WORLD for visualization
    T_WC1 = np.load(data_dir / "extrinsics" / f"{cfg.init_frame:06d}.npy")
    pts1_W = (T_WC1[:3, :3] @ pts1_C.reshape(-1, 3).T).T + T_WC1[:3, 3]
    colors1 = img1.reshape(-1, 3)
    
    valid = mask_bool.reshape(-1) & (z1_rec.reshape(-1) > 0.01) & (z1_rec.reshape(-1) < 2.0)
    rr.log("world/recovered_pc", rr.Points3D(pts1_W[valid], colors=colors1[valid], radii=0.001))

    # Log Normal Field as Arrows in 3D
    y_idx, x_idx = np.where(mask_bool)
    if len(x_idx) > 0:
        num_arrows = 2000
        indices = np.random.choice(len(x_idx), min(num_arrows, len(x_idx)), replace=False)
        sel_y, sel_x = y_idx[indices], x_idx[indices]
        origins = pts1_W.reshape(H, W, 3)[sel_y, sel_x]
        vectors = n_map[sel_y, sel_x] * 0.05
        rr.log("world/normal_field", rr.Arrows3D(origins=origins, vectors=vectors, colors=[0, 255, 0]))

    print("Finished.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
