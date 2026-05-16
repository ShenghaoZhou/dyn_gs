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
from uniflowmatch.models.ufm import UniFlowMatchConfidence
import rerun.blueprint as rrb

def setup_blueprint():
    blueprint = rrb.Blueprint(
        rrb.Tabs(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial2DView(name="UFM Flow", contents=["ufm/flow_masked"]),
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
    ufm_model: str = "infinity1096/UFM-Base"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    smooth_ksize: int = 5
    min_covis: float = 0.5

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

def predict_ufm_correspondences(model, source_image, target_image):
    """Predict correspondences between source and target images."""
    device = next(model.parameters()).device
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            # Input images should be [H, W, 3] and uint8
            result = model.predict_correspondences_batched(
                source_image=torch.from_numpy(source_image).to(device),
                target_image=torch.from_numpy(target_image).to(device),
            )
        flow_output = result.flow.flow_output[0].cpu().float()
        covisibility = result.covisibility.mask[0].cpu().float()
    return flow_output, covisibility

def main(cfg: Config):
    rr.init("exp_solve_n_d_dis_twoview_ufm", spawn=False)
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
    
    # 2. Load UFM and compute flow
    print(f"Loading UFM model {cfg.ufm_model}...")
    model = UniFlowMatchConfidence.from_pretrained(cfg.ufm_model).to(device).eval()
    
    print("Computing UFM flow...")
    flow_ufm, covis_ufm = predict_ufm_correspondences(model, img1, img2)
    flow = flow_ufm.permute(1, 2, 0).numpy() # [H, W, 2]
    covis = covis_ufm.numpy() # [H, W]
    
    
    flow_rgb = cv2.cvtColor(cv2.applyColorMap((np.linalg.norm(flow, axis=-1)*10).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    flow_rgb_masked = flow_rgb.copy()
    flow_rgb_masked[mask1 == 0] = 0
    rr.log("ufm/flow_masked", rr.Image(flow_rgb_masked))
    rr.log("ufm/covisibility", rr.Image(covis))

    # 3. Compute relative pose T_21
    T_21 = T_CO2 @ np.linalg.inv(T_CO1)
    R21 = T_21[:3, :3]
    t21 = T_21[:3, 3]
    
    # 4. Compute s = -1/z1 for each pixel
    K1_inv = np.linalg.inv(K1)
    grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    u1 = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1).reshape(-1, 3) 
    x1 = (K1_inv @ u1.T).T 
    
    u2 = u1[:, :2] + flow.reshape(-1, 2)
    u2_homog = np.concatenate([u2, np.ones((len(u2), 1))], axis=1)
    x2 = (np.linalg.inv(K2) @ u2_homog.T).T 
    
    Rx1 = (R21 @ x1.T).T
    x2_cross_Rx1 = np.cross(x2, Rx1)
    x2_cross_t = np.cross(x2, t21)
    
    s_num = np.sum(x2_cross_Rx1 * x2_cross_t, axis=1)
    s_den = np.sum(x2_cross_t * x2_cross_t, axis=1)
    s = s_num / (s_den + 1e-12)
    s_map = s.reshape(H, W)
    
    # 5. Compute spatial gradients of s
    s_map_smooth = cv2.GaussianBlur(s_map, (cfg.smooth_ksize, cfg.smooth_ksize), 0)
    ds_du = cv2.Sobel(s_map_smooth, cv2.CV_64F, 1, 0, ksize=3)
    ds_dv = cv2.Sobel(s_map_smooth, cv2.CV_64F, 0, 1, ksize=3)
    
    # 6. Solve for v = n/d per pixel
    m_row1 = K1_inv[:, 0]
    m_row2 = K1_inv[:, 1]
    
    M = np.zeros((H, W, 3, 3))
    M[:, :, 0, :] = m_row1
    M[:, :, 1, :] = m_row2
    M[:, :, 2, :] = x1.reshape(H, W, 3)
    
    rhs = np.stack([ds_du, ds_dv, s_map], axis=-1)
    v_map = np.linalg.solve(M, rhs)
    
    # 7. Recover n and d
    v_norm = np.linalg.norm(v_map, axis=-1, keepdims=True)
    n_map = v_map / (v_norm + 1e-12)
    d_map = 1.0 / (v_norm + 1e-12)
    
    # 8. Visualization
    rr.log("init/image", rr.Image(img1))
    rr.log("target/image", rr.Image(img2))
    
    mask_bool = (mask1 > 0) & (covis > cfg.min_covis)
    
    v_map_vis = v_map.copy()
    v_map_vis[~mask_bool] = 0
    v_map_norm = (v_map_vis - v_map_vis.min()) / (v_map_vis.max() - v_map_vis.min() + 1e-8)
    rr.log("v_field/rgb", rr.Image((v_map_norm * 255).astype(np.uint8)))
    
    n_vis = (n_map + 1.0) / 2.0
    n_vis[~mask_bool] = 0
    rr.log("estimated/normal", rr.Image((n_vis.clip(0, 1) * 255).astype(np.uint8)))
    
    d_vis = d_map[..., 0]
    d_vis[~mask_bool] = 0
    rr.log("estimated/distance_d", rr.Image(depth_to_rgb(d_vis)))
    
    # Compare with GT
    depth1_path = data_dir / "depth_dyn" / f"{cfg.init_frame:06d}.npy"
    if depth1_path.exists():
        depth1_gt = np.load(depth1_path)
        depth1_gt_torch = torch.from_numpy(depth1_gt).float().to(device)
        K1_torch = torch.from_numpy(K1).float().to(device)
        pts1_C_gt_torch = unproject_depth(depth1_gt_torch, K1_torch, H, W)
        
        normal1_gt_torch, _ = d2n_tblr(pts1_C_gt_torch.permute(2, 0, 1).unsqueeze(0))
        normal1_gt = -normal1_gt_torch[0].permute(1, 2, 0).cpu().numpy()
        pts1_C_gt_np = pts1_C_gt_torch.cpu().numpy()
        d1_gt = -np.sum(normal1_gt * pts1_C_gt_np, axis=-1)
        
        n1_gt_vis = (normal1_gt + 1.0) / 2.0
        n1_gt_vis[~mask1.astype(bool)] = 0
        rr.log("gt/normal", rr.Image((n1_gt_vis.clip(0, 1) * 255).astype(np.uint8)))
        
        d1_gt_vis = d1_gt.copy()
        d1_gt_vis[~mask1.astype(bool)] = 0
        rr.log("gt/distance_d", rr.Image(depth_to_rgb(d1_gt_vis)))

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

    # 3D Point Cloud
    z1_rec = -1.0 / (s_map + 1e-12)
    z1_rec[~mask_bool] = 0
    pts1_C = (z1_rec.reshape(-1, 1) * x1).reshape(H, W, 3)
    T_WC1 = np.load(data_dir / "extrinsics" / f"{cfg.init_frame:06d}.npy")
    pts1_W = (T_WC1[:3, :3] @ pts1_C.reshape(-1, 3).T).T + T_WC1[:3, 3]
    colors1 = img1.reshape(-1, 3)
    valid = mask_bool.reshape(-1) & (z1_rec.reshape(-1) > 0.01) & (z1_rec.reshape(-1) < 2.0)
    rr.log("world/recovered_pc", rr.Points3D(pts1_W[valid], colors=colors1[valid], radii=0.001))

    # Log Normal Field as Arrows in 3D
    # Sample some points for arrows to avoid clutter
    y_idx, x_idx = np.where(mask_bool)
    if len(x_idx) > 0:
        num_arrows = 2000
        indices = np.random.choice(len(x_idx), min(num_arrows, len(x_idx)), replace=False)
        sel_y, sel_x = y_idx[indices], x_idx[indices]
        
        origins = pts1_W.reshape(H, W, 3)[sel_y, sel_x]
        vectors = n_map[sel_y, sel_x] * 0.05 # 5cm arrows
        rr.log("world/normal_field", rr.Arrows3D(origins=origins, vectors=vectors, colors=[0, 255, 0]))

    print("Finished.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
