import numpy as np
import cv2
import rerun as rr
import torch
from pathlib import Path
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import torch.nn.functional as F
from gs_dyn_obj.utils.init import d2n_tblr, unproject_depth

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    target_frame: int = 40
    device: str = "cuda"
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    patch_size: int = 8
    num_samples: int = 10
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    dis_coarsest_scale: int = 8
    dis_variational_refinement_iter: int = 5

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

def sample_grid_on_mask(mask, num_samples):
    """Samples points on the object mask."""
    yy, xx = np.where(mask > 0)
    if len(xx) == 0:
        return np.array([])
    
    indices = np.random.choice(len(xx), min(num_samples, len(xx)), replace=False)
    pts = np.stack([xx[indices], yy[indices]], axis=-1).astype(np.float32)
    return pts

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

def get_patch(img, pt, size):
    """Extracts a patch centered at pt."""
    x, y = int(round(pt[0])), int(round(pt[1]))
    h, w = img.shape[:2]
    r = size // 2
    
    x0, x1 = max(0, x - r), min(w, x + r)
    y0, y1 = max(0, y - r), min(h, y + r)
    
    patch = img[y0:y1, x0:x1]
    
    # Pad if out of bounds
    if patch.shape[0] < size or patch.shape[1] < size:
        pad_y0 = max(0, r - y)
        pad_y1 = max(0, y + r - h)
        pad_x0 = max(0, r - x)
        pad_x1 = max(0, x + r - w)
        patch = np.pad(patch, ((pad_y0, pad_y1), (pad_x0, pad_x1), (0, 0)), mode='constant')
        
    return patch

def main(cfg: Config):
    rr.init("test_dis_twoview", spawn=False)
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)
    
    data_dir = Path(cfg.data_root)
    
    # 1. Load data for both frames
    def load_frame_data(frame_idx):
        stem = f"{frame_idx:06d}"
        img = np.array(cv2.imread(str(data_dir / "images" / f"{stem}.png"))[..., ::-1])
        mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        T_CO = load_pose_T_CO(cfg.data_root, frame_idx)
        return img, mask, K, T_CO

    img_init, mask_init, K1, T_CO1 = load_frame_data(cfg.init_frame)
    img_target, mask_target, K2, T_CO2 = load_frame_data(cfg.target_frame)
    
    H, W = img_init.shape[:2]
    device = torch.device(cfg.device)

    # 2. Run DIS densely on the mask area
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    dis.setVariationalRefinementIterations(cfg.dis_variational_refinement_iter)
    gray_init = cv2.cvtColor(img_init, cv2.COLOR_RGB2GRAY)
    gray_target = cv2.cvtColor(img_target, cv2.COLOR_RGB2GRAY)
    
    print("Computing DIS flow...")
    flow = dis.calc(gray_init, gray_target, None)
    
    # 3. Sample points on mask
    pts_init = sample_grid_on_mask(mask_init, cfg.num_samples)
    if len(pts_init) == 0:
        print("No mask found in init frame.")
        return

    # 4. Find matches using flow
    delta = interpolate_flow(flow, pts_init)
    pts_target = pts_init + delta
    
    # 5. Visualization
    H, W = img_init.shape[:2]
    combined_img = np.hstack([img_init, img_target])
    
    rr.log("images/init", rr.Image(img_init))
    rr.log("images/target", rr.Image(img_target))
    rr.log("images/combined", rr.Image(combined_img))
    
    # Draw lines on combined image
    line_strips = []
    for p0, p1 in zip(pts_init, pts_target):
        p1_offset = p1 + np.array([W, 0])
        line_strips.append([p0, p1_offset])
    
    rr.log("images/combined/matches", rr.LineStrips2D(line_strips, colors=[0, 255, 0], radii=2))
    rr.log("images/combined/pts_init", rr.Points2D(pts_init, colors=[255, 0, 0], radii=4))
    rr.log("images/combined/pts_target", rr.Points2D(pts_target + np.array([W, 0]), colors=[0, 0, 255], radii=4))

    # Log individual patches
    for i in range(len(pts_init)):
        patch_init = get_patch(img_init, pts_init[i], cfg.patch_size)
        patch_target = get_patch(img_target, pts_target[i], cfg.patch_size)
        
        # Concat patches for visualization
        patch_combined = np.hstack([patch_init, patch_target])
        rr.log(f"patches/match_{i:02d}", rr.Image(patch_combined))

    # Also log dense flow visualization
    hsv = np.zeros((H, W, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    # Mask the flow
    flow_rgb_masked = flow_rgb.copy()
    flow_rgb_masked[mask_init == 0] = 0
    rr.log("flow/dense", rr.Image(flow_rgb))
    rr.log("flow/dense_masked", rr.Image(flow_rgb_masked))

    # 6. Flow Error Reporting (Logic from exp_solve_n_d_dis_twoview_direct.py)
    depth1_path = data_dir / "depth_dyn" / f"{cfg.init_frame:06d}.npy"
    if depth1_path.exists() and T_CO1 is not None and T_CO2 is not None:
        print("Computing flow error reporting...")
        depth1_gt = np.load(depth1_path)
        
        # 10. Derive GT Optical Flow from GT n/d
        T_21 = T_CO2 @ np.linalg.inv(T_CO1)
        R21 = T_21[:3, :3]
        t21 = T_21[:3, 3]
        K1_inv = np.linalg.inv(K1)

        depth1_gt_torch = torch.from_numpy(depth1_gt).float().to(device)
        K1_torch = torch.from_numpy(K1).float().to(device)
        pts1_C_gt_torch = unproject_depth(depth1_gt_torch, K1_torch, H, W) # [H, W, 3]
        
        # Compute GT normal
        normal1_gt_torch, _ = d2n_tblr(pts1_C_gt_torch.permute(2, 0, 1).unsqueeze(0))
        normal1_gt = -normal1_gt_torch[0].permute(1, 2, 0).cpu().numpy() # [H, W, 3]
        
        # Compute GT distance d: d = -n^T p
        pts1_C_gt_np = pts1_C_gt_torch.cpu().numpy()
        d1_gt = -np.sum(normal1_gt * pts1_C_gt_np, axis=-1)
        
        # v_gt = n_gt / d_gt
        v_gt = normal1_gt / (d1_gt[..., None] + 1e-12) # [H, W, 3]
        v_gt_flat = v_gt.reshape(-1, 3)
        
        # H_batch = R - t * v^T
        tnT_d_gt = np.matmul(t21.reshape(3, 1), v_gt_flat.reshape(-1, 1, 3)) # [HW, 3, 3]
        H_batch_gt = R21.reshape(1, 3, 3) - tnT_d_gt
        H_batch_gt_pix = K2 @ H_batch_gt @ K1_inv
        
        # Grid for projection
        grid_y, grid_x = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        u1 = np.stack([grid_x, grid_y, np.ones_like(grid_x)], axis=-1).reshape(-1, 3) # [HW, 3]
        u1_homog_vec = u1.reshape(-1, 3, 1)
        
        # Project x1 through H
        u2_homog_gt = np.matmul(H_batch_gt_pix, u1_homog_vec).squeeze(-1) # [HW, 3]
        u2_gt = u2_homog_gt[:, :2] / (u2_homog_gt[:, 2:3] + 1e-12)
        flow_gt = (u2_gt - u1[:, :2]).reshape(H, W, 2)
        
        # Visualize GT Flow
        flow_gt_rgb = cv2.cvtColor(cv2.applyColorMap((np.linalg.norm(flow_gt, axis=-1)*10).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        flow_gt_rgb[mask_init == 0] = 0
        rr.log("gt/flow", rr.Image(flow_gt_rgb))
        
        # Flow Error
        flow_error = np.linalg.norm(flow - flow_gt, axis=-1)
        flow_error[mask_init == 0] = 0
        rr.log("error/flow", rr.Image(depth_to_rgb(flow_error, max_val=10.0)))
        
        mean_flow_err = np.median(flow_error[mask_init > 0])
        print(f"Median Flow Error: {mean_flow_err:.4f} pixels")

    print("Finished.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
