import torch
import torch.nn.functional as F
import numpy as np
import cv2
import rerun as rr
from pathlib import Path
import tyro
from dataclasses import dataclass
from batch_normal_integration import normal_integration_batch
from gs_dyn_obj.utils.init import unproject_depth
from run_single_view_loss import normal_from_depth_image
import time

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    frame_idx: int = 35
    device: str = "cuda"
    cg_max_iter: int = 5000
    cg_tol: float = 1e-3

def load_pose_T_WC(data_root, frame_idx):
    ext_file = Path(data_root) / "extrinsics" / f"{frame_idx:06d}.npy"
    if not ext_file.exists():
        return np.eye(4)
    return np.load(ext_file)

def setup_rerun():
    rr.init("test_normal_integration", spawn=False)
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")

def main(cfg: Config):
    setup_rerun()
    device = torch.device(cfg.device)
    data_dir = Path(cfg.data_root)
    stem = f"{cfg.frame_idx:06d}"

    # 1. Load Data
    print(f"Loading data for frame {cfg.frame_idx}...")
    image = cv2.imread(str(data_dir / "images" / f"{stem}.png"))[..., ::-1]
    mask = cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
    mask_bool = (mask > 0)
    
    # MoGE Normal Prior
    normal_moge = np.load(data_dir / "moge_normal" / f"{stem}.npy")
    if normal_moge.shape[0] == 3 and normal_moge.shape[1] != 3:
        normal_moge = normal_moge.transpose(1, 2, 0)
    
    # MoGE Depth Prior (for alignment check)
    depth_moge = np.load(data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{stem}.npy")
    
    K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
    T_WC = load_pose_T_WC(cfg.data_root, cfg.frame_idx)
    
    H, W = image.shape[:2]
    K_torch = torch.from_numpy(K).float().to(device)
    mask_t = torch.from_numpy(mask_bool).to(device)
    
    # 2. Alignment Check (Correct MoGE normal flip)
    print("Checking normal alignment...")
    depth_ref_normal = normal_from_depth_image(
        torch.from_numpy(depth_moge).float().to(device),
        K_torch
    ).permute(2, 0, 1) # [3, H, W] in camera space
    
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
    
    print(f"Detected Flip: {best_flip}, Alignment Dot: {best_dot:.4f}")
    normal_moge_corrected = normal_moge * np.array(best_flip).reshape(1, 1, 3)
    normal_moge_corr_t = torch.from_numpy(normal_moge_corrected).float().to(device)

    # 3. Normal Integration
    print("Starting normal integration...")
    # normal_integration_batch expects:
    # normal_map: [H, W, 3]
    # normal_mask: [N, H, W]
    # returns: [num_pixels]
    
    tic = time.time()
    depth_integrated_flat = normal_integration_batch(
        normal_moge_corr_t,
        mask_t.unsqueeze(0),
        K=K_torch,
        cg_max_iter=cfg.cg_max_iter,
        cg_tol=cfg.cg_tol,
        verbose=True
    )
    toc = time.time()
    print(f"Integration finished in {toc - tic:.3f} sec")
    
    # Reshape depth back to image
    depth_integrated = torch.zeros((H, W), device=device)
    depth_integrated[mask_t] = depth_integrated_flat
    
    # 4. Point Cloud Logging
    print("Logging to Rerun...")
    
    # Integrated Point Cloud
    valid_mask = (depth_integrated > 0.01) & mask_t
    pts_C = unproject_depth(depth_integrated, K_torch, H, W).reshape(-1, 3).cpu().numpy()
    valid_flat = valid_mask.reshape(-1).cpu().numpy()
    pts_W = (T_WC[:3, :3] @ pts_C[valid_flat].T).T + T_WC[:3, 3]
    
    colors = image.reshape(-1, 3)[valid_flat] / 255.0
    rr.log("world/integrated_pc", rr.Points3D(pts_W, colors=colors, radii=0.001))
    
    # Prior (MoGE) Point Cloud for Comparison
    pts_C_moge = unproject_depth(torch.from_numpy(depth_moge).float().to(device), K_torch, H, W).reshape(-1, 3).cpu().numpy()
    valid_moge = (depth_moge.reshape(-1) > 0.01) & mask_bool.reshape(-1)
    pts_W_moge = (T_WC[:3, :3] @ pts_C_moge[valid_moge].T).T + T_WC[:3, 3]
    colors_moge = image.reshape(-1, 3)[valid_moge] / 255.0
    rr.log("world/moge_prior_pc", rr.Points3D(pts_W_moge, colors=colors_moge, radii=0.001))

    # 2D Visualizations
    rr.log("2d/image", rr.Image(image))
    
    normal_vis = ((normal_moge_corrected + 1) / 2).clip(0, 1)
    normal_vis[~mask_bool] = 0
    rr.log("2d/normal_prior", rr.Image((normal_vis * 255).astype(np.uint8)))
    
    depth_vis = depth_integrated.cpu().numpy()
    depth_vis_scaled = (depth_vis - depth_vis[mask_bool].min()) / (depth_vis[mask_bool].max() - depth_vis[mask_bool].min() + 1e-6)
    depth_vis_scaled[~mask_bool] = 0
    rr.log("2d/depth_integrated", rr.Image((depth_vis_scaled * 255).astype(np.uint8)))

    # Log Camera
    rr.log("world/camera", rr.Pinhole(
        image_from_camera=K,
        width=W,
        height=H
    ))
    rr.log("world/camera", rr.Transform3D(
        translation=T_WC[:3, 3],
        mat3x3=T_WC[:3, :3]
    ))

    print("Done.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
