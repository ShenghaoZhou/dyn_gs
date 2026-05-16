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
import rerun.blueprint as rrb
import random
import threading
import time
import json

from obj_gs_mapping import GSMapping, MappingConfig, MiniCam, get_scaled_cam, build_rotation_from_normal
from gs_dyn_obj.gs_param import GSParam
from gs_dyn_obj.grouped_gs import RGB2SH
from gs_dyn_obj.gs_rendering import render_2dgs
from gs_dyn_obj.utils.init import unproject_depth, d2n_tblr

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    init_frame: int = 30
    n_frames: int = 50
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    no_vis: bool = False

def load_object_pose_world(clip_dir, frame_idx):
    poses_file = Path(clip_dir) / "object_poses.txt"
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

def load_frame_data_gt(clip_dir, frame_idx):
    stem = f"{frame_idx:06d}"
    img_path = clip_dir / "images" / f"{stem}.png"
    if not img_path.exists():
        img_path = clip_dir / "images" / f"{frame_idx:05d}.jpg"
    if not img_path.exists():
        return None
    
    img = np.array(cv2.imread(str(img_path))[..., ::-1])
    
    # GT depth/mask
    mask_gt_path = clip_dir / "obj_masks" / f"{stem}.png"
    depth_gt_path = clip_dir / "depth_dyn" / f"{stem}.npy"
    
    if not mask_gt_path.exists():
        return None
    
    mask = np.array(cv2.imread(str(mask_gt_path), cv2.IMREAD_GRAYSCALE))
    
    if not depth_gt_path.exists():
        return None
    
    depth = np.load(depth_gt_path)
    if depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    K = np.load(clip_dir / "intrinsics" / f"{stem}.npy")
    T_CW_gt = np.load(clip_dir / "extrinsics" / f"{stem}.npy")
    T_WO_gt = load_object_pose_world(clip_dir, frame_idx)
    
    if T_WO_gt is None:
        return None
        
    T_CiO_gt = T_CW_gt @ T_WO_gt
    
    return {
        "image": img, 
        "mask": mask, 
        "depth": depth, 
        "K": K, 
        "T_CO_gt": T_CiO_gt, 
        "frame_idx": frame_idx
    }

def init_gs_from_gt(fd, device):
    img = fd["image"]
    mask = fd["mask"] > 0
    depth = fd["depth"]
    K = fd["K"]
    T_CO = fd["T_CO_gt"]
    
    H, W = img.shape[:2]
    y, x = np.where(mask & (depth > 0))
    if len(y) == 0:
        return None
        
    # Sample points
    max_pts = 8000
    if len(y) > max_pts:
        perm = np.random.permutation(len(y))[:max_pts]
        y, x = y[perm], x[perm]
        
    z = depth[y, x]
    pts_c = np.stack([(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z], axis=-1)
    
    # Transform to Object space
    T_OC = np.linalg.inv(T_CO)
    pts_o = (pts_c @ T_OC[:3, :3].T) + T_OC[:3, 3]
    colors = img[y, x] / 255.0
    
    num_pts = pts_o.shape[0]
    means = torch.from_numpy(pts_o).float().to(device)
    colors_sh = RGB2SH(torch.from_numpy(colors).float().to(device))
    
    # Normals from depth
    full_pts_c = unproject_depth(torch.from_numpy(depth).float().to(device), torch.from_numpy(K).float().to(device), H, W)
    normals_c, _ = d2n_tblr(full_pts_c.permute(2, 0, 1).unsqueeze(0))
    normals_c = -F.normalize(normals_c[0][:, y, x].permute(1, 0), dim=1).cpu().numpy()
    normals_o = (normals_c @ T_OC[:3, :3].T)
    
    quats = build_rotation_from_normal(torch.from_numpy(normals_o).float().to(device))
    
    scales = torch.log(torch.ones((num_pts, 2), device=device) * 0.01)
    opacity = torch.logit(torch.ones((num_pts, 1), device=device) * 0.99)
    
    return GSParam(means, quats, scales, colors_sh, opacity)

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("gs_mapping_gt_loading", spawn=False)
        if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
        
    clip_dir = Path(cfg.data_root) / cfg.clip_id
    device = torch.device(cfg.device)
    
    # Load first frame to initialize GS
    f0 = load_frame_data_gt(clip_dir, cfg.init_frame)
    if f0 is None:
        print(f"Failed to load initial frame {cfg.init_frame}")
        return
        
    gs_params = init_gs_from_gt(f0, device)
    if gs_params is None:
        print("Failed to initialize GS params")
        return
    
    print(f"Initialized GS with {len(gs_params.means)} points from frame {cfg.init_frame}")
    
    pbar = tqdm(range(cfg.n_frames), desc="GT Loading & Visualization")
    for i in pbar:
        idx = cfg.init_frame + i
        fd = load_frame_data_gt(clip_dir, idx)
        if fd is None: break
        
        # Render the GS using GT pose
        with torch.no_grad():
            T_CO_t = torch.from_numpy(fd["T_CO_gt"]).float().to(device)
            K_t = torch.from_numpy(fd["K"]).float().to(device)
            H, W = fd["image"].shape[:2]
            
            render_image, _, _, render_alpha = render_2dgs(
                gs_params.means, F.normalize(gs_params.quats), 
                torch.exp(gs_params.scales), gs_params.colors, 
                torch.sigmoid(gs_params.opacity),
                viewmat=T_CO_t, K=K_t, width=W, height=H
            )
            
            render_image = render_image.clamp(0, 1)
            target_image = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0
            # Mask evaluation
            gt_mask = torch.from_numpy(fd["mask"] > 0).to(device).float()
            pred_mask = render_alpha[0] # [H, W]
            intersection = (gt_mask * pred_mask).sum()
            union = gt_mask.sum() + pred_mask.sum() - intersection
            iou = (intersection / (union + 1e-6)).item()
            mask_loss = F.l1_loss(pred_mask, gt_mask).item()

            if gt_mask.any():
                mse = torch.mean((render_image[:, gt_mask > 0.5] - target_image[:, gt_mask > 0.5])**2)
                psnr = -10.0 * torch.log10(mse + 1e-10)
                pbar.set_postfix({"PSNR": f"{psnr.item():.2f}", "IOU": f"{iou:.3f}", "MaskL1": f"{mask_loss:.3f}"})

            if not cfg.no_vis:
                rr.set_time("frame_idx", sequence=idx)
                
                # Log inputs
                rr.log("input/image", rr.Image(fd["image"]))
                rr.log("input/mask", rr.Image(fd["mask"]))
                
                # Log render
                render_np = (render_image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                rr.log("render/image", rr.Image(render_np))
                rr.log("render/mask", rr.Image((pred_mask.cpu().numpy() * 255).astype(np.uint8)))
                
                # Log object points in object space
                means_np = gs_params.means.cpu().numpy()
                colors_sh0 = gs_params.colors.cpu()
                if colors_sh0.dim() == 3: colors_sh0 = colors_sh0[:, 0, :]
                colors_rgb = (colors_sh0 * 0.28209 + 0.5).clamp(0, 1).numpy()
                rr.log("object/points", rr.Points3D(means_np, colors=colors_rgb))
                
                # Log camera in object space
                T_OC = np.linalg.inv(fd["T_CO_gt"])
                rr.log("object/camera", rr.Transform3D(translation=T_OC[:3, 3], mat3x3=T_OC[:3, :3]))
                rr.log("object/camera", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
                
                # Optional: transform points to camera space for visualization
                # T_OC = np.linalg.inv(fd["T_CO_gt"])
                # pts_c = (means_np @ fd["T_CO_gt"][:3, :3].T) + fd["T_CO_gt"][:3, 3]
                # rr.log("camera/points", rr.Points3D(pts_c, colors=colors_rgb))

if __name__ == "__main__":
    main(tyro.cli(Config))
