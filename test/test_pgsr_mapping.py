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
import random
import time
import json
import matplotlib.pyplot as plt
import io
from PIL import Image

# From local project
from obj_gs_mapping import GSMapping, MappingConfig, build_rotation_from_normal
from gs_dyn_obj.gs_param import GSParam
from gs_dyn_obj.grouped_gs import RGB2SH
from gs_dyn_obj.gs_rendering import render_2dgs
from gs_dyn_obj.utils.init import unproject_depth, d2n_tblr
from gs_dyn_obj.obj_gs import ObjectGS
from run_multi_view_loss import compute_multi_view_loss
from run_single_view_loss import compute_single_view_loss

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    init_frame: int = 30
    n_frames: int = 30
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    no_vis: bool = False
    
    # Optimization Parameters
    num_steps_per_frame: int = 100
    lr_means: float = 1e-4
    lr_quats: float = 1e-4
    lr_scales: float = 1e-3
    lr_colors: float = 1e-3
    lr_opacity: float = 5e-2
    
    # Loss Weights
    lambda_rgb: float = 1.0
    lambda_ssim: float = 0.2
    lambda_sv: float = 0.015
    lambda_mask: float = 0.1
    lambda_mv_photo: float = 1.0
    lambda_mv_ncc: float = 0.1
    
    # Keyframe management
    kf_window_size: int = 10
    kf_sample_num: int = 2

class CameraDevice:
    def __init__(self, K, T_CO, width, height, ncc_scale=1.0):
        self.device = T_CO.device
        self.world_view_transform = T_CO.t().contiguous()
        self.K = K
        self.width = width
        self.height = height
        self.ncc_scale = ncc_scale
        
    def get_k(self, scale=1.0):
        K = self.K.clone()
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] *= scale
        K[1, 2] *= scale
        return K

    def get_inv_k(self, scale=1.0):
        return torch.inverse(self.get_k(scale))

    def get_calib_matrix_nerf(self, scale=1.0):
        return self.get_k(scale), self.world_view_transform

def load_object_pose_world(clip_dir, frame_idx):
    poses_file = Path(clip_dir) / "object_poses.txt"
    if not poses_file.exists(): return None
    with open(poses_file, "r") as f:
        lines = f.readlines()
    if frame_idx >= len(lines): return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])]) # x, y, z, w
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4)
    T_WO[:3, :3] = r_WO; T_WO[:3, 3] = t_WO
    return T_WO

def load_frame_data_gt(clip_dir, frame_idx):
    stem = f"{frame_idx:06d}"
    img_path = clip_dir / "images" / f"{stem}.png"
    if not img_path.exists():
        img_path = clip_dir / "images" / f"{frame_idx:05d}.jpg"
    if not img_path.exists(): return None
    img = np.array(cv2.imread(str(img_path))[..., ::-1])
    mask_gt_path = clip_dir / "obj_masks" / f"{stem}.png"
    depth_gt_path = clip_dir / "depth_dyn" / f"{stem}.npy"
    if not mask_gt_path.exists(): return None
    mask = np.array(cv2.imread(str(mask_gt_path), cv2.IMREAD_GRAYSCALE))
    if not depth_gt_path.exists(): return None
    depth = np.load(depth_gt_path)
    if depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    K = np.load(clip_dir / "intrinsics" / f"{stem}.npy")
    T_CW_gt = np.load(clip_dir / "extrinsics" / f"{stem}.npy")
    T_WO_gt = load_object_pose_world(clip_dir, frame_idx)
    if T_WO_gt is None: return None
    T_CO_gt = T_CW_gt @ T_WO_gt
    return {"image": img, "mask": mask, "depth": depth, "K": K, "T_CO_gt": T_CO_gt, "frame_idx": frame_idx}

def init_gs_from_gt(fd, device):
    img = fd["image"]
    mask = fd["mask"] > 0
    depth = fd["depth"]
    K = fd["K"]
    T_CO = fd["T_CO_gt"]
    H, W = img.shape[:2]
    y, x = np.where(mask & (depth > 0))
    if len(y) == 0: return None
    max_pts = 30000
    if len(y) > max_pts:
        perm = np.random.permutation(len(y))[:max_pts]
        y, x = y[perm], x[perm]
    z = depth[y, x]
    pts_c = np.stack([(x - K[0, 2]) * z / K[0, 0], (y - K[1, 2]) * z / K[1, 1], z], axis=-1)
    T_OC = np.linalg.inv(T_CO)
    pts_o = (pts_c @ T_OC[:3, :3].T) + T_OC[:3, 3]
    colors = img[y, x] / 255.0
    num_pts = pts_o.shape[0]
    means = torch.from_numpy(pts_o).float().to(device); means.requires_grad = True
    colors_sh = RGB2SH(torch.from_numpy(colors).float().to(device)); colors_sh.requires_grad = True
    full_pts_c = unproject_depth(torch.from_numpy(depth).float().to(device), torch.from_numpy(K).float().to(device), H, W)
    normals_c, _ = d2n_tblr(full_pts_c.permute(2, 0, 1).unsqueeze(0))
    normals_c = -F.normalize(normals_c[0][:, y, x].permute(1, 0), dim=1).cpu().numpy()
    normals_o = (normals_c @ T_OC[:3, :3].T)
    quats = build_rotation_from_normal(torch.from_numpy(normals_o).float().to(device)); quats.requires_grad = True
    f = 0.5 * (K[0, 0] + K[1, 1])
    sizes = z / f
    scales = torch.log(torch.from_numpy(sizes).float().to(device).view(-1, 1).repeat(1, 2)); scales.requires_grad = True
    opacity = torch.logit(torch.ones((num_pts, 1), device=device) * 0.5); opacity.requires_grad = True
    return GSParam(means, quats, scales, colors_sh, opacity)

def ssim_loss(img1, img2, window_size=11):
    from gs_dyn_obj.utils.ssim import ssim as ssim_func
    return 1.0 - ssim_func(img1, img2, window_size)

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("pgsr_mapping", spawn=False)
        if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
    clip_dir = Path(cfg.data_root) / cfg.clip_id
    device = torch.device(cfg.device)
    f0 = load_frame_data_gt(clip_dir, cfg.init_frame)
    if f0 is None: return
    gs_params = init_gs_from_gt(f0, device)
    if gs_params is None: return
    optimizer = torch.optim.Adam([
        {'params': [gs_params.means], 'lr': cfg.lr_means},
        {'params': [gs_params.quats], 'lr': cfg.lr_quats},
        {'params': [gs_params.scales], 'lr': cfg.lr_scales},
        {'params': [gs_params.colors], 'lr': cfg.lr_colors},
        {'params': [gs_params.opacity], 'lr': cfg.lr_opacity},
    ])
    keyframes = []; online_psnrs = []
    pbar = tqdm(range(cfg.n_frames), desc="PGSR Mapping")
    for i in pbar:
        idx = cfg.init_frame + i
        fd = load_frame_data_gt(clip_dir, idx)
        if fd is None: break
        H, W = fd["image"].shape[:2]
        T_CO_t = torch.from_numpy(fd["T_CO_gt"]).float().to(device)
        K_t = torch.from_numpy(fd["K"]).float().to(device)
        target_image = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0
        target_mask = torch.from_numpy(fd["mask"] > 0).to(device)
        target_image_gray = target_image.mean(dim=0, keepdim=True)
        cam_curr = CameraDevice(K_t, T_CO_t, W, H)
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        all_pixels = torch.stack([grid_x, grid_y], dim=-1).float().reshape(-1, 2)
        mask_flat = target_mask.reshape(-1)
        pixels_masked = all_pixels[mask_flat]
        for step in range(cfg.num_steps_per_frame):
            optimizer.zero_grad()
            img_curr, depth_curr, normal_curr, alpha_curr = render_2dgs(
                gs_params.means, F.normalize(gs_params.quats), 
                torch.exp(gs_params.scales), gs_params.colors, 
                torch.sigmoid(gs_params.opacity),
                viewmat=T_CO_t, K=K_t, width=W, height=H
            )
            loss_photo = cfg.lambda_rgb * F.l1_loss(img_curr * target_mask, target_image * target_mask) + cfg.lambda_ssim * ssim_loss(img_curr, target_image)
            loss_sv = compute_single_view_loss(normal_curr, depth_curr, target_image, cam_curr, mask=target_mask)
            loss_mask = F.l1_loss(alpha_curr[0], target_mask.float())
            loss_mv = torch.tensor(0.0, device=device)
            if len(keyframes) > 0:
                kf_indices = random.sample(range(len(keyframes)), min(len(keyframes), cfg.kf_sample_num))
                for kf_idx in kf_indices:
                    kf = keyframes[kf_idx]
                    img_nea, _, _, alpha_nea = render_2dgs(gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales), gs_params.colors, torch.sigmoid(gs_params.opacity), viewmat=kf["camera"].world_view_transform.t(), K=kf["camera"].K, width=kf["camera"].width, height=kf["camera"].height)
                    loss_mv += cfg.lambda_mv_photo * F.l1_loss(img_nea * kf["mask"], kf["image"] * kf["mask"]) / len(kf_indices)
                    n_C_gs, d_gs = ObjectGS.gs_to_planar_params(gs_params, T_CO_t)
                    n_map, _ = ObjectGS.render_custom_attribute(gs_params, n_C_gs, T_CO_t, K_t, W, H)
                    d_map = ObjectGS.render_custom_attribute(gs_params, d_gs.repeat(1, 3), T_CO_t, K_t, W, H)[0][0:1]
                    ncc, ncc_mask = compute_multi_view_loss(target_image_gray, kf["image_gray"].unsqueeze(0), n_map, d_map, cam_curr, kf["camera"], pixels=pixels_masked, valid_indices=mask_flat)
                    if ncc_mask.any(): loss_mv += cfg.lambda_mv_ncc * ncc[ncc_mask].mean() / len(kf_indices)
            total_loss = loss_photo + cfg.lambda_sv * loss_sv + cfg.lambda_mask * loss_mask + loss_mv
            total_loss.backward(); optimizer.step()
            with torch.no_grad(): gs_params.quats.data = F.normalize(gs_params.quats.data, p=2, dim=-1)
        with torch.no_grad():
            img_final, _, _, _ = render_2dgs(gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales), gs_params.colors, torch.sigmoid(gs_params.opacity), viewmat=T_CO_t, K=K_t, width=W, height=H)
            mse = torch.mean((img_final[:, target_mask] - target_image[:, target_mask])**2)
            psnr = -10.0 * torch.log10(mse + 1e-10); online_psnrs.append(psnr.item())
            pbar.set_postfix({"PSNR": f"{psnr.item():.2f}", "GS": len(gs_params.means)})
            if not cfg.no_vis:
                rr.set_time("frame_idx", sequence=idx)
                rr.log("input/image", rr.Image(fd["image"]))
                rr.log("render/image", rr.Image((img_final.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)))
        if len(keyframes) >= cfg.kf_window_size: keyframes.pop(0)
        keyframes.append({"camera": cam_curr, "image": target_image.detach(), "mask": target_mask.detach(), "image_gray": target_image_gray.detach()[0], "frame_idx": idx})
    print(f"\nAverage Online PSNR: {np.mean(online_psnrs):.2f} dB")
    final_psnrs = []
    for i in tqdm(range(cfg.n_frames), desc="Final Evaluation"):
        idx = cfg.init_frame + i; fd = load_frame_data_gt(clip_dir, idx)
        if fd is None: break
        with torch.no_grad():
            T_CO_t = torch.from_numpy(fd["T_CO_gt"]).float().to(device); K_t = torch.from_numpy(fd["K"]).float().to(device)
            target_image = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0; target_mask = torch.from_numpy(fd["mask"] > 0).to(device)
            img_render, _, _, _ = render_2dgs(gs_params.means, F.normalize(gs_params.quats), torch.exp(gs_params.scales), gs_params.colors, torch.sigmoid(gs_params.opacity), viewmat=T_CO_t, K=K_t, width=W, height=H)
            if target_mask.any():
                mse = torch.mean((img_render[:, target_mask] - target_image[:, target_mask])**2)
                final_psnrs.append((-10.0 * torch.log10(mse + 1e-10)).item())
    if final_psnrs:
        avg_final_psnr = np.mean(final_psnrs)
        print(f"\nAverage Final PSNR: {avg_final_psnr:.2f} dB")
        with open(f"results_pgsr_{cfg.clip_id}.json", "w") as f: json.dump({"clip_id": cfg.clip_id, "avg_online_psnr": np.mean(online_psnrs), "avg_final_psnr": avg_final_psnr, "gs_count": len(gs_params.means)}, f, indent=4)

if __name__ == "__main__":
    main(tyro.cli(Config))
