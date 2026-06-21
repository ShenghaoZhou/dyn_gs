import os
import queue
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional, List, Tuple
import random
import threading

from gs_dyn_obj.grouped_gs import GaussianSuperPrimitive, RGB2SH
from gs_dyn_obj.gs_param import GSParam
from gs_dyn_obj.utils.init import unproject_depth, d2n_tblr
from gs_dyn_obj.gs_rendering import render_2dgs
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, quaternion_to_matrix

from run_single_view_loss import compute_single_view_loss
from run_multi_view_loss import compute_multi_view_loss

# Import PGSR-style loss utilities
import sys
sys.path.append(str(Path(__file__).parent / "third_party" / "PGSR"))
from utils.loss_utils import ssim, lncc, get_img_grad_weight

@dataclass
class MappingConfig:
    device: str = "cuda"
    lr_means: float = 5e-4
    lr_quats: float = 1e-3
    lr_scales: float = 1e-2
    lr_colors: float = 2.5e-3
    lr_opacity: float = 0.05
    
    num_steps_per_frame: int = 300
    pyr_levels: int = 2
    pyr_interval: int = 30
    
    kf_every: int = 5
    window_size: int = 20 # Increased window for better global consistency
    sample_num: int = 15000
    
    # Loss Weights
    lambda_dssim: float = 0.8
    lambda_sv: float = 0.015
    multi_view_photo_weight: float = 1.0
    multi_view_ncc_weight: float = 1.0
    multi_view_geo_weight: float = 0.5
    
    use_mask_loss: bool = True
    mask_loss_weight: float = 1.0
    
    # PGSR Parameters
    multi_view_min_dis: float = 0.02
    multi_view_pixel_noise_th: float = 1.0
    multi_view_patch_size: int = 3
    
    densify_every: int = 5
    prune_opacity_th: float = 0.01
    prune_screen_size_th: float = 100.0
    
    near_plane: float = 0.01
    far_plane: float = 100.0
    
    gs_type: str = "2d" # "2d" or "3d"
    use_pgsr: bool = True
    fix_color: bool = False
    fix_scale: bool = False
    
    # Virtual Camera
    use_virtul_cam: bool = True
    virtul_cam_prob: float = 0.5
    multi_view_max_dis: float = 0.1
    
    use_ray_dist: bool = True

class MiniCam:
    def __init__(self, K, extrin, width, height, ncc_scale=1.0):
        self.K = torch.from_numpy(K).float().cuda() if isinstance(K, np.ndarray) else K
        # world_view_transform should be the transpose of the world-to-camera (or object-to-camera) matrix
        extrin_t = torch.from_numpy(extrin).float().cuda() if isinstance(extrin, np.ndarray) else extrin
        self.world_view_transform = extrin_t.t().contiguous()
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

def get_scaled_cam(cam, scale):
    new_K = cam.get_k(scale)
    new_w = int(cam.width * scale)
    new_h = int(cam.height * scale)
    new_cam = MiniCam(new_K, cam.world_view_transform.t().cpu().numpy(), new_w, new_h, cam.ncc_scale)
    return new_cam

def get_lapla_norm(img, kernel=None):
    device = img.device
    laplacian_kernel = (
        torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=device, dtype=torch.float32
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )
    laplacian_kernel = laplacian_kernel.repeat(1, img.shape[0], 1, 1)
    laplacian = F.conv2d(img[None], laplacian_kernel, padding="same")
    laplacian_norm = torch.linalg.vector_norm(
        laplacian, ord=1, dim=1, keepdim=True)
    
    laplacian_norm[..., :, 0] = 0
    laplacian_norm[..., :, -1] = 0
    laplacian_norm[..., 0, :] = 0
    laplacian_norm[..., -1, :] = 0
    
    if kernel is not None:
        return F.conv2d(laplacian_norm, kernel, padding="same")[0, 0].clamp(0, 1)
    return laplacian_norm[0, 0].clamp(0, 1)

def get_depth_normal(d, K):
    H, W = d.shape[-2:]
    fy, fx = K[1, 1], K[0, 0]
    cy, cx = K[1, 2], K[0, 2]
    y, x = torch.meshgrid(torch.arange(H, device=d.device), torch.arange(W, device=d.device), indexing='ij')
    pts_c = torch.stack([(x - cx) * d[0] / fx, (y - cy) * d[0] / fy, d[0]], dim=-1)
    dz_dx = torch.gradient(pts_c[..., 0], dim=1)[0]
    dz_dy = torch.gradient(pts_c[..., 1], dim=0)[0]
    v1 = torch.stack([torch.ones_like(dz_dx), torch.zeros_like(dz_dx), dz_dx], dim=-1)
    v2 = torch.stack([torch.zeros_like(dz_dy), torch.ones_like(dz_dy), dz_dy], dim=-1)
    n = torch.cross(v1, v2, dim=-1)
    return -F.normalize(n, dim=-1).permute(2, 0, 1) # Point towards camera

def sample_patches(img, p):
    H, W = img.shape[-2:]
    p_norm = p.clone()
    p_norm[..., 0] = 2 * p_norm[..., 0] / (W - 1) - 1.0
    p_norm[..., 1] = 2 * p_norm[..., 1] / (H - 1) - 1.0
    val = F.grid_sample(img[None], p_norm.reshape(1, -1, 1, 2), align_corners=True)
    return val.reshape(-1, p.shape[1])

def compute_geo_consistency(cam1, cam2, depth1, depth2, pixel_noise_th=1.0):
    H, W = cam1.height, cam1.width
    device = depth1.device
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    pixels = torch.stack([grid_x, grid_y], dim=-1).float()
    K1_inv = cam1.get_inv_k().to(device)
    pixels_homo = torch.cat([pixels, torch.ones_like(pixels[..., :1])], dim=-1)
    pts1_c = depth1.reshape(H, W, 1) * (pixels_homo @ K1_inv.t())
    W2C1 = cam1.world_view_transform.t().to(device)
    C2W1 = torch.inverse(W2C1)
    pts1_w = (pts1_c @ C2W1[:3, :3].t()) + C2W1[:3, 3]
    W2C2 = cam2.world_view_transform.t().to(device)
    pts2_c = (pts1_w @ W2C2[:3, :3].t()) + W2C2[:3, 3]
    K2 = cam2.get_k().to(device)
    pts2_p = pts2_c @ K2.t()
    u2 = pts2_p[..., 0] / (pts2_p[..., 2] + 1e-6)
    v2 = pts2_p[..., 1] / (pts2_p[..., 2] + 1e-6)
    z2_proj = pts2_p[..., 2]
    grid2 = torch.stack([2 * u2 / (W - 1) - 1, 2 * v2 / (H - 1) - 1], dim=-1)
    depth2_sampled = F.grid_sample(depth2.reshape(1, 1, H, W), grid2.unsqueeze(0), align_corners=True).squeeze()
    pixels2_homo = torch.stack([u2, v2, torch.ones_like(u2)], dim=-1)
    K2_inv = cam2.get_inv_k().to(device)
    pts2_sampled_c = depth2_sampled.unsqueeze(-1) * (pixels2_homo @ K2_inv.t())
    C2W2 = torch.inverse(W2C2)
    pts2_sampled_w = (pts2_sampled_c @ C2W2[:3, :3].t()) + C2W2[:3, 3]
    pts1_back_c = (pts2_sampled_w @ W2C1[:3, :3].t()) + W2C1[:3, 3]
    pts1_back_p = pts1_back_c @ cam1.K.to(device).t()
    u1_back = pts1_back_p[..., 0] / (pts1_back_p[..., 2] + 1e-6)
    v1_back = pts1_back_p[..., 1] / (pts1_back_p[..., 2] + 1e-6)
    pixel_noise = torch.sqrt((u1_back - pixels[..., 0])**2 + (v1_back - pixels[..., 1])**2)
    d_mask = (z2_proj > 0.1) & (depth2_sampled > 0.1) & (pixel_noise < pixel_noise_th)
    weights = (1.0 / torch.exp(pixel_noise)).detach()
    weights[~d_mask] = 0
    return d_mask, weights, pixel_noise

def gen_virtul_cam(cam, trans_noise=0.1, deg_noise=15.0):
    W2C_T = cam.world_view_transform
    W2C = W2C_T.t()
    C2W = torch.inverse(W2C)
    translation_perturbation = torch.randn(3, device=C2W.device) * trans_noise
    rotation_perturbation = torch.randn(3, device=C2W.device) * deg_noise
    rx, ry, rz = torch.deg2rad(rotation_perturbation)
    Rx = torch.tensor([[1, 0, 0], [0, torch.cos(rx), -torch.sin(rx)], [0, torch.sin(rx), torch.cos(rx)]], device=C2W.device)
    Ry = torch.tensor([[torch.cos(ry), 0, torch.sin(ry)], [0, 1, 0], [-torch.sin(ry), 0, torch.cos(ry)]], device=C2W.device)
    Rz = torch.tensor([[torch.cos(rz), -torch.sin(rz), 0], [torch.sin(rz), torch.cos(rz), 0], [0, 0, 1]], device=C2W.device)
    R_perturbation = Rz @ Ry @ Rx
    C2W_new = C2W.clone()
    C2W_new[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W_new[:3, 3] = C2W[:3, 3] + translation_perturbation
    W2C_new = torch.inverse(C2W_new)
    return MiniCam(cam.K, W2C_new.t().cpu().numpy(), cam.width, cam.height, cam.ncc_scale)

def build_rotation_from_normal(normal):
    device = normal.device
    num_pts = normal.shape[0]
    ref_axis1 = torch.randn(num_pts, 3, device=device)
    ref_axis1 = F.normalize(ref_axis1, dim=1)
    rotation_axis1 = ref_axis1 - (normal * ref_axis1).sum(dim=1, keepdim=True) * normal
    rotation_axis1 = F.normalize(rotation_axis1, dim=1)
    rotation_axis2 = torch.cross(normal, rotation_axis1, dim=1)
    rotation_axis2 = F.normalize(rotation_axis2, dim=1)
    R = torch.stack([rotation_axis1, rotation_axis2, normal], dim=-1)
    det = torch.linalg.det(R)
    flip_mask = det < 0
    R[flip_mask, :, 1] *= -1
    return matrix_to_quaternion(R)

def init_gs_from_tracker_points(points, colors, device, normals=None, ray_o=None, ray_d=None, ray_dist=None, gs_type="2d", fx=None, fy=None):
    num_pts = points.shape[0]
    means = torch.from_numpy(points).float().to(device).requires_grad_(True)
    colors_sh = RGB2SH(torch.from_numpy(colors).float().to(device)).requires_grad_(True)
    if normals is not None:
        quats = build_rotation_from_normal(torch.from_numpy(normals).float().to(device)).requires_grad_(True)
    else:
        quats = torch.zeros((num_pts, 4), device=device); quats[:, 0] = 1.0
        quats = quats.requires_grad_(True)
    
    # Determine scale dynamically using perspective projection (matching SceneModel logic)
    if fx is None or fy is None:
        f_avg = 500.0
    else:
        f_avg = 0.5 * (fx + fy)
        
    pixel_size = 1.5
    angular_factor = pixel_size / f_avg
    
    if ray_dist is not None:
        ray_dist_t = torch.from_numpy(ray_dist).float().to(device) if isinstance(ray_dist, np.ndarray) else ray_dist.float().to(device)
        physical_scale = angular_factor * ray_dist_t.squeeze(-1)
    else:
        physical_scale = torch.ones(num_pts, device=device) * (angular_factor * 0.5)
        
    physical_scale = physical_scale.clamp(1e-6, 1e6)
    
    if gs_type == "3d" or gs_type == "normal":
        scales = torch.log(physical_scale.unsqueeze(-1).repeat(1, 3)).requires_grad_(True)
    else:
        scales = torch.log(physical_scale.unsqueeze(-1).repeat(1, 2)).requires_grad_(True)
    opacity = torch.logit(torch.ones((num_pts, 1), device=device) * 0.7).requires_grad_(True)
    
    if ray_o is not None: ray_o = ray_o.to(device)
    if ray_d is not None: ray_d = ray_d.to(device)
    if ray_dist is not None: ray_dist = ray_dist.to(device)
        
    return GSParam(means, quats, scales, colors_sh, opacity, ray_o, ray_d, ray_dist)

class GSMapping:
    def __init__(self, cfg: MappingConfig, initial_gs: GSParam):
        self.cfg = cfg
        self.gs_params = initial_gs
        self.data_queue = mp.Queue(maxsize=1)
        self.stop_event = mp.Event()
        self.last_finished_frame = -1
        self.lock = threading.Lock()

    def setup_optimizer(self):
        params = [
            {"params": [self.gs_params.quats], "lr": self.cfg.lr_quats},
            {"params": [self.gs_params.opacity], "lr": self.cfg.lr_opacity},
        ]
        if self.cfg.use_ray_dist and self.gs_params.ray_dist is not None:
            params.append({"params": [self.gs_params.ray_dist], "lr": self.cfg.lr_means})
        else:
            params.append({"params": [self.gs_params.means], "lr": self.cfg.lr_means})
            
        if not self.cfg.fix_scale:
            params.append({"params": [self.gs_params.scales], "lr": self.cfg.lr_scales})
        if not self.cfg.fix_color:
            params.append({"params": [self.gs_params.colors], "lr": self.cfg.lr_colors})
        
        self.optimizer = torch.optim.Adam(params)

    def update(self, frame_data):
        cpu_frame_data = {}
        for k, v in frame_data.items():
            if isinstance(v, torch.Tensor):
                cpu_frame_data[k] = v.detach().cpu()
            else:
                cpu_frame_data[k] = v
        try:
            self.data_queue.put_nowait(cpu_frame_data)
        except queue.Full:
            pass

    def stop(self):
        self.stop_event.set()

    def run(self):
        try:
            print("GS Mapping process started.")
            self.device = torch.device(self.cfg.device)
            self.gs_params.to(self.device)
            
            # Ensure parameters are optimizable
            if self.cfg.use_ray_dist and self.gs_params.ray_dist is not None:
                self.gs_params.ray_dist.requires_grad = True
                self.gs_params.means.requires_grad = False
            else:
                self.gs_params.means.requires_grad = True
                if self.gs_params.ray_dist is not None:
                    self.gs_params.ray_dist.requires_grad = False
                
            self.gs_params.quats.requires_grad = True
            self.gs_params.opacity.requires_grad = True
            
            if self.cfg.fix_scale:
                self.gs_params.scales.requires_grad = False
            else:
                self.gs_params.scales.requires_grad = True
                
            if self.cfg.fix_color:
                self.gs_params.colors.requires_grad = False
            else:
                self.gs_params.colors.requires_grad = True
            
            self.setup_optimizer()
            self.keyframes = []
            
            radius = 3
            self.disc_kernel = torch.zeros(1, 1, 2 * radius + 1, 2 * radius + 1, device=self.device)
            ky, kx = torch.meshgrid(torch.arange(-radius, radius + 1), torch.arange(-radius, radius + 1), indexing="ij")
            self.disc_kernel[0, 0, torch.sqrt(kx**2 + ky**2) <= radius + 0.5] = 1
            self.disc_kernel = self.disc_kernel / self.disc_kernel.sum()
            self.device = torch.device(self.cfg.device)
            self.gs_params.to(self.device)

            frame_count = 0
            while not self.stop_event.is_set():
                try:
                    # In real-time mode, we want to process the latest frame and skip older ones in the queue
                    frame_data = None
                    while not self.data_queue.empty():
                        frame_data = self.data_queue.get_nowait()
                    
                    if frame_data is None:
                        try:
                            frame_data = self.data_queue.get(timeout=0.1)
                        except queue.Empty:
                            continue
                except Exception as e:
                    if not self.stop_event.is_set():
                        print(f"Error in mapping loop: {e}")
                    continue
                
                self.optimize_frame(frame_data)
                frame_count += 1
                self.last_finished_frame = frame_data["frame_idx"]
                if frame_count % 5 == 0:
                    print(f"Mapping Progress: Processed {frame_count} frames. GS count: {len(self.gs_params.means)}")
        except KeyboardInterrupt:
            print("GS Mapping process received KeyboardInterrupt.")
        except Exception as e:
            print(f"GS Mapping process encountered an error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Cleaning up GS Mapping resources...")
            self.stop_event.set()
            self.keyframes = []
            if hasattr(self, 'optimizer'):
                del self.optimizer
            torch.cuda.empty_cache()
            print("GS Mapping process terminated.")

    def densify(self, image_t, depth_np, mask_t, cam, T_CO_t):
        with torch.no_grad():
            render_mode = "3dgs" if self.cfg.gs_type == "3d" else "normal"
            render_image, render_depth, _, _ = self.gs_params.render(T_CO_t, cam.K, cam.width, cam.height, mode=render_mode)
            render_depth = render_depth.squeeze()
            depth_curr_t = torch.from_numpy(depth_np).float().to(self.device)
            overlap_mask = (render_depth > 0) & (depth_curr_t > 0) & mask_t
            # if overlap_mask.sum() > 100:
            #     depth_curr_t *= torch.median(render_depth[overlap_mask] / depth_curr_t[overlap_mask])
            init_proba, penalty = get_lapla_norm(image_t, self.disc_kernel), get_lapla_norm(render_image, self.disc_kernel)
            sample_mask = (torch.rand_like(init_proba) < (init_proba - penalty) * 4.0) & mask_t & (depth_curr_t > 0)
            depth_closer = (depth_curr_t < (render_depth - 0.01)) | (render_depth == 0)
            sample_mask |= (depth_closer & mask_t & (depth_curr_t > 0) & (torch.rand_like(init_proba) < 0.2))
            
            if sample_mask.sum() > 0:
                y, x = torch.where(sample_mask)
                if len(y) > 5000:
                    perm = torch.randperm(len(y), device=self.device)[:5000]
                    y, x = y[perm], x[perm]
                z = depth_curr_t[y, x]
                if len(z) == 0:
                    print("[DEBUG] No valid points to densify.")
                    return
                
                fx, fy, cx, cy = cam.K[0, 0], cam.K[1, 1], cam.K[0, 2], cam.K[1, 2]
                pts_c = torch.stack([(x.float() - cx) * z / fx, (y.float() - cy) * z / fy, z], dim=-1)
                
                # NaN check for points
                if torch.isnan(pts_c).any():
                    valid = ~torch.isnan(pts_c).any(dim=-1)
                    if not valid.any(): return
                    y, x, z, pts_c = y[valid], x[valid], z[valid], pts_c[valid]

                full_pts_c = unproject_depth(depth_curr_t, cam.K, cam.height, cam.width)
                normals_c, _ = d2n_tblr(full_pts_c.permute(2, 0, 1).unsqueeze(0))
                
                # Robust normal extraction
                extracted_normals = normals_c[0][:, y, x].permute(1, 0)
                new_normals_c = -F.normalize(extracted_normals, dim=1, eps=1e-6)
                
                # NaN check for normals
                if torch.isnan(new_normals_c).any():
                    valid = ~torch.isnan(new_normals_c).any(dim=-1)
                    if not valid.any(): return
                    y, x, z, pts_c, new_normals_c = y[valid], x[valid], z[valid], pts_c[valid], new_normals_c[valid]
                T_OC_t = torch.inverse(T_CO_t)
                new_means = torch.einsum('ij,nj->ni', T_OC_t[:3, :3], pts_c) + T_OC_t[:3, 3]
                new_normals_o = F.normalize(torch.einsum('ij,nj->ni', T_OC_t[:3, :3], new_normals_c), dim=1)
                new_colors = RGB2SH(image_t[:, y, x].permute(1, 0))
                new_quats = build_rotation_from_normal(new_normals_o)
                sampled_init_proba = init_proba[y, x].clamp_min(1e-6)
                new_sizes = (1.0 / torch.sqrt(sampled_init_proba)).clamp(2.0, cam.width / 5.0) * (z / ((fx+fy)/2))
                
                if self.cfg.gs_type == "3d":
                    new_scales = torch.log(new_sizes.view(-1, 1).repeat(1, 3).clamp(1e-6, 1e6))
                    # Make it slightly "thin" in the third dimension? or just symmetric?
                    # Let's make it symmetric for now, PGSR will optimize it.
                else:
                    new_scales = torch.log(new_sizes.view(-1, 1).repeat(1, 2).clamp(1e-6, 1e6))
                    
                new_opacity = torch.logit(torch.ones((len(new_means), 1), device=self.device) * 0.1)
                    
                new_means = new_means.requires_grad_(True)
                new_quats = new_quats.requires_grad_(True)
                new_scales = new_scales.requires_grad_(True)
                new_colors = new_colors.requires_grad_(True)
                new_opacity = new_opacity.requires_grad_(True)

                self.gs_params.means = torch.nn.Parameter(torch.cat([self.gs_params.means.data, new_means], dim=0))
                self.gs_params.quats = torch.nn.Parameter(torch.cat([self.gs_params.quats.data, new_quats], dim=0))
                self.gs_params.scales = torch.nn.Parameter(torch.cat([self.gs_params.scales.data, new_scales], dim=0))
                self.gs_params.colors = torch.nn.Parameter(torch.cat([self.gs_params.colors.data, new_colors], dim=0))
                self.gs_params.opacity = torch.nn.Parameter(torch.cat([self.gs_params.opacity.data, new_opacity], dim=0))
                
                if self.gs_params.ray_o is not None:
                    # Initialize rays for new points
                    new_ray_o = T_OC_t[:3, 3].view(1, 3).repeat(len(new_means), 1)
                    new_ray_d_c = F.normalize(pts_c, dim=1)
                    new_ray_d_o = F.normalize(torch.einsum('ij,nj->ni', T_OC_t[:3, :3], new_ray_d_c), dim=1)
                    new_ray_dist = torch.norm(pts_c, dim=-1, keepdim=True)
                    
                    self.gs_params.ray_o = torch.cat([self.gs_params.ray_o, new_ray_o], dim=0)
                    self.gs_params.ray_d = torch.cat([self.gs_params.ray_d, new_ray_d_o], dim=0)
                    if self.gs_params.ray_dist is not None:
                        self.gs_params.ray_dist = torch.nn.Parameter(torch.cat([self.gs_params.ray_dist.data, new_ray_dist], dim=0))

                self.setup_optimizer()

    def prune(self, T_CO_t, cam):
        with torch.no_grad():
            means_c = torch.einsum('ij,nj->ni', T_CO_t[:3, :3], self.gs_params.means) + T_CO_t[:3, 3]
            dist = means_c[:, 2].clamp_min(0.01) 
            screen_size = cam.K[0, 0] * torch.exp(self.gs_params.scales).max(dim=-1)[0] / dist
            opacity = torch.sigmoid(self.gs_params.opacity.squeeze(-1))
            valid_mask = (opacity > self.cfg.prune_opacity_th)
            valid_mask &= (screen_size < self.cfg.prune_screen_size_th)
            valid_mask &= (dist > self.cfg.near_plane) & (dist < self.cfg.far_plane)

            if valid_mask.sum() == 0: valid_mask = torch.ones_like(valid_mask)
            self.gs_params.means = torch.nn.Parameter(self.gs_params.means[valid_mask])
            self.gs_params.quats = torch.nn.Parameter(self.gs_params.quats[valid_mask])
            self.gs_params.scales = torch.nn.Parameter(self.gs_params.scales[valid_mask])
            self.gs_params.colors = torch.nn.Parameter(self.gs_params.colors[valid_mask])
            self.gs_params.opacity = torch.nn.Parameter(self.gs_params.opacity[valid_mask])
            
            if self.gs_params.ray_dist is not None:
                self.gs_params.ray_o = self.gs_params.ray_o[valid_mask]
                self.gs_params.ray_d = self.gs_params.ray_d[valid_mask]
                self.gs_params.ray_dist = torch.nn.Parameter(self.gs_params.ray_dist[valid_mask])
                
    def optimize_frame(self, frame_data, num_steps=None, frame_count=0):
        image, mask, depth = frame_data["image"], frame_data["mask"], frame_data["depth"]
        K, T_CO = frame_data["K"], frame_data["T_CiO"]
        
        H, W = image.shape[:2]
        image_t = torch.from_numpy(image).float().to(self.device).permute(2, 0, 1) / 255.0
        mask_t = torch.from_numpy(mask > 0).bool().to(self.device)
        T_CO_t = torch.from_numpy(T_CO).float().to(self.device)
        K_t = torch.from_numpy(K).float().to(self.device)
        
        cam = MiniCam(K, T_CO, W, H)
        image_gray_t = image_t.mean(0, keepdim=True)
        
        # Coarse-to-Fine Pyramid
        image_pyr = [image_t]
        mask_pyr = [mask_t.unsqueeze(0).float()]
        for _ in range(self.cfg.pyr_levels - 1):
            image_pyr.append(F.avg_pool2d(image_pyr[-1], 2))
            mask_pyr.append(F.avg_pool2d(mask_pyr[-1], 2))
        for i in range(len(mask_pyr)):
            mask_pyr[i] = (mask_pyr[i][0] > 0.5)
            
        # Optimization loop
        if num_steps is None: num_steps = self.cfg.num_steps_per_frame
        for step in range(num_steps):
            if hasattr(self, 'stop_event') and self.stop_event.is_set(): break
            
            curr_pyr_lvl = max(0, (self.cfg.pyr_levels - 1) - (step // self.cfg.pyr_interval))
            target_image, target_mask = image_pyr[curr_pyr_lvl], mask_pyr[curr_pyr_lvl]
            scale = 1.0 / (2**curr_pyr_lvl)
            curr_K = cam.get_k(scale)
            
            with self.lock:
                self.optimizer.zero_grad()
                
                if self.cfg.use_ray_dist and self.gs_params.ray_dist is not None:
                    # Update means from rays
                    self.gs_params.means = self.gs_params.ray_o + self.gs_params.ray_d * self.gs_params.ray_dist
                
                render_mode = "3dgs" if self.cfg.gs_type == "3d" else "normal"
                render_image, render_depth, render_normal, render_alpha = self.gs_params.render(
                    T_CO_t, curr_K, target_image.shape[2], target_image.shape[1],
                    mode=render_mode,
                    near_plane=self.cfg.near_plane, far_plane=self.cfg.far_plane
                )
            
            # Losses
            loss_l1 = F.l1_loss(render_image * target_mask, target_image * target_mask)
            loss_ssim = 1.0 - ssim((render_image * target_mask).unsqueeze(0), (target_image * target_mask).unsqueeze(0))
            loss_photo = (1.0 - self.cfg.lambda_dssim) * loss_l1 + self.cfg.lambda_dssim * loss_ssim
            
            render_alpha_sq = render_alpha.squeeze(0)
            if self.cfg.use_mask_loss:
                loss_mask = F.l1_loss(render_alpha_sq, target_mask.float()) * self.cfg.mask_loss_weight
            else:
                loss_mask = F.l1_loss(render_alpha_sq * (~target_mask).float(), torch.zeros_like(render_alpha_sq))
            
            # PGSR Losses (Single-view and Multi-view Consistency)
            loss_sv = torch.tensor(0.0, device=self.device)
            loss_mv = torch.tensor(0.0, device=self.device)
            
            if self.cfg.use_pgsr:
                # Single-view Consistency (Normal-Depth)
                loss_sv = compute_single_view_loss(
                    render_normal, render_depth, target_image, cam, 
                    weight=1.0, 
                    scale=scale, mask=target_mask
                )
                
                if step >= 30:
                    # Select keyframes or use virtual camera
                    potential_neighbors = [kf for kf in self.keyframes if np.linalg.norm(kf[0].world_view_transform[3, :3].cpu().numpy() - cam.world_view_transform[3, :3].cpu().numpy()) > self.cfg.multi_view_min_dis]
                    
                    selected_neighbors = []
                    if potential_neighbors:
                        selected_neighbors = random.sample(potential_neighbors, min(2, len(potential_neighbors)))
                    
                    # Virtual camera logic
                    if (self.cfg.use_virtul_cam and random.random() < self.cfg.virtul_cam_prob) or not selected_neighbors:
                        v_cam = gen_virtul_cam(cam, trans_noise=self.cfg.multi_view_max_dis)
                        selected_neighbors.append((v_cam, image_t, image_gray_t[0], render_depth[0].detach(), mask_t))
                    
                    if selected_neighbors:
                        yy, xx = torch.where(target_mask)
                        if len(yy) > 0:
                            perm = torch.randperm(len(yy), device=self.device)[:4096]
                            px, py = xx[perm], yy[perm]
                            sampled_pixels = torch.stack([px, py], dim=-1).float()
                            sampled_indices = py * target_image.shape[2] + px
                            
                            for kf in selected_neighbors:
                                kf_cam, kf_image_t, kf_image_gray_t, kf_depth_t, kf_mask_t = kf
                                render_mode = "3dgs" if self.cfg.gs_type == "3d" else "normal"
                                r_img_nea, r_depth_nea, _, r_alpha_nea = self.gs_params.render(
                                    kf_cam.world_view_transform.t(), kf_cam.K, kf_cam.width, kf_cam.height,
                                    mode=render_mode
                                )
                                loss_mv_photo = (1.0 - self.cfg.lambda_dssim) * F.l1_loss(r_img_nea[:, kf_mask_t], kf_image_t[:, kf_mask_t]) + \
                                                self.cfg.lambda_dssim * (1.0 - ssim((r_img_nea * kf_mask_t).unsqueeze(0), (kf_image_t * kf_mask_t).unsqueeze(0)))
                                loss_mv += self.cfg.multi_view_photo_weight * loss_mv_photo / len(selected_neighbors)
                                ncc, _ = compute_multi_view_loss(
                                    image_gray_t, kf_image_gray_t.unsqueeze(0), render_normal, render_depth,
                                    cam, kf_cam, pixels=sampled_pixels, valid_indices=sampled_indices
                                )
                                loss_mv += self.cfg.multi_view_ncc_weight * ncc.mean() / len(selected_neighbors)
                                if self.cfg.use_mask_loss:
                                    loss_mv += self.cfg.mask_loss_weight * F.l1_loss(r_alpha_nea[0], kf_mask_t.float()) / len(selected_neighbors)
                                if step >= 60:
                                    d_mask, weights, p_noise = compute_geo_consistency(cam, kf_cam, render_depth, r_depth_nea, self.cfg.multi_view_pixel_noise_th)
                                    if d_mask.any():
                                        loss_mv += self.cfg.multi_view_geo_weight * (weights * p_noise)[d_mask].mean() / len(selected_neighbors)
            
            total_loss = loss_photo + loss_mask + self.cfg.lambda_sv * loss_sv + loss_mv
            total_loss.backward()
            with self.lock:
                self.optimizer.step()
        
        # Maintenance
        if frame_count % self.cfg.kf_every == 0:
            with torch.no_grad():
                render_mode = "3dgs" if self.cfg.gs_type == "3d" else "normal"
                _, r_depth, _, _ = self.gs_params.render(T_CO_t, K_t, W, H, mode=render_mode)
            self.keyframes.append((cam, image_t.detach(), image_gray_t[0].detach(), r_depth[0].detach(), mask_t.detach()))
            if len(self.keyframes) > self.cfg.window_size: self.keyframes.pop(0)
        
        if frame_count > 0 and frame_count % self.cfg.densify_every == 0:
            before_count = len(self.gs_params.means)
            with self.lock:
                self.densify(image_t, depth, mask_t, cam, T_CO_t)
                self.prune(T_CO_t, cam)
            after_count = len(self.gs_params.means)
            print(f"Mapping Frame {frame_count}: GS count {before_count} -> {after_count}")

    def refine_pose(self, image_t, mask_t, cam, T_CO_init, steps=60):
        """Optimizes T_CO and scale to align the GS model with the current image using a pyramid approach."""
        import torch.optim as optim
        from utils.loss_utils import ssim
        device = self.cfg.device
        
        def matrix_to_se3(T):
            from scipy.spatial.transform import Rotation as R
            r = R.from_matrix(T[:3, :3].cpu().numpy()).as_rotvec()
            t = T[:3, 3].cpu().numpy()
            return torch.tensor(np.concatenate([r, t]), device=device, dtype=torch.float32, requires_grad=True)

        def se3_to_matrix(se3):
            from pytorch3d.transforms import axis_angle_to_matrix
            rot = axis_angle_to_matrix(se3[:3])
            trans = se3[3:6]
            T = torch.eye(4, device=device)
            T[:3, :3] = rot
            T[:3, 3] = trans
            return T

        se3 = matrix_to_se3(T_CO_init)
        # Add a scale parameter to handle metric drift
        log_scale = torch.zeros(1, device=device, requires_grad=True)
        optimizer = optim.Adam([{'params': [se3], 'lr': 1e-3}, {'params': [log_scale], 'lr': 1e-2}])
        
        target_image_full = image_t.permute(2, 0, 1) / 255.0
        mask_full = mask_t.unsqueeze(0)
        
        best_loss = float('inf')
        best_T = T_CO_init.clone()

        # 3-Level Pyramid: 1/4, 1/2, Full
        pyramid_steps = [steps // 3] * 3
        for level, level_steps in enumerate(pyramid_steps):
            down = 2 ** (2 - level)
            if down > 1:
                target_image = F.interpolate(target_image_full.unsqueeze(0), scale_factor=1/down, mode='bilinear')[0]
                mask = F.interpolate(mask_full.float().unsqueeze(0), scale_factor=1/down, mode='nearest')[0, 0] > 0
                curr_K = cam.K / down; curr_K[2, 2] = 1.0
                curr_w, curr_h = int(cam.width / down), int(cam.height / down)
            else:
                target_image = target_image_full
                mask = mask_t
                curr_K = cam.K
                curr_w, curr_h = cam.width, cam.height

            for _ in range(level_steps):
                optimizer.zero_grad()
                T_curr = se3_to_matrix(se3)
                scale = torch.exp(log_scale)
                
                # Apply scale to means for rendering
                scaled_means = self.gs_params.means * scale
                
                render_image, _, _, _ = render_2dgs(
                    scaled_means, F.normalize(self.gs_params.quats), 
                    torch.exp(self.gs_params.scales) * scale, self.gs_params.colors, 
                    torch.sigmoid(self.gs_params.opacity),
                    viewmat=T_curr, K=curr_K, width=curr_w, height=curr_h
                )
                
                l1 = F.l1_loss(render_image[:, mask], target_image[:, mask])
                s = 1.0 - ssim(render_image.unsqueeze(0), target_image.unsqueeze(0))
                loss = 0.8 * l1 + 0.2 * s
                
                loss.backward()
                optimizer.step()
                
                if level == 2 and loss.item() < best_loss:
                    best_loss = loss.item()
                    best_T = T_curr.detach().clone()
                    best_scale = scale.item()
        
        return best_T, best_scale

    def save_gs(self, path):
        self.gs_params.dump(Path(path))
        print(f"GS parameters saved to {path}")

def start_mapping_process(cfg, initial_gs):
    # Ensure spawn method is used for PyTorch multiprocessing
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
        
    mapper = GSMapping(cfg, initial_gs)
    p = mp.Process(target=mapper.run)
    p.start()
    return mapper, p
