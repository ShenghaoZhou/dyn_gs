"""
We use a set of GS to model a rigid object. 
"""

import numpy as np
import torch
import torch.nn.functional as F
from .gs_rendering import render_2dgs, render_3dgs
from .gs_param import GSParam
from gsplat import rasterization_2dgs
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply, quaternion_to_matrix
from .utils.ssim import image_loss
from pytorch3d.transforms.so3 import so3_exp_map
import rerun as rr
import io
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def trimmed_l1_loss(pred, gt, quantile=0.9):
    loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1)
    if loss.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    loss_at_quantile = torch.quantile(loss, quantile)
    trimmed_loss = loss[loss < loss_at_quantile].mean()
    return trimmed_loss

def masked_l1_loss(pred, gt, mask=None, normalize=True, quantile: float = 1.0):
    if mask is None:
        return trimmed_l1_loss(pred, gt, quantile)
    else:
        sum_loss = F.l1_loss(pred, gt, reduction="none").mean(dim=-1, keepdim=True)
        if quantile < 1:
            num = sum_loss.numel()
            if num == 0:
                return torch.tensor(0.0, device=pred.device)
            if num < 16_000_000:
                threshold = torch.quantile(sum_loss, quantile)
            else:
                sorted_loss, _ = torch.sort(sum_loss.reshape(-1))
                idxf = quantile * num
                idxi = int(idxf)
                threshold = sorted_loss[idxi] + (sorted_loss[min(idxi + 1, num - 1)] - sorted_loss[idxi]) * (idxf - idxi)
            quantile_mask = (sum_loss < threshold).squeeze(-1)
        else: 
            quantile_mask = torch.ones_like(sum_loss, dtype=torch.bool).squeeze(-1)

        ndim = sum_loss.shape[-1]
        if normalize:
            denom = ndim * torch.sum(mask[quantile_mask]) + 1e-8
            return torch.sum((sum_loss * mask)[quantile_mask]) / denom
        else:
            return torch.mean((sum_loss * mask)[quantile_mask])

class ObjectGS: 
    def __init__(self, gs_params: GSParam, T_W_O: np.ndarray, obj_scale: float, 
                 image_ref: torch.Tensor = None, T_C_O_ref: torch.Tensor = None):
        self.gs_params = gs_params
        self.T_W_O = T_W_O # Current World-from-Object
        self.T_W_O_init = T_W_O.copy() if isinstance(T_W_O, np.ndarray) else T_W_O.clone()
        self.obj_scale = obj_scale
        
        self.image_ref = image_ref # Reference image for warping (C, H, W)
        self.T_C_O_ref = T_C_O_ref # Camera-from-Object pose for image_ref
        self.alpha_ref = None # Rendered alpha for image_ref

    def render(self, viewmat, K, width, height, **kwargs):
        return self.gs_params.render(viewmat, K, width, height, **kwargs)

    def update_reference(self, image: torch.Tensor, T_C_O_ref: torch.Tensor, alpha: torch.Tensor = None):
        """Update the reference image and pose used for warping."""
        self.image_ref = image
        self.T_C_O_ref = T_C_O_ref
        self.alpha_ref = alpha

    @staticmethod
    def gs_to_planar_params(gs_params, T_CO):
        """
        Convert GS parameters to planar surface parameters (normal and distance) in camera view.
        """
        means = gs_params.means # [N, 3]
        quats = gs_params.quats # [N, 4]
        R_OG = quaternion_to_matrix(quats) # [N, 3, 3]
        n_O = R_OG[:, :, 2] # [N, 3]
        R_CO = T_CO[:3, :3]
        t_CO = T_CO[:3, 3]
        n_C = torch.einsum('ij,nj->ni', R_CO, n_O) # [N, 3]
        p_C = torch.einsum('ij,nj->ni', R_CO, means) + t_CO # [N, 3]
        d = -torch.sum(n_C * p_C, dim=1, keepdim=True) # [N, 1]
        return n_C, d

    @staticmethod
    def render_custom_attribute(gs_params, attr, T_CO, K, width, height, near_plane=0.01, far_plane=100.0):
        """
        Render a custom attribute using our render_2dgs wrapper.
        """
        img, depth, normal, alphas = render_2dgs(
            gs_params.means, gs_params.quats, gs_params.scales,
            gs_params.colors, gs_params.opacity,
            viewmat=T_CO, K=K, width=width, height=height,
            near_plane=near_plane, far_plane=far_plane,
            custom_colors=attr
        )
        return img, alphas

    @staticmethod
    def manual_backward_homography_warp(image_ref_t, n_curr, d_curr, T_ref_curr, K):
        """
        Perform backward homography warping using rendered normal and distance maps.
        image_ref_t: [C, H_ref, W_ref] torch tensor (0-1)
        """
        H, W = n_curr.shape[1], n_curr.shape[2]
        device = n_curr.device
        
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device), 
            indexing='ij'
        )
        u1 = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).float().reshape(-1, 3) # [HW, 3]
        
        R_inv = T_ref_curr[:3, :3]
        t_inv = T_ref_curr[:3, 3]
        
        K_inv = torch.inverse(K)
        dir1 = (K_inv @ u1.t()).t() # [HW, 3]
        
        n1 = n_curr.permute(1, 2, 0).reshape(-1, 3) # [HW, 3]
        d1 = d_curr.reshape(-1, 1) # [HW, 1]
        
        d1_safe = torch.where(torch.abs(d1) > 1e-6, d1, torch.ones_like(d1))
        n1_dot_dir1 = torch.sum(n1 * dir1, dim=1, keepdim=True)
        p0_scaled = torch.matmul(R_inv, dir1.t()).t() - (n1_dot_dir1 / d1_safe) * t_inv.view(1, 3)
        
        u0_homog = torch.matmul(K, p0_scaled.t()).t()
        u0_pix = u0_homog[:, :2] / (u0_homog[:, 2:3] + 1e-8)
        
        grid = u0_pix.reshape(1, H, W, 2)
        grid[..., 0] = 2.0 * grid[..., 0] / (W - 1) - 1.0
        grid[..., 1] = 2.0 * grid[..., 1] / (H - 1) - 1.0
        
        warped = F.grid_sample(image_ref_t.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        return warped[0]

    def optimize_wrt_image(self, T_C_W, image, K, mask=None, mask_ref=None,
                           update_ref=True,
                           lr=1e-3, num_steps=100, rr_vis=False,
                           near_plane=0.01, far_plane=10.0, gs_type="2d"):
        """Optimize T_W_O by maximizing photometric consistency via warping.
        
        Args:
            T_C_W: Camera pose w.r.t world (Camera-from-World)
            image: Target image observation
            K: Intrinsics
            update_ref: Whether to update self.image_ref and self.T_C_O_ref after optimization
        """
        device = self.gs_params.means.device
        
        # Convert inputs to torch tensors
        if isinstance(image, np.ndarray):
            image_gt = torch.from_numpy(image.copy()).float().to(device)
        else:
            image_gt = image.float().to(device)
        if image_gt.max() > 1.0: image_gt /= 255.0
        if image_gt.shape[0] != 3: image_gt = image_gt.permute(2, 0, 1)

        mask_gt = None
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_gt = torch.from_numpy(mask.copy()).float().to(device)
            else:
                mask_gt = mask.float().to(device)
            if mask_gt.max() > 1.0: mask_gt /= 255.0
            if mask_gt.dim() == 2: mask_gt = mask_gt.unsqueeze(0)
            elif mask_gt.shape[0] != 1: mask_gt = mask_gt.permute(2, 0, 1)

        mask_ref_t = None
        if mask_ref is not None:
            if isinstance(mask_ref, np.ndarray):
                mask_ref_t = torch.from_numpy(mask_ref.copy()).float().to(device)
            else:
                mask_ref_t = mask_ref.float().to(device)
            if mask_ref_t.max() > 1.0: mask_ref_t /= 255.0
            if mask_ref_t.dim() == 2: mask_ref_t = mask_ref_t.unsqueeze(0)
            elif mask_ref_t.shape[0] != 1: mask_ref_t = mask_ref_t.permute(2, 0, 1)

        if isinstance(K, np.ndarray): K_t = torch.from_numpy(K.copy()).float().to(device)
        else: K_t = K.float().to(device)
            
        if isinstance(T_C_W, np.ndarray): T_C_W_t = torch.from_numpy(T_C_W.copy()).float().to(device)
        else: T_C_W_t = T_C_W.detach().float().to(device)

        if isinstance(self.T_W_O, np.ndarray): T_W_O_t = torch.from_numpy(self.T_W_O.copy()).float().to(device)
        else: T_W_O_t = self.T_W_O.detach().float().to(device)

        # Clone Gaussians to avoid concurrent modification issues
        from types import SimpleNamespace
        params_cloned = SimpleNamespace(
            means=self.gs_params.means.detach().clone(),
            quats=self.gs_params.quats.detach().clone(),
            scales=self.gs_params.scales.detach().clone(),
            colors=self.gs_params.colors.detach().clone(),
            opacity=self.gs_params.opacity.detach().clone()
        )

        # Ensure we have a reference image
        if self.image_ref is None or self.T_C_O_ref is None:
            with torch.no_grad():
                T_C_O_init = T_C_W_t @ T_W_O_t
                render_mode = "3dgs" if gs_type == "3d" else "normal"
                img_ref, depth_ref, _, alpha_ref = self.gs_params.render(
                    T_C_O_init, K_t, width=image_gt.shape[2], height=image_gt.shape[1],
                    mode=render_mode, near_plane=near_plane, far_plane=far_plane
                )
                self.update_reference(img_ref, T_C_O_init)

        # Optimization variables: delta pose on top of T_W_O (World-frame delta)
        trans = torch.zeros(3, device=device, requires_grad=True)
        log_R = torch.zeros((1, 3), device=device, requires_grad=True)
        
        optimizer = torch.optim.AdamW([
            {"params": trans, "lr": lr * 1.5},
            {"params": log_R, "lr": lr}
        ])
        
        losses = []
        best_loss = float('inf')
        best_T_W_O = T_W_O_t.clone()
        best_T_C_O = (T_C_W_t @ T_W_O_t).detach().clone()
        best_warped = None

        H, W = image_gt.shape[1], image_gt.shape[2]

        for step in range(num_steps):
            optimizer.zero_grad()
            
            rot_mat_delta = so3_exp_map(log_R)[0]
            if torch.isnan(rot_mat_delta).any(): break
                
            T_delta = torch.eye(4, device=device)
            T_delta[:3, :3] = rot_mat_delta
            T_delta[:3, 3] = trans
            
            T_W_O_curr = T_delta @ T_W_O_t
            T_C_O_curr = T_C_W_t @ T_W_O_curr
            T_ref_curr = self.T_C_O_ref @ torch.inverse(T_C_O_curr)
            
            # Render surface parameters at current view for backward warping
            n_curr, d_curr = self.gs_to_planar_params(params_cloned, T_C_O_curr)
            n_map, alpha_map = self.render_custom_attribute(params_cloned, n_curr, T_C_O_curr, K_t, W, H, near_plane, far_plane)
            d_3ch_map, _ = self.render_custom_attribute(params_cloned, d_curr.repeat(1, 3), T_C_O_curr, K_t, W, H, near_plane, far_plane)
            d_map = d_3ch_map[0:1]
            
            # Warp reference image to current view
            img_ref_in = self.image_ref
            if mask_ref_t is not None:
                img_ref_in = img_ref_in * mask_ref_t

            warped = self.manual_backward_homography_warp(img_ref_in, n_map, d_map, T_ref_curr, K_t)
            warped = warped * alpha_map
            
            # Photometric loss (L1)
            res = (warped - image_gt)
            if mask_gt is not None:
                res = res * mask_gt
            
            loss = torch.mean(torch.abs(res))
            
            if torch.isnan(loss): break
                
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            losses.append(current_loss)
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_T_W_O = T_W_O_curr.detach().clone()
                best_T_C_O = T_C_O_curr.detach().clone()
                best_warped = warped.detach().clone()

            if step % 50 == 0:
                print(f"[{self.__class__.__name__}] Step {step}/{num_steps} - Loss: {current_loss:.6f}")

        # Update current pose
        self.T_W_O = best_T_W_O.cpu().numpy() if isinstance(self.T_W_O, np.ndarray) else best_T_W_O
        
        # Update reference if requested
        if update_ref:
            self.update_reference(image_gt.detach(), best_T_C_O)

        if rr_vis:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(losses, label='Training Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Warping Optimization Loss')
            ax.grid(True)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            rr.log("opt/loss_plot", rr.Image(np.array(Image.open(buf))))
            plt.close(fig)
            
            if best_warped is not None:
                rr.log("opt/warped", rr.Image(best_warped.permute(1, 2, 0).cpu().numpy().clip(0, 1)))

        return best_T_W_O, losses

    def optimize_wrt_image_lm(self, T_C_W, image, K, mask=None, mask_ref=None,
                               update_ref=True, rr_vis=False,
                               pyramid_levels=[(8, 10), (4, 10), (2, 10), (1, 10)],
                               damping=1.0, min_delta=1e-4, re_render=True,
                               re_render_interval=1,
                               near_plane=0.01, far_plane=10.0, gs_type="2d"):
        """Levenberg-Marquardt pose optimization using analytical Jacobians.
        """
        print(f"[{self.__class__.__name__}] Entering optimize_wrt_image_lm, mask is {mask is not None}")
        device = self.gs_params.means.device
        
        # Prep inputs
        if isinstance(image, np.ndarray):
            image_gt = torch.from_numpy(image.copy()).float().to(device)
        else:
            image_gt = image.float().to(device)
        if image_gt.max() > 1.0: image_gt /= 255.0
        if image_gt.shape[0] != 3: image_gt = image_gt.permute(2, 0, 1)
        H, W = image_gt.shape[1], image_gt.shape[2]
        
        mask_gt = None
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_gt = torch.from_numpy(mask.copy()).float().to(device)
            else:
                mask_gt = mask.float().to(device)
            if mask_gt.max() > 1.0: mask_gt /= 255.0
            if mask_gt.dim() == 2: mask_gt = mask_gt.unsqueeze(0)
            elif mask_gt.shape[0] != 1: mask_gt = mask_gt.permute(2, 0, 1)

        mask_ref_t = None
        if mask_ref is not None:
            if isinstance(mask_ref, np.ndarray):
                mask_ref_t = torch.from_numpy(mask_ref.copy()).float().to(device)
            else:
                mask_ref_t = mask_ref.float().to(device)
            if mask_ref_t.max() > 1.0: mask_ref_t /= 255.0
            if mask_ref_t.dim() == 2: mask_ref_t = mask_ref_t.unsqueeze(0)
            elif mask_ref_t.shape[0] != 1: mask_ref_t = mask_ref_t.permute(2, 0, 1)

        if isinstance(K, np.ndarray): K_t = torch.from_numpy(K.copy()).float().to(device)
        else: K_t = K.float().to(device)
        if isinstance(T_C_W, np.ndarray): T_C_W_t = torch.from_numpy(T_C_W.copy()).float().to(device)
        else: T_C_W_t = T_C_W.detach().float().to(device)
        if isinstance(self.T_W_O, np.ndarray): T_W_O_t = torch.from_numpy(self.T_W_O.copy()).float().to(device)
        else: T_W_O_t = self.T_W_O.detach().float().to(device)
        # Clone Gaussians to avoid concurrent modification issues
        from types import SimpleNamespace
        params_cloned = SimpleNamespace(
            means=self.gs_params.means.detach().clone(),
            quats=self.gs_params.quats.detach().clone(),
            scales=self.gs_params.scales.detach().clone(),
            colors=self.gs_params.colors.detach().clone(),
            opacity=self.gs_params.opacity.detach().clone()
        )

        if self.image_ref is None or self.T_C_O_ref is None:
            with torch.no_grad():
                T_C_O_init = T_C_W_t @ T_W_O_t
                render_mode = "3dgs" if gs_type == "3d" else "normal"
                img_ref, _, _, alpha_ref = self.gs_params.render(
                    T_C_O_init, K_t, width=image_gt.shape[2], height=image_gt.shape[1],
                    mode=render_mode, near_plane=near_plane, far_plane=far_plane
                )
                self.update_reference(img_ref, T_C_O_init, alpha_ref)
                
                # Render geometry maps for the first time
                n_c_init, d_c_init = self.gs_to_planar_params(params_cloned, T_C_O_init)
                n_map_full, alpha_map_full = self.render_custom_attribute(params_cloned, n_c_init, T_C_O_init, K_t, W, H, near_plane, far_plane)
                d_3ch_full, _ = self.render_custom_attribute(params_cloned, d_c_init.repeat(1, 3), T_C_O_init, K_t, W, H, near_plane, far_plane)
                d_map_full = d_3ch_full[0:1]

        H, W = image_gt.shape[1], image_gt.shape[2]
        curr_T_W_O = T_W_O_t.clone()
        
        all_losses = []
        best_warped = None
        best_alpha = None
        
        curr_damping = damping

        # If not re-rendering every step, render once at the very beginning
        if not re_render:
            with torch.no_grad():
                T_C_O_init = T_C_W_t @ curr_T_W_O
                n_c_init, d_c_init = self.gs_to_planar_params(self.gs_params, T_C_O_init)
                n_map_full, alpha_map_full = self.render_custom_attribute(self.gs_params, n_c_init, T_C_O_init, K_t, W, H, near_plane, far_plane)
                d_3ch_full, _ = self.render_custom_attribute(self.gs_params, d_c_init.repeat(1, 3), T_C_O_init, K_t, W, H, near_plane, far_plane)
                d_map_full = d_3ch_full[0:1]

        # Optimization loop
        for res_scale, num_steps in pyramid_levels:
            opt_h, opt_w = H // res_scale, W // res_scale
            
            with torch.no_grad():
                img_ref_o = F.interpolate(self.image_ref.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
                mask_ref_o = F.interpolate(mask_ref_t.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0] if mask_ref_t is not None else None
                # Apply reference mask to reference image
                if mask_ref_o is not None:
                    img_ref_o = img_ref_o * mask_ref_o

                img_gt_o = F.interpolate(image_gt.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
                mask_gt_o = F.interpolate(mask_gt.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0] if mask_gt is not None else None
                alpha_ref_o = F.interpolate(self.alpha_ref.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0] if self.alpha_ref is not None else None
                # Also apply reference mask to alpha if provided
                if mask_ref_o is not None and alpha_ref_o is not None:
                    alpha_ref_o = alpha_ref_o * mask_ref_o
                K_o = K_t.clone()
                K_o[0, 0] *= (opt_w / W); K_o[1, 1] *= (opt_h / H)
                K_o[0, 2] *= (opt_w / W); K_o[1, 2] *= (opt_h / H)
                
                # Apply mask to GT image
                if mask_gt_o is not None:
                    img_gt_o = img_gt_o * mask_gt_o
                
                if not re_render:
                    n_map_o = F.interpolate(n_map_full.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
                    d_map_o = F.interpolate(d_map_full.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
                    alpha_o = F.interpolate(alpha_map_full.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]

                # A. Image Gradients of reference image
                # We compute gradients on the reference image (Only once per resolution level)
                grad_x = torch.zeros_like(img_ref_o)
                grad_y = torch.zeros_like(img_ref_o)
                grad_x[:, :, 1:-1] = 0.5 * (img_ref_o[:, :, 2:] - img_ref_o[:, :, :-2])
                grad_y[:, 1:-1, :] = 0.5 * (img_ref_o[:, 2:, :] - img_ref_o[:, :-2, :])
                
                # Alpha map gradients for mask loss
                grad_alpha_x = torch.zeros_like(alpha_ref_o) if alpha_ref_o is not None else None
                grad_alpha_y = torch.zeros_like(alpha_ref_o) if alpha_ref_o is not None else None
                if alpha_ref_o is not None:
                    grad_alpha_x[:, :, 1:-1] = 0.5 * (alpha_ref_o[:, :, 2:] - alpha_ref_o[:, :, :-2])
                    grad_alpha_y[:, 1:-1, :] = 0.5 * (alpha_ref_o[:, 2:, :] - alpha_ref_o[:, :-2, :])

                # Cache grid for warping
                grid_y, grid_x = torch.meshgrid(torch.arange(opt_h, device=device), torch.arange(opt_w, device=device), indexing='ij')
                u1 = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).float().reshape(-1, 3) # [HW, 3]
                K_inv_u1 = (torch.inverse(K_o) @ u1.t()).t() # [HW, 3]
                dirs_o = u1.t().reshape(3, opt_h, opt_w) # [3, H, W] for unprojection

            for step in range(num_steps):
                with torch.no_grad():
                    T_W_O_curr = curr_T_W_O
                    T_C_O_curr = T_C_W_t @ T_W_O_curr
                    T_ref_curr = self.T_C_O_ref @ torch.inverse(T_C_O_curr)
                    
                    if re_render and step % re_render_interval == 0:
                        # Re-render geometry maps at current pose
                        n_c, d_c = self.gs_to_planar_params(params_cloned, T_C_O_curr)
                        n_map_o, alpha_o = self.render_custom_attribute(params_cloned, n_c, T_C_O_curr, K_o, opt_w, opt_h, near_plane, far_plane)
                        d_3ch_o, _ = self.render_custom_attribute(params_cloned, d_c.repeat(1, 3), T_C_O_curr, K_o, opt_w, opt_h, near_plane, far_plane)
                        d_map_o = d_3ch_o[0:1]

                    # 1. Forward warp to get residuals
                    # Use optimized warping (Skip meshgrid)
                    # warped = self.manual_backward_homography_warp(img_ref_o, n_map_o, d_map_o, T_ref_curr, K_o)
                    
                    # Manual warp logic here for speed (Inlined and optimized)
                    R_inv = T_ref_curr[:3, :3]
                    t_inv = T_ref_curr[:3, 3]
                    n1 = n_map_o.permute(1, 2, 0).reshape(-1, 3)
                    d1 = d_map_o.reshape(-1, 1)
                    d1_safe = torch.where(torch.abs(d1) > 1e-6, d1, torch.ones_like(d1))
                    n1_dot_dir1 = torch.sum(n1 * K_inv_u1, dim=1, keepdim=True)
                    p0_scaled = torch.matmul(R_inv, K_inv_u1.t()).t() - (n1_dot_dir1 / d1_safe) * t_inv.view(1, 3)
                    u0_homog = torch.matmul(K_o, p0_scaled.t()).t()
                    u0_pix = u0_homog[:, :2] / (u0_homog[:, 2:3] + 1e-8)
                    grid = u0_pix.reshape(1, opt_h, opt_w, 2)
                    grid[..., 0] = 2.0 * grid[..., 0] / (opt_w - 1) - 1.0
                    grid[..., 1] = 2.0 * grid[..., 1] / (opt_h - 1) - 1.0
                    warped = F.grid_sample(img_ref_o.unsqueeze(0), grid, mode='bilinear', padding_mode='zeros', align_corners=True)[0]
                    
                    res = (warped - img_gt_o) 
                    loss_val = torch.mean(torch.abs(res)).item()
                    all_losses.append(loss_val)
                    
                    if rr_vis and res_scale == 1:
                        best_warped = warped.detach()
                        best_alpha = alpha_o.detach()
                
                # B. Unproject current pixels to 3D using the fixed geometry maps
                # p_curr = -d / (n^T * dir) * dir
                y_coords, x_coords = torch.meshgrid(torch.arange(opt_h, device=device), torch.arange(opt_w, device=device), indexing='ij')
                coords = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=0).float() # [3, H, W]
                dirs = torch.inverse(K_o) @ coords.reshape(3, -1)
                dirs = dirs.reshape(3, opt_h, opt_w)
                
                n_dot_dir = torch.sum(n_map_o * dirs, dim=0, keepdim=True)
                # Avoid division by zero
                depth_curr = -d_map_o / (n_dot_dir + 1e-6)
                P_curr = dirs * depth_curr # [3, H, W]
                
                # C. Transform to reference frame
                R_rc = T_ref_curr[:3, :3]
                t_rc = T_ref_curr[:3, 3:4]
                P_ref = R_rc @ P_curr.reshape(3, -1) + t_rc # [3, N]
                P_ref = P_ref.reshape(3, opt_h, opt_w)
                
                # D. Project to reference image to sample gradients
                z_ref = P_ref[2:3].clamp(min=1e-3)
                u_ref = (K_o @ P_ref.reshape(3, -1)).reshape(3, opt_h, opt_w)
                u_ref = u_ref[:2] / z_ref
                
                # Sample gradients at u_ref
                # grid_sample expects [-1, 1]
                grid = u_ref.permute(1, 2, 0).clone()
                grid[..., 0] = (grid[..., 0] / (opt_w - 1)) * 2 - 1
                grid[..., 1] = (grid[..., 1] / (opt_h - 1)) * 2 - 1
                
                g_x_sampled = F.grid_sample(grad_x.unsqueeze(0), grid.unsqueeze(0), align_corners=True)[0]
                g_y_sampled = F.grid_sample(grad_y.unsqueeze(0), grid.unsqueeze(0), align_corners=True)[0]
                
                g_alpha_x_sampled = None
                g_alpha_y_sampled = None
                if grad_alpha_x is not None:
                    g_alpha_x_sampled = F.grid_sample(grad_alpha_x.unsqueeze(0), grid.unsqueeze(0), align_corners=True)[0]
                    g_alpha_y_sampled = F.grid_sample(grad_alpha_y.unsqueeze(0), grid.unsqueeze(0), align_corners=True)[0]
                
                # E. Projection Jacobian d_pi/dP_ref [2, 3] per pixel
                fx, fy = K_o[0, 0], K_o[1, 1]
                inv_z = 1.0 / z_ref
                inv_z2 = inv_z * inv_z
                
                # dp_ref/dP_ref is [2, 3, H, W]
                # [fx/z, 0, -fx*x/z^2]
                # [0, fy/z, -fy*y/z^2]
                dp_dP = torch.zeros(2, 3, opt_h, opt_w, device=device)
                dp_dP[0, 0] = fx * inv_z
                dp_dP[0, 2] = -fx * P_ref[0] * inv_z2
                dp_dP[1, 1] = fy * inv_z
                dp_dP[1, 2] = -fy * P_ref[1] * inv_z2
                
                # F. Pose Jacobian dP_ref/dxi = -[R | -R[P_curr]x]  [3, 6]
                # dP_ref/dv = -R
                # dP_ref/dw = R[P_curr]x
                
                # Precompute R * [P_curr]x
                # [P]x = [0 -z y; z 0 -x; -y x 0]
                x, y, z = P_curr[0], P_curr[1], P_curr[2]
                R = R_rc
                
                # Jacobian J = dI/du * dp/dP * dP/dxi  [3, 6, H, W]
                # (We sum over the 3 color channels)
                # First compute J_pose = dp/dP * dP/dxi  [2, 6, H, W]
                J_pose = torch.zeros(2, 6, opt_h, opt_w, device=device)
                
                # Translational part: dp/dP * (-R)
                for i in range(3): # v_i
                    # dp/dP @ (-R[:, i])
                    J_pose[:, i] = -(dp_dP[:, 0] * R[0, i] + dp_dP[:, 1] * R[1, i] + dp_dP[:, 2] * R[2, i])
                
                # Rotational part: dp/dP * (R[P]x)
                # R[P]x columns:
                # col 0 (rx): R @ [0, z, -y]^T = R0*0 + R1*z + R2*-y
                # col 1 (ry): R @ [-z, 0, x]^T
                # col 2 (rz): R @ [y, -x, 0]^T
                RPx0 = R[:, 1:2, None, None] * z[None] - R[:, 2:3, None, None] * y[None] # [3, H, W]
                RPx1 = -R[:, 0:1, None, None] * z[None] + R[:, 2:3, None, None] * x[None]
                RPx2 = R[:, 0:1, None, None] * y[None] - R[:, 1:2, None, None] * x[None]
                
                J_pose[:, 3] = dp_dP[:, 0] * RPx0[0, 0] + dp_dP[:, 1] * RPx0[1, 0] + dp_dP[:, 2] * RPx0[2, 0]
                J_pose[:, 4] = dp_dP[:, 0] * RPx1[0, 0] + dp_dP[:, 1] * RPx1[1, 0] + dp_dP[:, 2] * RPx1[2, 0]
                J_pose[:, 5] = dp_dP[:, 0] * RPx2[0, 0] + dp_dP[:, 1] * RPx2[1, 0] + dp_dP[:, 2] * RPx2[2, 0]
                
                # Final Jacobian: J is [3, 6, opt_h, opt_w]
                # J_c = g_x_c * J_pose_x + g_y_c * J_pose_y
                J = g_x_sampled.unsqueeze(1) * J_pose[0:1] + g_y_sampled.unsqueeze(1) * J_pose[1:2]
                # Apply alpha mask
                J = J * alpha_o # [3, 6, H, W]
                
                # Mask Jacobian and residuals
                if mask_gt_o is not None and g_alpha_x_sampled is not None:
                    J_mask = g_alpha_x_sampled.unsqueeze(1) * J_pose[0:1] + g_alpha_y_sampled.unsqueeze(1) * J_pose[1:2]
                    # Mask residual: alpha_warped - mask_gt
                    alpha_warped = F.grid_sample(alpha_ref_o.unsqueeze(0), grid.unsqueeze(0), align_corners=True)[0]
                    res_mask = (alpha_warped - mask_gt_o)
                    
                    # Log mask loss
                    mask_loss_val = torch.mean(torch.abs(res_mask)).item()
                    rr.log("opt/mask_loss", rr.Scalars(mask_loss_val))
                    if step == 0:
                        if res_scale == 1:
                            print(f"[{self.__class__.__name__}] Initial Mask Loss: {mask_loss_val:.6f}")
                    
                    # Combine Jacobians and residuals
                    # Note: Per user request, we do NOT use mask alignment gradients for pose
                    # We strictly use the provided GT mask to focus on the object area
                    J_combined = J * mask_gt_o
                    res_combined = res * mask_gt_o
                else:
                    J_combined = J
                    res_combined = res

                # G. Solve LM
                # Reshape for matrix ops
                # J_flat is [C, 6, N], res_flat is [C, N]
                J_flat = J_combined.flatten(2)
                res_flat = res_combined.flatten(1)
                
                # JTJ: [6, 6]. We sum over channels (c) and pixels (n)
                # Use einsum for efficiency: sum over channels (c) and pixels (n)
                JTJ = torch.einsum('cin,cjn->ij', J_flat, J_flat)
                
                # JTr: [6]. Sum over channels (c) and pixels (n)
                JTr = torch.einsum('cin,cn->i', J_flat, res_flat)
                
                diag = torch.diag(torch.diag(JTJ))
                delta = torch.linalg.solve(JTJ + curr_damping * diag + 1e-6 * torch.eye(6, device=device), -JTr)
                
                # Early stopping check
                if torch.norm(delta) < min_delta:
                    break
                
                dT = torch.eye(4, device=device)
                dT[:3, :3] = so3_exp_map(delta[3:6].unsqueeze(0))
                dT[:3, 3] = delta[0:3]
                
                T_C_O_curr = T_C_W_t @ curr_T_W_O
                T_C_O_new = dT @ T_C_O_curr
                
                # Evaluate new loss
                with torch.no_grad():
                    T_ref_new = self.T_C_O_ref @ torch.inverse(T_C_O_new)
                    if re_render:
                        n_c_new, d_c_new = self.gs_to_planar_params(self.gs_params, T_C_O_new)
                        n_map_new, alpha_new = self.render_custom_attribute(self.gs_params, n_c_new, T_C_O_new, K_o, opt_w, opt_h, near_plane, far_plane)
                        d_3ch_new, _ = self.render_custom_attribute(self.gs_params, d_c_new.repeat(1, 3), T_C_O_new, K_o, opt_w, opt_h, near_plane, far_plane)
                        d_map_new = d_3ch_new[0:1]
                    else:
                        n_map_new, d_map_new, alpha_new = n_map_o, d_map_o, alpha_o
                        
                    warped_new = self.manual_backward_homography_warp(img_ref_o, n_map_new, d_map_new, T_ref_new, K_o)
                    res_new = (warped_new - img_gt_o) * alpha_new
                    if mask_gt_o is not None and alpha_ref_o is not None:
                        # Re-compute grid for new pose to warp alpha_ref
                        y_c, x_c = torch.meshgrid(torch.arange(opt_h, device=device), torch.arange(opt_w, device=device), indexing='ij')
                        coords_new = torch.stack([x_c, y_c, torch.ones_like(x_c)], dim=0).float()
                        dirs_new = torch.inverse(K_o) @ coords_new.reshape(3, -1)
                        dirs_new = dirs_new.reshape(3, opt_h, opt_w)
                        n_dot_dir_new = torch.sum(n_map_new * dirs_new, dim=0, keepdim=True)
                        depth_new = -d_map_new / (n_dot_dir_new + 1e-6)
                        P_curr_new = dirs_new * depth_new
                        P_ref_new = T_ref_new[:3, :3] @ P_curr_new.reshape(3, -1) + T_ref_new[:3, 3:4]
                        P_ref_new = P_ref_new.reshape(3, opt_h, opt_w)
                        z_rn = P_ref_new[2:3].clamp(min=1e-3)
                        u_rn = (K_o @ P_ref_new.reshape(3, -1)).reshape(3, opt_h, opt_w)
                        u_rn = u_rn[:2] / z_rn
                        gn = u_rn.permute(1, 2, 0).clone()
                        gn[..., 0] = (gn[..., 0] / (opt_w - 1)) * 2 - 1
                        gn[..., 1] = (gn[..., 1] / (opt_h - 1)) * 2 - 1
                        
                        alpha_warped_new = F.grid_sample(alpha_ref_o.unsqueeze(0), gn.unsqueeze(0), align_corners=True)[0]
                        res_mask_new = (alpha_warped_new - mask_gt_o)
                        new_loss_val = torch.mean(torch.abs(res_new)).item() + 2.0 * torch.mean(torch.abs(res_mask_new)).item()
                    else:
                        new_loss_val = torch.mean(torch.abs(res_new)).item()
                    
                    if step == num_steps - 1:
                        print(f"[{self.__class__.__name__}] Level {res_scale} Step {step}: Loss {new_loss_val:.6f}")
                
                if new_loss_val < loss_val:
                    # Accept step
                    curr_T_W_O = torch.inverse(T_C_W_t) @ T_C_O_new
                    curr_damping /= 1.5
                    loss_val = new_loss_val # Update for next step
                else:
                    # Reject step
                    curr_damping *= 2.0
                    if curr_damping > 1e4: break
        
        self.T_W_O = curr_T_W_O.cpu().numpy() if isinstance(self.T_W_O, np.ndarray) else curr_T_W_O
        if update_ref:
            self.update_reference(image_gt.detach(), (T_C_W_t @ curr_T_W_O).detach())

        if rr_vis:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(all_losses, label='LM Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Analytical LM Warping Optimization Loss')
            ax.grid(True)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            rr.log("opt/loss_plot", rr.Image(np.array(Image.open(buf))))
            plt.close(fig)
            
            if best_warped is not None:
                # Mask with alpha to remove background "ghosts"
                warped_masked = best_warped * best_alpha if best_alpha is not None else best_warped
                rr.log("opt/warped", rr.Image(warped_masked.permute(1, 2, 0).cpu().numpy().clip(0, 1)))
            
        return curr_T_W_O, all_losses

    def optimize_wrt_image_lm_irls(self, T_C_W, image, K, mask=None, mask_ref=None,
                                  update_ref=True, rr_vis=False,
                                  pyramid_levels=[(8, 10), (4, 10), (2, 10), (1, 10)],
                                  damping=1.0, min_delta=1e-4,
                                  re_render=True, re_render_interval=1,
                                  loss_type='huber', # 'l1' or 'huber'
                                  huber_delta=0.01,
                                  near_plane=0.01, far_plane=10.0):
        """Robust LM implementation using IRLS (Iteratively Reweighted Least Squares).
        Supports L1 and Huber loss to handle outliers and rendering artifacts.
        """
        device = self.gs_params.means.device
        
        # Prep inputs
        if isinstance(image, np.ndarray):
            image_gt = torch.from_numpy(image.copy()).float().to(device)
        else:
            image_gt = image.float().to(device)
        if image_gt.max() > 1.0: image_gt /= 255.0
        if image_gt.shape[0] != 3: image_gt = image_gt.permute(2, 0, 1)
        
        mask_gt = None
        if mask is not None:
            if isinstance(mask, np.ndarray):
                mask_gt = torch.from_numpy(mask.copy()).float().to(device)
            else:
                mask_gt = mask.float().to(device)
            if mask_gt.max() > 1.0: mask_gt /= 255.0
            if mask_gt.dim() == 2: mask_gt = mask_gt.unsqueeze(0)
            elif mask_gt.shape[0] != 1: mask_gt = mask_gt.permute(2, 0, 1)

        mask_ref_t = None
        if mask_ref is not None:
            if isinstance(mask_ref, np.ndarray):
                mask_ref_t = torch.from_numpy(mask_ref.copy()).float().to(device)
            else:
                mask_ref_t = mask_ref.float().to(device)
            if mask_ref_t.max() > 1.0: mask_ref_t /= 255.0
            if mask_ref_t.dim() == 2: mask_ref_t = mask_ref_t.unsqueeze(0)
            elif mask_ref_t.shape[0] != 1: mask_ref_t = mask_ref_t.permute(2, 0, 1)

        if isinstance(K, np.ndarray): K_t = torch.from_numpy(K.copy()).float().to(device)
        else: K_t = K.float().to(device)
        if isinstance(T_C_W, np.ndarray): T_C_W_t = torch.from_numpy(T_C_W.copy()).float().to(device)
        else: T_C_W_t = T_C_W.detach().float().to(device)
        if isinstance(self.T_W_O, np.ndarray): T_W_O_t = torch.from_numpy(self.T_W_O.copy()).float().to(device)
        else: T_W_O_t = self.T_W_O.detach().float().to(device)

        if self.image_ref is None or self.T_C_O_ref is None:
            with torch.no_grad():
                T_C_O_init = T_C_W_t @ T_W_O_t
                img_ref, _, _, _ = render_2dgs(
                    self.gs_params.means, self.gs_params.quats, self.gs_params.scales,
                    self.gs_params.colors, self.gs_params.opacity,
                    viewmat=T_C_O_init, K=K_t, width=image_gt.shape[2], height=image_gt.shape[1],
                    near_plane=near_plane, far_plane=far_plane
                )
                self.update_reference(img_ref, T_C_O_init)

        H, W = image_gt.shape[1], image_gt.shape[2]
        curr_T_W_O = T_W_O_t.clone()
        
        all_losses = []
        best_warped = None
        best_alpha = None

        # Optimization loop
        for res_scale, num_steps in pyramid_levels:
            opt_h, opt_w = H // res_scale, W // res_scale
            
            with torch.no_grad():
                img_ref_o = F.interpolate(self.image_ref.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
                mask_ref_o = F.interpolate(mask_ref_t.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0] if mask_ref_t is not None else None
                # Apply reference mask to reference image
                if mask_ref_o is not None:
                    img_ref_o = img_ref_o * mask_ref_o

                img_gt_o = F.interpolate(image_gt.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
                mask_gt_o = F.interpolate(mask_gt.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0] if mask_gt is not None else None
                K_o = K_t.clone()
                K_o[0, 0] *= (opt_w / W); K_o[1, 1] *= (opt_h / H)
                K_o[0, 2] *= (opt_w / W); K_o[1, 2] *= (opt_h / H)

                # Apply mask to GT image
                if mask_gt_o is not None:
                    img_gt_o = img_gt_o * mask_gt_o

                if not re_render:
                    T_C_O_init = T_C_W_t @ curr_T_W_O
                    n_c_init, d_c_init = self.gs_to_planar_params(self.gs_params, T_C_O_init)
                    n_map_f, alpha_f = self.render_custom_attribute(self.gs_params, n_c_init, T_C_O_init, K_t, W, H, near_plane, far_plane)
                    d_3ch_f, _ = self.render_custom_attribute(self.gs_params, d_c_init.repeat(1, 3), T_C_O_init, K_t, W, H, near_plane, far_plane)
                    d_map_f = d_3ch_f[0:1]
                    
                    n_map_o = F.interpolate(n_map_f.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
                    d_map_o = F.interpolate(d_map_f.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]
                    alpha_o = F.interpolate(alpha_f.unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0]

            # --- Analytical Jacobian (Reference Gradients computed once per scale) ---
            grad_x = torch.zeros_like(img_ref_o)
            grad_y = torch.zeros_like(img_ref_o)
            grad_x[:, :, 1:-1] = 0.5 * (img_ref_o[:, :, 2:] - img_ref_o[:, :, :-2])
            grad_y[:, 1:-1, :] = 0.5 * (img_ref_o[:, 2:, :] - img_ref_o[:, :-2, :])
            
            grad_alpha_x, grad_alpha_y = None, None
            if self.alpha_ref is not None:
                alpha_ref_o = F.interpolate(self.alpha_ref.unsqueeze(0).unsqueeze(0), size=(opt_h, opt_w), mode='bilinear')[0, 0]
                grad_alpha_x = torch.zeros_like(alpha_ref_o)
                grad_alpha_y = torch.zeros_like(alpha_ref_o)
                grad_alpha_x[:, 1:-1] = 0.5 * (alpha_ref_o[:, 2:] - alpha_ref_o[:, :-2])
                grad_alpha_y[1:-1, :] = 0.5 * (alpha_ref_o[2:, :] - alpha_ref_o[:-2, :])

            for step in range(num_steps):
                with torch.no_grad():
                    T_W_O_curr = curr_T_W_O
                    T_C_O_curr = T_C_W_t @ T_W_O_curr
                    T_ref_curr = self.T_C_O_ref @ torch.inverse(T_C_O_curr)
                    
                    if re_render and step % re_render_interval == 0:
                        # Re-render geometry maps
                        n_c, d_c = self.gs_to_planar_params(self.gs_params, T_C_O_curr)
                        n_map_o, alpha_o = self.render_custom_attribute(self.gs_params, n_c, T_C_O_curr, K_o, opt_w, opt_h, near_plane, far_plane)
                        d_3ch_o, _ = self.render_custom_attribute(self.gs_params, d_c.repeat(1, 3), T_C_O_curr, K_o, opt_w, opt_h, near_plane, far_plane)
                        d_map_o = d_3ch_o[0:1]

                    # Residuals
                    warped = self.manual_backward_homography_warp(img_ref_o, n_map_o, d_map_o, T_ref_curr, K_o)
                    res = (warped - img_gt_o) 
                    
                    # Robust weights (IRLS)
                    abs_res = torch.abs(res)
                    if loss_type == 'huber':
                        weights = torch.where(abs_res <= huber_delta, torch.ones_like(abs_res), huber_delta / abs_res.clamp(min=1e-6))
                    elif loss_type == 'l1':
                        weights = 1.0 / abs_res.clamp(min=1e-4)
                    else:
                        weights = torch.ones_like(abs_res)
                    
                    # Also weight by alpha and GT mask to ignore background
                    weights = weights * alpha_o
                    if mask_gt_o is not None:
                        weights = weights * mask_gt_o
                    
                    loss_val = torch.mean(abs_res).item()
                    all_losses.append(loss_val)
                    
                    if rr_vis and res_scale == 1:
                        best_warped = warped.detach()
                        best_alpha = alpha_o.detach()

                # Unproject current pixels
                y_coords, x_coords = torch.meshgrid(torch.arange(opt_h, device=device), torch.arange(opt_w, device=device), indexing='ij')
                coords = torch.stack([x_coords, y_coords, torch.ones_like(x_coords)], dim=0).float()
                dirs = torch.inverse(K_o) @ coords.reshape(3, -1)
                dirs = dirs.reshape(3, opt_h, opt_w)
                n_dot_dir = torch.sum(n_map_o * dirs, dim=0, keepdim=True)
                depth_curr = -d_map_o / (n_dot_dir + 1e-6)
                P_curr = dirs * depth_curr
                
                # Transform to reference
                R_rc = T_ref_curr[:3, :3]
                t_rc = T_ref_curr[:3, 3:4]
                P_ref = R_rc @ P_curr.reshape(3, -1) + t_rc
                P_ref = P_ref.reshape(3, opt_h, opt_w)
                z_ref = P_ref[2:3].clamp(min=1e-3)
                u_ref = (K_o @ P_ref.reshape(3, -1)).reshape(3, opt_h, opt_w)
                u_ref = u_ref[:2] / z_ref
                
                # Sample gradients
                grid = u_ref.permute(1, 2, 0).clone()
                grid[..., 0] = (grid[..., 0] / (opt_w - 1)) * 2 - 1
                grid[..., 1] = (grid[..., 1] / (opt_h - 1)) * 2 - 1
                g_x_sampled = F.grid_sample(grad_x.unsqueeze(0), grid.unsqueeze(0), align_corners=True)[0]
                g_y_sampled = F.grid_sample(grad_y.unsqueeze(0), grid.unsqueeze(0), align_corners=True)[0]
                
                g_alpha_x_sampled = None
                g_alpha_y_sampled = None
                if grad_alpha_x is not None:
                    g_alpha_x_sampled = F.grid_sample(grad_alpha_x.unsqueeze(0), grid.unsqueeze(0), align_corners=True)[0]
                    g_alpha_y_sampled = F.grid_sample(grad_alpha_y.unsqueeze(0), grid.unsqueeze(0), align_corners=True)[0]
                
                # Pose Jacobian
                fx, fy = K_o[0, 0], K_o[1, 1]
                inv_z = 1.0 / z_ref; inv_z2 = inv_z * inv_z
                dp_dP = torch.zeros(2, 3, opt_h, opt_w, device=device)
                dp_dP[0, 0] = fx * inv_z; dp_dP[0, 2] = -fx * P_ref[0] * inv_z2
                dp_dP[1, 1] = fy * inv_z; dp_dP[1, 2] = -fy * P_ref[1] * inv_z2
                
                x, y, z = P_curr[0], P_curr[1], P_curr[2]
                R = R_rc
                J_pose = torch.zeros(2, 6, opt_h, opt_w, device=device)
                for i in range(3): J_pose[:, i] = -(dp_dP[:, 0] * R[0, i] + dp_dP[:, 1] * R[1, i] + dp_dP[:, 2] * R[2, i])
                RPx0 = R[:, 1:2, None, None] * z[None] - R[:, 2:3, None, None] * y[None]
                RPx1 = -R[:, 0:1, None, None] * z[None] + R[:, 2:3, None, None] * x[None]
                RPx2 = R[:, 0:1, None, None] * y[None] - R[:, 1:2, None, None] * x[None]
                J_pose[:, 3] = dp_dP[:, 0] * RPx0[0, 0] + dp_dP[:, 1] * RPx0[1, 0] + dp_dP[:, 2] * RPx0[2, 0]
                J_pose[:, 4] = dp_dP[:, 0] * RPx1[0, 0] + dp_dP[:, 1] * RPx1[1, 0] + dp_dP[:, 2] * RPx1[2, 0]
                J_pose[:, 5] = dp_dP[:, 0] * RPx2[0, 0] + dp_dP[:, 1] * RPx2[1, 0] + dp_dP[:, 2] * RPx2[2, 0]
                
                # Final Jacobian [3, 6, H, W]
                J = g_x_sampled.unsqueeze(1) * J_pose[0:1] + g_y_sampled.unsqueeze(1) * J_pose[1:2]
                J_flat = J.reshape(3, 6, -1) 
                res_flat = res.reshape(3, -1)
                weights_flat = weights.reshape(3, -1)
                
                # JTJ = sum_pix,chan w * J^T * J
                JTJ = torch.einsum('cin,cn,cjn->ij', J_flat, weights_flat, J_flat)
                JTr = torch.einsum('cin,cn,cn->i', J_flat, weights_flat, res_flat)
                
                diag = torch.diag(torch.diag(JTJ))
                delta = torch.linalg.solve(JTJ + damping * diag + 1e-6 * torch.eye(6, device=device), -JTr)
                
                # Early stopping check
                if torch.norm(delta) < min_delta:
                    break
                
                dT = torch.eye(4, device=device)
                dT[:3, :3] = so3_exp_map(delta[3:6].unsqueeze(0))
                dT[:3, 3] = delta[0:3]
                
                T_C_O_curr = T_C_W_t @ curr_T_W_O
                T_C_O_new = dT @ T_C_O_curr
                
                # Evaluate new loss (Huber/L1 weighted)
                with torch.no_grad():
                    T_ref_new = self.T_C_O_ref @ torch.inverse(T_C_O_new)
                    n_c_new, d_c_new = self.gs_to_planar_params(self.gs_params, T_C_O_new)
                    n_map_new, alpha_new = self.render_custom_attribute(self.gs_params, n_c_new, T_C_O_new, K_o, opt_w, opt_h, near_plane, far_plane)
                    d_3ch_new, _ = self.render_custom_attribute(self.gs_params, d_c_new.repeat(1, 3), T_C_O_new, K_o, opt_w, opt_h, near_plane, far_plane)
                    d_map_new = d_3ch_new[0:1]
                    warped_new = self.manual_backward_homography_warp(img_ref_o, n_map_new, d_map_new, T_ref_new, K_o)
                    res_new = (warped_new - img_gt_o) * alpha_new
                    if mask_gt_o is not None:
                        res_new = res_new * mask_gt_o
                    abs_res_new = torch.abs(res_new)
                    if loss_type == 'huber':
                        w_new = torch.where(abs_res_new <= huber_delta, torch.ones_like(abs_res_new), huber_delta / abs_res_new.clamp(min=1e-6))
                    elif loss_type == 'l1':
                        w_new = 1.0 / abs_res_new.clamp(min=1e-4)
                    else:
                        w_new = torch.ones_like(abs_res_new)
                    new_loss_val = torch.mean(w_new * abs_res_new * alpha_new).item()
                
                if new_loss_val < loss_val:
                    # Accept step
                    curr_T_W_O = torch.inverse(T_C_W_t) @ T_C_O_new
                    damping /= 1.5
                    loss_val = new_loss_val
                else:
                    # Reject step
                    damping *= 2.0
                    if damping > 1e4: break
        
        self.T_W_O = curr_T_W_O.cpu().numpy() if isinstance(self.T_W_O, np.ndarray) else curr_T_W_O
        if update_ref: self.update_reference(image_gt.detach(), (T_C_W_t @ curr_T_W_O).detach())

        if rr_vis:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(all_losses, label='LM-IRLS Loss')
            ax.set_xlabel('Step'); ax.set_ylabel('Loss'); ax.grid(True)
            buf = io.BytesIO(); fig.savefig(buf, format='png', dpi=100); buf.seek(0)
            rr.log("opt/loss_plot", rr.Image(np.array(Image.open(buf)))); plt.close(fig)
            if best_warped is not None:
                warped_masked = best_warped * best_alpha if best_alpha is not None else best_warped
                rr.log("opt/warped", rr.Image(warped_masked.permute(1, 2, 0).cpu().numpy().clip(0, 1)))
        return curr_T_W_O, all_losses

    def optimize_wrt_image_lm_hybrid(self, T_C_W, image, K, mask=None, mask_ref=None,
                                    adam_steps=30, lm_steps=20,
                                    lr=1e-3, damping=1.0, 
                                    pyramid_levels=None,
                                    re_render=True, re_render_interval=1,
                                    irls=False,
                                    update_ref=True, rr_vis=False, gs_type="2d"):
        """Hybrid optimization: Adam for global convergence, LM for local refinement.
        """
        # 1. Adam Phase
        print(f"[{self.__class__.__name__}] Starting Adam phase ({adam_steps} steps)...")
        T_W_O_adam, losses_adam = self.optimize_wrt_image(
            T_C_W, image, K, mask=mask, lr=lr, num_steps=adam_steps, 
            update_ref=False, rr_vis=False, gs_type=gs_type
        )
        # T_W_O is updated by the call above
        
        # 2. LM Phase
        print(f"[{self.__class__.__name__}] Starting LM phase ({lm_steps} steps)...")
        if pyramid_levels is None:
            pyramid_levels = [(1, lm_steps)]

        if irls:
            T_W_O_lm, losses_lm = self.optimize_wrt_image_lm_irls(
                T_C_W, image, K, mask=mask, mask_ref=mask_ref, update_ref=update_ref, rr_vis=rr_vis,
                pyramid_levels=pyramid_levels,
                damping=damping, re_render=re_render, re_render_interval=re_render_interval
            )
        else:
            T_W_O_lm, losses_lm = self.optimize_wrt_image_lm(
                T_C_W, image, K, mask=mask, mask_ref=mask_ref, update_ref=update_ref, rr_vis=rr_vis,
                pyramid_levels=pyramid_levels,
                damping=damping, re_render=re_render, re_render_interval=re_render_interval,
                gs_type=gs_type
            )
        
        return T_W_O_lm, losses_adam + losses_lm

    def compute_track_2d_loss(self, T_C_W_query, K_query, T_W_O_query,
                              T_C_W_target, K_target, T_W_O_target,
                              tracks_2d_target, mask_query,
                              track_weights=None, quantile=0.98):
        """
        Compute the 2D track loss between a query frame and a target frame.
        
        Args:
            T_C_W_query: Camera-from-World pose of the query frame [4, 4]
            K_query: Intrinsics of the query frame [3, 3]
            T_W_O_query: World-from-Object pose of the query frame [4, 4]
            T_C_W_target: Camera-from-World pose of the target frame [4, 4]
            K_target: Intrinsics of the target frame [3, 3]
            T_W_O_target: World-from-Object pose of the target frame [4, 4] (Optimization variable)
            tracks_2d_target: GT 2D positions in target frame [H, W, 2] (e.g. u_query + flow)
            mask_query: Binary mask of the object in the query frame [H, W]
            track_weights: Optional weights for each track [H, W]
            quantile: Quantile for robust loss
            
        Returns:
            Scalar loss value
        """
        device = self.gs_params.means.device
        H, W = mask_query.shape[0], mask_query.shape[1]
        
        # 1. Render object-frame 3D positions in the query frame
        T_C_O_query = T_C_W_query @ T_W_O_query
        means_O_map, alpha_query = self.render_custom_attribute(
            self.gs_params, self.gs_params.means, T_C_O_query, K_query, W, H
        ) # means_O_map: [3, H, W]
        
        # 2. Transform these 3D points to the target camera frame
        # P_C_target = T_C_W_target @ T_W_O_target @ P_O
        T_C_O_target = T_C_W_target @ T_W_O_target
        
        P_O_flat = means_O_map.reshape(3, -1) # [3, HW]
        P_C_target_flat = T_C_O_target[:3, :3] @ P_O_flat + T_C_O_target[:3, 3:4] # [3, HW]
        
        # 3. Project to target image plane
        # u_target_pred = K_target @ P_C_target
        u_target_pred_homog = K_target @ P_C_target_flat # [3, HW]
        z_target = u_target_pred_homog[2:3, :].clamp(min=1e-6)
        u_target_pred = u_target_pred_homog[:2, :] / z_target # [2, HW]
        u_target_pred = u_target_pred.reshape(2, H, W).permute(1, 2, 0) # [H, W, 2]
        
        # 4. Compute loss
        # We only care about pixels where the object is present and rendered
        combined_mask = (mask_query > 0.5) & (alpha_query > 0.5)
        
        if track_weights is None:
            track_weights = torch.ones_like(mask_query)
            
        # Reshape for masked_l1_loss
        pred_flat = u_target_pred[combined_mask]
        gt_flat = tracks_2d_target[combined_mask]
        weights_flat = track_weights[combined_mask].unsqueeze(-1)
        
        loss = masked_l1_loss(
            pred_flat, 
            gt_flat, 
            mask=weights_flat, 
            quantile=quantile
        ) / max(H, W)
        
        return loss

    def optimize_wrt_flow(self, T_C_W_query, K_query, T_W_O_query, mask_query,
                          T_C_W_target, K_target, flow,
                          lr=1e-3, num_steps=100, rr_vis=False,
                          quantile=0.98):
        """
        Optimize T_W_O of the target frame using optical flow from a query frame.
        
        Args:
            T_C_W_query: Camera pose of query frame
            K_query: Intrinsics of query frame
            T_W_O_query: Object pose of query frame
            mask_query: Object mask in query frame
            T_C_W_target: Camera pose of target frame
            K_target: Intrinsics of target frame
            flow: Optical flow from query to target [H, W, 2]
        """
        device = self.gs_params.means.device
        
        # Prepare target tracks: u_query + flow
        H, W = mask_query.shape[0], mask_query.shape[1]
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        u_query = torch.stack([grid_x, grid_y], dim=-1).float() # [H, W, 2]
        tracks_2d_target = u_query + flow
        
        # Optimization variables
        trans = torch.zeros(3, device=device, requires_grad=True)
        log_R = torch.zeros((1, 3), device=device, requires_grad=True)
        
        optimizer = torch.optim.AdamW([
            {"params": trans, "lr": lr * 1.5},
            {"params": log_R, "lr": lr}
        ])
        
        T_W_O_target_init = torch.from_numpy(self.T_W_O).float().to(device) if isinstance(self.T_W_O, np.ndarray) else self.T_W_O
        
        losses = []
        for step in range(num_steps):
            optimizer.zero_grad()
            
            rot_mat_delta = so3_exp_map(log_R)[0]
            T_delta = torch.eye(4, device=device)
            T_delta[:3, :3] = rot_mat_delta
            T_delta[:3, 3] = trans
            
            T_W_O_target_curr = T_delta @ T_W_O_target_init
            
            loss = self.compute_track_2d_loss(
                T_C_W_query, K_query, T_W_O_query,
                T_C_W_target, K_target, T_W_O_target_curr,
                tracks_2d_target, mask_query,
                quantile=quantile
            )
            
            if torch.isnan(loss): break
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if step % 50 == 0:
                print(f"[{self.__class__.__name__} Flow] Step {step}/{num_steps} - Loss: {loss.item():.6f}")
                
        # Update current pose
        with torch.no_grad():
            rot_mat_delta = so3_exp_map(log_R)[0]
            T_delta = torch.eye(4, device=device)
            T_delta[:3, :3] = rot_mat_delta
            T_delta[:3, 3] = trans
            best_T_W_O = T_delta @ T_W_O_target_init
            self.T_W_O = best_T_W_O.cpu().numpy() if isinstance(self.T_W_O, np.ndarray) else best_T_W_O
            
        return best_T_W_O, losses

if __name__ == "__main__":
    pass
