"""
We use a set of GS to model a rigid object. 
"""

import numpy as np
import torch
from .gs_rendering import render_2dgs
from .gs_param import GSParam
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply
from .utils.ssim import image_loss
from pytorch3d.transforms.so3 import so3_exp_map
import rerun as rr
import io
from PIL import Image
import matplotlib.pyplot as plt

class ObjectGS: 
    def __init__(self, gs_params: GSParam, T_W_O: np.ndarray, obj_scale: float):
        self.gs_params = gs_params
        self.T_W_O = T_W_O
        self.obj_scale = obj_scale

    def optimize_wrt_image(self, image, K, T_C_O_init, 
                           lr=1e-3, num_steps=100, rr_vis=False,
                           static_bg_img=None, mask=None, hand_mask=None,
                           near_plane=0.01, far_plane=10.0):
        """Optimize T_C_O by transforming GS primitives.
           This follows the logic from grouped_gs.py:track_to_frame_with_static_bg.
           It transforms GS means and quats explicitly.
        """
        device = self.gs_params.means.device
        
        # Convert inputs to torch tensors
        if isinstance(image, np.ndarray):
            image_gt = torch.from_numpy(image.copy()).float().to(device)
        else:
            image_gt = image.float().to(device)
        
        if image_gt.max() > 1.0:
            image_gt = image_gt / 255.0
        if image_gt.shape[0] != 3:
            image_gt = image_gt.permute(2, 0, 1)
            
        if isinstance(K, np.ndarray):
            K_t = torch.from_numpy(K.copy()).float().to(device)
        else:
            K_t = K.float().to(device)
            
        if isinstance(T_C_O_init, np.ndarray):
            T_C_O_t = torch.from_numpy(T_C_O_init.copy()).float().to(device)
        else:
            T_C_O_t = T_C_O_init.float().to(device)

        # Initial camera-space GS
        means_o = self.gs_params.means
        quats_o = self.gs_params.quats
        
        # Transform object-space GS to camera-space starting from T_C_O_init
        means_c_init = torch.einsum('ij,nj->ni', T_C_O_t[:3, :3], means_o) + T_C_O_t[:3, 3]
        quat_C_O_init = matrix_to_quaternion(T_C_O_t[:3, :3])
        quats_c_init = quaternion_multiply(quat_C_O_init.unsqueeze(0), quats_o)
        
        # Optimization variables: delta pose from T_C_O_init
        trans = torch.zeros(3, device=device, requires_grad=True)
        log_R = torch.zeros((1, 3), device=device, requires_grad=True)
        
        optimizer = torch.optim.AdamW([
            {"params": trans, "lr": lr},
            {"params": log_R, "lr": lr}
        ])
        
        with torch.no_grad():
            center = means_c_init.mean(0, keepdim=True)
            xyz_delta = means_c_init - center
            
        bg = torch.zeros(3, device=device)
        
        valid_loss_mask = torch.ones_like(image_gt[0:1, :, :], dtype=torch.bool)
        if mask is not None:
             valid_loss_mask = valid_loss_mask & mask.unsqueeze(0).to(device)
        if hand_mask is not None:
             valid_loss_mask = valid_loss_mask & ~hand_mask.unsqueeze(0).to(device)

        if static_bg_img is not None:
            static_bg_t = static_bg_img.float().to(device)
            if static_bg_t.max() > 1.0:
                static_bg_t = static_bg_t / 255.0
            if static_bg_t.shape[0] != 3:
                static_bg_t = static_bg_t.permute(2, 0, 1)
        else:
            static_bg_t = None

        losses = []
        best_loss = float('inf')
        best_trans = None
        best_log_R = None

        for step in range(num_steps):
            optimizer.zero_grad()
            
            rot_mat = so3_exp_map(log_R)[0]
            if torch.isnan(rot_mat).any():
                break
                
            new_means = center + torch.einsum("ij,nj->ni", rot_mat, xyz_delta) + trans
            delta_quat = matrix_to_quaternion(rot_mat).unsqueeze(0)
            new_quats = quaternion_multiply(delta_quat, quats_c_init)
            
            render_image, render_depth, render_normal = render_2dgs(
                new_means, new_quats, self.gs_params.scales,
                self.gs_params.colors, self.gs_params.opacity,
                viewmat=torch.eye(4, device=device),
                K=K_t,
                width=image_gt.shape[2], height=image_gt.shape[1],
                bg=bg,
                near_plane=near_plane,
                far_plane=far_plane
            )
            
            # Alpha blend with static background if available
            if static_bg_t is not None:
                mask_bg = (render_image == bg[:, None, None]).all(dim=0)
                render_image[:, mask_bg] = static_bg_t[:, mask_bg]
            
            loss = image_loss(render_image * valid_loss_mask, image_gt * valid_loss_mask)
            
            if torch.isnan(loss):
                break
                
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            losses.append(current_loss)
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_trans = trans.detach().clone()
                best_log_R = log_R.detach().clone()
                best_render = render_image.detach().clone()

        if rr_vis:
            # Create and log loss plot similar to grouped_gs.py
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(losses, label='Training Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Optimization Loss Over Steps')
            ax.grid(True)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            loss_plot_image = Image.open(buf)
            loss_plot_array = np.array(loss_plot_image)
            rr.log("loss_plot", rr.Image(loss_plot_array))
            plt.close(fig)
            
            if best_render is not None:
                rr.log("opt_rendered", rr.Image(best_render.permute(1, 2, 0).cpu().numpy().clip(0, 1)))

        # Compute final T_C_O using best parameters
        with torch.no_grad():
            if best_log_R is not None:
                rot_mat_final = so3_exp_map(best_log_R)[0]
                # T_delta: X_c = R_delta(X_c0 - c) + c + t = R_delta X_c0 + (I - R_delta)c + t
                t_delta = (torch.eye(3, device=device) - rot_mat_final) @ center.squeeze() + best_trans
                
                T_delta = torch.eye(4, device=device)
                T_delta[:3, :3] = rot_mat_final
                T_delta[:3, 3] = t_delta
                
                T_C_O_final = T_delta @ T_C_O_t
            else:
                T_C_O_final = T_C_O_t
            
        return T_C_O_final, losses

    def optimize_pose_wrt_image(self, image, K, T_C_O_init, 
                             lr=1e-3, num_steps=100, rr_vis=False,
                             static_bg_img=None, mask=None, hand_mask=None,
                             near_plane=0.01, far_plane=10.0):
        """Optimize T_C_O directly as a viewmat.
           The object remains the same (no need to compute center and quat for points).
           Each step updates the viewmat to view the object from different angles.
        """
        device = self.gs_params.means.device
        
        # Convert inputs to torch tensors
        if isinstance(image, np.ndarray):
            image_gt = torch.from_numpy(image.copy()).float().to(device)
        else:
            image_gt = image.float().to(device)
            
        if image_gt.max() > 1.0:
            image_gt = image_gt / 255.0
        if image_gt.shape[0] != 3:
            image_gt = image_gt.permute(2, 0, 1)
            
        if isinstance(K, np.ndarray):
            K_t = torch.from_numpy(K.copy()).float().to(device)
        else:
            K_t = K.float().to(device)
            
        if isinstance(T_C_O_init, np.ndarray):
            T_C_O_t = torch.from_numpy(T_C_O_init.copy()).float().to(device)
        else:
            T_C_O_t = T_C_O_init.float().to(device)

        # Optimization variables (delta pose from init)
        # We optimize T_delta such that T_C_O_new = T_delta @ T_C_O_init
        trans = torch.zeros(3, device=device, requires_grad=True)
        log_R = torch.zeros((1, 3), device=device, requires_grad=True)
        
        optimizer = torch.optim.AdamW([
            {"params": trans, "lr": lr},
            {"params": log_R, "lr": lr}
        ])
        
        valid_loss_mask = torch.ones_like(image_gt[0:1, :, :], dtype=torch.bool)
        if mask is not None:
             valid_loss_mask = valid_loss_mask & mask.unsqueeze(0).to(device)
        if hand_mask is not None:
             valid_loss_mask = valid_loss_mask & ~hand_mask.unsqueeze(0).to(device)

        if static_bg_img is not None:
            static_bg_t = static_bg_img.float().to(device)
            if static_bg_t.max() > 1.0:
                static_bg_t = static_bg_t / 255.0
            if static_bg_t.shape[0] != 3:
                static_bg_t = static_bg_t.permute(2, 0, 1)
        else:
            static_bg_t = None

        bg = torch.zeros(3, device=device)
        losses = []
        best_loss = float('inf')
        best_trans = None
        best_log_R = None

        for step in range(num_steps):
            optimizer.zero_grad()
            
            rot_mat_delta = so3_exp_map(log_R)[0]
            if torch.isnan(rot_mat_delta).any():
                break
                
            T_delta = torch.eye(4, device=device)
            T_delta[:3, :3] = rot_mat_delta
            T_delta[:3, 3] = trans
            
            current_T_C_O = T_delta @ T_C_O_t
            
            # Render using current_T_C_O as viewmat
            # Object parameters remain exactly as initialized
            render_image, _, _ = render_2dgs(
                self.gs_params.means, self.gs_params.quats, self.gs_params.scales,
                self.gs_params.colors, self.gs_params.opacity,
                viewmat=current_T_C_O,
                K=K_t,
                width=image_gt.shape[2], height=image_gt.shape[1],
                bg=bg,
                near_plane=near_plane,
                far_plane=far_plane
            )
            
            if static_bg_t is not None:
                mask_bg = (render_image == bg[:, None, None]).all(dim=0)
                render_image[:, mask_bg] = static_bg_t[:, mask_bg]
            
            loss = image_loss(render_image * valid_loss_mask, image_gt * valid_loss_mask)
            
            if torch.isnan(loss):
                break
                
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            losses.append(current_loss)
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_trans = trans.detach().clone()
                best_log_R = log_R.detach().clone()
                best_render = render_image.detach().clone()

        if rr_vis:
            # Create and log loss plot similar to grouped_gs.py
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(losses, label='Training Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Pose Optimization Loss Over Steps')
            ax.grid(True)
            
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            loss_plot_image = Image.open(buf)
            loss_plot_array = np.array(loss_plot_image)
            rr.log("loss_plot", rr.Image(loss_plot_array))
            plt.close(fig)
            
            if best_render is not None:
                rr.log("opt_rendered", rr.Image(best_render.permute(1, 2, 0).cpu().numpy().clip(0, 1)))

        # Final T_C_O using best parameters
        with torch.no_grad():
            if best_log_R is not None:
                rot_mat_final = so3_exp_map(best_log_R)[0]
                T_delta_final = torch.eye(4, device=device)
                T_delta_final[:3, :3] = rot_mat_final
                T_delta_final[:3, 3] = best_trans
                T_C_O_final = T_delta_final @ T_C_O_t
            else:
                T_C_O_final = T_C_O_t
                
        return T_C_O_final, losses
