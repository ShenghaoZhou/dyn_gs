"""
This is the reproduction of Summer Bosch project
"Gaussian SuperPrimitive"

For a rigid body object, we propose to model it as a set of GS, termed GSP.
We use a segmented image + depth + surface normal to initialize it.

During the human object interaction, this GSP is tracked by optimizing its
6DoF pose over dynamic video.

We want to refactor the code to do GSP tracking only.

In terms of tracking, this approach is similar to "tracking by detection",
since we don't pose constraint on the full trajectory
Each 6DoF pose is optimized independently.
While we can use the same origin for 6 DoF pose,
we define them as small relative transform between two consecutive frames.
so that each pose change is small, and the optimization is easier.

We may directly do that for each frame sequentially

Another benefit of this approach: between frame, the change is small,
so we can track successfully from frame 0 to 1. Then in frame 1, we not only use
it to update the object geometry, but also use frame 1 geometry for tracking to frame 2.

In this approach, we may even model "non-rigid" motion, as between frames, ridig motion is a good enough approximation
"""
import numpy as np
import torch
import torch.nn.functional as F
from .utils.vis import vis_2dgs_rerun
from .gs_rendering import render_2dgs, render_3dgs
from .utils.init import unproject_depth, d2n_tblr
from .gs_param import GSParam
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion, quaternion_multiply
from .utils.ssim import image_loss
import rerun as rr
import io
from PIL import Image
import matplotlib.pyplot as plt
from pytorch3d.transforms.so3 import (
    so3_exp_map,
    so3_relative_angle,
)
import time 

def RGB2SH(rgb):
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

class GaussianSuperPrimitive:
    def __init__(self, image, mask, depth, extrin, K, normal=None):
        # we build up the initial GS from segmented RGB and depth
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        depth = torch.from_numpy(depth.copy()).float()
        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).bool()
        
        # Filter mask by depth validity
        mask = mask & (depth > 0)
        
        K = torch.from_numpy(K.copy()).float()
        extrin = torch.from_numpy(extrin.copy()).float()
        xyz = unproject_depth(depth, K, depth.shape[0], depth.shape[1])

        f = (0.5 * (K[0, 0] + K[1, 1]))
        
        if normal is None:
            normal_from_depth, valid_mask = d2n_tblr(
                xyz.permute((2, 0, 1)).unsqueeze(0))
            # Further filter mask by normal validity (optional, but safer)
            # Get normals and validity mask from depth
            # We DONT filter the initialization mask by valid_mask here, as it is too aggressive
            # and discards valid object parts that are thin (like fingers) or near edges.
            vm = valid_mask.squeeze(0).squeeze(0)
            
            means = xyz[mask].reshape(-1, 3)
            colors = image[mask].reshape(-1, 3) / 255.0
            sizes = depth[mask] / f # focal length based sizing
            
            # Extract normals for all masked pixels.
            # d2n_tblr currently produce +Z for surfaces facing the camera (+Z is forward in RDF).
            # We will flip this to -Z so they point at the camera.
            normal = normal_from_depth[0, :, mask].transpose(0, 1)
            
            # Fallback for boundary pixels where d2n_tblr failed to find neighbors.
            norm_mags = torch.norm(normal, dim=1)
            invalid_nn = norm_mags < 1e-6
            if invalid_nn.any():
                normal[invalid_nn] = torch.tensor([0.0, 0.0, -1.0], device=normal.device)
            
            # Flip and normalize to ensure they are front-facing unit vectors.
            normal = -normal
            normal = F.normalize(normal, dim=1)
        else:
            normal = torch.from_numpy(
                normal.copy()).float()
            
            means = xyz[mask].reshape(-1, 3)
            colors = image[mask].reshape(-1, 3) / 255.0
            sizes = depth[mask] / f
            normal = normal[mask].reshape(-1, 3)
            normal = F.normalize(normal, dim=1)

        ref_axis1 = torch.randn(normal.shape[0], 3)
        ref_axis1 = ref_axis1 / torch.norm(ref_axis1, dim=1, keepdim=True)
        rotation_axis1 = ref_axis1 - \
            (normal * ref_axis1).sum(dim=1, keepdim=True) * normal
        rotation_axis1 = rotation_axis1 / \
            torch.norm(rotation_axis1, dim=1, keepdim=True)
        rotation_axis2 = torch.cross(normal, rotation_axis1, dim=1)
        rotation_axis2 = rotation_axis2 / \
            torch.norm(rotation_axis2, dim=1, keepdim=True)
        gs_rotations = torch.stack(
            [rotation_axis1, rotation_axis2, normal], dim=1).transpose(1, 2)
        gs_rotations[torch.det(gs_rotations) < 0] *= -1

        pose = extrin.inverse()
        gs_means = torch.einsum('ij,nj->ni', pose[:3, :3], means) + pose[:3, 3]
        gs_rotations = torch.einsum(
            'ij,njk->nik', pose[:3, :3], gs_rotations)

        gs_quats = matrix_to_quaternion(gs_rotations)

        gs_scales = torch.ones(
            gs_means.shape[0], 2).cuda() * sizes.view(-1, 1).cuda()
        gs_opacity = torch.ones(gs_means.shape[0], 1).cuda() * 0.5
        self.gs_params = GSParam(
            gs_means.to(device),
            gs_quats.to(device),
            gs_scales.to(device),
            colors.to(device),
            gs_opacity.to(device)
        )

    def track_to_frame(self, frame_next, extrin_next, K_next, mask=None,
                       bg=torch.zeros(3), lr=1e-2, num_steps=100, rr_vis=False,
                       loss_increase_tol=0.0):
        """Optimize 6 DoF pose of object to align with next frame observation

        Args:
            frame_next (torch.Tensor): Next frame observation
            extrin_next (torch.Tensor): Next frame extrinsics
            K_next (torch.Tensor): Next frame intrinsics
            loss_increase_tol (float): Early stop if loss rises above previous by this amount
        """
        trans = torch.zeros(3,
                            device=frame_next.device)
        trans.requires_grad = True
        # which rotation representation for optimization?
        rot6d = torch.zeros(6,
                            device=frame_next.device)
        rot6d[[0, -2]] = 1.0  # initialize to identity
        rot6d.requires_grad = True
        optimizer = torch.optim.AdamW([trans, rot6d], lr=lr)
        image_gt = frame_next.permute(2, 0, 1) / 255.0
        if mask is not None:
            image_gt[:, ~mask] = bg[:, None].to(frame_next.device)
        with torch.no_grad():
            center = self.gs_params.means.mean(0, keepdim=True)
            xyz_delta = self.gs_params.means - center
            base_quats = self.gs_params.quats.detach()
        gs_params_tmp = self.gs_params.clone()
        prev_loss = None
        best_loss = float('inf')
        for step in range(num_steps):
            if rr_vis:
                rr.set_time("opt_step", sequence=step)
            optimizer.zero_grad()
            rot_mat = rotation_6d_to_matrix(rot6d)

            new_means = center + \
                torch.einsum("ij,nj->ni", rot_mat, xyz_delta) + trans
            # self.gs_params.means = new_means
            # render_image, render_depth, render_normal = self.gs_params.render(
            #     viewmat=extrin_next,
            #     K=K_next,
            #     width=frame_next.shape[1], height=frame_next.shape[0],
            #     scaling_modifier=1.0,
            #     bg=bg
            # )
            # detach base_quats to avoid reusing the autograd graph across steps
            delta_quat = matrix_to_quaternion(rot_mat).unsqueeze(0)
            new_quats = quaternion_multiply(delta_quat, base_quats)
            render_image, render_depth, render_normal = render_2dgs(
                new_means, new_quats, self.gs_params.scales,
                self.gs_params.colors, self.gs_params.opacity,
                viewmat=extrin_next,
                K=K_next,
                width=frame_next.shape[1], height=frame_next.shape[0],
                scaling_modifier=1.0,
                bg=bg
            )

            # render_image, render_depth, render_normal = render_2dgs(
            #     new_means, self.gs_params.quats, self.gs_params.scales,
            #     self.gs_params.colors, self.gs_params.opacity,
            #     viewmat=extrin_next,
            #     K=K_next,
            #     width=frame_next.shape[1], height=frame_next.shape[0],
            #     scaling_modifier=1.0,
            #     bg=bg
            # )
            if rr_vis:
                if step % 10 == 0:
                    gs_params_tmp.means = new_means

                    with torch.no_grad():
                        rr.log("debug_render_image",
                               rr.Image(
                                   render_image.permute(
                                       1, 2, 0).cpu().numpy()
                               ))
                        rr.log("debug_error_image",
                               rr.DepthImage(
                                   torch.square(render_image-image_gt).permute(
                                       1, 2, 0).sum(dim=-1).cpu().numpy()
                               ))
                        vis_2dgs_rerun(gs_params_tmp, name="debug_gs",
                                       points_only=True)
            # image_tensor[:, ~obj_mask_next] = bg[:, None].cuda()
            loss = image_loss(render_image, image_gt)
            current_loss = loss.item()

            # stop early if loss rises noticeably
            if prev_loss is not None and current_loss > prev_loss + loss_increase_tol:
                if rr_vis:
                    rr.log("log", rr.TextLog(
                        f"Early stop at step {step}: loss increased from {prev_loss:.6f} to {current_loss:.6f}"))
                break

            prev_loss = current_loss
            loss.backward()
            optimizer.step()

        # TODO: return estimated pose?
        rot_mat = rotation_6d_to_matrix(rot6d)
        delta_pose = torch.eye(4)
        delta_pose[:3, :3] = rot_mat
        delta_pose[:3, 3] = trans
        return delta_pose

    def track_to_frame_with_static_bg(self, frame_next, extrin_next, K_next, static_bg_img, mask=None, hand_mask=None,
                                      lr=1e-2, num_steps=100, rr_vis=False,
                                      convergence_threshold=5e-5, patience=3,
                                      near_plane=5e-4, far_plane=10.0,
                                      loss_increase_tol=0.0):
        """Optimize 6 DoF pose of object to align with next frame observation

        Args:
            frame_next (torch.Tensor): Next frame observation
            extrin_next (torch.Tensor): Next frame extrinsics
            K_next (torch.Tensor): Next frame intrinsics
            static_bg_img (torch.Tensor): Static background image
            mask (torch.Tensor, optional): Object mask
            hand_mask (torch.Tensor, optional): Hand mask to exclude from loss
            lr (float): Learning rate
            num_steps (int): Maximum optimization steps
            rr_vis (bool): Enable rerun visualization
            convergence_threshold (float): Loss change threshold for early stopping
            patience (int): Number of steps to wait before stopping if loss doesn't improve
            loss_increase_tol (float): Early stop if loss rises above previous by this amount
        """
        bg = torch.zeros(3, device=frame_next.device)
        trans = torch.zeros(3,
                            device=frame_next.device)
        trans.requires_grad = True
        # which rotation representation for optimization?
        # rot6d = torch.zeros(6,
        #                     device=frame_next.device)
        # rot6d[[0, -2]] = 1.0  # initialize to identity
        # rot6d.requires_grad = True
        log_R = torch.zeros((1, 3), device=frame_next.device)
        log_R.requires_grad = True

        # optimizer = torch.optim.AdamW([
        #     {"params": trans, "lr": lr},
        #     {"params": log_R, "lr": lr}
        # ])
        # optimizer = torch.optim.SGD([
        #     {"params": trans, "lr": 3e-3},
        #     {"params": log_R, "lr": 1e-2}
        # ])
        # optimizer = torch.optim.SGD([
        #     {"params": trans, "lr": 5e-2},
        #     {"params": log_R, "lr": 1e-2}
        # ])
        # optimizer = torch.optim.AdamW([
        #     {"params": trans, "lr": 3e-3},
        #     {"params": log_R, "lr": 1e-2}
        # ])
        optimizer = torch.optim.AdamW([
                {"params": trans, "lr": 3e-3},
                {"params": log_R, "lr": 1e-3}
            ])
        image_gt = frame_next.permute(2, 0, 1) / 255.0
        valid_loss_mask = torch.ones_like(image_gt[0:1, :, :], dtype=torch.bool)
        if mask is not None:
            # image_gt[:, ~mask] = bg[:, None].to(frame_next.device)
            valid_loss_mask = valid_loss_mask & mask.unsqueeze(0)
        if hand_mask is not None:
            # image_gt[:, hand_mask_tensor] = bg[:, None].to(frame_next.device)
            valid_loss_mask = valid_loss_mask & ~hand_mask.unsqueeze(0)
        static_bg_img = static_bg_img.to(frame_next.device)
        with torch.no_grad():
            center = self.gs_params.means.mean(0, keepdim=True)
            xyz_delta = self.gs_params.means - center
            base_quats = self.gs_params.quats.detach()
        gs_params_tmp = self.gs_params.clone()

        # Early stopping variables
        prev_loss = None
        best_loss = float('inf')

        patience_counter = 0
        converged = False
        losses = []

        with torch.autograd.set_detect_anomaly(False):
            for step in range(num_steps):
                if rr_vis:
                    rr.set_time("opt_step", sequence=step)
                optimizer.zero_grad()
                # rot_mat = rotation_6d_to_matrix(rot6d)

                rot_mat = so3_exp_map(log_R)[0]
                if torch.isnan(rot_mat).any():
                    raise ValueError("NaN encountered in rotation matrix during optimization.")
                new_means = center + \
                    torch.einsum("ij,nj->ni", rot_mat, xyz_delta) + trans

                delta_quat = matrix_to_quaternion(rot_mat).unsqueeze(0)
                new_quats = quaternion_multiply(delta_quat, base_quats)
                render_image, render_depth, render_normal = render_2dgs(
                    new_means, new_quats, self.gs_params.scales,
                    self.gs_params.colors, self.gs_params.opacity,
                    viewmat=extrin_next,
                    K=K_next,
                    width=frame_next.shape[1], height=frame_next.shape[0],
                    scaling_modifier=1.0,
                    bg=bg,
                    near_plane=near_plane,
                    far_plane=far_plane
                )
                
                # TODO: alpha blend static bg and rendered dynamic fg
                # with torch.no_grad():
                mask_bg = (render_image == bg[:, None, None]).all(dim=0)
                render_image[:, mask_bg] = static_bg_img[:, mask_bg]
                # render_image[:, mask_bg] = 0.7 * static_bg_img[:, mask_bg] + 0.3 * render_image[:, mask_bg]
                # render_image[:, ~mask_bg] = 0.7 * render_image[:, ~mask_bg] + 0.3 * static_bg_img[:, ~mask_bg]
                # render_image = 0.7 * render_image + 0.3 * static_bg_img

                if rr_vis:
                    if step % 10 == 0:
                        gs_params_tmp.means = new_means

                        with torch.no_grad():
                            # rr.log("debug_render_image",
                            #     rr.Image(
                            #         render_image.permute(
                            #             1, 2, 0).cpu().numpy().clip(0, 1)
                            #     ))
                            rr.log("debug_error_image",
                                rr.DepthImage(
                                    torch.square(render_image-image_gt).permute(
                                        1, 2, 0).sum(dim=-1).cpu().numpy()
                                ))
                            vis_2dgs_rerun(gs_params_tmp, name="debug_gs",
                                           points_only=True)
                            # rr.log("debug_gt_image", rr.Image(
                            #     image_gt.permute(1, 2, 0).cpu().numpy()))
                            # rr.log("debug_bg_image", rr.Image(
                            #     static_bg_img.permute(1, 2, 0).cpu().numpy().clip(0, 1)))
                            if prev_loss is not None:
                                rr.log("train_loss", rr.Scalars(prev_loss))
                # image_tensor[:, ~obj_mask_next] = bg[:, None].cuda()

                loss = image_loss(render_image * valid_loss_mask, image_gt * valid_loss_mask)
                current_loss = loss.item()
                
                # Check for NaN in loss
                if torch.isnan(loss):
                    rr.log("log", rr.TextLog(f"Warning: NaN detected in loss at step {step}. Skipping this step.", 
                                             rr.TextLogLevel.ERROR))
                    optimizer.zero_grad()
                    break
                
                print(f"Step {step} - Loss: {current_loss:.6f}")
                losses.append(current_loss)
                # stop if loss exceeds minimal loss so far by tolerance
                if best_loss < float('inf') and current_loss > best_loss + loss_increase_tol and step > 50:
                    rr.log("log", rr.TextLog(
                        f"Early stop at step {step}: loss {current_loss:.6f} exceeded best {best_loss:.6f} by > tol {loss_increase_tol:.6f}", 
                        level=rr.TextLogLevel.WARN))
                    # Restore best parameters before breaking
                    if best_trans is not None and best_log_R is not None:
                        trans.data.copy_(best_trans)
                        log_R.data.copy_(best_log_R)
                    break

                # Early stopping check by convergence window
                if prev_loss is not None:
                    loss_change = abs(prev_loss - current_loss)
                    if loss_change < convergence_threshold:
                        patience_counter += 1
                        if patience_counter >= patience:
                            rr.log("log",
                                rr.TextLog(
                                    f"Early stopping at step {step}: Loss converged (change < {convergence_threshold})"))
                            converged = True
                            prev_loss = current_loss
                            break
                    else:
                        patience_counter = 0

                # track best loss so far
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_trans = trans.detach().clone()
                    best_log_R = log_R.detach().clone()

                prev_loss = current_loss
                loss.backward()
                
                # Clip gradients to prevent NaN propagation
                # torch.nn.utils.clip_grad_norm_(trans, max_norm=1.0)
                # torch.nn.utils.clip_grad_norm_(log_R, max_norm=1.0)
                
                # Check for NaN in gradients
                if trans.grad is not None and torch.isnan(trans.grad).any():
                    print(f"Warning: NaN detected in trans gradient at step {step}. Skipping this step.")
                    optimizer.zero_grad()
                    break
                if log_R.grad is not None and torch.isnan(log_R.grad).any():
                    print(f"Warning: NaN detected in log_R gradient at step {step}. Skipping this step.")
                    optimizer.zero_grad()
                    break
                
                optimizer.step()
                
        # TODO: return estimated pose?
        # rot_mat = rotation_6d_to_matrix(rot6d)
        rot_mat = so3_exp_map(best_log_R)[0]
        if torch.isnan(rot_mat).any():
            raise ValueError("NaN encountered in rotation matrix during optimization.")
        delta_pose = torch.eye(4)
        delta_pose[:3, :3] = rot_mat
        delta_pose[:3, 3] = best_trans
        rr.log("log", rr.TextLog(f"optimization finished at step {step}, loss: {loss.item():.6f}, converged: {converged}"))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(losses, label='Training Loss')
        ax.set_xlabel('Step')
        ax.set_ylabel('Loss')
        ax.set_title('Optimization Loss Over Steps')
        ax.grid(True)

        # Convert plot to image for rerun logging
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        loss_plot_image = Image.open(buf)
        loss_plot_array = np.array(loss_plot_image)
        rr.log("loss_plot", rr.Image(loss_plot_array))
        plt.close(fig)
        
        if step >= num_steps - 1 and not converged:
            converged = True        
        return delta_pose.detach(), converged, loss.item()

    def track_to_frame_with_static_bg_global(self, frame_next, extrin_next, K_next, static_bg_img, mask=None, hand_mask=None,
                                        lr=1e-2, num_steps=100, rr_vis=False,
                                        convergence_threshold=5e-5, patience=3,
                                        near_plane=5e-4, far_plane=10.0,
                                        loss_increase_tol=0.0):
            """Optimize 6 DoF pose of object to align with next frame observation

            Args:
                frame_next (torch.Tensor): Next frame observation
                extrin_next (torch.Tensor): Next frame extrinsics
                K_next (torch.Tensor): Next frame intrinsics
                static_bg_img (torch.Tensor): Static background image
                mask (torch.Tensor, optional): Object mask
                hand_mask (torch.Tensor, optional): Hand mask to exclude from loss
                lr (float): Learning rate
                num_steps (int): Maximum optimization steps
                rr_vis (bool): Enable rerun visualization
                convergence_threshold (float): Loss change threshold for early stopping
                patience (int): Number of steps to wait before stopping if loss doesn't improve
                loss_increase_tol (float): Early stop if loss rises above previous by this amount
            """
            bg = torch.zeros(3, device=frame_next.device)
            trans = torch.zeros(3,
                                device=frame_next.device)
            trans.requires_grad = True
            # which rotation representation for optimization?
            # rot6d = torch.zeros(6,
            #                     device=frame_next.device)
            # rot6d[[0, -2]] = 1.0  # initialize to identity
            # rot6d.requires_grad = True
            log_R = torch.zeros((1, 3), device=frame_next.device)
            log_R.requires_grad = True

            # optimizer = torch.optim.AdamW([
            #     {"params": trans, "lr": lr},
            #     {"params": log_R, "lr": lr}
            # ])
            # optimizer = torch.optim.AdamW([
            #     {"params": trans, "lr": 5e-3},
            #     {"params": log_R, "lr": 1e-3}
            # ])

            optimizer = torch.optim.AdamW([
                {"params": trans, "lr": 1e-2},
                {"params": log_R, "lr": 3e-3}
            ])
            
            image_gt = frame_next.permute(2, 0, 1) / 255.0
            valid_loss_mask = torch.ones_like(image_gt[0:1, :, :], dtype=torch.bool)
            if mask is not None:
                # image_gt[:, ~mask] = bg[:, None].to(frame_next.device)
                valid_loss_mask = valid_loss_mask & mask.unsqueeze(0)
            if hand_mask is not None:
                # image_gt[:, hand_mask_tensor] = bg[:, None].to(frame_next.device)
                valid_loss_mask = valid_loss_mask & ~hand_mask.unsqueeze(0)
            static_bg_img = static_bg_img.to(frame_next.device)
            with torch.no_grad():
                # center = self.gs_params.means.mean(0, keepdim=True)
                # xyz_delta = self.gs_params.means - center
                base_quats = self.gs_params.quats.detach()
            gs_params_tmp = self.gs_params.clone()

            # Early stopping variables
            prev_loss = None
            best_loss = float('inf')

            patience_counter = 0
            converged = False
            losses = []
            with torch.autograd.set_detect_anomaly(False):
                for step in range(num_steps):
                    if rr_vis:
                        rr.set_time("opt_step", sequence=step)
                    optimizer.zero_grad()
                    # rot_mat = rotation_6d_to_matrix(rot6d)
                    rot_mat = so3_exp_map(log_R)[0]
                    if torch.isnan(rot_mat).any():
                        raise ValueError("NaN encountered in rotation matrix during optimization.")
                    # new_means = center + \
                    #     torch.einsum("ij,nj->ni", rot_mat, xyz_delta) + trans
                    new_means = torch.einsum("ij,nj->ni", rot_mat, self.gs_params.means) + trans

                    delta_quat = matrix_to_quaternion(rot_mat).unsqueeze(0)
                    new_quats = quaternion_multiply(delta_quat, base_quats)
                    render_image, render_depth, render_normal = render_2dgs(
                        new_means, new_quats, self.gs_params.scales,
                        self.gs_params.colors, self.gs_params.opacity,
                        viewmat=extrin_next,
                        K=K_next,
                        width=frame_next.shape[1], height=frame_next.shape[0],
                        scaling_modifier=1.0,
                        bg=bg,
                        near_plane=near_plane,
                        far_plane=far_plane
                    )

                    # TODO: alpha blend static bg and rendered dynamic fg
                    # with torch.no_grad():
                    mask_bg = (render_image == bg[:, None, None]).all(dim=0)
                    render_image[:, mask_bg] = static_bg_img[:, mask_bg]
                    # render_image[:, mask_bg] = 0.7 * static_bg_img[:, mask_bg] + 0.3 * render_image[:, mask_bg]
                    # render_image[:, ~mask_bg] = 0.7 * render_image[:, ~mask_bg] + 0.3 * static_bg_img[:, ~mask_bg]
                    # render_image = 0.7 * render_image + 0.3 * static_bg_img

                    if rr_vis:
                        if step % 10 == 0:
                            gs_params_tmp.means = new_means

                            with torch.no_grad():
                                # rr.log("debug_render_image",
                                #     rr.Image(
                                #         render_image.permute(
                                #             1, 2, 0).cpu().numpy().clip(0, 1)
                                #     ))
                                rr.log("debug_error_image",
                                    rr.DepthImage(
                                        torch.square(render_image-image_gt).permute(
                                            1, 2, 0).sum(dim=-1).cpu().numpy()
                                    ))
                                vis_2dgs_rerun(gs_params_tmp, name="debug_gs",
                                            points_only=True)
                                # rr.log("debug_gt_image", rr.Image(
                                #     image_gt.permute(1, 2, 0).cpu().numpy()))
                                # rr.log("debug_bg_image", rr.Image(
                                #     static_bg_img.permute(1, 2, 0).cpu().numpy().clip(0, 1)))
                                if prev_loss is not None:
                                    rr.log("train_loss", rr.Scalars(prev_loss))
                    # image_tensor[:, ~obj_mask_next] = bg[:, None].cuda()
                    loss = image_loss(render_image * valid_loss_mask, image_gt * valid_loss_mask)
                    current_loss = loss.item()
                    
                    # Check for NaN in loss
                    if torch.isnan(loss):
                        rr.log("log", rr.TextLog(f"Warning: NaN detected in loss at step {step}. Skipping this step.", 
                                                rr.TextLogLevel.ERROR))
                        optimizer.zero_grad()
                        break
                    
                    print(f"Step {step} - Loss: {current_loss:.6f}")
                    losses.append(current_loss)
                    # stop if loss exceeds minimal loss so far by tolerance
                    if best_loss < float('inf') and current_loss > best_loss + loss_increase_tol and step > 50:
                        rr.log("log", rr.TextLog(
                            f"Early stop at step {step}: loss {current_loss:.6f} exceeded best {best_loss:.6f} by > tol {loss_increase_tol:.6f}", 
                            level=rr.TextLogLevel.WARN))
                        # Restore best parameters before breaking
                        if best_trans is not None and best_log_R is not None:
                            trans.data.copy_(best_trans)
                            log_R.data.copy_(best_log_R)
                        break

                    # Early stopping check by convergence window
                    if prev_loss is not None:
                        loss_change = abs(prev_loss - current_loss)
                        if loss_change < convergence_threshold:
                            patience_counter += 1
                            if patience_counter >= patience:
                                rr.log("log",
                                    rr.TextLog(
                                        f"Early stopping at step {step}: Loss converged (change < {convergence_threshold})"))
                                converged = True
                                prev_loss = current_loss
                                break
                        else:
                            patience_counter = 0

                    # track best loss so far
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_trans = trans.detach().clone()
                        best_log_R = log_R.detach().clone()

                    prev_loss = current_loss

                    loss.backward()
                    
                    # Clip gradients to prevent NaN propagation
                    # torch.nn.utils.clip_grad_norm_(trans, max_norm=1.0)
                    # torch.nn.utils.clip_grad_norm_(log_R, max_norm=1.0)
                    
                    # Check for NaN in gradients
                    if trans.grad is not None and torch.isnan(trans.grad).any():
                        rr.log("log",
                            rr.Textog(
                            f"Warning: NaN detected in trans gradient at step {step}. Skipping this step."))
                        optimizer.zero_grad()
                        break
                    if log_R.grad is not None and torch.isnan(log_R.grad).any():
                        
                        
                        rr.log("log",  
                        rr.TextLog(
                        f"Warning: NaN detected in log_R gradient at step {step}. Skipping this step."))
                        optimizer.zero_grad()
                        break
                    
                    optimizer.step()

            # TODO: return estimated pose?
            # rot_mat = rotation_6d_to_matrix(rot6d)
            rot_mat = so3_exp_map(best_log_R)[0]
            if torch.isnan(rot_mat).any():
                raise ValueError("NaN encountered in rotation matrix during optimization.")
            delta_pose = torch.eye(4)
            delta_pose[:3, :3] = rot_mat
            delta_pose[:3, 3] = best_trans
            rr.log("log", rr.TextLog(f"optimization finished at step {step}, loss: {loss.item():.6f}, converged: {converged}"))

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(losses, label='Training Loss')
            ax.set_xlabel('Step')
            ax.set_ylabel('Loss')
            ax.set_title('Optimization Loss Over Steps')
            ax.grid(True)

            # Convert plot to image for rerun logging
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            loss_plot_image = Image.open(buf)
            loss_plot_array = np.array(loss_plot_image)
            rr.log("loss_plot", rr.Image(loss_plot_array))
            plt.close(fig)
            
            if step >= num_steps - 1 and not converged:
                converged = True        
            return delta_pose.detach(), converged, loss.item()

    def refine_geometry(self, frame, depth):
        """
        Assume gsp is registered to the current frame,
        we want to update its geometry
        This is "region growing" of gs, to better cover the object surface
        in the current frame
        """
        # we may be able to densify GS directly
        # but it is good to have depth prior
        # TODO:  we are not sure how well this optimization is
        pass

    def apply_pose(self, pose):
        pose = pose.to(self.gs_params.means.device)
        with torch.no_grad():
            center = self.gs_params.means.mean(0, keepdim=True)
            xyz_delta = self.gs_params.means - center
        # new_means = center + \
        #         torch.einsum("ij,nj->ni", rot_mat, xyz_delta) + trans
        self.gs_params.means = center + torch.einsum(
            "ij,nj->ni", pose[:3, :3], xyz_delta) + pose[:3, 3]
        self.gs_params.quats = quaternion_multiply(
            matrix_to_quaternion(pose[:3, :3]), self.gs_params.quats)
        

        return self


class GaussianSuperPrimitive3D:
    def __init__(self, gs_means, gs_quats, gs_scales, f_dc, gs_opacity):
        # we build up the initial GS from segmented RGB and depth
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gs_params = GSParam(
            gs_means.detach().to(device),
            gs_quats.detach().to(device),
            gs_scales.detach().to(device),
            f_dc.detach().to(device),
            gs_opacity.detach().to(device)
        )

    def track_to_frame(self, frame_next, extrin_next, K_next, mask=None,
                       bg=torch.zeros(3), lr=1e-2, num_steps=100, rr_vis=False,
                       rel_tol=1e-4,
                       ):
        """Optimize 6 DoF pose of object to align with next frame observation

        Args:
            frame_next (torch.Tensor): Next frame observation
            extrin_next (torch.Tensor): Next frame extrinsics
            K_next (torch.Tensor): Next frame intrinsics
        """
        trans = torch.zeros(3,
                            device=frame_next.device)
        trans.requires_grad = True
        # which rotation representation for optimization?
        rot6d = torch.zeros(6,
                            device=frame_next.device)
        rot6d[[0, -2]] = 1.0  # initialize to identity
        rot6d.requires_grad = True
        optimizer = torch.optim.AdamW([trans, rot6d], lr=lr)
        image_gt = frame_next.permute(2, 0, 1) / 255.0
        if mask is not None:
            image_gt[:, ~mask] = bg[:, None].to(frame_next.device)
        with torch.no_grad():
            center = self.gs_params.means.mean(0, keepdim=True)
            xyz_delta = self.gs_params.means - center
            base_quats = self.gs_params.quats.detach()
        gs_params_tmp = self.gs_params.clone()
        for step in range(num_steps):
            if rr_vis:
                rr.set_time("opt_step", sequence=step)
            optimizer.zero_grad()
            rot_mat = rotation_6d_to_matrix(rot6d)

            new_means = center + \
                torch.einsum("ij,nj->ni", rot_mat, xyz_delta) + trans
            # self.gs_params.means = new_means
            # render_image, render_depth, render_normal = self.gs_params.render(
            #     viewmat=extrin_next,
            #     K=K_next,
            #     width=frame_next.shape[1], height=frame_next.shape[0],
            #     scaling_modifier=1.0,
            #     bg=bg
            # )
            # detach base_quats to avoid reusing the autograd graph across steps
            # delta_quat = matrix_to_quaternion(rot_mat).unsqueeze(0)
            # new_quats = quaternion_multiply(delta_quat, base_quats)
            # render_image = render_3dgs(
            #     new_means, new_quats, self.gs_params.scales,
            #     self.gs_params.colors, self.gs_params.opacity,
            #     viewmat=extrin_next,
            #     K=K_next,
            #     width=frame_next.shape[1], height=frame_next.shape[0],
            #     scaling_modifier=1.0,
            #     bg=bg
            # )

            render_image, render_depth, render_normal = render_2dgs(
                new_means, self.gs_params.quats, self.gs_params.scales,
                self.gs_params.colors, self.gs_params.opacity,
                viewmat=extrin_next,
                K=K_next,
                width=frame_next.shape[1], height=frame_next.shape[0],
                scaling_modifier=1.0,
                bg=bg
            )

            if rr_vis:
                if step % 10 == 0:
                    gs_params_tmp.means = new_means

                    with torch.no_grad():
                        rr.log("debug_render_image",
                               rr.Image(
                                   render_image.permute(
                                       1, 2, 0).cpu().numpy()
                               ))
                        rr.log("debug_error_image",
                               rr.DepthImage(
                                   torch.square(render_image-image_gt).permute(
                                       1, 2, 0).sum(dim=-1).cpu().numpy()
                               ))
                        vis_2dgs_rerun(gs_params_tmp, name="debug_gs",
                                       points_only=True)
            # image_tensor[:, ~obj_mask_next] = bg[:, None].cuda()
            loss = image_loss(render_image, image_gt)
            # print(f"Step {step} - Loss: {loss.item():.6f}")
            loss.backward()
            optimizer.step()

        # TODO: return estimated pose?
        rot_mat = rotation_6d_to_matrix(rot6d)
        delta_pose = torch.eye(4)
        delta_pose[:3, :3] = rot_mat
        delta_pose[:3, 3] = trans
        return delta_pose

    @classmethod
    def from_depth(cls, image, mask, depth, extrin, K):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        depth = torch.from_numpy(depth.copy()).float()
        image = torch.from_numpy(image.copy()).float()
        mask = torch.from_numpy(mask.copy()).bool()
        K = torch.from_numpy(K.copy()).float()
        extrin = torch.from_numpy(extrin.copy()).float()
        xyz = unproject_depth(depth, K, depth.shape[0], depth.shape[1])

        # Filter mask by valid depth
        valid_depth_mask = depth > 0
        mask = mask & valid_depth_mask
        
        means = xyz[mask].reshape(-1, 3)
        colors = image[mask].reshape(-1, 3) / 255.0
        f = (0.5 * (K[0, 0] + K[1, 1]))
        sizes = (depth[mask] / f).clamp(min=1e-6) # Clamp small scales

        normal_from_depth, valid_mask = d2n_tblr(
            xyz.permute((2, 0, 1)).unsqueeze(0))
        normal = normal_from_depth[..., mask].view(-1, 3)
        normal = normal / torch.norm(normal, dim=1, keepdim=True)
        ref_axis1 = torch.randn(normal.shape[0], 3)
        ref_axis1 = ref_axis1 / torch.norm(ref_axis1, dim=1, keepdim=True)
        rotation_axis1 = ref_axis1 - \
            (normal * ref_axis1).sum(dim=1, keepdim=True) * normal
        rotation_axis1 = rotation_axis1 / \
            torch.norm(rotation_axis1, dim=1, keepdim=True)
        rotation_axis2 = torch.cross(normal, rotation_axis1, dim=1)
        rotation_axis2 = rotation_axis2 / \
            torch.norm(rotation_axis2, dim=1, keepdim=True)
        gs_rotations = torch.stack(
            [rotation_axis1, rotation_axis2, normal], dim=1).transpose(1, 2)
        gs_rotations[torch.det(gs_rotations) < 0] *= -1

        pose = extrin.inverse()
        gs_means = torch.einsum('ij,nj->ni', pose[:3, :3], means) + pose[:3, 3]
        gs_rotations = torch.einsum(
            'ij,njk->nik', pose[:3, :3], gs_rotations)

        gs_quats = matrix_to_quaternion(gs_rotations)

        gs_scales = torch.ones(
            gs_means.shape[0], 3).cuda() * sizes.view(-1, 1).cuda()
        gs_opacity = torch.ones(gs_means.shape[0], 1).cuda() * 0.5

        SH = RGB2SH(colors)
        return cls(
            gs_means.to(device),
            gs_quats.to(device),
            gs_scales.to(device),
            SH.to(device),
            gs_opacity.to(device)
        )

    def track_to_frame_with_static_bg(self, frame_next, extrin_next, K_next, static_bg_img, mask=None, hand_mask=None,
                                      lr=1e-2, num_steps=100, rr_vis=False,
                                      convergence_threshold=5e-5, patience=3,
                                      near_plane=5e-4, far_plane=10.0,
                                      loss_increase_tol=0.0):
        """Optimize 6 DoF pose of object to align with next frame observation

        Args:
            frame_next (torch.Tensor): Next frame observation
            extrin_next (torch.Tensor): Next frame extrinsics
            K_next (torch.Tensor): Next frame intrinsics
            static_bg_img (torch.Tensor): Static background image
            mask (torch.Tensor, optional): Object mask
            hand_mask (torch.Tensor, optional): Hand mask to exclude from loss
            lr (float): Learning rate
            num_steps (int): Maximum optimization steps
            rr_vis (bool): Enable rerun visualization
            convergence_threshold (float): Loss change threshold for early stopping
            patience (int): Number of steps to wait before stopping if loss doesn't improve
            loss_increase_tol (float): Early stop if loss rises above previous by this amount
        """
        bg = torch.zeros(3, device=frame_next.device)
        trans = torch.zeros(3,
                            device=frame_next.device)
        trans.requires_grad = True
        # which rotation representation for optimization?
        # rot6d = torch.zeros(6,
        #                     device=frame_next.device)
        # rot6d[[0, -2]] = 1.0  # initialize to identity
        # rot6d.requires_grad = True
        log_R = torch.zeros((1, 3), device=frame_next.device)
        log_R.requires_grad = True

        # optimizer = torch.optim.AdamW([
        #     {"params": trans, "lr": lr},
        #     {"params": log_R, "lr": lr}
        # ])
        optimizer = torch.optim.AdamW([
                {"params": trans, "lr": 1e-2},
                {"params": log_R, "lr": 3e-3}
            ])
       
        image_gt = frame_next.permute(2, 0, 1) / 255.0
        valid_loss_mask = torch.ones_like(
            image_gt[0:1, :, :], dtype=torch.bool)
        if mask is not None:
            # image_gt[:, ~mask] = bg[:, None].to(frame_next.device)
            valid_loss_mask = valid_loss_mask & mask.unsqueeze(0)
        if hand_mask is not None:
            # image_gt[:, hand_mask_tensor] = bg[:, None].to(frame_next.device)
            valid_loss_mask = valid_loss_mask & ~hand_mask.unsqueeze(0)
        static_bg_img = static_bg_img.to(frame_next.device)
        with torch.no_grad():
            center = self.gs_params.means.mean(0, keepdim=True)
            xyz_delta = self.gs_params.means - center
            base_quats = self.gs_params.quats.detach()
        gs_params_tmp = self.gs_params.clone()

        # Early stopping variables
        prev_loss = None
        best_loss = float('inf')

        patience_counter = 0
        converged = False
        losses = []
        with torch.autograd.set_detect_anomaly(False):
            for step in range(num_steps):
                if rr_vis:
                    rr.set_time("opt_step", sequence=step)
                optimizer.zero_grad()
                # rot_mat = rotation_6d_to_matrix(rot6d)
                rot_mat = so3_exp_map(log_R)[0]
                if torch.isnan(rot_mat).any():
                    raise ValueError(
                        "NaN encountered in rotation matrix during optimization.")
                new_means = center + \
                    torch.einsum("ij,nj->ni", rot_mat, xyz_delta) + trans

                delta_quat = matrix_to_quaternion(rot_mat).unsqueeze(0)
                new_quats = quaternion_multiply(delta_quat, base_quats)
                render_image, _, _ = render_3dgs(
                    new_means, new_quats, self.gs_params.scales,
                    self.gs_params.colors, self.gs_params.opacity,
                    viewmat=extrin_next,
                    K=K_next,
                    width=frame_next.shape[1], height=frame_next.shape[0],
                    scaling_modifier=1.0,
                    bg=bg,
                    near_plane=near_plane,
                    far_plane=far_plane
                )

                mask_bg = (render_image == bg[:, None, None]).all(dim=0)
                render_image[:, mask_bg] = static_bg_img[:, mask_bg]

                if rr_vis:
                    if step % 10 == 0:
                        gs_params_tmp.means = new_means

                        with torch.no_grad():
                            # rr.log("debug_render_image",
                            #     rr.Image(
                            #         render_image.permute(
                            #             1, 2, 0).cpu().numpy().clip(0, 1)
                            #     ))
                            rr.log("debug_error_image",
                                   rr.DepthImage(
                                       torch.square(render_image-image_gt).permute(
                                           1, 2, 0).sum(dim=-1).cpu().numpy()
                                   ))
                            vis_2dgs_rerun(gs_params_tmp, name="debug_gs",
                                           points_only=True)
                            # rr.log("debug_gt_image", rr.Image(
                            #     image_gt.permute(1, 2, 0).cpu().numpy()))
                            # rr.log("debug_bg_image", rr.Image(
                            #     static_bg_img.permute(1, 2, 0).cpu().numpy().clip(0, 1)))
                            if prev_loss is not None:
                                rr.log("train_loss", rr.Scalars(prev_loss))
                # image_tensor[:, ~obj_mask_next] = bg[:, None].cuda()
                loss = image_loss(render_image * valid_loss_mask,
                                  image_gt * valid_loss_mask)
                current_loss = loss.item()

                # Check for NaN in loss
                if torch.isnan(loss):
                    rr.log("log", rr.TextLog(f"Warning: NaN detected in loss at step {step}. Skipping this step.",
                                             rr.TextLogLevel.ERROR))
                    optimizer.zero_grad()
                    break

                print(f"Step {step} - Loss: {current_loss:.6f}")
                losses.append(current_loss)
                # stop if loss exceeds minimal loss so far by tolerance
                if best_loss < float('inf') and current_loss > best_loss + loss_increase_tol and step > 50:
                    rr.log("log", rr.TextLog(
                        f"Early stop at step {step}: loss {current_loss:.6f} exceeded best {best_loss:.6f} by > tol {loss_increase_tol:.6f}",
                        level=rr.TextLogLevel.WARN))
                    # Restore best parameters before breaking
                    if best_trans is not None and best_log_R is not None:
                        trans.data.copy_(best_trans)
                        log_R.data.copy_(best_log_R)
                    break

                # Early stopping check by convergence window
                if prev_loss is not None:
                    loss_change = abs(prev_loss - current_loss)
                    if loss_change < convergence_threshold:
                        patience_counter += 1
                        if patience_counter >= patience:
                            rr.log("log",
                                   rr.TextLog(
                                       f"Early stopping at step {step}: Loss converged (change < {convergence_threshold})"))
                            converged = True
                            prev_loss = current_loss
                            break
                    else:
                        patience_counter = 0

                # track best loss so far
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_trans = trans.detach().clone()
                    best_log_R = log_R.detach().clone()

                prev_loss = current_loss

                loss.backward()

                # Clip gradients to prevent NaN propagation
                # torch.nn.utils.clip_grad_norm_(trans, max_norm=1.0)
                # torch.nn.utils.clip_grad_norm_(log_R, max_norm=1.0)

                # Check for NaN in gradients
                if trans.grad is not None and torch.isnan(trans.grad).any():
                    print(
                        f"Warning: NaN detected in trans gradient at step {step}. Skipping this step.")
                    optimizer.zero_grad()
                    break
                if log_R.grad is not None and torch.isnan(log_R.grad).any():
                    print(
                        f"Warning: NaN detected in log_R gradient at step {step}. Skipping this step.")
                    optimizer.zero_grad()
                    break

                optimizer.step()

        # TODO: return estimated pose?
        # rot_mat = rotation_6d_to_matrix(rot6d)
        rot_mat = so3_exp_map(best_log_R)[0]
        if torch.isnan(rot_mat).any():
            raise ValueError(
                "NaN encountered in rotation matrix during optimization.")
        delta_pose = torch.eye(4)
        delta_pose[:3, :3] = rot_mat
        delta_pose[:3, 3] = best_trans
        # rr.log("log", rr.TextLog(f"optimization finished at step {step}, loss: {loss.item():.6f}, converged: {converged}"))

        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.plot(losses, label='Training Loss')
        # ax.set_xlabel('Step')
        # ax.set_ylabel('Loss')
        # ax.set_title('Optimization Loss Over Steps')
        # ax.grid(True)

        # # Convert plot to image for rerun logging
        # buf = io.BytesIO()
        # fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        # buf.seek(0)
        # loss_plot_image = Image.open(buf)
        # loss_plot_array = np.array(loss_plot_image)
        # rr.log("loss_plot", rr.Image(loss_plot_array))
        # plt.close(fig)

        if step >= num_steps - 1 and not converged:
            converged = True
        return delta_pose.detach(), converged, loss.item()

    def track_to_frame_with_static_bg_global(self, frame_next, extrin_next, K_next, static_bg_img, mask=None, hand_mask=None,
                                      lr=1e-2, num_steps=100, rr_vis=False,
                                      convergence_threshold=5e-5, patience=3,
                                      near_plane=5e-4, far_plane=10.0,
                                      loss_increase_tol=0.0):
        """Optimize 6 DoF pose of object to align with next frame observation

        Args:
            frame_next (torch.Tensor): Next frame observation
            extrin_next (torch.Tensor): Next frame extrinsics
            K_next (torch.Tensor): Next frame intrinsics
            static_bg_img (torch.Tensor): Static background image
            mask (torch.Tensor, optional): Object mask
            hand_mask (torch.Tensor, optional): Hand mask to exclude from loss
            lr (float): Learning rate
            num_steps (int): Maximum optimization steps
            rr_vis (bool): Enable rerun visualization
            convergence_threshold (float): Loss change threshold for early stopping
            patience (int): Number of steps to wait before stopping if loss doesn't improve
            loss_increase_tol (float): Early stop if loss rises above previous by this amount
        """
        bg = torch.zeros(3, device=frame_next.device)
        trans = torch.zeros(3,
                            device=frame_next.device)
        trans.requires_grad = True
        # which rotation representation for optimization?
        # rot6d = torch.zeros(6,
        #                     device=frame_next.device)
        # rot6d[[0, -2]] = 1.0  # initialize to identity
        # rot6d.requires_grad = True
        log_R = torch.zeros((1, 3), device=frame_next.device)
        log_R.requires_grad = True

        # optimizer = torch.optim.AdamW([
        #     {"params": trans, "lr": lr},
        #     {"params": log_R, "lr": lr}
        # ])
        optimizer = torch.optim.AdamW([
            {"params": trans, "lr": 1e-2},
            {"params": log_R, "lr": 3e-3}
        ])
        
        image_gt = frame_next.permute(2, 0, 1) / 255.0
        valid_loss_mask = torch.ones_like(
            image_gt[0:1, :, :], dtype=torch.bool)
        if mask is not None:
            # image_gt[:, ~mask] = bg[:, None].to(frame_next.device)
            valid_loss_mask = valid_loss_mask & mask.unsqueeze(0)
        if hand_mask is not None:
            # image_gt[:, hand_mask_tensor] = bg[:, None].to(frame_next.device)
            valid_loss_mask = valid_loss_mask & ~hand_mask.unsqueeze(0)
        static_bg_img = static_bg_img.to(frame_next.device)
        with torch.no_grad():
            # center = self.gs_params.means.mean(0, keepdim=True)
            # xyz_delta = self.gs_params.means - center
            base_quats = self.gs_params.quats.detach()
        gs_params_tmp = self.gs_params.clone()

        # Early stopping variables
        prev_loss = None
        best_loss = float('inf')

        patience_counter = 0
        converged = False
        losses = []
        with torch.autograd.set_detect_anomaly(False):
            for step in range(num_steps):
                if rr_vis:
                    rr.set_time("opt_step", sequence=step)
                optimizer.zero_grad()
                # rot_mat = rotation_6d_to_matrix(rot6d)
                rot_mat = so3_exp_map(log_R)[0]
                if torch.isnan(rot_mat).any():
                    raise ValueError(
                        "NaN encountered in rotation matrix during optimization.")
                new_means = torch.einsum("ij,nj->ni", rot_mat, self.gs_params.means) + trans

                delta_quat = matrix_to_quaternion(rot_mat).unsqueeze(0)
                new_quats = quaternion_multiply(delta_quat, base_quats)
                render_image, _, _ = render_3dgs(
                    new_means, new_quats, self.gs_params.scales,
                    self.gs_params.colors, self.gs_params.opacity,
                    viewmat=extrin_next,
                    K=K_next,
                    width=frame_next.shape[1], height=frame_next.shape[0],
                    scaling_modifier=1.0,
                    bg=bg,
                    near_plane=near_plane,
                    far_plane=far_plane
                )

                # TODO: alpha blend static bg and rendered dynamic fg
                # with torch.no_grad():
                mask_bg = (render_image == bg[:, None, None]).all(dim=0)
                render_image[:, mask_bg] = static_bg_img[:, mask_bg]
                # render_image[:, mask_bg] = 0.7 * static_bg_img[:, mask_bg] + 0.3 * render_image[:, mask_bg]
                # render_image[:, ~mask_bg] = 0.7 * render_image[:, ~mask_bg] + 0.3 * static_bg_img[:, ~mask_bg]
                # render_image = 0.7 * render_image + 0.3 * static_bg_img

                if rr_vis:
                    if step % 10 == 0:
                        gs_params_tmp.means = new_means

                        with torch.no_grad():
                            # rr.log("debug_render_image",
                            #     rr.Image(
                            #         render_image.permute(
                            #             1, 2, 0).cpu().numpy().clip(0, 1)
                            #     ))
                            rr.log("debug_error_image",
                                   rr.DepthImage(
                                       torch.square(render_image-image_gt).permute(
                                           1, 2, 0).sum(dim=-1).cpu().numpy()
                                   ))
                            vis_2dgs_rerun(gs_params_tmp, name="debug_gs",
                                           points_only=True)
                            # rr.log("debug_gt_image", rr.Image(
                            #     image_gt.permute(1, 2, 0).cpu().numpy()))
                            # rr.log("debug_bg_image", rr.Image(
                            #     static_bg_img.permute(1, 2, 0).cpu().numpy().clip(0, 1)))
                            if prev_loss is not None:
                                rr.log("train_loss", rr.Scalars(prev_loss))
                # image_tensor[:, ~obj_mask_next] = bg[:, None].cuda()
                loss = image_loss(render_image * valid_loss_mask,
                                  image_gt * valid_loss_mask)
                current_loss = loss.item()

                # Check for NaN in loss
                if torch.isnan(loss):
                    rr.log("log", rr.TextLog(f"Warning: NaN detected in loss at step {step}. Skipping this step.",
                                             rr.TextLogLevel.ERROR))
                    optimizer.zero_grad()
                    break

                print(f"Step {step} - Loss: {current_loss:.6f}")
                losses.append(current_loss)
                # stop if loss exceeds minimal loss so far by tolerance
                if best_loss < float('inf') and current_loss > best_loss + loss_increase_tol and step > 50:
                    rr.log("log", rr.TextLog(
                        f"Early stop at step {step}: loss {current_loss:.6f} exceeded best {best_loss:.6f} by > tol {loss_increase_tol:.6f}",
                        level=rr.TextLogLevel.WARN))
                    # Restore best parameters before breaking
                    if best_trans is not None and best_log_R is not None:
                        trans.data.copy_(best_trans)
                        log_R.data.copy_(best_log_R)
                    break

                # Early stopping check by convergence window
                if prev_loss is not None:
                    loss_change = abs(prev_loss - current_loss)
                    if loss_change < convergence_threshold:
                        patience_counter += 1
                        if patience_counter >= patience:
                            rr.log("log",
                                   rr.TextLog(
                                       f"Early stopping at step {step}: Loss converged (change < {convergence_threshold})"))
                            converged = True
                            prev_loss = current_loss
                            break
                    else:
                        patience_counter = 0

                # track best loss so far
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_trans = trans.detach().clone()
                    best_log_R = log_R.detach().clone()

                prev_loss = current_loss

                loss.backward()

                # Clip gradients to prevent NaN propagation
                # torch.nn.utils.clip_grad_norm_(trans, max_norm=1.0)
                # torch.nn.utils.clip_grad_norm_(log_R, max_norm=1.0)

                # Check for NaN in gradients
                if trans.grad is not None and torch.isnan(trans.grad).any():
                    print(
                        f"Warning: NaN detected in trans gradient at step {step}. Skipping this step.")
                    optimizer.zero_grad()
                    break
                if log_R.grad is not None and torch.isnan(log_R.grad).any():
                    print(
                        f"Warning: NaN detected in log_R gradient at step {step}. Skipping this step.")
                    optimizer.zero_grad()
                    break

                optimizer.step()

        # TODO: return estimated pose?
        # rot_mat = rotation_6d_to_matrix(rot6d)
        rot_mat = so3_exp_map(best_log_R)[0]
        if torch.isnan(rot_mat).any():
            raise ValueError(
                "NaN encountered in rotation matrix during optimization.")
        delta_pose = torch.eye(4)
        delta_pose[:3, :3] = rot_mat
        delta_pose[:3, 3] = best_trans
        # rr.log("log", rr.TextLog(f"optimization finished at step {step}, loss: {loss.item():.6f}, converged: {converged}"))

        # fig, ax = plt.subplots(figsize=(10, 6))
        # ax.plot(losses, label='Training Loss')
        # ax.set_xlabel('Step')
        # ax.set_ylabel('Loss')
        # ax.set_title('Optimization Loss Over Steps')
        # ax.grid(True)

        # # Convert plot to image for rerun logging
        # buf = io.BytesIO()
        # fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        # buf.seek(0)
        # loss_plot_image = Image.open(buf)
        # loss_plot_array = np.array(loss_plot_image)
        # rr.log("loss_plot", rr.Image(loss_plot_array))
        # plt.close(fig)

        if step >= num_steps - 1 and not converged:
            converged = True
        return delta_pose.detach(), converged, loss.item()



    def refine_geometry(self, frame, depth):
        """
        Assume gsp is registered to the current frame,
        we want to update its geometry
        This is "region growing" of gs, to better cover the object surface
        in the current frame
        """
        # we may be able to densify GS directly
        # but it is good to have depth prior
        # TODO:  we are not sure how well this optimization is
        pass

    def apply_pose(self, pose):
        pose = pose.to(self.gs_params.means.device)
        with torch.no_grad():
            center = self.gs_params.means.mean(0, keepdim=True)
            xyz_delta = self.gs_params.means - center
            self.gs_params.means = center + torch.einsum(
            "ij,nj->ni", pose[:3, :3], xyz_delta) + pose[:3, 3]
            # self.gs_params.quats = quaternion_multiply(
            #     matrix_to_quaternion(pose[:3, :3]), self.gs_params.quats)
        return self
