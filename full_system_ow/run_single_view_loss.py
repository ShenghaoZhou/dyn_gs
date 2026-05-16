import torch
import torch.nn.functional as F
import numpy as np

def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz

def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_x, valid_y = torch.meshgrid(valid_x, valid_y, indexing='xy')
    
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H)
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3)
    # This actually computes points in CAMERA space in the provided implementation
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    return xyz_cam

def depth_pcd2normal(xyz, offset=None):
    hd, wd, _ = xyz.shape 
    if offset is not None:
        ix, iy = torch.meshgrid(
            torch.arange(wd), torch.arange(hd), indexing='xy')
        xy = (torch.stack((ix, iy), dim=-1)[1:-1,1:-1]).to(xyz.device)
        p_offset = torch.tensor([[0,1],[0,-1],[1,0],[-1,0]]).float().to(xyz.device)
        new_offset = p_offset[None,None] + offset.reshape(hd, wd, 4, 2)[1:-1,1:-1]
        xys = xy[:,:,None] + new_offset
        xys[..., 0] = 2 * xys[..., 0] / (wd - 1) - 1.0
        xys[..., 1] = 2 * xys[..., 1] / (hd - 1) - 1.0
        sampled_xyzs = torch.nn.functional.grid_sample(xyz.permute(2,0,1)[None], xys.reshape(1, -1, 1, 2))
        sampled_xyzs = sampled_xyzs.permute(0,2,3,1).reshape(hd-2,wd-2,4,3)
        bottom_point = sampled_xyzs[:,:,0]
        top_point = sampled_xyzs[:,:,1]
        right_point = sampled_xyzs[:,:,2]
        left_point = sampled_xyzs[:,:,3]
    else:
        bottom_point = xyz[..., 2:hd,   1:wd-1, :]
        top_point    = xyz[..., 0:hd-2, 1:wd-1, :]
        right_point  = xyz[..., 1:hd-1, 2:wd,   :]
        left_point   = xyz[..., 1:hd-1, 0:wd-2, :]
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal

def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix=None, offset=None):
    # depth: (H, W), intrinsic_matrix: (3, 3)
    xyz_cam = depth2point_world(depth, intrinsic_matrix) # Returns points in camera space
    xyz_cam = xyz_cam.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_cam, offset)
    return xyz_normal

def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape 
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img

def compute_single_view_loss(
    rendered_normal, 
    plane_depth, 
    gt_rgb, 
    camera, 
    weight=0.015,
    wo_image_weight=False,
    scale=1,
    mask=None
):
    """
    Standalone function to compute single-view consistency loss.
    
    Args:
        rendered_normal: Normal map rendered from Gaussians [3, H, W]
        plane_depth: Depth map rendered from Gaussians [1, H, W]
        gt_rgb: Ground truth RGB image [3, H, W]
        camera: Camera object/dict with get_calib_matrix_nerf(scale) method
        weight: Loss weight
        wo_image_weight: If True, do not use image gradient weighting
        scale: Downsampling scale
        mask: Optional boolean mask [H, W] or [1, H, W] to compute loss only on object pixels
        
    Returns:
        normal_loss: The computed consistency loss
    """
    intrinsic_matrix, extrinsic_matrix = camera.get_calib_matrix_nerf(scale=scale)
    depth = plane_depth.squeeze()
    
    # Compute normal from rendered depth (depth_normal)
    # normal_from_depth_image returns (H, W, 3)
    depth_normal = normal_from_depth_image(
        depth.to(intrinsic_matrix.device), 
        intrinsic_matrix.to(depth.device), 
        extrinsic_matrix.to(depth.device)
    )
    depth_normal = depth_normal.permute(2, 0, 1) # [3, H, W]
    
    # Image gradient weighting
    image_weight = (1.0 - get_img_grad_weight(gt_rgb))
    image_weight = (image_weight).clamp(0, 1).detach() ** 2
    
    # Absolute difference per pixel
    diff = (depth_normal.squeeze() - rendered_normal.squeeze()).abs().sum(0) # [H, W]
    
    if mask is not None:
        mask = mask.squeeze()
        if not wo_image_weight:
            normal_loss = weight * (image_weight * diff * mask).sum() / (mask.sum() + 1e-6)
        else:
            normal_loss = weight * (diff * mask).sum() / (mask.sum() + 1e-6)
    else:
        if not wo_image_weight:
            normal_loss = weight * (image_weight * diff).mean()
        else:
            normal_loss = weight * diff.mean()
    
    return normal_loss.mean()

if __name__ == "__main__":
    print("Single-view loss extraction script ready.")
    pass
