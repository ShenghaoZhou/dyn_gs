import torch
import torch.nn.functional as F
import numpy as np


def compute_geo_consistency(viewpoint_cam, nearest_cam, gaussians, render_pkg, nearest_render_pkg, pixel_noise_th, wo_use_geo_occ_aware):
    H, W = render_pkg['plane_depth'].squeeze().shape
    ix, iy = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

    pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
    pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
    map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
    
    pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
    pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
    R = torch.tensor(nearest_cam.R).float().cuda()
    T = torch.tensor(nearest_cam.T).float().cuda()
    pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
    pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
    pts_projections = torch.stack(
                [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
    pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
    if not wo_use_geo_occ_aware:
        d_mask = d_mask & (pixel_noise < pixel_noise_th)
        weights = (1.0 / torch.exp(pixel_noise)).detach()
        weights[~d_mask] = 0
    else:
        weights = torch.ones_like(pixel_noise)
        weights[~d_mask] = 0
    return d_mask, weights, pixel_noise, pixels


def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets, indexing='xy')[::-1], dim=-1).view(1, -1, 2)

def patch_warp(H, uv):
    B, P = uv.shape[:2]
    H = H.view(B, 3, 3)
    ones = torch.ones((B, P, 1), device=uv.device)
    homo_uv = torch.cat((uv, ones), dim=-1)

    grid_tmp = torch.einsum("bik,bpk->bpi", H, homo_uv)
    grid_tmp = grid_tmp.reshape(B, P, 3)
    grid = grid_tmp[..., :2] / (grid_tmp[..., 2:] + 1e-10)
    return grid

def lncc(ref, nea):
    # ref: [batch_size, total_patch_size]
    # nea: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    # Use conv2d to sum over the patch area
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask

def compute_multi_view_loss(
    ref_image_gray, 
    nea_image_gray, 
    ref_normal, 
    ref_distance, 
    ref_camera, 
    nea_camera, 
    patch_size=3, 
    sample_num=102400, 
    valid_indices=None,
    pixels=None
):
    """
    Standalone function to compute multi-view photometric (LNCC) loss.
    
    Args:
        ref_image_gray: Reference image in grayscale [C, H, W]
        nea_image_gray: Neighbor image in grayscale [C, H, W]
        ref_normal: Rendered normal map of reference view [3, H, W] or flattened [N, 3]
        ref_distance: Rendered distance map of reference view [1, H, W] or flattened [N]
        ref_camera: A dict or object with K, invK, R, T, W, H, etc.
        nea_camera: A dict or object with K, invK, R, T, W, H, etc.
        patch_size: Half-patch size (total_patch_size = (2*patch_size+1)^2)
        sample_num: Number of points to sample if valid_indices is None
        valid_indices: Indices of valid points to use for the loss
        pixels: Reference pixel coordinates corresponding to valid_indices
        
    Note: In training, ref_camera and nea_camera are expected to have:
          - world_view_transform (4x4)
          - get_k(scale), get_inv_k(scale)
          - ncc_scale
    """
    device = ref_image_gray.device
    total_patch_size = (patch_size * 2 + 1) ** 2
    
    # Precompute relative transformation
    # ref_to_nea_r = R_nea^T @ R_ref
    # ref_to_nea_t = -R_nea^T @ R_ref @ T_ref + T_nea
    ref_wvt = ref_camera.world_view_transform
    nea_wvt = nea_camera.world_view_transform
    
    ref_to_nea_r = nea_wvt[:3,:3].transpose(-1,-2) @ ref_wvt[:3,:3]
    ref_to_nea_t = -ref_to_nea_r @ ref_wvt[3,:3] + nea_wvt[3,:3]
    
    # Process inputs if they are maps
    if len(ref_normal.shape) == 3:
        ref_normal = ref_normal.permute(1, 2, 0).reshape(-1, 3)
    
    # Flatten distance/depth map regardless of dimensionality
    ref_distance = ref_distance.reshape(-1)
        
    if valid_indices is not None:
        ref_local_n = ref_normal[valid_indices]
        ref_local_d = ref_distance[valid_indices]
    else:
        # If no indices provided, we need pixels to sample
        # This part depends on how 'pixels' are defined in the caller
        ref_local_n = ref_normal
        ref_local_d = ref_distance
        
    if pixels is None:
        raise ValueError("Pixel coordinates must be provided for patch warping.")

    # Compute metric points in reference camera space: X = Z * K_inv * u
    # pixels is [N, 2]. We need to make it homogeneous.
    ref_u_homo = torch.cat([pixels, torch.ones((pixels.shape[0], 1), device=device)], dim=-1)
    ref_K_inv = ref_camera.get_inv_k(ref_camera.ncc_scale).to(device)
    ref_X_cam = ref_local_d[:, None] * (ref_u_homo @ ref_K_inv.t()) # [N, 3]
    
    # Compute plane distance: d = n^T * X
    # ref_local_n is [N, 3]. 
    plane_d = (ref_local_n * ref_X_cam).sum(dim=-1).clamp_min(1e-6) # [N]

    # Compute Homography: H = K_nea * (R - t*n^T/d) * K_ref_inv
    H_ref_to_nea = ref_to_nea_r[None] - \
        torch.matmul(ref_to_nea_t[None, :, None].expand(ref_local_d.shape[0], 3, 1), 
                    ref_local_n[:, :, None].expand(ref_local_d.shape[0], 3, 1).permute(0, 2, 1)) / plane_d[..., None, None]
    
    K_nea = nea_camera.get_k(nea_camera.ncc_scale).to(device)
    invK_ref = ref_camera.get_inv_k(ref_camera.ncc_scale).to(device)
    
    H_ref_to_nea = torch.matmul(K_nea[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_nea)
    H_ref_to_nea = H_ref_to_nea @ invK_ref
    
    # Patch sampling for reference frame
    offsets = patch_offsets(patch_size, device)
    # ncc_scale usually 1.0
    ori_pixels_patch = pixels.reshape(-1, 1, 2) / ref_camera.ncc_scale + offsets.float()
    
    H_ref, W_ref = ref_image_gray.shape[-2:]
    pixels_patch = ori_pixels_patch.clone()
    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W_ref - 1) - 1.0
    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H_ref - 1) - 1.0
    
    ref_gray_val = F.grid_sample(ref_image_gray[None], pixels_patch.view(1, -1, 1, 2), align_corners=True)
    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)
    
    # Warping to neighbor frame
    grid = patch_warp(H_ref_to_nea.reshape(-1, 3, 3), ori_pixels_patch)
    
    H_nea, W_nea = nea_image_gray.shape[-2:]
    grid[:, :, 0] = 2 * grid[:, :, 0] / (W_nea - 1) - 1.0
    grid[:, :, 1] = 2 * grid[:, :, 1] / (H_nea - 1) - 1.0
    
    sampled_nea_gray_val = F.grid_sample(nea_image_gray[None], 
                                         grid.reshape(1, -1, 1, 2), align_corners=True)
    sampled_nea_gray_val = sampled_nea_gray_val.reshape(-1, total_patch_size)
    
    # Compute LNCC
    ncc, ncc_mask = lncc(ref_gray_val, sampled_nea_gray_val)
    
    return ncc, ncc_mask

if __name__ == "__main__":
    print("Multi-view loss extraction script ready.")
    # Here you can implement a test case if needed
    pass
