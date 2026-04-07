import torch
import math
from gsplat import rasterization, rasterization_2dgs
import math

def getProjectionMatrix(znear, zfar, fovX, fovY, device="cpu"):
    """identical to the one in utils/graphics_utils.py, but with device argument
    """
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def render_2dgs(gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
                viewmat, K, width, height, near_plane: float = 0.01,
                far_plane: float = 100.0, scaling_modifier: float = 1.0,
                bg=torch.zeros(3)):
    device = gs_means.device
    
    # gsplat v1.4.0 has broadcasting bugs in non-packed mode when batch_dims is empty.
    # To work around this, we add a dummy batch dimension of size 1.
    viewmats = viewmat.unsqueeze(0).unsqueeze(0).contiguous() # [1, 1, 4, 4]
    Ks = K.unsqueeze(0).unsqueeze(0).contiguous() # [1, 1, 3, 3]
    
    means = gs_means.unsqueeze(0).contiguous()
    quats = gs_rotations.unsqueeze(0).contiguous()
    
    if gs_scales.shape[-1] == 2:
        scales = torch.cat([gs_scales, torch.zeros_like(gs_scales[..., :1])], dim=-1)
    else:
        scales = gs_scales
    scales = scales.unsqueeze(0).contiguous()
        
    opacities = gs_opacity.squeeze(-1).unsqueeze(0).contiguous()
    
    # Handle SH vs RGB
    if gs_colors.min() < -0.01 or gs_colors.max() > 1.01:
        C0 = 0.28209479177387814
        colors_rgb = gs_colors * C0 + 0.5
    else:
        colors_rgb = gs_colors

    # colors must be [B, C, N, 3] = [1, 1, N, 3]
    colors = colors_rgb.unsqueeze(0).unsqueeze(0).contiguous()
    
    # Use "RGB+ED" to get both color and expected depth (normalized by alpha)
    # rasterization_2dgs returns:
    # colors, alphas, normals, surf_normals, distort, median_depth, meta
    render_colors, render_alphas, render_normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
        means, quats, scales, opacities, colors, 
        viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane,
        render_mode="RGB+ED"
    )
    
    # render_colors is [B, C, H, W, 4] where the last channel is expected depth
    img = render_colors[0, 0, ..., :3].permute(2, 0, 1)
    depth = render_colors[0, 0, ..., 3:4].permute(2, 0, 1)
    normal = render_normals[0, 0].permute(2, 0, 1)
    
    # Handle background
    bg = bg.to(device)
    alphas = render_alphas[0, 0].permute(2, 0, 1)
    img = img + (1 - alphas) * bg[:, None, None]
    
    return img, depth, normal

def render_3dgs(gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
                viewmat, K, width, height, near_plane: float = 0.01,
                far_plane: float = 100.0, scaling_modifier: float = 1.0,
                bg=torch.zeros(3)):
    device = gs_means.device
    
    # Workaround for gsplat v1.4.0 bugs with zero batch dimensions
    viewmats = viewmat.unsqueeze(0).unsqueeze(0).contiguous() # [1, 1, 4, 4]
    Ks = K.unsqueeze(0).unsqueeze(0).contiguous() # [1, 1, 3, 3]
    
    means = gs_means.unsqueeze(0).contiguous()
    quats = gs_rotations.unsqueeze(0).contiguous()
    scales = gs_scales.unsqueeze(0).contiguous()
    opacities = gs_opacity.squeeze(-1).unsqueeze(0).contiguous()
    
    # Handle SH vs RGB
    if gs_colors.min() < -0.01 or gs_colors.max() > 1.01:
        C0 = 0.28209479177387814
        colors_rgb = gs_colors * C0 + 0.5
    else:
        colors_rgb = gs_colors

    # Use "RGB+ED" mode in gsplat to get expected depth automatically.
    # This is more efficient than manual channel concatenation and handles alpha normalization properly.
    render_colors_depth, render_alphas, meta = rasterization(
        means, quats, scales, opacities, colors_rgb.unsqueeze(0).unsqueeze(0).contiguous(),
        viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane,
        render_mode="RGB+ED"
    )
    
    # render_colors_depth is [B, C, H, W, 4] where the 4th channel is expected depth
    img = render_colors_depth[0, 0, ..., :3].permute(2, 0, 1)
    depth = render_colors_depth[0, 0, ..., 3:4].permute(2, 0, 1)
    
    # Handle background for color
    bg = bg.to(device)
    alphas = render_alphas[0, 0].permute(2, 0, 1)
    img = img + (1 - alphas) * bg[:, None, None]
    
    # 3DGS doesn't have native normals; returning zeros to match expected interface
    return img, depth, torch.zeros_like(img)

def render_2dgs_full(gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
                     viewmat, K, width, height, near_plane: float = 0.01,
                     far_plane: float = 100.0, scaling_modifier: float = 1.0,
                     bg=torch.zeros(3)):
    device = gs_means.device
    
    # Use 1-size batch dimension for gsplat compatibility
    viewmats = viewmat.unsqueeze(0).unsqueeze(0).contiguous()
    Ks = K.unsqueeze(0).unsqueeze(0).contiguous()
    
    means = gs_means.unsqueeze(0).contiguous()
    quats = gs_rotations.unsqueeze(0).contiguous()
    
    if gs_scales.shape[-1] == 2:
        scales = torch.cat([gs_scales, torch.zeros_like(gs_scales[..., :1])], dim=-1)
    else:
        scales = gs_scales
    scales = scales.unsqueeze(0).contiguous()
        
    opacities = gs_opacity.squeeze(-1).unsqueeze(0).contiguous()
    # Handle SH vs RGB
    if gs_colors.min() < -0.01 or gs_colors.max() > 1.01:
        C0 = 0.28209479177387814
        colors_rgb = gs_colors * C0 + 0.5
    else:
        colors_rgb = gs_colors

    # colors must be [B, C, N, 3] = [1, 1, N, 3]
    colors = colors_rgb.unsqueeze(0).unsqueeze(0).contiguous()
    
    # Use "RGB+ED" to get both color and expected depth
    render_colors, render_alphas, render_normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
        means, quats, scales, opacities, colors, 
        viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane,
        render_mode="RGB+ED"
    )
    
    img = render_colors[0, 0, ..., :3].permute(2, 0, 1)
    depth = render_colors[0, 0, ..., 3:4].permute(2, 0, 1)
    normal = render_normals[0, 0].permute(2, 0, 1)
    alpha = render_alphas[0, 0].permute(2, 0, 1)
    
    # Handle background
    bg = bg.to(device)
    img = img + (1 - alpha) * bg[:, None, None]
    
    # Matching extra returns: world_view_transform, full_proj_transform, radii, means2D
    # original world_view_transform is viewmat.transpose(0, 1)
    world_view_transform = viewmat.transpose(0, 1)
    # full_proj_transform is needed too. 
    # Since gsplat handles it internally, we might need to compute it manually if the caller needs it.
    # For now, let's just return what we can.
    radii = meta["radii"][0]
    means2D = meta["means2d"][0]
    
    device = gs_means.device
    
    # full_proj_transform calculation (matching getProjectionMatrix logic)
    FoVx = 2 * math.atan(width / (2 * K[0, 0].item()))
    FoVy = 2 * math.atan(height / (2 * K[1, 1].item()))
    
    # We can use the utility from gs_rendering if needed, 
    # but let's avoid cross-importing if we can.
    projection_matrix = getProjectionMatrix(
        znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device).transpose(0, 1)

    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    return img, depth, normal, alpha, world_view_transform, full_proj_transform, radii, means2D

def render_2dgs_visiblity(gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
                          viewmat, K, width, height,  near_plane: float = 0.01,
                          far_plane: float = 100.0, scaling_modifier: float = 1.0,
                          bg=torch.zeros(3)):
    viewmats = viewmat.unsqueeze(0).unsqueeze(0).contiguous()
    Ks = K.unsqueeze(0).unsqueeze(0).contiguous()
    
    means = gs_means.unsqueeze(0).contiguous()
    quats = gs_rotations.unsqueeze(0).contiguous()
    
    if gs_scales.shape[-1] == 2:
        scales = torch.cat([gs_scales, torch.zeros_like(gs_scales[..., :1])], dim=-1)
    else:
        scales = gs_scales
    scales = scales.unsqueeze(0).contiguous()
        
    opacities = gs_opacity.squeeze(-1).unsqueeze(0).contiguous()
    # Handle SH vs RGB
    if gs_colors.min() < -0.01 or gs_colors.max() > 1.01:
        C0 = 0.28209479177387814
        colors_rgb = gs_colors * C0 + 0.5
    else:
        colors_rgb = gs_colors

    # colors must be [B, C, N, 3] = [1, 1, N, 3]
    colors = colors_rgb.unsqueeze(0).unsqueeze(0).contiguous()
    
    _, _, _, _, _, _, meta = rasterization_2dgs(
        means, quats, scales, opacities, colors, 
        viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane
    )
    
    radii = meta["radii"][0]
    means2D = meta["means2d"][0]
    
    return radii, means2D
