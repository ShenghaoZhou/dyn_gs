import torch
import torch.nn.functional as F
import math
from gsplat import rasterization, rasterization_2dgs

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
                bg=torch.zeros(3), custom_colors=None, shs=None):
    device = gs_means.device
    if gs_means.shape[0] == 0:
        return torch.zeros(3, height, width, device=device), \
               torch.zeros(1, height, width, device=device), \
               torch.zeros(3, height, width, device=device), \
               torch.zeros(1, height, width, device=device)
    
    # working shapes for gsplat compatibility: [C, ...]
    viewmats = viewmat.unsqueeze(0).contiguous() # [1, 4, 4] (C=1)
    Ks = K.unsqueeze(0).contiguous() # [1, 3, 3] (C=1)
    
    # Handle SH vs RGB
    sh_degree = None
    if shs is not None:
        sh_degree = int(math.sqrt(shs.shape[1])) - 1
        if sh_degree == 0:
            # SH degree 0 is just RGB, convert SH to RGB and squeeze to [N, 3]
            colors_rgb = shs.squeeze(1) * 0.28209479177387814 + 0.5
            colors = colors_rgb.unsqueeze(0).contiguous() # [1, N, 3]
            sh_degree = None # Treat as RGB
        else:
            # Manual broadcast to [C, N, D, 3] where C=1
            colors = shs.unsqueeze(0).contiguous() 
    else:
        if custom_colors is not None:
            colors_rgb = custom_colors
        else:
            colors_rgb = torch.sigmoid(gs_colors)
        # Manual broadcast to [C, N, 3] where C=1
        colors = colors_rgb.unsqueeze(0).contiguous()
    
    # CRITICAL: Apply activations and normalization
    means = gs_means.contiguous() # [N, 3]
    quats = F.normalize(gs_rotations, dim=-1).contiguous() # [N, 4]
    
    # Handle padding for 2D scales
    if gs_scales.shape[-1] == 2:
        scales_raw = torch.cat([gs_scales, torch.zeros_like(gs_scales[..., :1]) - 10.0], dim=-1)
    else:
        scales_raw = gs_scales
    scales = torch.exp(scales_raw).contiguous() # [N, 3]
    
    opacities = torch.sigmoid(gs_opacity.squeeze(-1)).contiguous() # [N]
    
    render_colors, render_alphas, render_normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
        means, quats, scales, opacities, colors, 
        viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane,
        render_mode="RGB", sh_degree=sh_degree
    )
    
    # render_colors is [C, H, W, 3]
    img = render_colors[0].permute(2, 0, 1)
    depth = median_depth[0].permute(2, 0, 1)
    
    # render_normals is [C, H, W, 3]
    normal_world = render_normals[0] # [H, W, 3]
    R_cw = viewmat[:3, :3]
    normal_cam = torch.einsum('ij,hwj->hwi', R_cw, normal_world)
    normal = normal_cam.permute(2, 0, 1) # [3, H, W]
    
    # Handle background for colors
    bg = bg.to(device)
    alphas = render_alphas[0].permute(2, 0, 1)
    
    img = img + (1 - alphas) * bg[:, None, None]
    
    return img, depth, normal, alphas

def render_2dgs_full(gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
                viewmat, K, width, height, near_plane: float = 0.01,
                far_plane: float = 100.0, scaling_modifier: float = 1.0,
                bg=torch.zeros(3), custom_colors=None, shs=None):
    # This matches the signature of render_2dgs but returns more info for mapping
    img, depth, normal, alphas = render_2dgs(
        gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
        viewmat, K, width, height, near_plane, far_plane, scaling_modifier, bg, custom_colors, shs
    )
    
    # Standardized shaping for meta extraction
    device = gs_means.device
    viewmats = viewmat.unsqueeze(0).contiguous()
    Ks = K.unsqueeze(0).contiguous()
    means = gs_means.contiguous()
    quats = F.normalize(gs_rotations, dim=-1).contiguous()
    if gs_scales.shape[-1] == 2:
        scales_raw = torch.cat([gs_scales, torch.zeros_like(gs_scales[..., :1]) - 10.0], dim=-1)
    else:
        scales_raw = gs_scales
    scales = torch.exp(scales_raw).contiguous()
    opacities = torch.sigmoid(gs_opacity.squeeze(-1)).contiguous()
    
    if shs is not None:
        sh_degree = int(math.sqrt(shs.shape[1])) - 1
        if sh_degree == 0:
            colors = (shs.squeeze(1) * 0.28209479177387814 + 0.5).unsqueeze(0).contiguous()
            sh_degree = None
        else:
            colors = shs.unsqueeze(0).contiguous()
    else:
        colors = gs_colors.unsqueeze(0).contiguous()
        sh_degree = None

    _, _, _, _, _, _, meta = rasterization_2dgs(
        means, quats, scales, opacities, colors, 
        viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane,
        sh_degree=sh_degree
    )
    
    radii = meta["radii"][0]
    means2D = meta["means2d"][0]
    
    world_view_transform = viewmat.transpose(0, 1)
    FoVx = 2 * math.atan(width / (2 * K[0, 0].item()))
    FoVy = 2 * math.atan(height / (2 * K[1, 1].item()))
    projection_matrix = getProjectionMatrix(
        znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    return img, depth, normal, alphas, world_view_transform, full_proj_transform, radii, means2D

def render_2dgs_visiblity(gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
                          viewmat, K, width, height,  near_plane: float = 0.01,
                          far_plane: float = 100.0, scaling_modifier: float = 1.0,
                          bg=torch.zeros(3), shs=None):
    # standardized shaping
    viewmats = viewmat.unsqueeze(0).contiguous()
    Ks = K.unsqueeze(0).contiguous()
    means = gs_means.contiguous()
    quats = F.normalize(gs_rotations, dim=-1).contiguous()
    if gs_scales.shape[-1] == 2:
        scales_raw = torch.cat([gs_scales, torch.zeros_like(gs_scales[..., :1]) - 10.0], dim=-1)
    else:
        scales_raw = gs_scales
    scales = torch.exp(scales_raw).contiguous()
    opacities = torch.sigmoid(gs_opacity.squeeze(-1)).contiguous()
    
    if shs is not None:
        sh_degree = int(math.sqrt(shs.shape[1])) - 1
        if sh_degree == 0:
            colors = (shs.squeeze(1) * 0.28209479177387814 + 0.5).unsqueeze(0).contiguous()
            sh_degree = None
        else:
            colors = shs.unsqueeze(0).contiguous()
    else:
        colors = gs_colors.unsqueeze(0).contiguous()
        sh_degree = None

    _, _, _, _, _, _, meta = rasterization_2dgs(
        means, quats, scales, opacities, colors, 
        viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane,
        sh_degree=sh_degree
    )
    
    radii = meta["radii"][0]
    means2D = meta["means2d"][0]
    
    return radii, means2D

def render_3dgs_full(gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
                viewmat, K, width, height, near_plane: float = 0.01,
                far_plane: float = 100.0, scaling_modifier: float = 1.0,
                bg=torch.zeros(3), shs=None):
    from gsplat import rasterization
    device = gs_means.device
    
    # working shapes for gsplat compatibility: [C, ...]
    viewmats = viewmat.unsqueeze(0).contiguous() 
    Ks = K.unsqueeze(0).contiguous() 
    
    # Handle SH vs RGB
    sh_degree = None
    if shs is not None:
        sh_degree = int(math.sqrt(shs.shape[1])) - 1
        if sh_degree == 0:
            colors = (shs.squeeze(1) * 0.28209479177387814 + 0.5).unsqueeze(0).contiguous()
            sh_degree = None
        else:
            colors = shs.unsqueeze(0).contiguous()
    else:
        if gs_colors.min() < -0.01 or gs_colors.max() > 1.01:
            C0 = 0.28209479177387814
            colors_rgb = gs_colors * C0 + 0.5
        else:
            colors_rgb = gs_colors
        colors = colors_rgb.unsqueeze(0).contiguous()

    means = gs_means.contiguous()
    quats = F.normalize(gs_rotations, dim=-1).contiguous()
    scales = torch.exp(gs_scales).contiguous()
    opacities = torch.sigmoid(gs_opacity.squeeze(-1)).contiguous()

    render_colors, render_alphas, info = rasterization(
        means, quats, scales, opacities, colors,
        viewmats, Ks, width, height,
        sh_degree=sh_degree,
        near_plane=near_plane, far_plane=far_plane,
        render_mode="RGB"
    )
    
    img = render_colors[0].permute(2, 0, 1)
    alphas = render_alphas[0].permute(2, 0, 1)
    depth = torch.zeros(1, height, width, device=device) 
    normal = torch.zeros(3, height, width, device=device)
    
    bg = bg.to(device)
    img = img + (1 - alphas) * bg[:, None, None]
    
    radii = info["radii"][0]
    means2D = info["means2d"][0]
    
    # Matching render_2dgs_full return structure
    world_view_transform = viewmat.transpose(0, 1)
    FoVx = 2 * math.atan(width / (2 * K[0, 0].item()))
    FoVy = 2 * math.atan(height / (2 * K[1, 1].item()))
    # getProjectionMatrix needs to be defined or imported
    projection_matrix = getProjectionMatrix(
        znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device).transpose(0, 1)
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    return img, depth, normal, alphas, world_view_transform, full_proj_transform, radii, means2D

def render_3dgs(gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
                viewmat, K, width, height, near_plane: float = 0.01,
                far_plane: float = 100.0, scaling_modifier: float = 1.0,
                bg=torch.zeros(3), shs=None):
    img, depth, normal, alphas, _, _, _, _ = render_3dgs_full(
        gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
        viewmat, K, width, height, near_plane, far_plane, scaling_modifier, bg, shs
    )
    return img, depth, normal, alphas
