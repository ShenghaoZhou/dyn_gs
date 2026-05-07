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
                bg=torch.zeros(3), custom_colors=None):
    device = gs_means.device
    
    # working shapes for gsplat 1.5.3: means [N,3], viewmats [1,4,4], colors [1,N,3]
    means = gs_means.contiguous() # [N, 3]
    quats = gs_rotations.contiguous() # [N, 4]
    
    if gs_scales.shape[-1] == 2:
        scales = torch.cat([gs_scales, torch.zeros_like(gs_scales[..., :1])], dim=-1)
    else:
        scales = gs_scales
    scales = scales.contiguous() # [N, 3]
    opacities = gs_opacity.squeeze(-1).contiguous() # [N]
    
    viewmats = viewmat.unsqueeze(0).contiguous() # [1, 4, 4] (C=1)
    Ks = K.unsqueeze(0).contiguous() # [1, 3, 3] (C=1)
    
    # Handle SH vs RGB
    if custom_colors is not None:
        colors_rgb = custom_colors
    elif gs_colors.min() < -0.01 or gs_colors.max() > 1.01:
        C0 = 0.28209479177387814
        colors_rgb = gs_colors * C0 + 0.5
    else:
        colors_rgb = gs_colors

    # colors must be [C, N, D] = [1, N, D]
    colors = colors_rgb.unsqueeze(0).contiguous()
    
    # CRITICAL: Apply activations and normalization
    means = gs_means.contiguous()
    quats = F.normalize(gs_rotations, dim=-1).contiguous()
    
    # Handle padding for 2D scales
    if gs_scales.shape[-1] == 2:
        scales = torch.cat([gs_scales, torch.zeros_like(gs_scales[..., :1]) - 10.0], dim=-1) # -10.0 for near-zero scale
    else:
        scales = gs_scales
    scales = torch.exp(scales).contiguous()
    
    opacities = torch.sigmoid(gs_opacity.squeeze(-1)).contiguous()
    
    render_colors, render_alphas, render_normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
        means, quats, scales, opacities, colors, 
        viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane,
        render_mode="RGB"
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
    
    return img, depth, normal, alphas

def render_3dgs(gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
                viewmat, K, width, height, near_plane: float = 0.01,
                far_plane: float = 100.0, scaling_modifier: float = 1.0,
                bg=torch.zeros(3)):
    from pytorch3d.transforms import quaternion_to_matrix
    device = gs_means.device
    
    # 1. Compute Normals from Smallest Axis
    # Scales: [N, 3], Quats: [N, 4]
    # Identify index of smallest scale
    smallest_axis_idx = torch.argmin(gs_scales, dim=-1, keepdim=True) # [N, 1]
    # Get rotation matrices [N, 3, 3]
    R_on = quaternion_to_matrix(F.normalize(gs_rotations, dim=-1))
    # Extract the column corresponding to the smallest scale
    # normals_o = R_on.gather(2, smallest_axis_idx.unsqueeze(-1).expand(-1, 3, 1)).squeeze(-1)
    
    # Efficiently gather the column:
    # smallest_axis_idx is [N, 1]. We want R_on[:, :, smallest_axis_idx]
    # R_on is [N, 3, 3]. 
    batch_indices = torch.arange(gs_means.shape[0], device=device)
    normals_o = R_on[batch_indices, :, smallest_axis_idx.squeeze(-1)] # [N, 3]
    
    # Ensure normals point towards camera
    # T_CO = viewmat. [3, 3] is R_co.
    # pts_c = R_co @ pts_o + t_co.
    # Normal in cam: n_c = R_co @ n_o.
    R_co = viewmat[:3, :3]
    t_co = viewmat[:3, 3]
    means_c = torch.einsum('ij,nj->ni', R_co, gs_means) + t_co
    normals_c = torch.einsum('ij,nj->ni', R_co, normals_o)
    
    # Flip if pointing away from camera (dot product with view direction -means_c should be positive)
    # Actually, dot(n_c, -means_c) > 0  => dot(n_c, means_c) < 0
    cos_theta = (normals_c * means_c).sum(-1)
    normals_o = torch.where(cos_theta[..., None] > 0, -normals_o, normals_o)
    
    # 2. Setup Rasterization
    viewmats = viewmat.unsqueeze(0).unsqueeze(0).contiguous() # [1, 1, 4, 4]
    Ks = K.unsqueeze(0).unsqueeze(0).contiguous() # [1, 1, 3, 3]
    
    means = gs_means.unsqueeze(0).contiguous()
    quats = F.normalize(gs_rotations, dim=-1).unsqueeze(0).contiguous()
    scales = (torch.exp(gs_scales) * scaling_modifier).unsqueeze(0).contiguous()
    opacities = torch.sigmoid(gs_opacity.squeeze(-1)).unsqueeze(0).contiguous()
    
    # Handle SH vs RGB
    if gs_colors.min() < -0.01 or gs_colors.max() > 1.01 or gs_colors.abs().mean() < 0.2:
        C0 = 0.28209479177387814
        colors_rgb = gs_colors * C0 + 0.5
    else:
        colors_rgb = gs_colors
    
    # NaN Guard
    if torch.isnan(colors_rgb).any():
        colors_rgb = torch.where(torch.isnan(colors_rgb), torch.zeros_like(colors_rgb), colors_rgb)
    if torch.isnan(normals_o).any():
        normals_o = torch.where(torch.isnan(normals_o), torch.zeros_like(normals_o), normals_o)

    # Concatenate colors and normals for joint rasterization
    # colors_and_normals: [N, 6]
    colors_and_normals = torch.cat([colors_rgb, normals_o], dim=-1)
    
    # Render
    render_pkg, render_alphas, meta = rasterization(
        means, quats, scales, opacities, colors_and_normals.unsqueeze(0).unsqueeze(0).contiguous(),
        viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane,
        render_mode="RGB+ED"
    )
    
    # render_pkg is [B, C, H, W, 7] (3 color + 3 normal + 1 depth)
    img = render_pkg[0, 0, ..., :3].permute(2, 0, 1)
    normal_world = render_pkg[0, 0, ..., 3:6]
    depth = render_pkg[0, 0, ..., 6:7].permute(2, 0, 1)
    
    # Transform normals to camera space
    normal_cam = torch.einsum('ij,hwj->hwi', R_co, normal_world)
    normal = normal_cam.permute(2, 0, 1) 
    
    # Handle background for color
    bg = bg.to(device)
    alphas = render_alphas[0, 0].permute(2, 0, 1)
    img = img + (1 - alphas) * bg[:, None, None]
    
    return img, depth, normal, alphas

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
    # Handle SH vs RGB (heuristic detection)
    # SH coefficients for mid-range RGB values often fall within [0, 1], 
    # but their mean is typically closer to 0.0 than 0.5.
    if gs_colors.min() < -0.01 or gs_colors.max() > 1.01 or gs_colors.abs().mean() < 0.2:
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
    
    # Transform normals to camera space
    normal_world = render_normals[0, 0]
    R_cw = viewmat[:3, :3]
    normal_cam = torch.einsum('ij,hwj->hwi', R_cw, normal_world)
    normal = normal_cam.permute(2, 0, 1) 
    
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
