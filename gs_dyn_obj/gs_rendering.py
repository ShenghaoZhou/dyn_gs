from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import math
import torch

from diff_gaussian_rasterization import GaussianRasterizationSettings as DGSRasterizationSettings
from diff_gaussian_rasterization import GaussianRasterizer as DGSRasterizer
from diff_gaussian_rasterization import rasterize_gaussians as raterize_gaussians_3dgs
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
                viewmat, K, width, height,  near_plane: float = 0.01,
                far_plane: float = 100.0, scaling_modifier: float = 1.0,
                bg=torch.zeros(3)
                ):

    device = gs_means.device

    FoVx = 2 * math.atan(width / (2 * K[0, 0].item()))
    FoVy = 2 * math.atan(height / (2 * K[1, 1].item()))
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    world_view_transform = viewmat.transpose(0, 1).to(device)

    projection_matrix = getProjectionMatrix(
        znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device).transpose(0, 1)

    full_proj_transform = (world_view_transform.unsqueeze(
        0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    camera_center = world_view_transform.inverse()[3, :3]
    bg = bg.to(device)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = torch.zeros_like(gs_means, requires_grad=True, device=device)

    render_image, radii, allmap = rasterizer(
        means3D=gs_means,
        means2D=means2D,
        shs=None,
        colors_precomp=gs_colors,
        opacities=gs_opacity,
        scales=gs_scales,
        rotations=gs_rotations,
        cov3D_precomp=None
    )

    render_normal = allmap[2:5]

    # need to decide between median depth or expected depth
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # print("render depth median range: ", render_depth_median.min(),
    #       render_depth_median.max())
    # render_depth_median[render_depth_median > 1.8107923] = 1.8107923
    # plt.imshow(render_depth_median[0].cpu().numpy())
    # plt.title("rendered depth median")
    # plt.show()
    render_alpha = allmap[1:2]
    render_depth_expected = allmap[0:1]
    # Clamp alpha to avoid division by zero and NaN during backward pass
    render_depth_expected = (render_depth_expected / render_alpha.clamp(min=1e-8))
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    # according to the comment of 2DGS, we should use median depth for bounded scene,
    # and expected depth for unbounded scene
    render_depth = render_depth_expected

    return render_image, render_depth, render_normal


def render_2dgs_full(gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
                     viewmat, K, width, height,  near_plane: float = 0.01,
                     far_plane: float = 100.0, scaling_modifier: float = 1.0,
                     bg=torch.zeros(3)
                     ):

    device = gs_means.device

    FoVx = 2 * math.atan(width / (2 * K[0, 0].item()))
    FoVy = 2 * math.atan(height / (2 * K[1, 1].item()))
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    world_view_transform = viewmat.transpose(0, 1).to(device)

    projection_matrix = getProjectionMatrix(
        znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device).transpose(0, 1)

    full_proj_transform = (world_view_transform.unsqueeze(
        0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    camera_center = world_view_transform.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg.to(device),
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = torch.zeros_like(gs_means, requires_grad=True, device=device)

    render_image, radii, allmap = rasterizer(
        means3D=gs_means,
        means2D=means2D,
        shs=None,
        colors_precomp=gs_colors,
        opacities=gs_opacity,
        scales=gs_scales,
        rotations=gs_rotations,
        cov3D_precomp=None
    )

    render_normal = allmap[2:5]

    # need to decide between median depth or expected depth
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

    # print("render depth median range: ", render_depth_median.min(),
    #       render_depth_median.max())
    # render_depth_median[render_depth_median > 1.8107923] = 1.8107923
    # plt.imshow(render_depth_median[0].cpu().numpy())
    # plt.title("rendered depth median")
    # plt.show()
    render_alpha = allmap[1:2]
    render_depth_expected = allmap[0:1]
    # Clamp alpha to avoid division by zero and NaN during backward pass
    render_depth_expected = (render_depth_expected / render_alpha.clamp(min=1e-8))
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)
    # according to the comment of 2DGS, we should use median depth for bounded scene,
    # and expected depth for unbounded scene
    render_depth = render_depth_expected

    return render_image, render_depth, render_normal, render_alpha, \
        world_view_transform, full_proj_transform, radii, means2D


def render_2dgs_visiblity(gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
                          viewmat, K, width, height,  near_plane: float = 0.01,
                          far_plane: float = 100.0, scaling_modifier: float = 1.0,
                          bg=torch.zeros(3)
                          ):

    device = gs_means.device

    FoVx = 2 * math.atan(width / (2 * K[0, 0].item()))
    FoVy = 2 * math.atan(height / (2 * K[1, 1].item()))
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)

    world_view_transform = viewmat.transpose(0, 1).to(device)

    projection_matrix = getProjectionMatrix(
        znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device).transpose(0, 1)

    full_proj_transform = (world_view_transform.unsqueeze(
        0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)

    camera_center = world_view_transform.inverse()[3, :3]

    raster_settings = GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg.to(device),
        scale_modifier=scaling_modifier,
        viewmatrix=world_view_transform,
        projmatrix=full_proj_transform,
        sh_degree=0,
        campos=camera_center,
        prefiltered=False,
        debug=False,
        # pipe.debug
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means2D = torch.zeros_like(gs_means, requires_grad=True, device=device)

    render_image, radii, allmap = rasterizer(
        means3D=gs_means,
        means2D=means2D,
        shs=None,
        colors_precomp=gs_colors,
        opacities=gs_opacity,
        scales=gs_scales,
        rotations=gs_rotations,
        cov3D_precomp=None
    )

    return radii, means2D


def render_3dgs(gs_means, gs_rotations, gs_scales, gs_colors, gs_opacity,
                viewmat, K, width, height,  near_plane: float = 0.01,
                far_plane: float = 100.0, scaling_modifier: float = 1.0,
                bg=torch.zeros(3)):
    from diff_gaussian_rasterization import rasterize_gaussians
    device = gs_means.device
    
    # Input validation and clamping to prevent numerical issues
    gs_means = gs_means.contiguous()
    gs_opacity = torch.clamp(gs_opacity, 0.0, 1.0).contiguous()
    gs_scales = gs_scales.contiguous()
    gs_rotations = gs_rotations.contiguous()
    gs_colors = gs_colors.contiguous()

    FoVx = 2 * math.atan(width / (2 * K[0, 0].item()))
    FoVy = 2 * math.atan(height / (2 * K[1, 1].item()))
    tanfovx = math.tan(FoVx * 0.5)
    tanfovy = math.tan(FoVy * 0.5)
    bg = bg.to(device)
    
    world_view_transform = viewmat.transpose(0, 1).to(device).contiguous()
    cam_centre = world_view_transform.detach().inverse()[3, :3].contiguous()
    projection_matrix = getProjectionMatrix(
        znear=near_plane, zfar=far_plane, fovX=FoVx, fovY=FoVy, device=device).transpose(0, 1).to(device).contiguous()
    
    means2D = torch.zeros_like(gs_means, requires_grad=True, device=device)
    
    # Using degree 0 as default
    max_sh_degree = 0
    empty_tensor = torch.Tensor([]).to(device)

    raster_settings = DGSRasterizationSettings(
        image_height=int(height),
        image_width=int(width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg,
        scale_modifier=scaling_modifier,
        projmatrix=projection_matrix,
        sh_degree=max_sh_degree,
        campos=cam_centre,
        prefiltered=False,
        debug=False,
    )

    # Call directly to ensure precise argument passing
    color, invdepth, mainGaussID, radii = rasterize_gaussians(
        means3D=gs_means,
        means2D=means2D,
        dc=gs_colors,         # [N, 3] or [N, 1, 3]
        sh=empty_tensor,      # Empty if degree 0
        colors_precomp=empty_tensor,
        opacities=gs_opacity,
        scales=gs_scales,
        rotations=gs_rotations,
        cov3Ds_precomp=empty_tensor,
        viewmatrix=world_view_transform,
        raster_settings=raster_settings
    )
    
    # Synchronize to catch CUDA errors early
    torch.cuda.synchronize()
    
    render_depth = 1.0 / (invdepth + 1e-8)
    
    return color, render_depth, torch.zeros_like(color)








