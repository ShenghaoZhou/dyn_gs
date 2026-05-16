import torch
import torch.nn.functional as F


def d2n_tblr(points: torch.Tensor,
             k: int = 5,
             d_min: float = 1e-3,
             d_max: float = 10.0) -> torch.Tensor:
    """ points:     3D points in camera coordinates, shape: (B, 3, H, W)
        k:          neighborhood size
    """
    k = (k - 1) // 2

    B, _, H, W = points.size()
    points_pad = F.pad(points, (k, k, k, k), mode='constant',
                       value=0)             # (B, 3, k+H+k, k+W+k)
    valid_pad = (points_pad[:, 2:, :, :] > d_min) & (
        points_pad[:, 2:, :, :] < d_max)  # (B, 1, k+H+k, k+W+k)
    valid_pad = valid_pad.float()

    # vertical vector (top - bottom)
    vec_vert = points_pad[:, :, :H, k:k + W] - \
        points_pad[:, :, 2 * k:2 * k + H, k:k + W]   # (B, 3, H, W)

    # horizontal vector (left - right)
    vec_hori = points_pad[:, :, k:k + H, :W] - \
        points_pad[:, :, k:k + H, 2 * k:2 * k + W]   # (B, 3, H, W)

    # valid_mask (all five depth values - center/top/bottom/left/right should be valid)
    valid_mask = valid_pad[:, :, k:k + H, k:k + W] * \
        valid_pad[:, :, :H, k:k + W] * \
        valid_pad[:, :, 2 * k:2 * k + H, k:k + W] * \
        valid_pad[:, :, k:k + H, :W] * \
        valid_pad[:, :, k:k + H, 2 * k:2 * k + W]
    valid_mask = valid_mask > 0.5

    # get cross product (B, 3, H, W)
    cross_product = - torch.linalg.cross(vec_vert, vec_hori, dim=1)
    normal = F.normalize(cross_product, p=2.0, dim=1, eps=1e-12)

    return normal, valid_mask


def unproject_depth(depth, K, height, width):
    """
    Unproject depth image to 3D points.
    """
    u, v = torch.meshgrid(torch.arange(
        width, device=K.device), torch.arange(height, device=K.device), indexing='xy')
    pixels = torch.stack([u, v, torch.ones_like(u)],
                         dim=-1).float()  # shape (height, width, 3)
    # Manual 3x3 inverse for K to avoid CUSOLVER issues
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    K_inv = torch.zeros_like(K)
    K_inv[0, 0] = 1.0 / fx
    K_inv[1, 1] = 1.0 / fy
    K_inv[0, 2] = -cx / fx
    K_inv[1, 2] = -cy / fy
    K_inv[2, 2] = 1.0
    unprojected_normalized = torch.einsum('ij,hwj->hwi', K_inv, pixels)
    pts = unprojected_normalized * depth.unsqueeze(-1)
    return pts


def initialize_gs(frame, mask):
    pcd = unproject_depth(
        frame.depth, frame.K, frame.depth.shape[0], frame.depth.shape[1])
    mask = mask.view(frame.depth.shape[0], frame.depth.shape[1])
    depth = frame.depth[mask]
    normal = frame.normal[:, mask]
    image = frame.image[:, mask]
    pcd = pcd[mask]

    f = 0.5 * (frame.K[0, 0] + frame.K[1, 1])
    gs = initialize_2dgs(pcd, normal, frame.extrin, colors=image.T,
                         sizes=depth/f)
    # return GaussianSuperPrimitive(gs, depth_scale=1.0, confidence=0.0, anchor_id=frame.id)
    return gs
