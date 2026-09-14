"""
As-Rigid-As-Possible (ARAP) regularization for Gaussian Splatting.

Provides:
- k-NN neighbor graph construction on Gaussian means
- Edge-length preservation loss (local isometry)
- Quaternion rotation consistency loss
"""
from typing import Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F


def build_knn_graph(means: torch.Tensor, k: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build k-nearest-neighbor graph on Gaussian positions.
    
    Args:
        means: [N, 3] Gaussian positions
        k: Number of neighbors per point
        
    Returns:
        edges: [E, 2] int64 tensor of (i, j) index pairs
        ref_dists: [E] float tensor of reference edge lengths
    """
    N = means.shape[0]
    if N <= 1:
        empty_edges = torch.empty((0, 2), dtype=torch.long, device=means.device)
        empty_dists = torch.empty((0,), dtype=means.dtype, device=means.device)
        return empty_edges, empty_dists

    k = min(k, N - 1)
    
    # Pairwise distance matrix (chunked for memory efficiency if large)
    if N <= 50000:
        dists = torch.cdist(means.unsqueeze(0), means.unsqueeze(0)).squeeze(0)  # [N, N]
        dists.fill_diagonal_(float('inf'))
        _, nn_idx = dists.topk(k, dim=1, largest=False)  # [N, k]
    else:
        chunk_size = 10000
        nn_idx = torch.empty((N, k), dtype=torch.long, device=means.device)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_dists = torch.cdist(means[start:end].unsqueeze(0), means.unsqueeze(0)).squeeze(0)
            chunk_dists[:, start:end].fill_diagonal_(float('inf'))
            _, nn_idx[start:end] = chunk_dists.topk(k, dim=1, largest=False)
    
    # Build edge list: (i, j) pairs
    src = torch.arange(N, device=means.device).unsqueeze(1).expand(-1, k).reshape(-1)  # [N*k]
    dst = nn_idx.reshape(-1)  # [N*k]
    edges = torch.stack([src, dst], dim=1)  # [N*k, 2]
    
    # Compute reference distances (target distances to preserve)
    ref_dists = torch.norm(means[edges[:, 0]] - means[edges[:, 1]], dim=1)  # [N*k]
    
    return edges, ref_dists


def compute_arap_loss(
    means: torch.Tensor,
    edges: torch.Tensor,
    ref_dists: torch.Tensor,
) -> torch.Tensor:
    """Compute edge-length preservation (local isometry) loss.
    
    This is the core ARAP-style rigidity constraint: penalize changes
    in pairwise distances between neighboring Gaussians.
    
    L = (1/|E|) * sum_{(i,j)} ( ||p_i - p_j|| - d_ij^ref )^2
    
    Args:
        means: [N, 3] current Gaussian positions (differentiable)
        edges: [E, 2] neighbor index pairs (from build_knn_graph)
        ref_dists: [E] reference edge lengths (from build_knn_graph)
        
    Returns:
        Scalar loss value
    """
    if edges.numel() == 0 or ref_dists.numel() == 0:
        return torch.tensor(0.0, device=means.device)

    # Current edge vectors and lengths with epsilon for numerical stability in sqrt grad
    edge_vecs = means[edges[:, 0]] - means[edges[:, 1]]  # [E, 3]
    curr_dists = torch.sqrt(torch.sum(edge_vecs ** 2, dim=1) + 1e-8)  # [E]
    
    loss = torch.mean((curr_dists - ref_dists) ** 2)
    return loss


def compute_rotation_consistency_loss(
    quats: torch.Tensor,
    edges: torch.Tensor,
) -> torch.Tensor:
    """Encourage neighboring Gaussians to have similar orientations.
    
    L = (1/|E|) * sum_{(i,j)} (1 - |<q_i, q_j>|)
    
    Uses the geodesic quaternion distance: two quaternions q and -q
    represent the same rotation, so we use |dot product|.
    
    Args:
        quats: [N, 4] Gaussian quaternions (differentiable)
        edges: [E, 2] neighbor index pairs
        
    Returns:
        Scalar loss value
    """
    if edges.numel() == 0:
        return torch.tensor(0.0, device=quats.device)

    q_i = F.normalize(quats[edges[:, 0]], dim=1)  # [E, 4]
    q_j = F.normalize(quats[edges[:, 1]], dim=1)  # [E, 4]
    
    dot = torch.clamp((q_i * q_j).sum(dim=1).abs(), max=1.0)  # [E], in [0, 1]
    loss = torch.mean(1.0 - dot)
    return loss


def regularize_lifted_flow_with_arap(
    pts2d: np.ndarray,
    pts3d_canon: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    strain_thresh: float = 0.15,
    k_neighbors: int = 8,
    lambda_arap: float = 5.0,
    num_opt_steps: int = 25,
    device: str = "cuda"
) -> Tuple[Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """Regularize 2D-to-3D lifted optical flow tracks using ARAP isometry with canonical object points.

    1. Computes local ARAP edge strain on lifted points to filter out non-rigid flow/depth outliers (e.g. hand, occlusions).
    2. Optimizes depths along visual rays to preserve canonical pairwise edge lengths.
    3. Solves closed-form Kabsch 6-DoF rigid alignment between canonical points and regularized 3D points.

    Args:
        pts2d: [N, 2] 2D point positions in the current frame
        pts3d_canon: [N, 3] corresponding canonical 3D object coordinates
        depth: [H, W] current depth map
        K: [3, 3] camera intrinsic matrix
        strain_thresh: maximum allowed mean fractional edge strain before a track is deemed an outlier
        k_neighbors: number of nearest neighbors in canonical space
        lambda_arap: weight on edge-length preservation loss during depth optimization
        num_opt_steps: number of gradient optimization steps along visual rays
        device: computation device ('cuda' or 'cpu')

    Returns:
        T_arap: [4, 4] estimated 6-DoF object pose in camera frame (T_CiO), or None if insufficient points
        inlier_mask: [N] boolean array of surviving isometric tracks
        pts3d_reg: [M, 3] regularized 3D points in camera frame, or None
    """
    N = len(pts2d)
    if N < 8 or depth is None:
        return None, np.ones(N, dtype=bool), None

    h, w = depth.shape[:2]
    ix = np.clip(np.round(pts2d[:, 0]).astype(int), 0, w - 1)
    iy = np.clip(np.round(pts2d[:, 1]).astype(int), 0, h - 1)
    d_obs = depth[iy, ix]
    valid_d = (d_obs > 0.05) & np.isfinite(d_obs)

    if valid_d.sum() < 8:
        return None, valid_d, None

    K_inv = np.linalg.inv(K)
    rays = (K_inv @ np.stack([pts2d[:, 0], pts2d[:, 1], np.ones(N)], axis=0)).T  # [N, 3]
    pts3d_cam_raw = rays * d_obs[:, None]

    dev = torch.device(device if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    pts_canon_t = torch.from_numpy(pts3d_canon).float().to(dev)
    rays_t = torch.from_numpy(rays).float().to(dev)
    d_obs_t = torch.from_numpy(d_obs).float().to(dev)

    # 1. Build k-NN graph on canonical object points
    k = min(k_neighbors, N - 1)
    edges, ref_dists = build_knn_graph(pts_canon_t, k=k)
    if edges.numel() == 0:
        return None, valid_d, None

    src, dst = edges[:, 0], edges[:, 1]

    # Compute raw strain against canonical geometry
    pts3d_cam_raw_t = torch.from_numpy(pts3d_cam_raw).float().to(dev)
    raw_cam_dists = torch.norm(pts3d_cam_raw_t[src] - pts3d_cam_raw_t[dst], dim=1)
    raw_strain = torch.abs(raw_cam_dists - ref_dists) / (ref_dists + 1e-6)

    # Mean strain per point
    point_strain = torch.zeros(N, device=dev)
    point_deg = torch.zeros(N, device=dev)
    point_strain.scatter_add_(0, src, raw_strain)
    point_deg.scatter_add_(0, src, torch.ones_like(raw_strain))
    point_strain = point_strain / (point_deg + 1e-6)

    inlier_mask_t = (point_strain < strain_thresh) & torch.from_numpy(valid_d).to(dev)
    inlier_mask = inlier_mask_t.cpu().numpy()

    if inlier_mask.sum() < 6:
        # Fallback to valid depth
        inlier_mask = valid_d

    pts_canon_f = pts_canon_t[inlier_mask]
    rays_f = rays_t[inlier_mask]
    d_obs_f = d_obs_t[inlier_mask]
    M = len(pts_canon_f)

    if M < 6:
        return None, inlier_mask, None

    # 2. Rebuild graph on surviving inlier points
    k_f = min(k_neighbors, M - 1)
    edges_f, ref_dists_f = build_knn_graph(pts_canon_f, k=k_f)
    src_f, dst_f = edges_f[:, 0], edges_f[:, 1]

    # 3. Optimize depths along visual rays to satisfy local isometry
    z = d_obs_f.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([z], lr=0.005)

    for _ in range(num_opt_steps):
        optimizer.zero_grad()
        p_c = rays_f * z.unsqueeze(1)
        e_vecs = p_c[src_f] - p_c[dst_f]
        curr_dists = torch.sqrt(torch.sum(e_vecs ** 2, dim=1) + 1e-8)
        loss_data = torch.mean((z - d_obs_f) ** 2)
        loss_arap = torch.mean((curr_dists - ref_dists_f) ** 2)
        loss = loss_data + lambda_arap * loss_arap
        loss.backward()
        optimizer.step()

    z_opt = z.detach()
    p_c_opt = (rays_f * z_opt.unsqueeze(1)).cpu().numpy()
    pts_canon_np = pts_canon_f.cpu().numpy()

    # 4. Closed-form Kabsch alignment: R, t between canonical points and regularized 3D points
    p_O_mean = pts_canon_np.mean(axis=0)
    p_C_mean = p_c_opt.mean(axis=0)
    H = (pts_canon_np - p_O_mean).T @ (p_c_opt - p_C_mean)
    U, S, Vt = np.linalg.svd(H)
    R_mat = Vt.T @ U.T
    if np.linalg.det(R_mat) < 0:
        Vt[-1, :] *= -1
        R_mat = Vt.T @ U.T
    t_vec = p_C_mean - R_mat @ p_O_mean

    T_arap = np.eye(4)
    T_arap[:3, :3] = R_mat
    T_arap[:3, 3] = t_vec

    return T_arap, inlier_mask, p_c_opt
