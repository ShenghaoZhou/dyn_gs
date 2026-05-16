import numpy as np
import cv2
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from gs_dyn_obj.grouped_gs import GaussianSuperPrimitive
from gs_dyn_obj.gs_rendering_gsplat import render_2dgs
from pytorch3d.transforms import quaternion_to_matrix, matrix_to_quaternion
from pytorch3d.transforms.so3 import so3_exp_map
from gsplat import rasterization_2dgs
import matplotlib.pyplot as plt
import tyro
from dataclasses import dataclass
import io
from PIL import Image
from gs_dyn_obj.utils.init import d2n_tblr, unproject_depth
from scipy.spatial.transform import Rotation as R
from run_single_view_loss import compute_single_view_loss, normal_from_depth_image
import time

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    target_frame: int = 40
    near_plane: float = 0.01
    far_plane: float = 10.0
    device: str = "cuda"
    opt_lr: float = 1e-3
    opt_steps: int = 200
    patch_size: int = 3
    sample_num: int = 5000
    
    # Loss weights
    lambda_sv: float = 1.0     # single-view consistency
    lambda_prior: float = 1.0  # MoGE normal prior
    lambda_rgb: float = 1.0    # photometric loss at ref frame
    lambda_ncc: float = 1.0    # NCC multi-view loss
    lambda_tv: float = 0.01    # total variation loss on distance
    lambda_dist: float = 0.0   # penalty for distance deviation from initial
    
    # MOGE options
    use_moge_depth: bool = False
    init_from_depth_normal: bool = False
    
    # Optimization mode
    use_viewray: bool = True   # If False, directly optimize GS means

class CameraDevice:
    def __init__(self, K, T_CW, width, height):
        self.device = T_CW.device
        self.world_view_transform = T_CW.t() 
        self.K = K 
        self.width = width
        self.height = height
        
    def get_k(self, scale=1.0):
        K = self.K.clone()
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] *= scale
        K[1, 2] *= scale
        return K

    def get_calib_matrix_nerf(self, scale=1.0):
        return self.get_k(scale), self.world_view_transform.t()

def depth_to_rgb(depth, min_val=0.0, max_val=2.0):
    depth_norm = np.clip((depth - min_val) / (max_val - min_val), 0, 1)
    colormap = plt.get_cmap("magma")
    rgb = (colormap(depth_norm)[..., :3] * 255).astype(np.uint8)
    return rgb

# --- PGSR Loss Functions ---

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
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))
    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps
    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum
    cc = (cross * cross) / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    return ncc

def compute_multi_view_ncc_loss(
    ref_image_gray, nea_image_gray, ref_normal, ref_distance, T_nea_ref, K, 
    patch_size=3, valid_indices=None, pixels=None
):
    device = ref_image_gray.device
    total_patch_size = (patch_size * 2 + 1) ** 2
    R_rel = T_nea_ref[:3, :3]
    t_rel = T_nea_ref[:3, 3]
    if len(ref_normal.shape) == 3:
        ref_normal = ref_normal.permute(1, 2, 0).reshape(-1, 3)
    if len(ref_distance.shape) == 3:
        ref_distance = ref_distance.reshape(-1)
    elif len(ref_distance.shape) == 2:
        ref_distance = ref_distance.reshape(-1)
    if valid_indices is not None:
        ref_local_n = ref_normal[valid_indices]
        ref_local_d = ref_distance[valid_indices]
        pixels = pixels[valid_indices]
    else:
        ref_local_n = ref_normal
        ref_local_d = ref_distance
    if pixels is None: raise ValueError("Pixels must be provided.")
    K_inv = torch.inverse(K)
    H_ref_to_nea = R_rel[None] - \
        torch.matmul(t_rel[None, :, None].expand(ref_local_d.shape[0], 3, 1), 
                    ref_local_n[:, :, None].permute(0, 2, 1)) / ref_local_d[..., None, None]
    H_ref_to_nea = torch.matmul(K[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_nea)
    H_ref_to_nea = H_ref_to_nea @ K_inv
    offsets = patch_offsets(patch_size, device)
    ori_pixels_patch = pixels.reshape(-1, 1, 2) + offsets.float()
    H_ref, W_ref = ref_image_gray.shape[-2:]
    pixels_patch = ori_pixels_patch.clone()
    pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W_ref - 1) - 1.0
    pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H_ref - 1) - 1.0
    ref_gray_val = F.grid_sample(ref_image_gray[None, None], pixels_patch.view(1, -1, 1, 2), align_corners=True)
    ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)
    grid = patch_warp(H_ref_to_nea.reshape(-1, 3, 3), ori_pixels_patch)
    H_nea, W_nea = nea_image_gray.shape[-2:]
    grid[:, :, 0] = 2 * grid[:, :, 0] / (W_nea - 1) - 1.0
    grid[:, :, 1] = 2 * grid[:, :, 1] / (H_nea - 1) - 1.0
    sampled_nea_gray_val = F.grid_sample(nea_image_gray[None, None], grid.reshape(1, -1, 1, 2), align_corners=True)
    sampled_nea_gray_val = sampled_nea_gray_val.reshape(-1, total_patch_size)
    ncc_val = lncc(ref_gray_val, sampled_nea_gray_val)
    return ncc_val

def gs_to_planar_params(means, quats, T_CO):
    R_OG = quaternion_to_matrix(quats) # [N, 3, 3]
    n_O = R_OG[:, :, 2] # [N, 3]
    R_CO = T_CO[:3, :3]
    t_CO = T_CO[:3, 3]
    n_C = torch.einsum('ij,nj->ni', R_CO, n_O) # [N, 3]
    p_C = torch.einsum('ij,nj->ni', R_CO, means) + t_CO # [N, 3]
    d = -torch.sum(n_C * p_C, dim=1, keepdim=True) # [N, 1]
    return n_C, d

def render_custom_attribute(means, quats, scales, opacities, attr, T_CW, K, width, height, near_plane=0.01, far_plane=100.0):
    viewmats = T_CW.unsqueeze(0).unsqueeze(0).contiguous()
    Ks = K.unsqueeze(0).unsqueeze(0).contiguous()
    means = means.unsqueeze(0).contiguous()
    quats = quats.unsqueeze(0).contiguous()
    if scales.shape[-1] == 2:
        scales = torch.cat([scales, torch.zeros_like(scales[..., :1])], dim=-1).unsqueeze(0).contiguous()
    else:
        scales = scales.unsqueeze(0).contiguous()
    opacities = opacities.squeeze(-1).unsqueeze(0).contiguous()
    colors = attr.unsqueeze(0).unsqueeze(0).contiguous()
    render_colors, render_alphas, _, _, _, _, _ = rasterization_2dgs(
        means, quats, scales, opacities, colors,
        viewmats, Ks, width, height,
        near_plane=near_plane, far_plane=far_plane,
        render_mode="RGB"
    )
    return render_colors[0, 0].permute(2, 0, 1), render_alphas[0, 0].permute(2, 0, 1)

def load_object_pose_world(data_root, frame_idx):
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists(): return None
    with open(poses_file, "r") as f: lines = f.readlines()
    if frame_idx >= len(lines): return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])]) # x y z w
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4); T_WO[:3, :3] = r_WO; T_WO[:3, 3] = t_WO
    return T_WO

def load_pose_T_CO(data_root, frame_idx):
    """
    Compute Camera-to-Object (T_CO) pose.
    T_CO = T_CW * T_WO = inv(T_WC) * T_WO.
    Points in Object Space (P_O) transform to Camera Space (P_C) as: P_C = T_CO * P_O.
    """
    T_WO = load_object_pose_world(data_root, frame_idx)
    if T_WO is None: return None
    ext_file = Path(data_root) / "extrinsics" / f"{frame_idx:06d}.npy"
    if not ext_file.exists(): return T_WO 
    T_WC = np.load(ext_file)
    T_CW = np.linalg.inv(T_WC)
    return T_CW @ T_WO

def setup_blueprint():
    import rerun.blueprint as rrb
    return rrb.Blueprint(
        rrb.Tabs(
            rrb.Vertical(
                rrb.Grid(
                    rrb.Spatial2DView(name="Ref Image", contents=["gs/ref/image"]),
                    rrb.Spatial2DView(name="Ref Depth", contents=["gs/ref/depth"]),
                    rrb.Spatial2DView(name="Ref Normal", contents=["gs/ref/normal"]),
                    rrb.Spatial2DView(name="Tgt Image", contents=["gs/tgt/image"]),
                    rrb.Spatial2DView(name="Tgt Depth", contents=["gs/tgt/depth"]),
                    rrb.Spatial2DView(name="Tgt Normal", contents=["gs/tgt/normal"]),
                    grid_columns=3, name="2D Comparison"
                ),
                rrb.Horizontal(rrb.Spatial2DView(name="Loss Plot", contents=["opt/loss_plot"])),
                name="Optimization"
            ),
            rrb.Horizontal(rrb.Spatial3DView(name="3D Visualization", contents=["world/**"]), name="3D View")
        )
    )

def main(cfg: Config):
    rr.init("exp_gs_improve_normal_twoview", spawn=False)
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    rr.send_blueprint(setup_blueprint())
    
    device = torch.device(cfg.device)
    data_dir = Path(cfg.data_root)
    
    # 1. Load Data
    def load_frame_data(idx):
        stem = f"{idx:06d}"
        img = np.array(cv2.imread(str(data_dir / "images" / f"{stem}.png"))[..., ::-1])
        mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        if cfg.use_moge_depth: depth = np.load(data_dir / "moge_depth" / f"{stem}.npy")
        else: depth = np.load(data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{stem}.npy")
        normal = np.load(data_dir / "moge_normal" / f"{stem}.npy")
        if normal.shape[0] == 3 and normal.shape[1] != 3: normal = normal.transpose(1, 2, 0)
        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        T_CO = load_pose_T_CO(cfg.data_root, idx)
        T_WC = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        T_WO = load_object_pose_world(cfg.data_root, idx)
        return img, mask, depth, normal, K, T_CO, T_WC, T_WO

    img_ref, mask_ref, depth_ref, normal_ref, K_ref, T_CO_ref, T_WC_ref, T_WO_ref = load_frame_data(cfg.init_frame)
    img_tgt, mask_tgt, depth_tgt, normal_tgt, K_tgt, T_CO_tgt, T_WC_tgt, T_WO_tgt = load_frame_data(cfg.target_frame)
    
    # Load GT depth and convert to Object-Space
    def load_gt_pcd_O(idx, T_WC, T_WO, K, img, mask):
        stem = f"{idx:06d}"
        depth_gt_file = data_dir / "depth_dyn" / f"{stem}.npy"
        if not depth_gt_file.exists(): return None
        depth_gt = np.load(depth_gt_file)
        h, w = depth_gt.shape
        # Points in Camera Space
        xyz_C = unproject_depth(torch.from_numpy(depth_gt).float().to(device), 
                                torch.from_numpy(K).float().to(device), h, w).reshape(-1, 3).cpu().numpy()
        valid = (depth_gt.reshape(-1) > 0.01) & (mask.reshape(-1) > 0)
        pts_C_valid = xyz_C[valid]
        
        # Robust transform: P_C -> P_W -> P_O
        # P_W = T_WC * P_C
        # P_O = inv(T_WO) * P_W
        pts_W = (T_WC[:3, :3] @ pts_C_valid.T).T + T_WC[:3, 3]
        T_OW = np.linalg.inv(T_WO)
        pts_O = (T_OW[:3, :3] @ pts_W.T).T + T_OW[:3, 3]
        return pts_O, img.reshape(-1, 3)[valid]

    # Map GT/GS into the Global World frame as seen at the Reference Frame (T_WO_ref)
    def to_world_ref(pts_O):
        return (T_WO_ref[:3, :3] @ pts_O.T).T + T_WO_ref[:3, 3]

    def log_pcd_and_frames(idx, T_WC, T_WO, K, img, mask, name):
        pcd_O = load_gt_pcd_O(idx, T_WC, T_WO, K, img, mask)
        if pcd_O:
            pts_O, colors = pcd_O
            # 1. Log in Object Space
            rr.log(f"object_space/{name}", rr.Points3D(pts_O, colors=colors, radii=0.001))
            # 2. Log in World Space (as transformed to ref-frame world)
            rr.log(f"world/{name}", rr.Points3D(to_world_ref(pts_O), colors=colors, radii=0.001))
            
            # 3. Log Frames in World Space (relative to actual world of that frame)
            # We want to see if the object and camera match the points
            rr.log(f"world/frames/{name}/camera", rr.Transform3D(translation=T_WC[:3, 3], mat3x3=T_WC[:3, :3]))
            rr.log(f"world/frames/{name}/object", rr.Transform3D(translation=T_WO[:3, 3], mat3x3=T_WO[:3, :3]))
            
    log_pcd_and_frames(cfg.init_frame, T_WC_ref, T_WO_ref, K_ref, img_ref, mask_ref, "gt_ref")
    log_pcd_and_frames(cfg.target_frame, T_WC_tgt, T_WO_tgt, K_tgt, img_tgt, mask_tgt, "gt_tgt")

    H, W = img_ref.shape[:2]
    K_ref_t = torch.from_numpy(K_ref).float().to(device)
    T_CO_ref_t = torch.from_numpy(T_CO_ref).float().to(device)
    K_tgt_t = torch.from_numpy(K_tgt).float().to(device)
    T_CO_tgt_t = torch.from_numpy(T_CO_tgt).float().to(device)
    
    image_ref_t = torch.from_numpy(img_ref).float().to(device).permute(2, 0, 1) / 255.0
    image_ref_gray = image_ref_t.mean(0)
    image_tgt_t = torch.from_numpy(img_tgt).float().to(device).permute(2, 0, 1) / 255.0
    image_tgt_gray = image_tgt_t.mean(0)
    mask_ref_t = (torch.from_numpy(mask_ref).to(device) > 0)
    
    # Normal alignment for MoGE
    print("Initializing GS and aligning normals...")
    depth_ref_normal = normal_from_depth_image(torch.from_numpy(depth_ref).float().to(device), K_ref_t).permute(2, 0, 1)
    nm_t = torch.from_numpy(normal_ref).float().to(device).permute(2, 0, 1)
    best_dot = -1.0; best_flip = (1, 1, 1)
    ref_norm_unit = F.normalize(depth_ref_normal, dim=0)
    for fx in [1, -1]:
        for fy in [1, -1]:
            for fz in [1, -1]:
                fvec = torch.tensor([fx, fy, fz], device=device).view(3, 1, 1)
                nm_f = F.normalize(nm_t * fvec, dim=0)
                dot = (ref_norm_unit * nm_f).sum(0)[mask_ref_t].mean().item()
                if dot > best_dot: best_dot = dot; best_flip = (fx, fy, fz)
    
    normal_ref_corr = normal_ref * np.array(best_flip).reshape(1, 1, 3)
    init_normal = depth_ref_normal.permute(1, 2, 0).cpu().numpy() if cfg.init_from_depth_normal else normal_ref_corr
    
    gsp = GaussianSuperPrimitive(img_ref, mask_ref, depth_ref, T_CO_ref, K_ref, normal=init_normal)
    gs_params = gsp.gs_params
    normal_moge_t = torch.from_numpy(normal_ref_corr).float().to(device)
    
    # Ray parameterization setup (needed for NCC even if not using viewray for GS means)
    with torch.no_grad():
        T_OC_ref_t = torch.inverse(T_CO_ref_t)
        R_OC = T_OC_ref_t[:3, :3]
        t_OC = T_OC_ref_t[:3, 3]
        means_C = (T_CO_ref_t[:3, :3] @ gs_params.means.T).T + T_CO_ref_t[:3, 3]
        init_distances = torch.norm(means_C, dim=1)
        ray_dirs_C = means_C / (init_distances.unsqueeze(1) + 1e-8)
        ray_dirs_O = (R_OC @ ray_dirs_C.T).T

    # Optimization setup
    quats = torch.nn.Parameter(gs_params.quats.clone())
    scales = torch.nn.Parameter(gs_params.scales.clone())
    colors = torch.nn.Parameter(gs_params.colors.clone())
    opacity = torch.nn.Parameter(gs_params.opacity.clone())
    
    params = [
        {'params': [quats], 'lr': cfg.opt_lr},
        # {'params': [scales], 'lr': cfg.opt_lr},
        # {'params': [colors], 'lr': cfg.opt_lr},
        {'params': [opacity], 'lr': cfg.opt_lr},
    ]
    
    if cfg.use_viewray:
        distances = torch.nn.Parameter(init_distances.clone())
        params.append({'params': [distances], 'lr': cfg.opt_lr})
    else:
        # Define means as Parameter directly
        means_param = torch.nn.Parameter(gs_params.means.clone())
        params.append({'params': [means_param], 'lr': cfg.opt_lr})
        means_init_val = gs_params.means.clone()

    optimizer = torch.optim.Adam(params)
    
    cam_ref = CameraDevice(K_ref_t, T_CO_ref_t, W, H)
    loss_history = []
    pbar = tqdm(range(cfg.opt_steps))
    
    # Initial Points Viz
    with torch.no_grad():
        if cfg.use_viewray:
            means_init = init_distances.unsqueeze(1) * ray_dirs_O + t_OC.unsqueeze(0)
        else:
            means_init = means_init_val
        pts_W_init = to_world_ref(means_init.cpu().numpy())
        rr.log("world/object_gs_before", rr.Points3D(pts_W_init, colors=gs_params.colors.cpu().numpy().clip(0,1), radii=0.001))

    # Loop
    for step in pbar:
        optimizer.zero_grad()
        if cfg.use_viewray:
            means = distances.unsqueeze(1) * ray_dirs_O + t_OC.unsqueeze(0)
        else:
            means = means_param
        
        # 1. Render at Ref View
        img_r, d_r, n_r, a_r = render_2dgs(means, quats, scales, colors, opacity, T_CO_ref_t, K_ref_t, W, H, cfg.near_plane, cfg.far_plane)
        
        # Single-view loss
        loss_sv = compute_single_view_loss(n_r, d_r, image_ref_t, cam_ref, mask_ref_t)
        
        # Normal Prior
        n_r_n = F.normalize(n_r, dim=0); n_p_n = F.normalize(normal_moge_t.permute(2, 0, 1), dim=0)
        loss_prior = F.l1_loss(n_r_n * mask_ref_t, n_p_n * mask_ref_t, reduction='sum') / (mask_ref_t.sum() + 1e-6)
        
        # Photometric Ref
        loss_rgb = F.l1_loss(img_r * mask_ref_t, image_ref_t * mask_ref_t, reduction='sum') / (mask_ref_t.sum() + 1e-6)
        
        # 2. NCC Multi-view Loss (Ref to Tgt)
        # We need normal and distance maps at ref view
        # n_r and d_r are already rendered
        T_nea_ref = T_CO_tgt_t @ T_OC_ref_t 
        
        mask_flat = (a_r[0] > 0.5).reshape(-1)
        valid_indices = torch.where(mask_flat)[0]
        if len(valid_indices) > cfg.sample_num:
            perm = torch.randperm(len(valid_indices), device=device)[:cfg.sample_num]
            valid_indices = valid_indices[perm]
        
        grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
        pixels = torch.stack([grid_x, grid_y], dim=-1).float().reshape(-1, 2)
        
        loss_ncc = torch.tensor(0.0, device=device)
        if len(valid_indices) > 0:
            ncc_val = compute_multi_view_ncc_loss(image_ref_gray, image_tgt_gray, n_r, d_r, T_nea_ref, K_ref_t, 
                                                patch_size=cfg.patch_size, valid_indices=valid_indices, pixels=pixels)
            loss_ncc = ncc_val.mean()
            
        # TV and Distance losses
        if cfg.use_viewray:
            dist_img = torch.zeros((H, W), device=device); dist_img[mask_ref_t] = distances
            loss_tv = (dist_img[:, 1:] - dist_img[:, :-1]).abs().sum() + (dist_img[1:, :] - dist_img[:-1, :]).abs().sum()
            loss_tv /= (mask_ref_t.sum() + 1e-6)
            loss_dist = torch.mean(torch.square(distances - init_distances))
        else:
            # For direct means, we can regularize distance from initial points
            loss_dist = torch.mean(torch.square(means - means_init_val))
            # TV on means directly (as a fallback)
            loss_tv = torch.tensor(0.0, device=device) # TV on unstructured points is harder, skipped for now
            
        total_loss = cfg.lambda_sv * loss_sv + cfg.lambda_prior * loss_prior + \
                     cfg.lambda_rgb * loss_rgb + cfg.lambda_ncc * loss_ncc + \
                     cfg.lambda_tv * loss_tv + cfg.lambda_dist * loss_dist
        
        total_loss.backward()
        optimizer.step()
        with torch.no_grad(): quats.data = F.normalize(quats.data, p=2, dim=-1)
        
        loss_history.append(total_loss.item())
        pbar.set_description(f"L: {total_loss.item():.4f} NCC: {loss_ncc.item():.4f} SV: {loss_sv.item():.4f}")
        
        # Viz every 50 steps
        if step % 50 == 0 or step == cfg.opt_steps - 1:
            with torch.no_grad():
                # Render Tgt view for viz
                img_t_r, d_t_r, n_t_r, a_t_r = render_2dgs(means, quats, scales, colors, opacity, T_CO_tgt_t, K_tgt_t, W, H, cfg.near_plane, cfg.far_plane)
                rr.log("gs/ref/image", rr.Image(img_r.permute(1, 2, 0).cpu().numpy().clip(0,1)))
                rr.log("gs/ref/depth", rr.Image(depth_to_rgb(d_r[0].cpu().numpy())))
                rr.log("gs/ref/normal", rr.Image(((n_r.permute(1, 2, 0).cpu().numpy()+1)/2).clip(0,1)))
                rr.log("gs/tgt/image", rr.Image(img_t_r.permute(1, 2, 0).cpu().numpy().clip(0,1)))
                rr.log("gs/tgt/depth", rr.Image(depth_to_rgb(d_t_r[0].cpu().numpy())))
                rr.log("gs/tgt/normal", rr.Image(((n_t_r.permute(1, 2, 0).cpu().numpy()+1)/2).clip(0,1)))
                
                # Consistently log all GS outputs in Ref World frame
                pts_O = means.cpu().numpy()
                pts_W = to_world_ref(pts_O)
                rr.log("world/object_gs", rr.Points3D(pts_W, colors=colors.cpu().numpy().clip(0,1), radii=0.001))

    # Final Loss plot
    fig, ax = plt.subplots(); ax.plot(loss_history); ax.set_yscale('log')
    buf = io.BytesIO(); fig.savefig(buf, format='png'); buf.seek(0)
    rr.log("opt/loss_plot", rr.Image(np.array(Image.open(buf)))); plt.close(fig)
    print("Optimization Complete.")

if __name__ == "__main__":
    cfg = tyro.cli(Config); main(cfg)
