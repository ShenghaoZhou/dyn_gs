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
import matplotlib.pyplot as plt
import tyro
from dataclasses import dataclass
import io
from PIL import Image
from gs_dyn_obj.utils.init import d2n_tblr, unproject_depth
from scipy.spatial.transform import Rotation as R
from run_single_view_loss import compute_single_view_loss, normal_from_depth_image

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    frame_idx: int = 35
    near_plane: float = 0.01
    far_plane: float = 10.0
    device: str = "cuda"
    opt_lr: float = 1e-3
    opt_steps: int = 100
    
    # Loss weights
    lambda_sv: float = 1.0     # consistency between rendered normal and depth-derived normal
    lambda_prior: float = 1.0  # consistency with MoGE normal prior
    lambda_rgb: float = 1.0    # photometric loss
    
    # MOGE options
    use_moge_depth: bool = False
    init_from_depth_normal: bool = False

class CameraDevice:
    def __init__(self, K, T_CW, width, height):
        self.device = T_CW.device
        self.world_view_transform = T_CW.t() # [4, 4] transposed for GS convention in some places, 
                                             # but run_single_view_loss uses it as extrinsics
        self.K = K # [3, 3]
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
        # run_single_view_loss uses:
        # intrinsic_matrix, extrinsic_matrix = camera.get_calib_matrix_nerf(scale=scale)
        # depth_normal = normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix)
        # Note: normal_from_depth_image in run_single_view_loss doesn't actually use extrinsic_matrix
        # unless provided. It computes normals in camera space by default.
        return self.get_k(scale), self.world_view_transform.t() # return back to standard 4x4 if needed

def depth_to_rgb(depth, min_val=0.0, max_val=2.0):
    """Consistent colormapping for depth visualization."""
    depth_norm = np.clip((depth - min_val) / (max_val - min_val), 0, 1)
    colormap = plt.get_cmap("magma")
    rgb = (colormap(depth_norm)[..., :3] * 255).astype(np.uint8)
    return rgb

def load_object_pose_world(data_root, frame_idx):
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists():
        return None
    with open(poses_file, "r") as f:
        lines = f.readlines()
    if frame_idx >= len(lines):
        return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])]) # x y z w
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4)
    T_WO[:3, :3] = r_WO
    T_WO[:3, 3] = t_WO
    return T_WO

def load_pose_T_CO(data_root, frame_idx):
    T_WO = load_object_pose_world(data_root, frame_idx)
    if T_WO is None:
        return None
    ext_file = Path(data_root) / "extrinsics" / f"{frame_idx:06d}.npy"
    if not ext_file.exists():
        return T_WO 
    T_WC = np.load(ext_file) 
    T_CW = np.linalg.inv(T_WC)
    T_CO = T_CW @ T_WO
    return T_CO

def setup_blueprint():
    import rerun.blueprint as rrb
    return rrb.Blueprint(
        rrb.Tabs(
            rrb.Vertical(
                rrb.Grid(
                    # GT / Priors Row
                    rrb.Spatial2DView(name="GT Image (Masked)", contents=["gt/masked_image"]),
                    rrb.Spatial2DView(name="Depth Prior (Masked)", contents=["gt/masked_depth_prior"]),
                    rrb.Spatial2DView(name="Normal Prior (Masked)", contents=["gt/masked_normal_prior"]),
                    
                    # Before Opt Row
                    rrb.Spatial2DView(name="GS Image (Before)", contents=["gs/before/image"]),
                    rrb.Spatial2DView(name="GS Depth (Before)", contents=["gs/before/depth"]),
                    rrb.Spatial2DView(name="GS Normal (Before)", contents=["gs/before/normal"]),
                    
                    # After Opt Row
                    rrb.Spatial2DView(name="GS Image (After)", contents=["gs/after/image"]),
                    rrb.Spatial2DView(name="GS Depth (After)", contents=["gs/after/depth"]),
                    rrb.Spatial2DView(name="GS Normal (After)", contents=["gs/after/normal"]),
                    
                    grid_columns=3,
                    name="2D Comparison Grid"
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Loss Plot", contents=["opt/loss_plot"]),
                ),
                name="2D Comparison"
            ),
            rrb.Horizontal(
                rrb.Spatial3DView(name="3D Visualization", contents=["world/**"]),
                name="3D Comparison"
            )
        )
    )

def main(cfg: Config):
    rr.init("exp_gs_improve_normal", spawn=False)
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    rr.send_blueprint(setup_blueprint())
    
    device = torch.device(cfg.device)
    data_dir = Path(cfg.data_root)
    stem = f"{cfg.frame_idx:06d}"
    
    # 1. Load Data
    print(f"Loading data for frame {cfg.frame_idx}...")
    image = np.array(cv2.imread(str(data_dir / "images" / f"{stem}.png"))[..., ::-1])
    mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
    mask_bool = (mask > 0)
    
    # MoGE Priors
    if cfg.use_moge_depth:
        depth_moge = np.load(data_dir / "moge_depth" / f"{stem}.npy")
    else:
        depth_moge = np.load(data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{stem}.npy")
        
    normal_moge = np.load(data_dir / "moge_normal" / f"{stem}.npy")
    if normal_moge.shape[0] == 3 and normal_moge.shape[1] != 3:
        normal_moge = normal_moge.transpose(1, 2, 0)
    
    K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
    T_CO = load_pose_T_CO(cfg.data_root, cfg.frame_idx)
    T_WC = np.load(data_dir / "extrinsics" / f"{stem}.npy")
    T_WO = load_object_pose_world(cfg.data_root, cfg.frame_idx)
    
    # GT Depth for evaluation/visualization
    depth_gt_file = data_dir / "depth_dyn" / f"{stem}.npy"
    if depth_gt_file.exists():
        depth_gt = np.load(depth_gt_file)
    else:
        depth_gt = depth_moge
        
    H, W = image.shape[:2]
    K_torch = torch.from_numpy(K).float().to(device)
    T_CO_torch = torch.from_numpy(T_CO).float().to(device)
    image_t = torch.from_numpy(image).float().to(device).permute(2, 0, 1) / 255.0
    normal_moge_t = torch.from_numpy(normal_moge).float().to(device)
    mask_t = (torch.from_numpy(mask).to(device) > 0)
    
    # 2. Alignment Check and GS Initialization
    print("Analyzing normal alignment...")
    depth_ref_normal = normal_from_depth_image(
        torch.from_numpy(depth_moge).float().to(device),
        K_torch
    ).permute(2, 0, 1) # [3, H, W] in camera space
    
    nm_t = torch.from_numpy(normal_moge).float().to(device).permute(2, 0, 1)
    
    # Find best flip for MoGE to match depth-derived normals
    best_dot_depth = -1.0
    best_flip_depth = (1, 1, 1)
    ref_norm = F.normalize(depth_ref_normal, dim=0)
    for fx in [1, -1]:
        for fy in [1, -1]:
            for fz in [1, -1]:
                fvec = torch.tensor([fx, fy, fz], device=device).view(3, 1, 1)
                nm_f = F.normalize(nm_t * fvec, dim=0)
                dot = (ref_norm * nm_f).sum(dim=0)[mask_t].mean().item()
                if dot > best_dot_depth:
                    best_dot_depth = dot
                    best_flip_depth = (fx, fy, fz)
    
    print(f"Detected MoGE vs Depth Flip: {best_flip_depth}, Dot: {best_dot_depth:.4f}")
    normal_moge_corrected = normal_moge * np.array(best_flip_depth).reshape(1, 1, 3)
    
    print("Initializing GS...")
    if cfg.init_from_depth_normal:
        # Use depth-derived normal directly
        init_normal = depth_ref_normal.permute(1, 2, 0).cpu().numpy()
        print("Using depth-derived normal for GS initialization.")
    else:
        init_normal = normal_moge_corrected
        print("Using MoGE normal for GS initialization.")

    gsp = GaussianSuperPrimitive(image, mask, depth_moge, T_CO, K, normal=init_normal)
    gs_params = gsp.gs_params
    
    # Check if GS normals are flipped relative to MoGE (shouldn't happen with correct transform)
    with torch.no_grad():
        img_pre, depth_pre, normal_pre, alpha_pre = render_2dgs(
            gs_params.means, gs_params.quats, gs_params.scales, 
            gs_params.colors, gs_params.opacity, 
            T_CO_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane
        )
        
        nm_corr_t = torch.from_numpy(normal_moge_corrected).float().to(device).permute(2, 0, 1)
        dot_init = (F.normalize(normal_pre, dim=0) * F.normalize(nm_corr_t, dim=0)).sum(0)[mask_t].mean().item()
        print(f"GS vs Corrected MoGE Dot: {dot_init:.4f}")
        
        if dot_init < 0:
            print("Detected GS Normal flip! Correcting...")
            normal_moge_corrected *= -1
            normal_moge_t = torch.from_numpy(normal_moge_corrected).float().to(device)
        else:
            normal_moge_t = torch.from_numpy(normal_moge_corrected).float().to(device)

    # 3. Final Visualization Setup
    with torch.no_grad():
        img_pre, depth_pre, normal_pre, alpha_pre = render_2dgs(
            gs_params.means, gs_params.quats, gs_params.scales, 
            gs_params.colors, gs_params.opacity, 
            T_CO_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane
        )

        pts_O_pre = gs_params.means.cpu().numpy()
        pts_W_pre = (T_WO[:3, :3] @ pts_O_pre.T).T + T_WO[:3, 3]
        colors_pre = gs_params.colors.cpu().numpy().clip(0, 1)
        rr.log("world/object_gs_before", rr.Points3D(pts_W_pre, colors=colors_pre, radii=0.001))

    # Log GT / Priors (Masked)
    masked_image = image.copy()
    masked_image[~mask_bool] = 0
    rr.log("gt/masked_image", rr.Image(masked_image))
    
    depth_moge_vis = depth_to_rgb(depth_moge)
    depth_moge_vis[~mask_bool] = 0
    rr.log("gt/masked_depth_prior", rr.Image(depth_moge_vis))
    
    normal_moge_vis = ((normal_moge + 1) / 2).clip(0, 1)
    normal_moge_vis[~mask_bool] = 0
    rr.log("gt/masked_normal_prior", rr.Image((normal_moge_vis * 255).astype(np.uint8)))
    
    # Log Initial Renders
    rr.log("gs/before/image", rr.Image(img_pre.permute(1, 2, 0).cpu().numpy().clip(0,1)))
    rr.log("gs/before/depth", rr.Image(depth_to_rgb(depth_pre[0].cpu().numpy())))
    rr.log("gs/before/normal", rr.Image(((normal_pre.permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0,1)))

    # GT Point Cloud
    xyz_gt_C = unproject_depth(torch.from_numpy(depth_gt).float().to(device), K_torch, H, W).reshape(-1, 3).cpu().numpy()
    valid_gt = (depth_gt.reshape(-1) > 0.01) & (mask.reshape(-1) > 0)
    pts_W_gt = (T_WC[:3, :3] @ xyz_gt_C[valid_gt].T).T + T_WC[:3, 3]
    rr.log("world/object_gt", rr.Points3D(pts_W_gt, colors=image.reshape(-1, 3)[valid_gt], radii=0.001))

    # 4. Optimization
    print("Starting GS optimization...")
    means = torch.nn.Parameter(gs_params.means.clone())
    quats = torch.nn.Parameter(gs_params.quats.clone())
    scales = torch.nn.Parameter(gs_params.scales.clone())
    colors = torch.nn.Parameter(gs_params.colors.clone())
    opacity = torch.nn.Parameter(gs_params.opacity.clone())
    
    optimizer = torch.optim.Adam([
        {'params': [means], 'lr': cfg.opt_lr},
        {'params': [quats], 'lr': cfg.opt_lr * 0.1},
        # {'params': [scales], 'lr': cfg.opt_lr},
        # {'params': [colors], 'lr': cfg.opt_lr},
        # {'params': [opacity], 'lr': cfg.opt_lr},
    ])
    
    cam = CameraDevice(K_torch, T_CO_torch, W, H)
    
    loss_history = []
    pbar = tqdm(range(cfg.opt_steps))
    for step in pbar:
        optimizer.zero_grad()
        
        img_tgt, depth_tgt, normal_tgt, alpha_tgt = render_2dgs(
            means, quats, scales, colors, opacity, 
            T_CO_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane
        )
        
        # Ensure mask is 3D for broadcasting: [1, H, W]
        mask_3d = mask_t.unsqueeze(0)
        
        # 1. Single-view consistency loss
        loss_sv = compute_single_view_loss(
            rendered_normal=normal_tgt,
            plane_depth=depth_tgt,
            gt_rgb=image_t,
            camera=cam,
            mask=mask_t
        )
        
        # 2. Normal Prior Loss
        # Normalize both for consistent direction loss
        norm_tgt_n = F.normalize(normal_tgt, dim=0)
        norm_prior_n = F.normalize(normal_moge_t.permute(2, 0, 1), dim=0)
        diff_prior = F.l1_loss(norm_tgt_n * mask_3d, norm_prior_n * mask_3d, reduction='none')
        loss_prior = diff_prior.sum() / (mask_3d.sum() + 1e-6)
        
        # 3. Photometric Loss
        diff_rgb = F.l1_loss(img_tgt * mask_3d, image_t * mask_3d, reduction='none')
        loss_rgb = diff_rgb.sum() / (mask_3d.sum() + 1e-6)
        
        total_loss = cfg.lambda_sv * loss_sv + cfg.lambda_prior * loss_prior + cfg.lambda_rgb * loss_rgb
        
        total_loss.backward()
        optimizer.step()
        
        # Normalize quaternions
        with torch.no_grad():
            quats.data = F.normalize(quats.data, p=2, dim=-1)
            
        loss_history.append(total_loss.item())
        pbar.set_description(f"Loss: {total_loss.item():.4f} (SV: {loss_sv.item():.4f}, Prior: {loss_prior.item():.4f}, RGB: {loss_rgb.item():.4f})")

    # 5. Result Visualization
    print("Finalizing results...")
    with torch.no_grad():
        img_post, depth_post, normal_post, alpha_post = render_2dgs(
            means, quats, scales, colors, opacity, 
            T_CO_torch, K_torch, W, H, cfg.near_plane, cfg.far_plane
        )
        
        pts_O_post = means.cpu().numpy()
        pts_W_post = (T_WO[:3, :3] @ pts_O_post.T).T + T_WO[:3, 3]
        colors_post = colors.cpu().numpy().clip(0, 1)
        rr.log("world/object_gs_after", rr.Points3D(pts_W_post, colors=colors_post, radii=0.001))

    rr.log("gs/after/image", rr.Image(img_post.permute(1, 2, 0).cpu().numpy().clip(0,1)))
    rr.log("gs/after/depth", rr.Image(depth_to_rgb(depth_post[0].cpu().numpy())))
    rr.log("gs/after/normal", rr.Image(((normal_post.permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0,1)))

    # Loss plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(loss_history)
    ax.set_title("Optimization Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Total Loss")
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    rr.log("opt/loss_plot", rr.Image(np.array(Image.open(buf))))
    plt.close(fig)

    print("Done.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
