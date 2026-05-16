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
from run_multi_view_loss import compute_multi_view_loss
from run_single_view_loss import compute_single_view_loss

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    target_frame: int = 40
    near_plane: float = 0.01
    far_plane: float = 10.0
    device: str = "cuda"
    opt_lr: float = 1e-3
    opt_steps: int = 50
    lambda_mv: float = 1.0
    lambda_sv: float = 0.5
    lambda_rgb: float = 1.0

class CameraDevice:
    def __init__(self, K, T_CW, width, height, ncc_scale=1.0):
        self.device = T_CW.device
        self.world_view_transform = T_CW.t() # [4, 4] transposed for GS convention
        self.K = K # [3, 3]
        self.width = width
        self.height = height
        self.ncc_scale = ncc_scale
        
    def get_k(self, scale=1.0):
        K = self.K.clone()
        K[0, 0] *= scale
        K[1, 1] *= scale
        K[0, 2] *= scale
        K[1, 2] *= scale
        return K

    def get_inv_k(self, scale=1.0):
        return torch.inverse(self.get_k(scale))

    def get_calib_matrix_nerf(self, scale=1.0):
        return self.get_k(scale), self.world_view_transform

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
                rrb.Horizontal(
                    rrb.Spatial2DView(name="GT Target", contents=["gt/target/image"]),
                    rrb.Spatial2DView(name="GS Image Before", contents=["gs/before/image"]),
                    rrb.Spatial2DView(name="GS Image After", contents=["gs/after/image"]),
                    name="Images"
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="GS Depth Before", contents=["gs/before/depth"]),
                    rrb.Spatial2DView(name="GS Depth After", contents=["gs/after/depth"]),
                    name="Depths"
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="GS Normal Before", contents=["gs/before/normal"]),
                    rrb.Spatial2DView(name="GS Normal After", contents=["gs/after/normal"]),
                    name="Normals"
                ),
                name="2D Comparison"
            ),
            rrb.Vertical(
                rrb.Spatial2DView(name="Loss Plot", contents=["opt/loss_plot"]),
                name="Optimization"
            ),
            rrb.Horizontal(
                rrb.Spatial3DView(name="3D Workspace", contents=["world/**"]),
                name="3D Visualization"
            )
        )
    )

def main(cfg: Config):
    rr.init("test_pgsr_loss", spawn=False)
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
    rr.send_blueprint(setup_blueprint())
    
    device = torch.device(cfg.device)
    data_dir = Path(cfg.data_root)
    
    # 1. Load data for init frame (35)
    print(f"Loading init frame {cfg.init_frame}...")
    init_stem = f"{cfg.init_frame:06d}"
    image_init = np.array(cv2.imread(str(data_dir / "images" / f"{init_stem}.png"))[..., ::-1])
    mask_init = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{init_stem}.png"), cv2.IMREAD_GRAYSCALE))
    depth_init = np.load(data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{init_stem}.npy")
    K_init = np.load(data_dir / "intrinsics" / f"{init_stem}.npy")
    T_C_O_init = load_pose_T_CO(cfg.data_root, cfg.init_frame)
    
    # Init GS
    gsp = GaussianSuperPrimitive(image_init, mask_init, depth_init, T_C_O_init, K_init)
    gs_params = gsp.gs_params
    
    # 2. Load data for target frame (40)
    print(f"Loading target frame {cfg.target_frame}...")
    target_stem = f"{cfg.target_frame:06d}"
    image_target_gt = np.array(cv2.imread(str(data_dir / "images" / f"{target_stem}.png"))[..., ::-1])
    mask_target_gt = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{target_stem}.png"), cv2.IMREAD_GRAYSCALE))
    depth_target_gt = np.load(data_dir / "depth_dyn" / f"{target_stem}.npy")
    K_target = np.load(data_dir / "intrinsics" / f"{target_stem}.npy")
    
    T_WO_target_gt = load_object_pose_world(cfg.data_root, cfg.target_frame)
    ext_file_target = data_dir / "extrinsics" / f"{target_stem}.npy"
    T_WC_target_gt = np.load(ext_file_target)
    T_CW_target_gt = np.linalg.inv(T_WC_target_gt)
    T_C_O_target_gt = T_CW_target_gt @ T_WO_target_gt

    H, W = image_target_gt.shape[:2]
    K_target_torch = torch.from_numpy(K_target).float().to(device)
    T_C_O_target_torch = torch.from_numpy(T_C_O_target_gt).float().to(device)
    K_init_torch = torch.from_numpy(K_init).float().to(device)
    T_C_O_init_torch = torch.from_numpy(T_C_O_init).float().to(device)
    
    # Prep GT visual
    rr.log("gt/target/image", rr.Image(image_target_gt))
    
    # GT Point Cloud in World Frame
    xyz_target_gt_C = unproject_depth(torch.from_numpy(depth_target_gt).float().to(device), K_target_torch, H, W).reshape(-1, 3).cpu().numpy()
    colors_target_gt = image_target_gt.reshape(-1, 3)
    valid_gt = (depth_target_gt.reshape(-1) > 0.01)
    pts_W_gt = (T_WC_target_gt[:3, :3] @ xyz_target_gt_C[valid_gt].T).T + T_WC_target_gt[:3, 3]
    rr.log("world/gt_pc", rr.Points3D(pts_W_gt, colors=colors_target_gt[valid_gt], radii=0.001))

    # Cameras
    rr.log("world/camera_target", rr.Transform3D(mat3x3=T_WC_target_gt[:3,:3], translation=T_WC_target_gt[:3,3]))
    rr.log("world/camera_target", rr.Pinhole(image_from_camera=K_target, width=W, height=H))
    
    # GS Before Rendering & PC
    with torch.no_grad():
        img_pre, depth_pre, normal_pre, alpha_pre = render_2dgs(
            gs_params.means, gs_params.quats, gs_params.scales, 
            gs_params.colors, gs_params.opacity, 
            T_C_O_target_torch, K_target_torch, W, H
        )
        
        # Log Initial GS Point Cloud in World Frame
        T_WO_init = load_object_pose_world(cfg.data_root, cfg.init_frame)
        pts_O_pre = gs_params.means.cpu().numpy()
        colors_pre = gs_params.colors.cpu().numpy().clip(0, 1)
        pts_W_pre = (T_WO_init[:3, :3] @ pts_O_pre.T).T + T_WO_init[:3, 3]
        rr.log("world/gs_pc_before", rr.Points3D(pts_W_pre, colors=colors_pre, radii=0.001))

    rr.log("gs/before/image", rr.Image(img_pre.permute(1, 2, 0).cpu().numpy().clip(0,1)))
    rr.log("gs/before/depth", rr.Image(depth_to_rgb(depth_pre[0].cpu().numpy())))
    rr.log("gs/before/normal", rr.Image(((normal_pre.permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0,1)))

    # 4. Define optimization
    means = torch.nn.Parameter(gs_params.means.clone())
    quats = torch.nn.Parameter(gs_params.quats.clone())
    scales = torch.nn.Parameter(gs_params.scales.clone())
    colors = torch.nn.Parameter(gs_params.colors.clone())
    opacity = torch.nn.Parameter(gs_params.opacity.clone())
    
    optimizer = torch.optim.Adam([
        {'params': [means], 'lr': cfg.opt_lr},
        {'params': [quats], 'lr': cfg.opt_lr * 0.1},
        {'params': [scales], 'lr': cfg.opt_lr},
        {'params': [colors], 'lr': cfg.opt_lr},
        {'params': [opacity], 'lr': cfg.opt_lr},
    ])
    
    # Camera objects for losses
    cam_init = CameraDevice(K_init_torch, T_C_O_init_torch, W, H)
    cam_target = CameraDevice(K_target_torch, T_C_O_target_torch, W, H)
    
    image_init_t = torch.from_numpy(image_init).float().to(device).permute(2, 0, 1) / 255.0
    image_target_t = torch.from_numpy(image_target_gt).float().to(device).permute(2, 0, 1) / 255.0
    mask_target_t = (torch.from_numpy(mask_target_gt).to(device) > 0)
    image_target_t = image_target_t * mask_target_t.unsqueeze(0)
    
    # Pre-calculate pixel indices for LNCC
    grid_y, grid_x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    all_pixels = torch.stack([grid_x, grid_y], dim=-1).float().reshape(-1, 2)
    
    print("Starting optimization...")
    pbar = tqdm(range(cfg.opt_steps))
    loss_history = []
    for step in pbar:
        optimizer.zero_grad()
        
        # Render at target view
        img_tgt, depth_tgt, normal_tgt, alpha_tgt = render_2dgs(
            means, quats, scales, colors, opacity, 
            T_C_O_target_torch, K_target_torch, W, H
        )
        
        # Render at init view (needed for multi-view loss if we use it as reference)
        img_ref, depth_ref, normal_ref, alpha_ref = render_2dgs(
            means, quats, scales, colors, opacity, 
            T_C_O_init_torch, K_init_torch, W, H
        )
        
        # 1. RGB Loss (L1) at target view
        loss_rgb = F.l1_loss(img_tgt, image_target_t)
        
        # 2. Multi-view Loss (LNCC)
        # We'll use target as reference and init as neighbor
        ref_gray = img_tgt.mean(dim=0, keepdim=True)
        nea_gray = image_init_t.mean(dim=0, keepdim=True)
        # We need distance and normal maps for the reference view
        # compute_multi_view_loss expects distance [1, H, W] and normal [3, H, W]
        # It also needs pixels. We only use pixels within the mask.
        mask_tgt_t = (torch.from_numpy(mask_target_gt).to(device) > 0).reshape(-1)
        pixels_masked = all_pixels[mask_tgt_t]
        
        # distance in pgsr sense is the 'd' such that n^T p + d = 0? 
        # Actually run_multi_view_loss uses ref_local_d as the distance from origin.
        # Let's derive it from depth_tgt
        # Wait, run_multi_view_loss.py: 130: ref_local_n[:, :, None] / ref_local_d[..., None, None]
        # This is for planar warping. We need distance to plane.
        # Let's use the actual GS means to get n and d.
        
        # Calculate n_C and d for each GS in target view
        R_CO = T_C_O_target_torch[:3, :3]
        t_CO = T_C_O_target_torch[:3, 3]
        R_OG = quaternion_to_matrix(quats)
        n_O = R_OG[:, :, 2]
        n_C_gs = torch.einsum('ij,nj->ni', R_CO, n_O)
        p_C_gs = torch.einsum('ij,nj->ni', R_CO, means) + t_CO
        d_gs = -torch.sum(n_C_gs * p_C_gs, dim=1, keepdim=True)
        
        # But compute_multi_view_loss wants maps. Let's render them.
        from gs_to_planar_patch import render_custom_attribute
        # render_custom_attribute(gs_params, attr, T_CW, K, width, height)
        # We need a dummy object for gs_params to pass to the helper
        class DummyGS: pass
        gs_dummy = DummyGS()
        gs_dummy.means = means; gs_dummy.quats = quats; gs_dummy.scales = scales; gs_dummy.opacity = opacity
        
        n_map, _ = render_custom_attribute(gs_dummy, n_C_gs, T_C_O_target_torch, K_target_torch, W, H)
        d_map_3ch, _ = render_custom_attribute(gs_dummy, d_gs.repeat(1, 3), T_C_O_target_torch, K_target_torch, W, H)
        d_map = d_map_3ch[0:1]
        
        # Ensure d_map is safe for division
        d_map = torch.where(torch.abs(d_map) < 1e-4, 1e-4 * torch.sign(d_map + 1e-8), d_map)
        
        ncc, ncc_mask = compute_multi_view_loss(
            ref_image_gray=ref_gray,
            nea_image_gray=nea_gray,
            ref_normal=n_map,
            ref_distance=d_map,
            ref_camera=cam_target,
            nea_camera=cam_init,
            pixels=pixels_masked,
            valid_indices=mask_tgt_t
        )
        loss_mv = ncc[ncc_mask].mean() if ncc_mask.any() else torch.tensor(0.0, device=device)
        
        # 3. Single-view Loss
        loss_sv = compute_single_view_loss(
            rendered_normal=normal_tgt,
            plane_depth=depth_tgt, # or d_map? run_single_view_loss uses rendered depth
            gt_rgb=image_target_t,
            camera=cam_target
        )
        
        total_loss = cfg.lambda_rgb * loss_rgb + cfg.lambda_mv * loss_mv + cfg.lambda_sv * loss_sv
        loss_history.append(total_loss.item())
        
        total_loss.backward()
        optimizer.step()
        
        # Constraints
        with torch.no_grad():
            quats.data = F.normalize(quats.data, p=2, dim=-1)
        
        pbar.set_description(f"Loss: {total_loss.item():.4f} (RGB: {loss_rgb.item():.4f}, MV: {loss_mv.item():.4f}, SV: {loss_sv.item():.4f})")

    # 5. After Optimization Rendering
    print("Finalizing...")
    with torch.no_grad():
        img_post, depth_post, normal_post, alpha_post = render_2dgs(
            means, quats, scales, colors, opacity, 
            T_C_O_target_torch, K_target_torch, W, H
        )
    rr.log("gs/after/image", rr.Image(img_post.permute(1, 2, 0).cpu().numpy().clip(0,1)))
    rr.log("gs/after/depth", rr.Image(depth_to_rgb(depth_post[0].cpu().numpy())))
    rr.log("gs/after/normal", rr.Image(((normal_post.permute(1, 2, 0).cpu().numpy() + 1) / 2).clip(0,1)))
    
    # Log loss plot
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
    
    # Final GS Point Cloud in World Frame
    pts_O = means.detach()
    pts_W_post = (T_WO_target_gt[:3,:3] @ pts_O.cpu().numpy().T).T + T_WO_target_gt[:3,3]
    colors_post = colors.detach().cpu().numpy().clip(0,1)
    rr.log("world/gs_pc_after", rr.Points3D(pts_W_post, colors=colors_post, radii=0.001))

    print("Done.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
