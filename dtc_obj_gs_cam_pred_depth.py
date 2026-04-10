import numpy as np
import cv2
import rerun as rr
import torch
from pathlib import Path
from tqdm import tqdm
from gs_dyn_obj.grouped_gs import GaussianSuperPrimitive, GaussianSuperPrimitive3D
from gs_dyn_obj.obj_gs import ObjectGS
from gs_dyn_obj.utils.vis import vis_2dgs_rerun, vis_3dgs_rerun
import tyro
from dataclasses import dataclass
import rerun.blueprint as rrb
import matplotlib.pyplot as plt

def depth_to_rgb(depth, min_val=0.0, max_val=0.6):
    """Consistent colormapping for depth visualization."""
    depth_norm = np.clip((depth - min_val) / (max_val - min_val), 0, 1)
    colormap = plt.get_cmap("magma")
    rgb = (colormap(depth_norm)[..., :3] * 255).astype(np.uint8)
    rgb[depth == 0] = 0
    return rgb

@dataclass
class Config:
    data_root: str = "data/dtc_sample"
    near_plane: float = 0.01
    far_plane: float = 10.0
    device: str = "cuda"
    backend: str = "2dgs" # "2dgs" or "3dgs"
    renderer_backend: str = "gsplat" # "inria" or "gsplat"
    opt_lr: float = 1e-3
    opt_steps: int = 100
    mask_loss_weight: float = 0.0 # Weight for mask loss (rendered vs GT mask)
    use_previous_pose: bool = True
    rr_vis_internal: bool = True

def main(cfg: Config):
    rr.init("dtc_obj_gs_cam_pred_depth")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")

    # Define Rerun Blueprint for UI layout
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(origin="/world", name="3D Reconstruction"),
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/image", name="Current Image"),
                rrb.Spatial2DView(origin="/opt_rendered", name="GS Render (Optimized)"),
                name="Color Comparison"
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/loss_plot", name="Optimization Loss"),
                rrb.Spatial2DView(origin="/pred_depth_vis", name="Predicted Depth"),
                name="Training Progress"
            )
        ),
        collapse_panels=True
    )
    rr.send_blueprint(blueprint)
    
    data_dir = Path(cfg.data_root)
    
    # Load intrinsics
    K = np.load(data_dir / "intrinsics.npy")
    
    # Load and log GT point cloud for the scene (if available)
    points_gt_path = data_dir / "points.npy"
    if points_gt_path.exists():
        points_gt = np.load(points_gt_path)
        rr.log("world/gt_points", rr.Points3D(points_gt, colors=[200, 200, 200]), static=True)
    
    # Get all frames and sort numerically
    image_paths = sorted((data_dir / "image").glob("*.jpg"), key=lambda p: int(p.stem))
    
    obj_gs = None
    gt_cam_traj = []
    est_cam_traj_raw = []
    est_cam_traj_scaled = []
    
    T_C_W_prev = None
    T_W_C_init_gt = None
    scale_factor = 1.0 # Default to 1.0 if no GT available
    
    print(f"Total frames: {len(image_paths)}")
    
    for idx, img_path in enumerate(tqdm(image_paths)):
        stem = img_path.stem
        rr.set_time("frame_idx", sequence=idx)
        rr.set_time("timestamp", sequence=int(stem))
        
        # Load frame data
        image = np.array(cv2.imread(str(img_path))[..., ::-1])
        mask_path = data_dir / "mask" / f"{stem}.png"
        mask = np.array(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE))
        mask_bool = (mask > 0)
        
        pose_path = data_dir / "pose" / f"{stem}.npy"
        T_W_C_gt = np.load(pose_path) # Dataset provides camera-to-world pose
        T_C_W_gt = np.linalg.inv(T_W_C_gt)
        
        gt_cam_traj.append(T_W_C_gt[:3, 3].tolist())
        
        # Log basic observation
        rr.log("image", rr.Image(image).compress(jpeg_quality=50))
        rr.log("world/camera_gt", rr.Transform3D(
            translation=T_W_C_gt[:3, 3],
            mat3x3=T_W_C_gt[:3, :3],
            relation=rr.TransformRelation.ParentFromChild
        ))
        
        # Initial initialization from the first valid mask/pred_depth
        if obj_gs is None:
            pred_depth_path = data_dir / "depth" / f"{stem}.npy"
            gt_depth_path = data_dir / "obj_depth_gt" / f"{stem}.npy"
            
            if not pred_depth_path.exists():
                continue
            
            pred_depth = np.load(pred_depth_path)
            # Resize predicted depth to match current image/mask size
            if pred_depth.shape != mask.shape:
                pred_depth = cv2.resize(pred_depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            rr.log("pred_depth_vis", rr.Image(depth_to_rgb(pred_depth, max_val=2.0)).compress(jpeg_quality=50))
            
            if mask.sum() > 0 and pred_depth.sum() > 0:
                # 1. Compute Scale Factor using GT depth (if available)
                if gt_depth_path.exists():
                    depth_gt = np.load(gt_depth_path)
                    valid_mask = (mask > 0) & (depth_gt > 0) & (pred_depth > 0)
                    if valid_mask.sum() > 0:
                        scale_factor = np.median(depth_gt[valid_mask]) / np.median(pred_depth[valid_mask])
                        print(f"Computed scale factor from first frame: {scale_factor:.4f}")
                
                # 2. Initialize GS using Predicted Depth
                if cfg.backend == "2dgs":
                    gsp = GaussianSuperPrimitive(image, mask, pred_depth, T_C_W_gt, K)
                    vis_2dgs_rerun(gsp.gs_params, name="world/gs_points", points_only=True)
                else:
                    gsp = GaussianSuperPrimitive3D.from_depth(image, mask, pred_depth, T_C_W_gt, K)
                    vis_3dgs_rerun(gsp.gs_params, name="world/gs_points", points_only=True)
                
                # Create ObjectGS
                obj_gs = ObjectGS(gsp.gs_params, T_W_O=np.eye(4), obj_scale=1.0)
                
                T_C_W_prev = T_C_W_gt
                T_W_C_init_gt = T_W_C_gt.copy()
                
                est_cam_traj_raw.append(T_W_C_gt[:3, 3].tolist())
                est_cam_traj_scaled.append(T_W_C_gt[:3, 3].tolist())
                
                print(f"Initialized ObjectGS ({cfg.backend}) with Predicted Depth at frame {idx}.")
                continue
            else:
                continue
        
        # Estimate pose for current frame
        T_C_W_init = T_C_W_prev # Always use tracking logic as requested
            
        # Optimization
        image_torch = torch.from_numpy(image).float().to(cfg.device)
        K_torch = torch.from_numpy(K).float().to(cfg.device)
        mask_torch = torch.from_numpy(mask_bool).bool().to(cfg.device)
        
        T_C_W_est_torch, losses = obj_gs.optimize_wrt_image(
            image_torch, K_torch, T_C_W_init,
            lr=cfg.opt_lr,
            num_steps=cfg.opt_steps,
            rr_vis=cfg.rr_vis_internal,
            mask=mask_torch,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            backend=cfg.renderer_backend,
            mask_loss_weight=cfg.mask_loss_weight
        )
        
        T_C_W_est = T_C_W_est_torch.detach().cpu().numpy()
        T_W_C_est = np.linalg.inv(T_C_W_est)
        
        # 3. Store and Scale Trajectories
        # Raw estimated position in "predicted depth" scale
        est_pos_raw = T_W_C_est[:3, 3]
        est_cam_traj_raw.append(est_pos_raw.tolist())
        
        # Scaled position to metric scale: 
        # Displacement from start position is scaled
        displacement = est_pos_raw - T_W_C_init_gt[:3, 3]
        scaled_pos = T_W_C_init_gt[:3, 3] + scale_factor * displacement
        est_cam_traj_scaled.append(scaled_pos.tolist())
        
        T_C_W_prev = T_C_W_est
        
        # Log estimated camera (unscaled for current coordinate system visualization)
        rr.log("world/camera_est", rr.Transform3D(
            translation=est_pos_raw,
            mat3x3=T_W_C_est[:3, :3],
            relation=rr.TransformRelation.ParentFromChild
        ))
        
        # Log scaled camera (metric)
        rr.log("world/camera_est_metric", rr.Transform3D(
            translation=scaled_pos,
            mat3x3=T_W_C_est[:3, :3],
            relation=rr.TransformRelation.ParentFromChild
        ))
        
        # Log trajectories to Rerun
        rr.log("world/gt_traj", rr.LineStrips3D([gt_cam_traj], colors=[[0, 255, 0]]))
        rr.log("world/est_traj_raw", rr.LineStrips3D([est_cam_traj_raw], colors=[[255, 0, 0]]))
        rr.log("world/est_traj_scaled", rr.LineStrips3D([est_cam_traj_scaled], colors=[[255, 165, 0]])) # Orange
        
        # Error metrics relative to scaled trajectory
        t_err_scaled = np.linalg.norm(scaled_pos - T_W_C_gt[:3, 3])
        rr.log("error/t_err_scaled", rr.Scalars(t_err_scaled))

    print(f"Finished processing sequence. scale_factor={scale_factor:.4f}")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
