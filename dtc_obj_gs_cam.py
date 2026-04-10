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
    noise_translation: float = 0.0  # Set to > 0 to test robustness
    noise_rotation: float = 0.0     # Set to > 0 to test robustness
    mask_loss_weight: float = 0.0 # Weight for mask loss (rendered vs GT mask)
    use_previous_pose: bool = True
    rr_vis_internal: bool = True

def add_noise_to_pose(T, noise_t, noise_r):
    T_noisy = T.copy()
    if noise_t > 0:
        T_noisy[:3, 3] += np.random.normal(0, noise_t, 3)
    
    if noise_r > 0:
        axis = np.random.normal(0, 1, 3)
        axis /= np.linalg.norm(axis)
        angle = np.random.normal(0, noise_r)
        R_noise, _ = cv2.Rodrigues(axis * angle)
        T_noisy[:3, :3] = R_noise @ T[:3, :3]
    return T_noisy

def main(cfg: Config):
    rr.init("dtc_obj_gs_cam")
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
        print(f"Logged {len(points_gt)} world GT points.")
    
    # Get all frames and sort numerically
    image_paths = sorted((data_dir / "image").glob("*.jpg"), key=lambda p: int(p.stem))
    
    obj_gs = None
    gt_cam_traj = []
    est_cam_traj = []
    
    T_C_W_prev = None
    
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
        
        # Initial initialization from the first valid mask/depth
        if obj_gs is None:
            depth_path = data_dir / "obj_depth_gt" / f"{stem}.npy"
            if not depth_path.exists():
                continue
            depth_gt = np.load(depth_path)
            if mask.sum() > 0 and depth_gt.sum() > 0:
                if cfg.backend == "2dgs":
                    gsp = GaussianSuperPrimitive(image, mask, depth_gt, T_C_W_gt, K)
                    vis_2dgs_rerun(gsp.gs_params, name="world/gs_points", points_only=True)
                else:
                    gsp = GaussianSuperPrimitive3D.from_depth(image, mask, depth_gt, T_C_W_gt, K)
                    vis_3dgs_rerun(gsp.gs_params, name="world/gs_points", points_only=True)
                
                # Wrap in ObjectGS. We treat the world space (at frame 0) as our reference frame.
                obj_gs = ObjectGS(gsp.gs_params, T_W_O=np.eye(4), obj_scale=1.0)
                
                T_C_W_prev = T_C_W_gt
                est_cam_traj.append(T_W_C_gt[:3, 3].tolist())
                
                print(f"Initialized ObjectGS ({cfg.backend}) at frame {idx}.")
                continue
            else:
                continue
        
        # Estimate pose for current frame
        if cfg.use_previous_pose:
            T_C_W_init = T_C_W_prev
        else:
            T_C_W_init = add_noise_to_pose(T_C_W_gt, cfg.noise_translation, cfg.noise_rotation)
            
        # Optimization
        image_torch = torch.from_numpy(image).float().to(cfg.device)
        K_torch = torch.from_numpy(K).float().to(cfg.device)
        mask_torch = torch.from_numpy(mask_bool).bool().to(cfg.device)
        
        # Note: optimize_wrt_image is defined in ObjectGS (gs_dyn_obj/obj_gs.py)
        # It optimizes T_C_O (camera-to-object pose). 
        # Since our GS is in world space, T_C_O is T_C_W.
        T_C_W_est_torch, losses = obj_gs.optimize_wrt_image(
            image_torch, K_torch, T_C_W_init,
            lr=cfg.opt_lr,
            num_steps=cfg.opt_steps,
            rr_vis=cfg.rr_vis_internal,
            mask=mask_torch, # We use the object mask to guide the camera pose estimation
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            backend=cfg.renderer_backend,
            mask_loss_weight=cfg.mask_loss_weight
        )
        
        T_C_W_est = T_C_W_est_torch.detach().cpu().numpy()
        T_W_C_est = np.linalg.inv(T_C_W_est)
        
        est_cam_traj.append(T_W_C_est[:3, 3].tolist())
        T_C_W_prev = T_C_W_est
        
        # Log estimated camera
        rr.log("world/camera_est", rr.Transform3D(
            translation=T_W_C_est[:3, 3],
            mat3x3=T_W_C_est[:3, :3],
            relation=rr.TransformRelation.ParentFromChild
        ))
        rr.log("world/camera_est/image", rr.Pinhole(
            resolution=(image.shape[1], image.shape[0]),
            image_from_camera=K,
            camera_xyz=rr.ViewCoordinates.RDF
        ))
        
        # Log trajectories
        rr.log("world/gt_traj", rr.LineStrips3D([gt_cam_traj], colors=[[0, 255, 0]]))
        rr.log("world/est_traj", rr.LineStrips3D([est_cam_traj], colors=[[255, 0, 0]]))
        
        # Error metrics
        t_err = np.linalg.norm(T_W_C_est[:3, 3] - T_W_C_gt[:3, 3])
        rr.log("error/t_err", rr.Scalars(t_err))
        
        # Translation distance from origin
        rr.log("traj/dist_gt", rr.Scalars(np.linalg.norm(T_W_C_gt[:3, 3])))
        rr.log("traj/dist_est", rr.Scalars(np.linalg.norm(T_W_C_est[:3, 3])))

    print("Finished processing sequence.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
