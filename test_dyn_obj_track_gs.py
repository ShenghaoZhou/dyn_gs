import cv2
import numpy as np
import rerun as rr
import torch
from data import HOT3DDataLoader
from pathlib import Path
import tyro
from dataclasses import dataclass
from tqdm import tqdm
import torch.nn.functional as F
import trimesh

from gs_dyn_obj.grouped_gs import GaussianSuperPrimitive
from gs_dyn_obj.utils.vis import vis_2dgs_rerun

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    data_seq: str = "clip-002793"
    depth_model: str = "DA3-BASE"
    dyn_opt_steps: int = 100
    dyn_opt_lr: float = 1e-4 # Updated to match final
    dyn_opt_patience: int = 5
    dyn_opt_thresh: float = 1e-6
    dyn_opt_loss_increase_tol: float = 5e-4
    near_plane: float = 0.01
    far_plane: float = 10.0
    
def rr_log_camera(image, K, extrin, name, static=False, color=None):
    rr.log(f"{name}", rr.Transform3D(
        translation=extrin[:3, 3], mat3x3=extrin[:3, :3],
        relation=rr.TransformRelation.ChildFromParent
    ), static=static)
    if color is None:
        rr.log(f"{name}/image", rr.Pinhole(
            resolution=(image.shape[1], image.shape[0]),
            image_from_camera=K,
            camera_xyz=rr.ViewCoordinates.RDF,
        ), static=static)
    else:
        rr.log(f"{name}/image", rr.Pinhole(
            resolution=(image.shape[1], image.shape[0]),
            image_from_camera=K,
            camera_xyz=rr.ViewCoordinates.RDF,
            color=color,
        ), static=static)
    rr.log(f"{name}/image", rr.Image(
        image).compress(jpeg_quality=30), static=static)

def estimate_scale(P, V):
    # P: (N, 3) GSP points in O-space
    # V: (M, 3) Mesh vertices in O-space
    P_c = P - P.mean(axis=0)
    V_c = V - V.mean(axis=0)
    dist_P = np.linalg.norm(P_c, axis=1)
    dist_V = np.linalg.norm(V_c, axis=1)
    if len(dist_P) == 0 or len(dist_V) == 0:
        return 1.0
    return np.median(dist_V) / np.median(dist_P)

def main(cfg: Config):
    data_loader = HOT3DDataLoader(Path(cfg.data_root) / cfg.data_seq, depth_model=cfg.depth_model)
    rr.init("dynamic_object_tracking_gs")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")

    print(f"Processing sequence {cfg.data_seq} with GS tracking...")
    
    dyn_obj = None
    T_W_O_est = None
    s_align = 1.0
    gt_cam_traj = []
    est_cam_traj = []
    est_cam_traj_scaled = []
    
    mesh = data_loader.obj_mesh
    
    for idx in tqdm(range(len(data_loader))):
        rr.set_time("frame_idx", sequence=idx)
        frame = data_loader[idx]
        image = frame["image"]
        extrin = frame["extrin"]
        K = frame["K"]
        obj_mask = frame["obj_mask"]
        hand_mask = frame["hand_mask"]
        
        # Log basic data
        rr.log("image", rr.Image(image).compress(jpeg_quality=50))
        rr.log("image/mask", rr.Image(obj_mask))

        if obj_mask.sum() == 0:
            if dyn_obj is not None:
                # Still log the object-centric view if possible, or just skip
                pass
            continue

        # Initialization
        if dyn_obj is None:
            if "depth" not in frame or frame["depth"] is None:
                print(f"Skipping frame {idx} because depth is missing for initialization.")
                continue
            
            # Initialize GaussianSuperPrimitive
            # Note: We need a segmentation mask and depth
            depth = frame["depth"]
            
            # Create dyn_obj
            dyn_obj = GaussianSuperPrimitive(image, obj_mask, depth, extrin, K)
            print(f"Initialized Dynamic Object with {len(dyn_obj.gs_params.means)} Gaussians.")
            
            # Initialize estimated pose and scale
            T_W_O_gt = data_loader.get_obj_pose(idx)
            T_W_O_est = T_W_O_gt.copy()
            
            means_W = dyn_obj.gs_params.means.detach().cpu().numpy()
            T_Ogt_W = np.linalg.inv(T_W_O_gt)
            means_O = (T_Ogt_W[:3, :3] @ means_W.T).T + T_Ogt_W[:3, 3]
            s_align = estimate_scale(means_O, mesh.vertices)
            
            # Log initial mesh
            for p in ["object", "object_scaled"]:
                vertex_colors = None
                if hasattr(mesh.visual, 'vertex_colors'):
                    vertex_colors = mesh.visual.vertex_colors[:, :3]
                rr.log(f"{p}/gt_mesh", rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    triangle_indices=mesh.faces,
                    vertex_colors=vertex_colors,
                ), static=True)

            # Define a helper for logging (or just do it here)
            T_W_C = np.linalg.inv(extrin)
            T_O_C = T_Ogt_W @ T_W_C
            gt_cam_traj.append(T_O_C[:3, 3])
            est_cam_traj.append(T_O_C[:3, 3])
            est_cam_traj_scaled.append(T_O_C[:3, 3] * s_align)

            # Log initial object-centric
            for p, s, traj in [("object", 1.0, est_cam_traj), ("object_scaled", s_align, est_cam_traj_scaled)]:
                rr.log(f"{p}/gt_camera", rr.Transform3D(
                    translation=T_O_C[:3, 3], mat3x3=T_O_C[:3, :3],
                    relation=rr.TransformRelation.ChildFromParent
                ))
                rr.log(f"{p}/est_camera", rr.Transform3D(
                    translation=T_O_C[:3, 3] * s, mat3x3=T_O_C[:3, :3],
                    relation=rr.TransformRelation.ChildFromParent
                ))
                rr.log(f"{p}/est_points", rr.Points3D(means_O * s, colors=dyn_obj.gs_params.colors.detach().cpu().numpy()))
                rr.log(f"{p}/gt_traj", rr.LineStrip3D(np.array(gt_cam_traj), color=[0, 255, 0]))
                rr.log(f"{p}/est_traj", rr.LineStrip3D(np.array(traj), color=[255, 0, 0]))

            # Initial GSP log
            vis_2dgs_rerun(dyn_obj.gs_params, name="gsp/init", points_only=True)
            continue

        # Tracking (from prev frame to current frame)
        # We need the observation at the current frame
        image_track = torch.from_numpy(image).float().cuda()
        extrin_track = torch.from_numpy(extrin).float().cuda()
        K_track = torch.from_numpy(K).float().cuda()
        hand_mask_tensor = torch.from_numpy(hand_mask).bool().cuda()

        # Prepare static_bg_img
        # Using the current image as background for tracking (simplified)
        static_bg_img = image_track.permute(2, 0, 1) / 255.0
        # Optional: mask out the current object area in the background image
        # but since we want to track TO this image, maybe it's better to keep it?
        # Actually, track_to_frame_with_static_bg will blend GS over it.
        
        # Pre-capture center for pose update
        prev_center = dyn_obj.gs_params.means.mean(0).detach().cpu().numpy()

        # Optimization
        delta_pose, converged, loss = dyn_obj.track_to_frame_with_static_bg(
            image_track, extrin_track, K_track, static_bg_img,
            mask=None, # Loss mask for the whole image
            hand_mask=hand_mask_tensor,
            lr=cfg.dyn_opt_lr,
            num_steps=cfg.dyn_opt_steps,
            rr_vis=False,
            patience=cfg.dyn_opt_patience,
            convergence_threshold=cfg.dyn_opt_thresh,
            loss_increase_tol=cfg.dyn_opt_loss_increase_tol,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane
        )
        
        T_W_O_gt = data_loader.get_obj_pose(idx)

        if converged:
            # Update dyn_obj pose
            dyn_obj = dyn_obj.apply_pose(delta_pose)
            rr.log("log", rr.TextLog(f"Frame {idx}: Tracking converged with loss {loss:.4f}"))
            
            # Update estimated cumulative pose
            R = delta_pose[:3, :3].cpu().numpy()
            t = delta_pose[:3, 3].cpu().numpy()
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = (np.eye(3) - R) @ prev_center + t
            
            if T_W_O_est is None:
                # This case shouldn't happen if initialized properly, but for safety:
                T_W_O_est = T_W_O_gt.copy()
            else:
                T_W_O_est = M @ T_W_O_est
        else:
            rr.log("log", rr.TextLog(f"Frame {idx}: Tracking failed to converge or high loss {loss:.4f}", level=rr.TextLogLevel.WARN))
            
        if T_W_O_est is None:
            T_W_O_est = T_W_O_gt.copy()
            
        # Object-centric visualization
        T_W_C = np.linalg.inv(extrin) # extrin is T_C_W
        
        # GT relative camera
        T_Ogt_C = np.linalg.inv(T_W_O_gt) @ T_W_C
        gt_cam_traj.append(T_Ogt_C[:3, 3])
        
        # Estimated relative camera
        T_Oest_C = np.linalg.inv(T_W_O_est) @ T_W_C
        est_cam_traj.append(T_Oest_C[:3, 3])

        # Estimated points in object space
        means_W = dyn_obj.gs_params.means.detach().cpu().numpy()
        T_Oest_W = np.linalg.inv(T_W_O_est)
        means_Oest = (T_Oest_W[:3, :3] @ means_W.T).T + T_Oest_W[:3, 3]
        

        # Log Unscaled view
        rr.log("object/gt_camera", rr.Transform3D(
            translation=T_Ogt_C[:3, 3], mat3x3=T_Ogt_C[:3, :3],
            relation=rr.TransformRelation.ChildFromParent
        ))
        rr.log("object/est_camera", rr.Transform3D(
            translation=T_Oest_C[:3, 3], mat3x3=T_Oest_C[:3, :3],
            relation=rr.TransformRelation.ChildFromParent
        ))
        rr.log("object/est_points", rr.Points3D(means_Oest, colors=dyn_obj.gs_params.colors.detach().cpu().numpy()))
        rr.log("object/gt_traj", rr.LineStrip3D(gt_cam_traj, color=[0, 255, 0]))
        rr.log("object/est_traj", rr.LineStrip3D(est_cam_traj, color=[255, 0, 0]))

        # Log Scaled view
        means_Oest_scaled = means_Oest * s_align
        T_Oest_C_scaled = T_Oest_C.copy()
        T_Oest_C_scaled[:3, 3] *= s_align
        est_cam_traj_scaled.append(T_Oest_C_scaled[:3, 3])
        
        rr.log("object_scaled/gt_camera", rr.Transform3D(
            translation=T_Ogt_C[:3, 3], mat3x3=T_Ogt_C[:3, :3],
            relation=rr.TransformRelation.ChildFromParent
        ))
        rr.log("object_scaled/est_camera", rr.Transform3D(
            translation=T_Oest_C_scaled[:3, 3], mat3x3=T_Oest_C_scaled[:3, :3],
            relation=rr.TransformRelation.ChildFromParent
        ))
        rr.log("object_scaled/est_points", rr.Points3D(means_Oest_scaled, colors=dyn_obj.gs_params.colors.detach().cpu().numpy()))
        rr.log("object_scaled/gt_traj", rr.LineStrip3D(gt_cam_traj, color=[0, 255, 0])) # GT traj same as unscaled
        rr.log("object_scaled/est_traj", rr.LineStrip3D(est_cam_traj_scaled, color=[255, 0, 0]))

        # Log rendered image for comparison
        # Use object's own render method
        render_pkg = dyn_obj.gs_params.render(
            extrin_track, K_track, image.shape[1], image.shape[0],
            near_plane=cfg.near_plane, far_plane=cfg.far_plane,
            bg=torch.zeros(3, device="cuda")
        )
        rendered_image = render_pkg[0].permute(1, 2, 0).cpu().detach().numpy()
        rendered_image = (rendered_image.clip(0, 1) * 255).astype(np.uint8)
        rr.log("rendered_object", rr.Image(rendered_image).compress(jpeg_quality=50))

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
