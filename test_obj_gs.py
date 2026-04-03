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
from gs_dyn_obj.obj_gs import ObjectGS
from gs_dyn_obj.utils.vis import vis_2dgs_rerun
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    data_seq: str = "clip-002793"
    depth_model: str = "DA3-BASE"
    dyn_opt_steps: int = 100
    dyn_opt_lr: float = 1e-3
    near_plane: float = 0.01
    far_plane: float = 10.0
    noise_translation: float = 0.02  # 2cm noise
    noise_rotation: float = 0.05     # ~3 degrees noise

def add_noise_to_pose(T, noise_t, noise_r):
    T_noisy = T.copy()
    T_noisy[:3, 3] += np.random.normal(0, noise_t, 3)
    
    # Simple axis-angle noise
    axis = np.random.normal(0, 1, 3)
    axis /= np.linalg.norm(axis)
    angle = np.random.normal(0, noise_r)
    R_noise, _ = cv2.Rodrigues(axis * angle)
    
    T_noisy[:3, :3] = R_noise @ T[:3, :3]
    return T_noisy

def main(cfg: Config):
    data_loader = HOT3DDataLoader(Path(cfg.data_root) / cfg.data_seq, depth_model=cfg.depth_model)
    rr.init("object_gs_pose_optimization")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")

    print(f"Testing ObjectGS:optimize_pose_wrt_image on {cfg.data_seq}")
    
    obj_gs = None
    mesh = data_loader.obj_mesh
    gt_cam_traj = []
    est_cam_traj = []
    noisy_cam_traj = []

    for idx in tqdm(range(len(data_loader))):
        rr.set_time("frame_idx", sequence=idx)
        frame = data_loader[idx]
        image = frame["image"]
        T_C_W = frame["extrin"]
        K = frame["K"]
        obj_mask = frame["obj_mask"]
        hand_mask = frame["hand_mask"]
        
        if obj_mask.sum() == 0:
            continue

        T_W_O_gt = data_loader.get_obj_pose(idx)

        # Initialization
        if obj_gs is None:
            if "depth" not in frame or frame["depth"] is None:
                continue
            
            # 1. Initialize GS in World space using GaussianSuperPrimitive
            gsp = GaussianSuperPrimitive(image, obj_mask, frame["depth"], T_C_W, K)
            
            # 2. Transform GS to Object space
            T_O_W = np.linalg.inv(T_W_O_gt)
            T_O_W_t = torch.from_numpy(T_O_W).float().cuda()
            
            # Transform means
            means_O = torch.einsum('ij,nj->ni', T_O_W_t[:3, :3], gsp.gs_params.means) + T_O_W_t[:3, 3]
            
            # Transform quaternions
            quat_O_W = matrix_to_quaternion(T_O_W_t[:3, :3])
            quats_O = quaternion_multiply(quat_O_W.unsqueeze(0), gsp.gs_params.quats)
            
            # Update gs_params to be in Object space
            gsp.gs_params.means = means_O
            gsp.gs_params.quats = quats_O
            
            # 3. Create ObjectGS
            obj_gs = ObjectGS(gsp.gs_params, T_W_O_gt, obj_scale=1.0)
            
            # Log mesh once in object frame
            vertex_colors = None
            if hasattr(mesh.visual, 'vertex_colors'):
                vertex_colors = mesh.visual.vertex_colors[:, :3]
            rr.log("object/gt_mesh", rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_colors=vertex_colors,
            ), static=True)
            print(f"Initialized ObjectGS with {len(obj_gs.gs_params.means)} Gaussians in object local frame.")

        # Compute GT T_C_O
        T_C_O_gt = T_C_W @ T_W_O_gt
        
        # Create noisy initial estimate
        T_C_O_init = add_noise_to_pose(T_C_O_gt, cfg.noise_translation, cfg.noise_rotation)
        
        # Prepare data for optimization
        image_torch = torch.from_numpy(image).float().cuda()
        K_torch = torch.from_numpy(K).float().cuda()
        hand_mask_torch = torch.from_numpy(hand_mask).bool().cuda()
        
        # Optimize pose
        T_C_O_est, losses = obj_gs.optimize_wrt_image(
            image_torch, K_torch, T_C_O_init,
            static_bg_img=image_torch, # Use image as background for tracking
            hand_mask=hand_mask_torch,
            lr=cfg.dyn_opt_lr,
            num_steps=cfg.dyn_opt_steps,
            rr_vis=True
        )
        T_C_O_est_np = T_C_O_est.detach().cpu().numpy()

        # Object-centric visualization: Fixed object at origin, moving cameras
        # Camera pose in object frame is T_O_C = inv(T_C_O)
        T_O_C_gt = np.linalg.inv(T_C_O_gt)
        T_O_C_est = np.linalg.inv(T_C_O_est_np)
        T_O_C_noisy = np.linalg.inv(T_C_O_init)
        
        gt_cam_traj.append(T_O_C_gt[:3, 3])
        est_cam_traj.append(T_O_C_est[:3, 3])
        noisy_cam_traj.append(T_O_C_noisy[:3, 3])

        # Log to Rerun
        rr.log("object/gt_camera", rr.Transform3D(
            translation=T_O_C_gt[:3, 3], mat3x3=T_O_C_gt[:3, :3],
            relation=rr.TransformRelation.ChildFromParent
        ))
        rr.log("object/est_camera", rr.Transform3D(
            translation=T_O_C_est[:3, 3], mat3x3=T_O_C_est[:3, :3],
            relation=rr.TransformRelation.ChildFromParent
        ))
        rr.log("object/noisy_camera", rr.Transform3D(
            translation=T_O_C_noisy[:3, 3], mat3x3=T_O_C_noisy[:3, :3],
            relation=rr.TransformRelation.ChildFromParent
        ))
        
        rr.log("object/est_points", rr.Points3D(
            obj_gs.gs_params.means.detach().cpu().numpy(), 
            colors=obj_gs.gs_params.colors.detach().cpu().numpy()
        ))
        
        rr.log("object/gt_traj", rr.LineStrips3D(gt_cam_traj, colors=[0, 255, 0]))
        rr.log("object/est_traj", rr.LineStrips3D(est_cam_traj, colors=[255, 0, 0]))
        rr.log("object/noisy_traj", rr.LineStrips3D(noisy_cam_traj, colors=[100, 100, 100]))

        # Log errors
        t_err = np.linalg.norm(T_C_O_est_np[:3, 3] - T_C_O_gt[:3, 3])
        rr.log("opt/t_error", rr.Scalars(t_err))
        
        # Rotation error (approx)
        R_err_mat = T_C_O_est_np[:3, :3].T @ T_C_O_gt[:3, :3]
        angle_err = np.arccos(np.clip((np.trace(R_err_mat) - 1) / 2, -1, 1))
        rr.log("opt/r_error", rr.Scalars(np.degrees(angle_err)))

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
