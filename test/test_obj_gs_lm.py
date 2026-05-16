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
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    data_seq: str = "clip-002793"
    depth_model: str = "DA3-BASE"
    dyn_opt_steps: int = 15
    damping: float = 1.0
    res_scale: int = 4
    near_plane: float = 0.01
    far_plane: float = 10.0
    noise_translation: float = 0.02  # 2cm noise
    noise_rotation: float = 0.05     # ~3 degrees noise
    use_gt_depth: bool = False

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
    if cfg.use_gt_depth:
        cfg.depth_model = "GT"
    data_loader = HOT3DDataLoader(Path(cfg.data_root) / cfg.data_seq, depth_model=cfg.depth_model)
    rr.init("object_gs_pose_optimization_lm")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")

    print(f"Testing ObjectGS:optimize_wrt_image_lm on {cfg.data_seq}")
    
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
        
        if obj_mask.sum() == 0:
            continue

        T_W_O_gt = data_loader.get_obj_pose(idx)

        # Initialization
        if obj_gs is None:
            if "depth" not in frame or frame["depth"] is None:
                continue
            
            gsp = GaussianSuperPrimitive(image, obj_mask, frame["depth"], T_C_W, K)
            
            T_O_W = np.linalg.inv(T_W_O_gt)
            T_O_W_t = torch.from_numpy(T_O_W).float().cuda()
            
            means_O = torch.einsum('ij,nj->ni', T_O_W_t[:3, :3], gsp.gs_params.means) + T_O_W_t[:3, 3]
            quat_O_W = matrix_to_quaternion(T_O_W_t[:3, :3])
            quats_O = quaternion_multiply(quat_O_W.unsqueeze(0), gsp.gs_params.quats)
            
            gsp.gs_params.means = means_O
            gsp.gs_params.quats = quats_O
            
            obj_gs = ObjectGS(gsp.gs_params, T_W_O_gt, obj_scale=1.0)
            print(f"Initialized ObjectGS with {len(obj_gs.gs_params.means)} Gaussians.")

        # Compute GT T_C_O
        T_C_O_gt = T_C_W @ T_W_O_gt
        T_C_W_torch = torch.from_numpy(T_C_W).float().cuda()
        
        # Optimize pose using LM
        image_torch = torch.from_numpy(image).float().cuda()
        K_torch = torch.from_numpy(K).float().cuda()
        
        T_W_O_est, info = obj_gs.optimize_wrt_image_lm(
            T_C_W_torch, image_torch, K_torch,
            update_ref=True,
            pyramid_levels=[(cfg.res_scale, cfg.dyn_opt_steps)],
            damping=cfg.damping,
            rr_vis=True
        )
        
        T_W_O_est_np = T_W_O_est.detach().cpu().numpy() if torch.is_tensor(T_W_O_est) else T_W_O_est
        T_C_O_est_np = T_C_W @ T_W_O_est_np
        
        # Error metrics
        t_err = np.linalg.norm(T_C_O_est_np[:3, 3] - T_C_O_gt[:3, 3])
        rr.log("opt/t_error", rr.Scalars(t_err))

        T_O_C_gt = np.linalg.inv(T_C_O_gt)
        T_O_C_est = np.linalg.inv(T_C_O_est_np)
        
        gt_cam_traj.append(T_O_C_gt[:3, 3])
        est_cam_traj.append(T_O_C_est[:3, 3])

        # Rotation error
        R_err_mat = T_C_O_est_np[:3, :3].T @ T_C_O_gt[:3, :3]
        angle_err = np.arccos(np.clip((np.trace(R_err_mat) - 1) / 2, -1, 1))
        rr.log("opt/r_error", rr.Scalars(np.degrees(angle_err)))
        
        print(f"Frame {idx} - T_err: {t_err:.4f}m, R_err: {np.degrees(angle_err):.2f} deg")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
