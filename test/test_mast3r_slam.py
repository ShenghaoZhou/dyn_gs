import os
import sys
import numpy as np
import cv2
import torch
import rerun as rr
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import time
import logging
import multiprocessing

# Add project root and MASt3R-SLAM to sys.path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
mast3r_path = current_dir / "third_party" / "MASt3R-SLAM"
sys.path.append(str(mast3r_path))

import lietorch
from mast3r_slam.global_opt import FactorGraph
from mast3r_slam.config import load_config, config, set_global_config
from mast3r_slam.frame import Mode, SharedKeyframes, SharedStates, create_frame
from mast3r_slam.mast3r_utils import (
    load_mast3r,
    mast3r_inference_mono,
)
from mast3r_slam.tracker import FrameTracker

@dataclass
class Config:
    data_root: str = str(Path(__file__).parent / "data/hot3d_clips_processed")
    clip_id: str = "clip-003312"
    init_frame: int = 0
    n_frames_track: int = 30
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    no_vis: bool = False
    stride: int = 1
    mast3r_config: str = "config/base.yaml"
    ignore_background: bool = True
    background_threshold: float = 0.01

def load_object_pose_world(data_root, frame_idx):
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists(): 
        logging.warning(f"Poses file not found: {poses_file}")
        return None
    with open(poses_file, "r") as f: lines = f.readlines()
    if frame_idx >= len(lines): 
        logging.warning(f"Frame index {frame_idx} out of range (len={len(lines)})")
        return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])]) # x y z w
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4)
    T_WO[:3, :3] = r_WO; T_WO[:3, 3] = t_WO
    return T_WO

def load_frame_data(data_dir, frame_idx):
    img_path = Path(data_dir) / "images" / f"{frame_idx:06d}.png"
    mask_path = Path(data_dir) / "obj_masks" / f"{frame_idx:06d}.png"
    
    if not img_path.exists():
        img_path = Path(data_dir) / "images" / f"{frame_idx:06d}.jpg"
        if not img_path.exists():
            logging.warning(f"Image not found for frame {frame_idx} in {data_dir}/images/")
            return None
            
    img = np.array(cv2.imread(str(img_path))[..., ::-1])
    mask = np.array(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)) if mask_path.exists() else None
    
    if mask is None:
        logging.warning(f"Mask not found for frame {frame_idx} at {mask_path}")
        return None
        
    if mask is not None and mask.shape != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    intrinsics_path = Path(data_dir) / "intrinsics" / f"{frame_idx:06d}.npy"
    extrinsics_path = Path(data_dir) / "extrinsics" / f"{frame_idx:06d}.npy"
    
    if not intrinsics_path.exists():
        logging.warning(f"Intrinsics not found: {intrinsics_path}")
        return None
    if not extrinsics_path.exists():
        logging.warning(f"Extrinsics not found: {extrinsics_path}")
        return None
        
    K = np.load(intrinsics_path)
    T_CW_gt = np.load(extrinsics_path)
    T_WO_gt = load_object_pose_world(data_dir, frame_idx)
    
    return {"image": img, "mask": mask, "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": frame_idx}

def main(cfg: Config):
    logging.basicConfig(level=logging.INFO)
    if not cfg.no_vis:
        rr.init("MASt3R_SLAM_HOT3D", spawn=False)
        if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
        rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    # Load MASt3R-SLAM config
    load_config(cfg.mast3r_config)
    config["dataset"]["ignore_background"] = cfg.ignore_background
    config["dataset"]["background_threshold"] = cfg.background_threshold
    # Lower thresholds for small objects in HOT3D
    config["tracking"]["min_match_frac"] = 0.01
    config["local_opt"]["min_match_frac"] = 0.01
    config["use_calib"] = True # Use intrinsics for HOT3D
    
    device = cfg.device
    model = load_mast3r(device=device)
    
    data_dir = Path(cfg.data_root) / cfg.clip_id
    
    # Sample first frame to get correct H, W for SharedKeyframes
    fd_init = load_frame_data(data_dir, cfg.init_frame)
    if fd_init is None:
        raise ValueError(f"Could not load initial frame {cfg.init_frame}")
    
    H, W = fd_init["image"].shape[:2]
    img_size = 512 # Default MASt3R size
    
    # MASt3R uses 512 internally for the model, but create_frame resizes/crops it.
    # We use a dummy frame to get the exact dimensions used by the model.
    dummy_frame = create_frame(0, fd_init["image"] / 255.0, lietorch.Sim3.Identity(1).to(device), img_size=img_size, device=device)
    h_model, w_model = dummy_frame.img.shape[-2:]
    
    manager = multiprocessing.Manager()
    keyframes = SharedKeyframes(manager, h_model, w_model) # Match model output dimensions
    tracker = FrameTracker(model, keyframes, device)
    
    # FactorGraph needs intrinsics if use_calib is True
    # We'll update K per frame if needed, but for now we take first frame's K
    # MASt3R-SLAM expects K_frame (scaled to img_size)
    from mast3r_slam.dataloader import Intrinsics
    K_orig = fd_init["K"]
    # Intrinsics.from_calib handles the scaling. 
    # calib format: [fx, fy, cx, cy]
    calib = [K_orig[0,0], K_orig[1,1], K_orig[0,2], K_orig[1,2]]
    mast3r_intrinsics = Intrinsics.from_calib(img_size, W, H, calib)
    K_scaled = torch.from_numpy(mast3r_intrinsics.K_frame).to(device, dtype=torch.float32)
    keyframes.set_intrinsics(K_scaled)
    
    factor_graph = FactorGraph(model, keyframes, K_scaled, device)
    
    all_dist_errors = []
    all_rot_errors = []
    traj_obj_est_C, traj_obj_gt_C = [], []

    print(f"\n>>> Starting Sequential MASt3R-SLAM on {cfg.clip_id}")
    
    for i in range(cfg.n_frames_track):
        idx = cfg.init_frame + i * cfg.stride
        fd = load_frame_data(data_dir, idx)
        if fd is None: break
        
        color_rgb = fd["image"]
        mask = fd["mask"]
        K = fd["K"]
        T_WO_gt = fd["T_WO_gt"]
        T_CW_gt = fd["T_CW_gt"]
        T_CO_gt = T_CW_gt @ T_WO_gt if T_WO_gt is not None else None
        
        print(f"\n--- Processing Frame {idx} ({i+1}/{cfg.n_frames_track}) ---")
        
        # Initial pose guess
        # In object-centric frame, T_WC is Camera to Object transform (T_OC)
        # So T_WC = inv(T_CO)
        if i == 0:
            T_OC_gt_mat = np.linalg.inv(T_CO_gt)
            t = T_OC_gt_mat[:3, 3]
            q = R.from_matrix(T_OC_gt_mat[:3, :3]).as_quat() # x, y, z, w
            
            # Initialize Sim3 from parameter vector: [t, q, s]
            pose_vec = torch.zeros(1, 8, device=device)
            pose_vec[0, :3] = torch.from_numpy(t).float().to(device)
            pose_vec[0, 3:7] = torch.from_numpy(q).float().to(device)
            pose_vec[0, 7] = 1.0 # scale
            T_WC_init = lietorch.Sim3(pose_vec)
        else:
            # Use last frame's pose as guess
            T_WC_init = last_T_WC
            
        # Create Frame object
        # We need to manually inject the mask since create_frame computes it from image thresholding
        frame = create_frame(idx, color_rgb / 255.0, T_WC_init, img_size=img_size, device=device)
        
        # Inject GT mask (resized to match frame image dimensions)
        h_f, w_f = frame.img.shape[-2:]
        mask_resized = cv2.resize(mask, (w_f, h_f), interpolation=cv2.INTER_NEAREST)
        frame.mask = torch.from_numpy(mask_resized > 0).float().to(device)
        
        if i == 0:
            # Initialize
            X_init, C_init = mast3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            # No optimization for first frame
        else:
            # Track
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            
            if add_new_kf:
                print(f"[INFO] Adding new keyframe at frame {idx}")
                keyframes.append(frame)
                # Global Optimization (Sequential)
                # Add factors to all previous keyframes
                kf_idx = [len(keyframes) - 1]
                prev_idx = list(range(len(keyframes) - 1))
                factor_graph.add_factors(prev_idx, kf_idx, config["local_opt"]["min_match_frac"])
                factor_graph.solve_GN_rays()
        
        last_T_WC = frame.T_WC
        T_OC_est = frame.T_WC.matrix()[0].detach().cpu().numpy()
        T_CO_est = np.linalg.inv(T_OC_est)
        
        # Log to Rerun
        if not cfg.no_vis:
            rr.set_time("frame_idx", sequence=idx)
            rr.log("input/image", rr.Image(color_rgb))
            
            # Camera in Object Frame
            rr.log("object/camera_est", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
            traj_obj_est_C.append(T_OC_est[:3, 3])
            rr.log("object/traj_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[[0, 255, 0]], radii=0.003))
            
            if T_CO_gt is not None:
                T_OC_gt = np.linalg.inv(T_CO_gt)
                traj_obj_gt_C.append(T_OC_gt[:3, 3])
                rr.log("object/camera_gt", rr.Transform3D(mat3x3=T_OC_gt[:3, :3], translation=T_OC_gt[:3, 3]))
                rr.log("object/traj_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[255, 255, 255]], radii=0.003))

            # Reconstructed points (Object frame)
            # Show points from all keyframes
            unique_kf_idx = factor_graph.get_unique_kf_idx()
            all_pts = []
            for kf_idx_tensor in unique_kf_idx:
                kf_idx = kf_idx_tensor.item()
                kf = keyframes[kf_idx]
                if kf.X_canon is not None:
                    # X_canon is in camera frame
                    X_cam = kf.X_canon # (N, 3)
                    T_OC_kf = kf.T_WC.matrix()[0].detach().cpu().numpy()
                    X_obj = (T_OC_kf[:3, :3] @ X_cam.detach().cpu().numpy().T + T_OC_kf[:3, 3:4]).T
                    
                    # Confidence-based filtering for visualization
                    C = kf.get_average_conf().detach().cpu().numpy()
                    mask_vis = C > config["tracking"]["C_conf"]
                    if mask_vis.any():
                        all_pts.append(X_obj[mask_vis.squeeze()])
            
            if all_pts:
                pts_cat = np.concatenate(all_pts, axis=0)
                rr.log("object/points", rr.Points3D(pts_cat, radii=0.002, colors=[0, 255, 0]))

        # Metrics
        if T_CO_gt is not None:
            dist_error = np.linalg.norm(T_CO_est[:3, 3] - T_CO_gt[:3, 3])
            rot_error = np.rad2deg(np.arccos(np.clip((np.trace(T_CO_est[:3, :3] @ T_CO_gt[:3, :3].T) - 1) / 2, -1, 1)))
            all_dist_errors.append(dist_error)
            all_rot_errors.append(rot_error)
            print(f"Frame {idx} - ATE: {dist_error:.4f}m, RotErr: {rot_error:.4f}deg")

    if all_dist_errors:
        print("\n" + "="*40)
        print(f"MASt3R-SLAM Results for {cfg.clip_id}:")
        print(f"  Mean ATE: {np.mean(all_dist_errors):.4f} m")
        print(f"  Mean Rot Error: {np.mean(all_rot_errors):.2f} deg")
        print("="*40)

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
