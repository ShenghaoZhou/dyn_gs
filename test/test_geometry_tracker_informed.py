import numpy as np
import cv2
import rerun as rr
import torch
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import rerun.blueprint as rrb

from geometric_tracker import GeometricTracker

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 0
    n_frames: int = 100
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    
    # Tracking Parameters
    grid_spacing: int = 12
    feature_type: str = "grid" # "grid", "orb", or "gftt"
    n_features: int = 1000
    max_pnp_points: int = -1
    ransac_thresh: float = 1.0
    kf_disparity_thresh: float = 20.0
    kf_min_interval: int = 8
    kf_max_interval: int = 15
    max_keyframes: int = 20
    triangulate_thresh: int = 5
    triangulate: bool = False
    
    do_refine: bool = True
    no_vis: bool = False
    use_gt_depth: bool = False
    
    # Informed Parameters
    use_informed_filtering: bool = True
    skip_pnp: bool = False
    informed_thresh: float = 50.0
    guess_type: str = "GT-Cam" # "GT-Cam", "GT-Cam+CV-Obj"

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

def load_frame_data(data_dir, frame_idx, use_gt_depth=False):
    stem = f"{frame_idx:06d}"
    img_path = data_dir / "images" / f"{stem}.png"
    if not img_path.exists(): return None
    img = np.array(cv2.imread(str(img_path))[..., ::-1])
    
    if use_gt_depth:
        mask_path = data_dir / "obj_masks" / f"{stem}.png"
        depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
    else:
        mask_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
        if not mask_path.exists(): mask_path = data_dir / "obj_masks" / f"{stem}.png"
        depth_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
        
    if not mask_path.exists(): return None
    mask = np.array(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE))
    
    if not depth_path.exists(): return None
    depth = np.load(depth_path)
    if depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
    T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
    T_WO_gt = load_object_pose_world(data_dir, frame_idx)
    return {
        "image": img, "mask": mask, "depth": depth,
        "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": frame_idx
    }

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("test_geometry_tracker_informed", spawn=False)
        if cfg.rerun_url:
            rr.connect_grpc(cfg.rerun_url)
        
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial3DView(name="Object-Centric View", origin="/object"),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Input Image", origin="/input/image"),
                    rrb.Spatial2DView(name="Mask", origin="/input/mask"),
                ),
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
        rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    data_dir = Path(cfg.data_root)
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    
    geo_tracker = GeometricTracker(cfg)
    
    f0 = load_frame_data(data_dir, cfg.init_frame, cfg.use_gt_depth)
    if f0 is None:
        print(f"Failed to load initial frame {cfg.init_frame}")
        return

    # Initial Object-in-Camera pose (GT for anchor)
    T_C0O_gt = f0["T_CW_gt"] @ f0["T_WO_gt"] if f0["T_WO_gt"] is not None else np.eye(4)
    T_WO0 = f0["T_WO_gt"]
    
    print(f"[Debug] Initial K matrix:\n{f0['K']}")
    geo_tracker.poses[f0["frame_idx"]] = T_C0O_gt
    geo_tracker.K_dict[f0["frame_idx"]] = f0["K"]
    geo_tracker.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], f0["depth"], f0["K"], T_C0O_gt)
    geo_tracker.keyframes.append(f0["frame_idx"])
    
    # History for informed guess (winning strategy: GT-Cam + CV-Obj)
    history_T_WO_gt = [T_WO0]
    
    last_kf_idx = cfg.init_frame
    last_kf_gray = cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY)
    
    traj_obj_est_C = []
    traj_obj_gt_C = []

    prev_f = f0
    for i in tqdm(range(1, cfg.n_frames), desc="Tracking"):
        idx = cfg.init_frame + i
        fd = load_frame_data(data_dir, idx, cfg.use_gt_depth)
        if fd is None: break
        
        flow = dis.calc(cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY), None)
        
        # Winning Strategy: GT-Cam + CV-Obj Guess
        if cfg.guess_type == "GT-Cam+CV-Obj":
            if len(history_T_WO_gt) >= 2:
                T_prev = history_T_WO_gt[-1]
                T_prev_prev = history_T_WO_gt[-2]
                T_WO_guess = T_prev @ np.linalg.inv(T_prev_prev) @ T_prev
            else:
                T_WO_guess = history_T_WO_gt[-1]
            T_guess = fd["T_CW_gt"] @ T_WO_guess
        elif cfg.guess_type == "GT-Cam":
            # Camera moves with GT, Object stays at initial position
            T_guess = fd["T_CW_gt"] @ T_WO0
        else:
            # Default fallback (identity motion in object frame)
            T_guess = geo_tracker.poses[idx-1].copy()
        
        success = geo_tracker.step_informed(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], T_guess, flow, skip_pnp=cfg.skip_pnp)
        if not success: 
            print(f"Tracking failed at frame {idx}")
            break

        # Maintain feature count
        active_tids = [tid for tid, t in geo_tracker.tracks.items() if idx in t['obs']]
        if len(active_tids) < cfg.n_features:
            geo_tracker.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], geo_tracker.poses[idx], align=False)
            
        # Adaptive Keyframe Selection using DIS flow disparity
        curr_gray = cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY)
        flow_kf_curr = dis.calc(last_kf_gray, curr_gray, None)
        
        # Calculate average disparity on the object mask
        mask_curr = fd["mask"] > 0
        if np.any(mask_curr):
            # Compute magnitude of flow within the mask
            disp_map = np.linalg.norm(flow_kf_curr, axis=-1)
            # Use median disparity for robustness to flow outliers
            avg_disparity = np.median(disp_map[mask_curr])
        else:
            avg_disparity = 0
            
        frames_since_kf = idx - last_kf_idx
        
        # Heuristic: enough motion (disparity) OR too many frames passed
        is_kf = (avg_disparity > cfg.kf_disparity_thresh and frames_since_kf >= cfg.kf_min_interval) or \
                (frames_since_kf >= cfg.kf_max_interval)

        if is_kf:
            print(f"[Keyframe] Frame {idx}: Disparity {avg_disparity:.1f}px, frames since KF {frames_since_kf}")
            
            geo_tracker.keyframes.append(idx)
            geo_tracker.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], geo_tracker.poses[idx], align=False)
            geo_tracker.run_ba()
            
            # Update last keyframe state
            last_kf_idx = idx
            last_kf_gray = curr_gray

        # Update trajectories (camera in object frame)
        T_OC_est = np.linalg.inv(geo_tracker.poses[idx])
        traj_obj_est_C.append(T_OC_est[:3, 3])
        
        if fd["T_WO_gt"] is not None:
            T_CiO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"]
            T_OC_gt = np.linalg.inv(T_CiO_gt)
            traj_obj_gt_C.append(T_OC_gt[:3, 3])
            history_T_WO_gt.append(fd["T_WO_gt"])
            
            # Debug info
            dist_error = np.linalg.norm(T_OC_est[:3, 3] - T_OC_gt[:3, 3])
            R_rel = T_OC_est[:3, :3].T @ T_OC_gt[:3, :3]
            rot_error_deg = np.rad2deg(np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)))
            if i % 10 == 0 or i == 1:
                print(f"[Debug] Frame {idx}: Err {dist_error:.4f}m, {rot_error_deg:.2f} deg | Cam Est {T_OC_est[:3, 3]}")

        if not cfg.no_vis:
            rr.set_time("frame", sequence=idx)
            rr.log("input/image", rr.Image(fd["image"]))
            rr.log("input/mask", rr.Image(fd["mask"]))
            
            # estimated camera position in world frame (for viz)
            T_WC_est = f0["T_WO_gt"] @ T_OC_est
            
            rr.log("object/camera_est", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
            rr.log("world/camera_est", rr.Transform3D(mat3x3=T_WC_est[:3, :3], translation=T_WC_est[:3, 3]))
            rr.log("world/camera_est", rr.Pinhole(image_from_camera=fd["K"], width=fd["image"].shape[1], height=fd["image"].shape[0]))
            
            # Log current tracks
            active_points = []
            active_colors = []
            reprojected_2d = []
            for tid, t in geo_tracker.tracks.items():
                if idx in t['obs']:
                    pt_O = t['pt3d']
                    active_points.append(pt_O)
                    color = [(tid * 13) % 256, (tid * 71) % 256, (tid * 113) % 256]
                    active_colors.append(color)
                    
                    T_CiO = geo_tracker.poses[idx]
                    pt_Ci = (T_CiO[:3, :3] @ pt_O) + T_CiO[:3, 3]
                    if pt_Ci[2] > 0.01:
                        uv = fd["K"] @ (pt_Ci / pt_Ci[2])
                        reprojected_2d.append(uv[:2])
            
            if active_points:
                rr.log("object/points/active", rr.Points3D(active_points, colors=active_colors, radii=0.002))
                if reprojected_2d:
                    rr.log("input/image/reprojections", rr.Points2D(reprojected_2d, colors=active_colors, radii=2.0))
            
            all_points = np.array([t['pt3d'] for t in geo_tracker.tracks.values()])
            rr.log("object/points/all", rr.Points3D(all_points, colors=[200, 200, 200], radii=0.001))
            
            # Trajectories
            rr.log("object/traj/camera_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[[255, 0, 255]], radii=0.001))
            if traj_obj_gt_C:
                rr.log("object/traj/camera_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[0, 255, 0]], radii=0.001))

        prev_f = fd

    print("Test complete.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
