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
from uniflowmatch.models.ufm import UniFlowMatchConfidence

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    init_frame: int = 0
    n_frames: int = 150
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    
    # UFM Parameters
    use_ufm: bool = True
    ufm_model: str = "infinity1096/UFM-Base"
    
    # Tracking Parameters
    grid_spacing: int = 12
    feature_type: str = "grid" # "grid", "orb", or "gftt"
    n_features: int = 1200
    max_pnp_points: int = -1
    ransac_thresh: float = 1.0
    
    # Keyframe Parameters
    kf_disparity_thresh: float = 20.0  # Median pixel displacement
    kf_min_interval: int = 8          # Prevents poor triangulation baseline
    kf_max_interval: int = 15         # Prevents excessive drift
    kf_overlap_thresh: float = 0.6    # Refresh map if < 60% tracks survived
    max_keyframes: int = 20
    
    # Triangulation Parameters
    triangulate: bool = True
    triangulate_thresh: int = 3 # Start refining after 3 frames of track length
    triangulate_parallax_thresh: float = 15.0 # Minimum pixel displacement for triangulation
    
    do_refine: bool = True
    no_vis: bool = False
    use_gt_depth: bool = False
    
    # Informed Parameters
    use_informed_filtering: bool = True
    skip_pnp: bool = False
    informed_thresh: float = 20.0 
    guess_type: str = "GT-Cam+CV-Obj" # "CV", "GT-Cam", "GT-Cam+CV-Obj"

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

def load_frame_data(data_dir, frame_idx):
    stem = f"{frame_idx:06d}"
    img_path = data_dir / "images" / f"{stem}.png"
    if not img_path.exists(): return None
    img = np.array(cv2.imread(str(img_path))[..., ::-1])
    
    # Always load both if available
    # Inferred depth/mask
    mask_est_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
    if not mask_est_path.exists(): mask_est_path = data_dir / "obj_masks" / f"{stem}.png"
    depth_est_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
    
    # GT depth/mask
    mask_gt_path = data_dir / "obj_masks" / f"{stem}.png"
    depth_gt_path = data_dir / "depth_dyn" / f"{stem}.npy"

    if not mask_est_path.exists(): return None
    mask = np.array(cv2.imread(str(mask_est_path), cv2.IMREAD_GRAYSCALE))
    
    depth = None
    if depth_est_path.exists():
        depth = np.load(depth_est_path)
        if depth.shape != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    depth_gt = None
    if depth_gt_path.exists():
        depth_gt = np.load(depth_gt_path)
        if depth_gt.shape != img.shape[:2]:
            depth_gt = cv2.resize(depth_gt, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
    T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
    T_WO_gt = load_object_pose_world(data_dir, frame_idx)
    
    return {
        "image": img, "mask": mask, "depth": depth, "depth_gt": depth_gt,
        "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": frame_idx
    }

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("test_geometry_tracker_informed_ufm", spawn=False)
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

    data_dir = Path(cfg.data_root) / cfg.clip_id
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    
    ufm_model = None
    if cfg.use_ufm:
        print(f"Loading UFM model {cfg.ufm_model}...")
        ufm_model = UniFlowMatchConfidence.from_pretrained(cfg.ufm_model).to(cfg.device).eval()
    
    geo_tracker = GeometricTracker(cfg)
    
    f0 = load_frame_data(data_dir, cfg.init_frame)
    if f0 is None:
        print(f"Failed to load initial frame {cfg.init_frame}")
        return

    # Use GT depth ONLY if requested for initialization anchoring
    init_depth = f0["depth_gt"] if cfg.use_gt_depth else f0["depth"]
    if init_depth is None:
        init_depth = f0["depth_gt"] # fallback
        print("[Warning] Using GT depth for initialization as inferred depth is missing.")

    # Initial Object-in-Camera pose (GT for anchor)
    T_C0O_gt = f0["T_CW_gt"] @ f0["T_WO_gt"] if f0["T_WO_gt"] is not None else np.eye(4)
    T_WO0 = f0["T_WO_gt"]
    
    print(f"[Debug] Initial K matrix:\n{f0['K']}")
    geo_tracker.poses[f0["frame_idx"]] = T_C0O_gt
    geo_tracker.K_dict[f0["frame_idx"]] = f0["K"]
    geo_tracker.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], init_depth, f0["K"], T_C0O_gt)
    geo_tracker.keyframes.append(f0["frame_idx"])
    
    # History for informed guess
    history_T_WO_gt = [T_WO0]
    last_kf_idx = cfg.init_frame
    last_kf_gray = cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY)
    
    traj_obj_est_C = []
    traj_obj_gt_C = []

    all_dist_errors = []
    all_rot_errors = []

    prev_f = f0
    for i in tqdm(range(1, cfg.n_frames), desc="Tracking"):
        idx = cfg.init_frame + i
        fd = load_frame_data(data_dir, idx)
        if fd is None: break
        
        if cfg.use_ufm:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                    result = ufm_model.predict_correspondences_batched(
                        source_image=torch.from_numpy(prev_f["image"]).to(cfg.device),
                        target_image=torch.from_numpy(fd["image"]).to(cfg.device),
                    )
                flow = result.flow.flow_output[0].cpu().float().permute(1, 2, 0).numpy()
                vis_mask = result.covisibility.mask[0].cpu().float().numpy()
        else:
            flow = dis.calc(cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY), None)
            vis_mask = None
        
        # Pose Guess (Relative Motion Model)
        if cfg.guess_type == "GT-Cam+CV-Obj":
            # Uses GT Camera Pose + Constant Velocity Object Pose (Informed Tracking)
            if len(history_T_WO_gt) >= 2:
                T_prev = history_T_WO_gt[-1]
                T_prev_prev = history_T_WO_gt[-2]
                T_WO_guess = T_prev @ np.linalg.inv(T_prev_prev) @ T_prev
            else:
                T_WO_guess = history_T_WO_gt[-1]
            T_guess = fd["T_CW_gt"] @ T_WO_guess
        elif cfg.guess_type == "GT-Cam":
            # Uses GT Camera Pose, Object stays static in world
            T_guess = fd["T_CW_gt"] @ T_WO0
        elif cfg.guess_type == "CV":
            # Pure Constant Velocity in Camera-Object Relative Frame (No GT)
            if i >= 2:
                T_prev = geo_tracker.poses[idx-1]
                T_prev_prev = geo_tracker.poses[idx-2]
                T_guess = T_prev @ np.linalg.inv(T_prev_prev) @ T_prev
            else:
                T_guess = geo_tracker.poses[idx-1].copy()
        else:
            T_guess = geo_tracker.poses[idx-1].copy()
        
        # 1. Prediction-based Feature Recovery (Match Projections)
        # Recover points that flow might have lost but the guess says are visible
        recovered = geo_tracker.match_projections(idx, fd["image"], fd["mask"], fd["K"], T_guess)
        
        # 2. Tracking Step (Flow + Filtering + PnP)
        success, n_inliers = geo_tracker.step_informed(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], T_guess, flow, vis_mask_prev_curr=vis_mask, skip_pnp=cfg.skip_pnp)
        if not success: 
            print(f"Tracking failed at frame {idx}")
            break

        # Maintain feature count
        active_tids = [tid for tid, t in geo_tracker.tracks.items() if idx in t['obs']]
        if len(active_tids) < cfg.n_features:
            geo_tracker.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], geo_tracker.poses[idx], align=True)
            
        # 3. Hybrid Keyframe Selection (Feature Survival + Geometric Baseline)
        curr_gray = cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY)
        flow_kf_curr = dis.calc(last_kf_gray, curr_gray, None)
        mask_curr = fd["mask"] > 0
        
        # 3a. Geometric Disparity (Median)
        if np.any(mask_curr):
            disp_map = np.linalg.norm(flow_kf_curr, axis=-1)
            avg_disparity = np.median(disp_map[mask_curr])
        else:
            avg_disparity = 0
            
        # 3b. Feature Overlap Ratio
        active_tids = [tid for tid, t in geo_tracker.tracks.items() if idx in t['obs']]
        if active_tids:
            overlap_count = sum(1 for tid in active_tids if last_kf_idx in geo_tracker.tracks[tid]['obs'])
            overlap_ratio = overlap_count / len(active_tids)
        else:
            overlap_ratio = 0
            
        frames_since_kf = idx - last_kf_idx
        
        # Trigger Condition:
        # - Need minimum frames (8) to allow some motion for triangulation
        # - Trigger if motion is enough (20px) OR features are dying (overlap < 60%)
        # - Safety temporal fallback (15)
        is_kf = (frames_since_kf >= cfg.kf_min_interval and (avg_disparity > cfg.kf_disparity_thresh or overlap_ratio < cfg.kf_overlap_thresh)) or \
                (frames_since_kf >= cfg.kf_max_interval)

        if is_kf:
            print(f"[Keyframe] Frame {idx}: Disp {avg_disparity:.1f}px, Overlap {overlap_ratio:.1f}, Recovered {recovered}, Active {len(active_tids)}")
            geo_tracker.keyframes.append(idx)
            geo_tracker.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], geo_tracker.poses[idx], align=True)
            # BA now handles triangulation internally if configured
            geo_tracker.run_ba()
            
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
            
            dist_error = np.linalg.norm(T_OC_est[:3, 3] - T_OC_gt[:3, 3])
            R_rel = T_OC_est[:3, :3].T @ T_OC_gt[:3, :3]
            rot_error_deg = np.rad2deg(np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)))
            
            all_dist_errors.append(dist_error)
            all_rot_errors.append(rot_error_deg)

            if i % 10 == 0 or i == 1:
                print(f"[Debug] Frame {idx}: Err {dist_error:.4f}m, {rot_error_deg:.2f} deg | Rec {recovered} | Active {len(active_tids)}")

        if not cfg.no_vis:
            rr.set_time("frame", sequence=idx)
            rr.log("input/image", rr.Image(fd["image"]))
            rr.log("input/mask", rr.Image(fd["mask"]))
            
            T_WC_est = f0["T_WO_gt"] @ T_OC_est
            rr.log("object/camera_est", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
            rr.log("world/camera_est", rr.Transform3D(mat3x3=T_WC_est[:3, :3], translation=T_WC_est[:3, 3]))
            rr.log("world/camera_est", rr.Pinhole(image_from_camera=fd["K"], width=fd["image"].shape[1], height=fd["image"].shape[0]))
            
            active_points, active_colors, reprojected_2d = [], [], []
            for tid, t in geo_tracker.tracks.items():
                if idx in t['obs']:
                    active_points.append(t['pt3d'])
                    color = [(tid * 13) % 256, (tid * 71) % 256, (tid * 113) % 256]
                    active_colors.append(color)
                    p_Ci = (geo_tracker.poses[idx][:3, :3] @ t['pt3d']) + geo_tracker.poses[idx][:3, 3]
                    if p_Ci[2] > 0.01:
                        uv = fd["K"] @ (p_Ci / p_Ci[2])
                        reprojected_2d.append(uv[:2])
            
            if active_points:
                rr.log("object/points/active", rr.Points3D(active_points, colors=active_colors, radii=0.002))
                if reprojected_2d:
                    rr.log("input/image/reprojections", rr.Points2D(reprojected_2d, colors=active_colors, radii=2.0))
            
            all_points = np.array([t['pt3d'] for t in geo_tracker.tracks.values()])
            rr.log("object/points/all", rr.Points3D(all_points, colors=[200, 200, 200], radii=0.001))
            rr.log("object/traj/camera_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[[255, 0, 255]], radii=0.001))
            if traj_obj_gt_C:
                rr.log("object/traj/camera_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[0, 255, 0]], radii=0.001))

        prev_f = fd

    if all_dist_errors:
        print("\n" + "="*30)
        print(f"Final ATE: {np.mean(all_dist_errors):.4f} m")
        print(f"Final Rot Error: {np.mean(all_rot_errors):.2f} deg")
        print("="*30)

    print("Test complete.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
