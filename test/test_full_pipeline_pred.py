import numpy as np
import cv2
import rerun as rr
import torch
import torch.nn.functional as F
import time
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import pyceres
import matplotlib.pyplot as plt
import rerun.blueprint as rrb
import torch.multiprocessing as mp
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass
import poselib
import pycolmap.cost_functions
from geometric_tracker import GeometricTracker, run_ba, interpolate_flow, sample_grid_on_mask, compute_prior_flow
from gs_dyn_obj.gs_rendering import render_2dgs
from gs_dyn_obj.utils.init import unproject_depth, d2n_tblr

from gs_dyn_obj.obj_gs import ObjectGS
from gs_dyn_obj.grouped_gs import GaussianSuperPrimitive
from gs_dyn_obj.gs_param import GSParam

from obj_gs_mapping import MappingConfig, start_mapping_process, init_gs_from_tracker_points

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 0
    fps: float = 30.0
    hist_sec: float = 1.0
    pred_sec: float = 1.0
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    
    # Tracking Parameters
    grid_spacing: int = 6
    feature_type: str = "grid" # "grid", "orb", or "gftt"
    n_features: int = 2000
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
    use_informed_filtering: bool = False
    triangulate_thresh: int = 3 # Start refining after 3 frames of track length
    
    # GS Tracking Parameters
    gs_lr: float = 5e-4
    gs_opt_steps: int = 100
    near_plane: float = 0.01
    far_plane: float = 10.0
    
    no_vis: bool = False
    re_render: bool = False
    
    # Depth source
    use_gt_depth: bool = False
    
    do_refine: bool = True
    
    # Informed Tracking
    use_informed_filtering: bool = False # Disabled for Stage 1 robustness
    informed_thresh: float = 50.0 # Looser threshold
    skip_pnp: bool = False
    
    # Mapping Parameters
    run_mapping: bool = False

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

@dataclass
class Metrics:
    mask_loss: float = 0.0
    masked_psnr: float = 0.0

def compute_metrics(img_gt, img_rend, mask_gt, mask_rend):
    img_gt_f = img_gt.astype(np.float32) / 255.0
    img_rend_f = img_rend.astype(np.float32)
    
    mask_gt_f = (mask_gt > 0).astype(np.float32)
    mask_rend_f = mask_rend.astype(np.float32)
    
    # Mask Loss (L1)
    mask_loss = np.mean(np.abs(mask_gt_f - mask_rend_f))
    
    # Masked PSNR
    mask_overlap = (mask_gt_f > 0.5)
    if np.any(mask_overlap):
        mse = np.mean((img_gt_f[mask_overlap] - img_rend_f[mask_overlap]) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 1e-10 else 100.0
    else:
        psnr = 0.0
        
    return Metrics(mask_loss=mask_loss, masked_psnr=psnr)

def main(cfg: Config):
    n_hist_frames = int(cfg.hist_sec * cfg.fps)
    n_pred_frames = int(cfg.pred_sec * cfg.fps)
    total_frames = n_hist_frames + n_pred_frames
    
    if not cfg.no_vis:
        rr.init("test_full_pipeline_pred", spawn=False)
        if cfg.rerun_url:
            rr.connect_grpc(cfg.rerun_url)
        
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(name="World-Centric View", origin="/world"),
                    rrb.Spatial3DView(name="Object-Centric View", origin="/object"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="GT Image", origin="/input/image"),
                    rrb.Spatial2DView(name="Render", origin="/object/camera_est/image/render"),
                    rrb.Spatial2DView(name="Prediction Comparison", origin="/prediction/side_by_side"),
                ),
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    data_dir = Path(cfg.data_root)
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)

    def load_frame_data(frame_idx):
        stem = f"{frame_idx:06d}"
        img_path = data_dir / "images" / f"{stem}.png"
        if not img_path.exists(): return None
        img = np.array(cv2.imread(str(img_path))[..., ::-1])
        mask_gt = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        
        if cfg.use_gt_depth:
            depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
            mask_path = data_dir / "obj_masks" / f"{stem}.png"
        else:
            mask_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
            if not mask_path.exists(): mask_path = data_dir / "obj_masks" / f"{stem}.png"
            depth_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
            
        mask = np.array(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE))
        if not depth_path.exists(): return None
        depth = np.load(depth_path)
        if depth.shape != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        T_WO_gt = load_object_pose_world(cfg.data_root, frame_idx)
        return {
            "image": img, "mask": mask, "mask_gt": mask_gt, "depth": depth,
            "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": frame_idx
        }

    # Tracking State
    poses_W_O_est = {}
    traj_world_gt_C = []
    traj_world_gt_O = []
    traj_world_est_C = []
    traj_world_est_O = []
    traj_obj_est_C = []
    traj_obj_gt_C = []

    f0 = load_frame_data(cfg.init_frame)
    if f0 is None: return
    H, W = f0["image"].shape[:2]

    # Stage 1: Geometric Initialization
    n_init_frames = 15
    print(f"Stage 1: Geometric Initialization (Frames 0-{n_init_frames-1})")
    geo_tracker = GeometricTracker(cfg)
    # Ensure n_features is sufficient
    geo_tracker.cfg.n_features = 1000 
    geo_tracker.cfg.ransac_thresh = 3.0
    T_C0O_gt = f0["T_CW_gt"] @ f0["T_WO_gt"] if f0["T_WO_gt"] is not None else np.eye(4)
    geo_tracker.poses[f0["frame_idx"]] = T_C0O_gt
    geo_tracker.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], f0["depth"], f0["K"], T_C0O_gt)
    geo_tracker.keyframes.append(f0["frame_idx"])
    geo_tracker.K_dict[f0["frame_idx"]] = f0["K"]

    print(f"DEBUG: Frame 0 GT T_CO t: {T_C0O_gt[:3, 3]}")
    
    prev_f = f0
    for k in range(1, n_init_frames):
        fd = load_frame_data(cfg.init_frame + k)
        if fd is None: break
        
        flow = dis.calc(cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY), None)
        T_prev = geo_tracker.poses[fd["frame_idx"]-1]
        T_guess = T_prev @ np.linalg.inv(geo_tracker.poses[fd["frame_idx"]-2]) @ T_prev if k >= 2 else T_prev.copy()
        geo_tracker.step_informed(fd["frame_idx"], fd["image"], fd["mask"], fd["depth"], fd["K"], T_guess, flow)
            
        prev_f = fd
    
    geo_tracker.keyframes.append(prev_f["frame_idx"])
    geo_tracker.run_ba()
    
    # Stage 2: GS Initialization
    print("Stage 2: GS Initialization")
    T_CiO_init = geo_tracker.poses[prev_f["frame_idx"]]
    K_init = prev_f["K"]
    
    ref_depths, obs_depths = [], []
    for tid, t in geo_tracker.tracks.items():
        p_O = t['pt3d']
        p_Ci = T_CiO_init[:3, :3] @ p_O + T_CiO_init[:3, 3]
        if p_Ci[2] <= 0.01: continue
        uv_hom = K_init @ p_Ci
        u, v = uv_hom[:2] / uv_hom[2]
        ix, iy = int(round(u)), int(round(v))
        if 0 <= ix < W and 0 <= iy < H:
            d_obs = prev_f["depth"][iy, ix]
            if d_obs > 0:
                ref_depths.append(p_Ci[2])
                obs_depths.append(d_obs)
    
    s_map = np.median(np.array(ref_depths) / np.array(obs_depths)) if len(ref_depths) > 5 else 1.0
    print(f"DEBUG: GS Initialization s_map: {s_map:.4f} (from {len(ref_depths)} points)")
    refined_depth = prev_f["depth"] * s_map
    gsp_init = GaussianSuperPrimitive(prev_f["image"], prev_f["mask"], refined_depth, T_CiO_init, K_init)
    T_WO_init_est = np.linalg.inv(prev_f["T_CW_gt"]) @ T_CiO_init
    T_WO_init_gt = prev_f["T_WO_gt"]
    obj_gs = ObjectGS(gsp_init.gs_params, T_WO_init_est, obj_scale=1.0)
    poses_W_O_est[prev_f["frame_idx"]] = T_WO_init_est
    
    # Check initial pose error at the same frame
    if prev_f["T_WO_gt"] is not None:
        T_CiO_gt_init = prev_f["T_CW_gt"] @ prev_f["T_WO_gt"]
        pos_err = np.linalg.norm(T_CiO_init[:3, 3] - T_CiO_gt_init[:3, 3])
        # Rotation error
        R_err_mat = T_CiO_init[:3, :3] @ np.linalg.inv(T_CiO_gt_init[:3, :3])
        rot_err = np.arccos(np.clip((np.trace(R_err_mat) - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / np.pi
        print(f"Initial Pose Error at frame {prev_f['frame_idx']}: {pos_err:.4f}m, rot: {rot_err:.2f}deg")

    # Stage 3: History Tracking
    print(f"Stage 3: History Tracking ({cfg.hist_sec} sec, {n_hist_frames} frames)")
    last_kf_idx = prev_f["frame_idx"]
    last_kf_gray = cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY)
    
    # Profiling stats
    stats = {"gs_opt": [], "geo_step": [], "pnp": []}
    all_metrics = []
    
    for i in tqdm(range(n_init_frames, n_hist_frames), desc="History"):
        fd = load_frame_data(cfg.init_frame + i)
        if fd is None: break
        
        idx = fd["frame_idx"]
        prev_idx = idx - 1
        
        # 1. Tracking with GeometricTracker
        # Compute flow from previous frame
        flow = dis.calc(cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY), 
                        cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY), None)
        
        # Constant velocity guess with damping
        T_prev = geo_tracker.poses[prev_idx]
        T_prev_prev = geo_tracker.poses.get(prev_idx - 1, T_prev)
        # Use a dampened version of the motion model
        v = np.linalg.inv(T_prev_prev) @ T_prev
        T_guess = T_prev @ v
        
        t0 = time.time()
        success = geo_tracker.step_informed(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], T_guess, flow)
        stats["geo_step"].append(time.time() - t0)
        
        T_CiO = geo_tracker.poses[idx]
        T_WO = np.linalg.inv(fd["T_CW_gt"]) @ T_CiO
        obj_gs.T_W_O = T_WO
        poses_W_O_est[idx] = T_WO

        # 3. Photometric Refinement (Using it as a refinement, not primary tracking)
        if cfg.gs_lr > 0 and success: # Only refine if geo tracking was successful
            t0 = time.time()
            # Use smaller damping and more levels for better convergence
            obj_gs.optimize_wrt_image_lm(fd["T_CW_gt"], fd["image"], fd["K"], mask=fd["mask"], 
                                       pyramid_levels=[(2, 5), (1, 5)], damping=1.0)
            stats["gs_opt"].append(time.time() - t0)
            # Update tracker from GS refinement
            T_WO = obj_gs.T_W_O
            poses_W_O_est[idx] = T_WO
            T_CiO = fd["T_CW_gt"] @ T_WO
            geo_tracker.poses[idx] = T_CiO
        
        # Compare with GT
        if fd["T_WO_gt"] is not None:
            T_CiO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"]
            pos_err = np.linalg.norm(T_CiO[:3, 3] - T_CiO_gt[:3, 3])
            R_err_mat = T_CiO[:3, :3] @ np.linalg.inv(T_CiO_gt[:3, :3])
            rot_err = np.arccos(np.clip((np.trace(R_err_mat) - 1.0) / 2.0, -1.0, 1.0)) * 180.0 / np.pi
            if i % 5 == 0:
                print(f"Frame {idx}: Tracker Pose Error: pos={pos_err:.4f}m, rot={rot_err:.2f}deg")
        
        # Maintenance
        if len([tid for tid, t in geo_tracker.tracks.items() if idx in t['obs']]) < cfg.n_features:
            geo_tracker.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], T_CiO, align=True)
        
        # Keyframe logic
        curr_gray = cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY)
        if (idx - last_kf_idx) >= cfg.kf_max_interval:
            geo_tracker.keyframes.append(idx)
            geo_tracker.run_ba()
            last_kf_idx = idx; last_kf_gray = curr_gray

        if not cfg.no_vis:
            log_frame(fd, T_CiO, obj_gs, traj_world_gt_C, traj_world_gt_O, traj_world_est_C, traj_world_est_O, traj_obj_est_C, traj_obj_gt_C)
        prev_f = fd

    # Stage 4: Prediction
    print(f"Stage 4: Prediction ({cfg.pred_sec} sec, {n_pred_frames} frames)")
    
    # Store end-of-history states
    T_WO_hist_est = geo_tracker.poses[prev_f["frame_idx"]] # Using T_CiO as proxy for O in C
    T_WO_hist_gt = prev_f["T_WO_gt"]
    T_CW_hist_gt = prev_f["T_CW_gt"]
    T_CiO_hist_est = T_WO_hist_est # In this script's convention, poses[idx] is T_CiO
    
    # For prediction, we use the last estimated pose as a base and apply GT relative motion
    T_CiO_pred = T_CiO_hist_est.copy()
    
    for i in tqdm(range(n_hist_frames, total_frames), desc="Prediction"):
        fd = load_frame_data(cfg.init_frame + i)
        if fd is None: break
        
        idx = fd["frame_idx"]
        
        # Calculate GT Object Pose Change relative to end of history
        # Delta_T_O = inv(T_WO_hist_gt) @ T_WO_curr_gt  (Relative motion in object frame)
        if fd["T_WO_gt"] is not None and T_WO_hist_gt is not None:
            # T_WO_hist_est is really T_CiO_hist_est in camera frame. 
            # We want T_CiO_pred = T_CW_gt @ T_WO_pred
            # T_WO_pred = T_WO_hist_est_in_world @ Delta_T_O_world ??
            
            # More robust: relative motion in Object frame
            # Apply GT relative motion to ESTIMATED initial pose
            rel_to_init = np.linalg.inv(T_WO_init_gt) @ fd["T_WO_gt"]
            T_WO_render = T_WO_init_est @ rel_to_init
            T_CiO_render = fd["T_CW_gt"] @ T_WO_render
        else:
            T_CiO_render = fd["T_CW_gt"] @ np.linalg.inv(T_CW_hist_gt) @ T_CiO_hist_est
            
        # Render GS model from this pose
        with torch.no_grad():
            img_rend, _, _, render_alpha = render_2dgs(
                obj_gs.gs_params.means, obj_gs.gs_params.quats, obj_gs.gs_params.scales,
                obj_gs.gs_params.colors, obj_gs.gs_params.opacity,
                viewmat=torch.from_numpy(T_CiO_render).float().cuda(),
                K=torch.from_numpy(fd["K"]).float().cuda(),
                width=W, height=H
            )
            if i == n_hist_frames:
                print(f"DEBUG: T_CiO_render (frame {idx}):\n{T_CiO_render}", flush=True)
                print(f"DEBUG: Gaussian means stats: mean={obj_gs.gs_params.means.mean(dim=0).cpu().numpy()}, std={obj_gs.gs_params.means.std(dim=0).cpu().numpy()}", flush=True)
                print(f"DEBUG: Rendered alpha max: {render_alpha.max().item()}", flush=True)

            img_rend_np = img_rend.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            mask_rend_np = render_alpha.squeeze().cpu().numpy().clip(0, 1)
            
        # Metrics
        metrics = compute_metrics(fd["image"], img_rend_np, fd["mask_gt"], mask_rend_np)
        all_metrics.append(metrics)
        
        # Side-by-side comparison
        img_gt = fd["image"] / 255.0
        img_gt_masked = img_gt * (fd["mask_gt"][:, :, None] > 0.5)
        h_side = np.hstack([img_gt_masked, img_rend_np])
        
        if not cfg.no_vis:
            rr.set_time("frame", sequence=idx)
            rr.log("prediction/side_by_side", rr.Image(h_side))
            rr.log("prediction/mask_loss", rr.Scalars(metrics.mask_loss))
            rr.log("prediction/masked_psnr", rr.Scalars(metrics.masked_psnr))
            log_frame(fd, T_CiO_render, obj_gs, traj_world_gt_C, traj_world_gt_O, traj_world_est_C, traj_world_est_O, traj_obj_est_C, traj_obj_gt_C)
        
        # Save first prediction frame for inspection
        if i == n_hist_frames:
            print(f"DEBUG: Saving side-by-side for i={i}", flush=True)
            save_img = (h_side[:, :, ::-1] * 255).astype(np.uint8)
            success = cv2.imwrite("/home/shzhou/project/dyn_gs_exp/pred_side_by_side.png", save_img)
            print(f"Saved side-by-side comparison to pred_side_by_side.png, success={success}", flush=True)

    print("\nPerformance Bottlenecks (Average per frame):")
    for k, v in stats.items():
        if v:
            print(f"  {k}: {np.mean(v)*1000:.2f} ms")
    
    if all_metrics:
        avg_mask_loss = np.mean([m.mask_loss for m in all_metrics])
        avg_masked_psnr = np.mean([m.masked_psnr for m in all_metrics])
        print(f"\nAverage Prediction Metrics:")
        print(f"  Mask Loss: {avg_mask_loss:.6f}")
        print(f"  Masked PSNR: {avg_masked_psnr:.2f} dB")

    print("Pipeline complete.")

def log_frame(fd, T_CiO, obj_gs, traj_world_gt_C, traj_world_gt_O, traj_world_est_C, traj_world_est_O, traj_obj_est_C, traj_obj_gt_C):
    H, W = fd["image"].shape[:2]
    rr.set_time("frame", sequence=fd["frame_idx"])
    rr.log("input/image", rr.Image(fd["image"]))
    
    T_WC_gt = np.linalg.inv(fd["T_CW_gt"])
    rr.log("world/camera_gt", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
    rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WC_gt[:3, :3], translation=T_WC_gt[:3, 3]))
    
    if fd["T_WO_gt"] is not None:
        rr.log("world/object_gt", rr.Transform3D(mat3x3=fd["T_WO_gt"][:3, :3], translation=fd["T_WO_gt"][:3, 3]))
        traj_world_gt_O.append(fd["T_WO_gt"][:3, 3])
        rr.log("world/traj/object_gt", rr.LineStrips3D([np.array(traj_world_gt_O)], colors=[[0, 255, 255]], radii=0.001))

    T_WO_est = T_WC_gt @ T_CiO
    rr.log("world/object_est", rr.Transform3D(mat3x3=T_WO_est[:3, :3], translation=T_WO_est[:3, 3]))
    traj_world_est_O.append(T_WO_est[:3, 3])
    rr.log("world/traj/object_est", rr.LineStrips3D([np.array(traj_world_est_O)], colors=[[0, 0, 255]], radii=0.001))

    rr.log("world/object_est/points", rr.Points3D(obj_gs.gs_params.means.detach().cpu().numpy(), 
                                               colors=obj_gs.gs_params.colors.detach().cpu().numpy(), radii=0.002))
    
    traj_world_gt_C.append(T_WC_gt[:3, 3])
    rr.log("world/traj/camera_gt", rr.LineStrips3D([np.array(traj_world_gt_C)], colors=[[0, 255, 0]], radii=0.001))

    # Object-Centric View
    rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    T_OC_est = np.linalg.inv(T_CiO)
    rr.log("object/camera_est", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
    rr.log("object/camera_est", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
    
    with torch.no_grad():
        img_rend, _, _, _ = render_2dgs(
            obj_gs.gs_params.means, obj_gs.gs_params.quats, obj_gs.gs_params.scales,
            obj_gs.gs_params.colors, obj_gs.gs_params.opacity,
            viewmat=torch.from_numpy(T_CiO).float().cuda(),
            K=torch.from_numpy(fd["K"]).float().cuda(),
            width=W, height=H
        )
        img_rend_np = img_rend.permute(1, 2, 0).cpu().numpy().clip(0, 1)
        rr.log("object/camera_est/image/render", rr.Image(np.ascontiguousarray(img_rend_np)))

    traj_obj_est_C.append(T_OC_est[:3, 3])
    rr.log("object/traj/camera_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[[255, 0, 255]], radii=0.001))

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
