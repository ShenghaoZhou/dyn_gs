import numpy as np
import cv2
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import rerun.blueprint as rrb
import threading
import time

from geometric_tracker import GeometricTracker
from gs_dyn_obj.obj_gs import ObjectGS
from gs_dyn_obj.gs_param import GSParam
from gs_dyn_obj.gs_rendering import render_2dgs
from src.obj_gs_mapping import MappingConfig, GSMapping, init_gs_from_tracker_points, unproject_depth, d2n_tblr

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
    
    # Tracking Parameters
    grid_spacing: int = 8
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
    guess_type: str = "GS" # "CV", "GT-Cam", "GT-Cam+CV-Obj", "GS"
    
    # GS Parameters
    gs_setup: str = "B" # "A": GT-Cam + GS-obj, "B": GT-Cam + CV-obj + GS-obj
    n_init_frames: int = 8
    use_ray_dist: bool = True
    near_plane: float = 0.01
    far_plane: float = 10.0
    opt_method: str = "LM" # "LM", "Hybrid", or "Adam"

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
    
    mask_est_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
    if not mask_est_path.exists(): mask_est_path = data_dir / "obj_masks" / f"{stem}.png"
    depth_est_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
    
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
        rr.init("test_geometry_tracker_informed_gs", spawn=False)
        if cfg.rerun_url:
            rr.connect_grpc(cfg.rerun_url)
        
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Spatial3DView(name="Object-Centric View", origin="/object"),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="Input Image", origin="/input/image"),
                    rrb.Spatial2DView(name="GS Render", origin="/object/render"),
                ),
            ),
            collapse_panels=True,
        )
        rr.send_blueprint(blueprint)
        rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    data_dir = Path(cfg.data_root) / cfg.clip_id
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    
    geo_tracker = GeometricTracker(cfg)
    
    f0 = load_frame_data(data_dir, cfg.init_frame)
    if f0 is None:
        print(f"Failed to load initial frame {cfg.init_frame}")
        return

    init_depth = f0["depth_gt"] if cfg.use_gt_depth else f0["depth"]
    if init_depth is None:
        init_depth = f0["depth_gt"] # fallback
        print("[Warning] Using GT depth for initialization as inferred depth is missing.")

    T_C0O_gt = f0["T_CW_gt"] @ f0["T_WO_gt"] if f0["T_WO_gt"] is not None else np.eye(4)
    T_WO0 = f0["T_WO_gt"]
    
    geo_tracker.poses[f0["frame_idx"]] = T_C0O_gt
    geo_tracker.K_dict[f0["frame_idx"]] = f0["K"]
    geo_tracker.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], init_depth, f0["K"], T_C0O_gt)
    geo_tracker.keyframes.append(f0["frame_idx"])
    
    history_T_WO_est = [T_WO0]
    last_kf_idx = cfg.init_frame
    last_kf_gray = cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY)
    
    traj_obj_est_C = []
    traj_obj_gt_C = []
    all_dist_errors = []
    all_rot_errors = []

    # GS Mapping State
    obj_gs = None
    mapper = None
    mapping_thread = None
    s_map = 1.0

    try:
        prev_f = f0
        for i in tqdm(range(1, cfg.n_frames), desc="Tracking"):
            idx = cfg.init_frame + i
            fd = load_frame_data(data_dir, idx)
            if fd is None: break
            
            flow = dis.calc(cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY), None)
            
            # 1. GS Initialization (after n_init_frames)
            if i == cfg.n_init_frames:
                print(f"\nStage 2: GS Initialization at frame {idx}")
                T_CiO_init = geo_tracker.poses[idx-1]
                K_init = prev_f["K"]
                h, w = prev_f["image"].shape[:2]
                
                # Estimate scale from tracker points
                ref_depths, obs_depths = [], []
                for tid, t in geo_tracker.tracks.items():
                    p_O = t['pt3d']
                    p_Ci = T_CiO_init[:3, :3] @ p_O + T_CiO_init[:3, 3]
                    if p_Ci[2] <= 0.01: continue
                    uv_hom = K_init @ p_Ci
                    u, v = uv_hom[:2] / uv_hom[2]
                    ix, iy = int(round(u)), int(round(v))
                    if 0 <= ix < w and 0 <= iy < h:
                        d_obs = prev_f["depth"][iy, ix]
                        if d_obs > 0:
                            ref_depths.append(p_Ci[2])
                            obs_depths.append(d_obs)
                
                if len(ref_depths) > 5:
                    s_map = np.median(np.array(ref_depths) / np.array(obs_depths))
                    print(f"GS Initialization: Refined depth scale = {s_map:.4f}")
                
                refined_depth = prev_f["depth"] * s_map
                mask_init = prev_f["mask"] > 0
                y, x = np.where(mask_init & (refined_depth > 0))
                perm = np.random.permutation(len(y))[:8000]
                y, x = y[perm], x[perm]
                z = refined_depth[y, x]
                pts_c = np.stack([(x - K_init[0, 2]) * z / K_init[0, 0], (y - K_init[1, 2]) * z / K_init[1, 1], z], axis=-1)
                T_OC_init = np.linalg.inv(T_CiO_init)
                pts_o = (pts_c @ T_OC_init[:3, :3].T) + T_OC_init[:3, 3]
                colors0 = prev_f["image"][y, x] / 255.0
                
                torch.cuda.empty_cache()
                full_pts_c = unproject_depth(torch.from_numpy(refined_depth).float().to(cfg.device), torch.from_numpy(K_init).float().to(cfg.device), h, w)
                normals_c, _ = d2n_tblr(full_pts_c.permute(2, 0, 1).unsqueeze(0))
                normals_c = -F.normalize(normals_c[0][:, y, x].permute(1, 0), dim=1).cpu().numpy()
                normals_o = (normals_c @ T_OC_init[:3, :3].T)
                
                ray_o_o, ray_d_o, ray_dist = None, None, None
                if cfg.use_ray_dist:
                    ray_o_o = torch.from_numpy(T_OC_init[:3, 3]).float().view(1, 3).repeat(len(pts_o), 1)
                    fx, fy, cx, cy = K_init[0, 0], K_init[1, 1], K_init[0, 2], K_init[1, 2]
                    ray_d_c = np.stack([(x - cx) / fx, (y - cy) / fy, np.ones_like(x)], axis=-1)
                    ray_d_c = ray_d_c / np.linalg.norm(ray_d_c, axis=-1, keepdims=True)
                    ray_d_o = torch.from_numpy(ray_d_c @ T_OC_init[:3, :3].T).float()
                    ray_dist = torch.norm(torch.from_numpy(pts_c).float(), dim=-1, keepdim=True)

                gs_params = init_gs_from_tracker_points(pts_o, colors0, cfg.device, normals=normals_o, ray_o=ray_o_o, ray_d=ray_d_o, ray_dist=ray_dist)
                map_cfg = MappingConfig(device=cfg.device, num_steps_per_frame=15, pyr_levels=1, use_ray_dist=cfg.use_ray_dist)
                mapper = GSMapping(map_cfg, gs_params)
                obj_gs = ObjectGS(gs_params, np.eye(4), obj_scale=1.0)
                
                mapping_thread = threading.Thread(target=mapper.run)
                mapping_thread.daemon = True # Make it daemon as a fallback
                mapping_thread.start()

            # 2. Pose Guessing
            if cfg.gs_setup == "A":
                T_WO_guess_init = history_T_WO_est[-1]
            else: # Setup B: Constant Velocity
                if len(history_T_WO_est) >= 2:
                    T_prev = history_T_WO_est[-1]
                    T_prev_prev = history_T_WO_est[-2]
                    T_WO_guess_init = T_prev @ np.linalg.inv(T_prev_prev) @ T_prev
                else:
                    T_WO_guess_init = history_T_WO_est[-1]
            
            T_guess_init = fd["T_CW_gt"] @ T_WO_guess_init

            if obj_gs is not None:
                # 2a. GS Pose Guess
                obj_gs.T_W_O = T_WO_guess_init
                if cfg.opt_method == "Hybrid":
                    T_WO_gs, losses = obj_gs.optimize_wrt_image_lm_hybrid(
                        fd["T_CW_gt"], fd["image"], fd["K"], mask=fd["mask"],
                        adam_steps=30, lm_steps=15,
                        update_ref=True, rr_vis=True
                    )
                elif cfg.opt_method == "Adam":
                    T_WO_gs, losses = obj_gs.optimize_wrt_image(
                        fd["T_CW_gt"], fd["image"], fd["K"], mask=fd["mask"],
                        num_steps=50, lr=1e-3,
                        update_ref=True, rr_vis=True
                    )
                else: # Default to LM
                    T_WO_gs, losses = obj_gs.optimize_wrt_image_lm(
                        fd["T_CW_gt"], fd["image"], fd["K"], mask=fd["mask"],
                        near_plane=cfg.near_plane, far_plane=cfg.far_plane, 
                        pyramid_levels=[(4, 5), (2, 5), (1, 5)],
                        damping=10.0,
                        update_ref=True, re_render=True, rr_vis=True
                    )
                T_WO_gs = obj_gs.T_W_O
                T_guess = fd["T_CW_gt"] @ T_WO_gs
            else:
                T_guess = T_guess_init

            # 3. Geometric Step
            recovered = geo_tracker.match_projections(idx, fd["image"], fd["mask"], fd["K"], T_guess)
            success = geo_tracker.step_informed(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], T_guess, flow, skip_pnp=cfg.skip_pnp)
            if not success: 
                print(f"Tracking failed at frame {idx}")
                break

            # Maintain features
            active_tids = [tid for tid, t in geo_tracker.tracks.items() if idx in t['obs']]
            if len(active_tids) < cfg.n_features:
                geo_tracker.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], geo_tracker.poses[idx], align=True)
                
            # Keyframe Selection
            curr_gray = cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY)
            flow_kf_curr = dis.calc(last_kf_gray, curr_gray, None)
            mask_curr = fd["mask"] > 0
            avg_disparity = np.median(np.linalg.norm(flow_kf_curr, axis=-1)[mask_curr]) if np.any(mask_curr) else 0
            overlap_count = sum(1 for tid in active_tids if last_kf_idx in geo_tracker.tracks[tid]['obs']) if active_tids else 0
            overlap_ratio = overlap_count / len(active_tids) if active_tids else 0
            frames_since_kf = idx - last_kf_idx
            is_kf = (frames_since_kf >= cfg.kf_min_interval and (avg_disparity > cfg.kf_disparity_thresh or overlap_ratio < cfg.kf_overlap_thresh)) or \
                    (frames_since_kf >= cfg.kf_max_interval)

            if is_kf:
                geo_tracker.keyframes.append(idx)
                geo_tracker.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], geo_tracker.poses[idx], align=True)
                geo_tracker.run_ba()
                last_kf_idx = idx
                last_kf_gray = curr_gray

            # 4. Update Mapper
            if mapper is not None:
                mapping_data = {
                    "image": fd["image"], "mask": fd["mask"], "depth": fd["depth"] * s_map,
                    "K": fd["K"], "T_CiO": geo_tracker.poses[idx], "frame_idx": idx, "is_keyframe": is_kf
                }
                mapper.update(mapping_data)

            # Update History
            T_OC_est = np.linalg.inv(geo_tracker.poses[idx])
            traj_obj_est_C.append(T_OC_est[:3, 3])
            T_WO_est = np.linalg.inv(fd["T_CW_gt"]) @ geo_tracker.poses[idx]
            history_T_WO_est.append(T_WO_est)
            
            if fd["T_WO_gt"] is not None:
                T_CiO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"]
                T_OC_gt = np.linalg.inv(T_CiO_gt)
                traj_obj_gt_C.append(T_OC_gt[:3, 3])
                dist_error = np.linalg.norm(T_OC_est[:3, 3] - T_OC_gt[:3, 3])
                R_rel = T_OC_est[:3, :3].T @ T_OC_gt[:3, :3]
                rot_error_deg = np.rad2deg(np.arccos(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)))
                all_dist_errors.append(dist_error)
                all_rot_errors.append(rot_error_deg)
                if i % 10 == 0:
                    print(f"[Debug] Frame {idx}: Err {dist_error:.4f}m, {rot_error_deg:.2f} deg | Rec {recovered} | Active {len(active_tids)}")

            if not cfg.no_vis:
                rr.set_time("frame", sequence=idx)
                rr.log("input/image", rr.Image(fd["image"]))
                rr.log("object/camera_est", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
                rr.log("object/traj/camera_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[[255, 0, 255]], radii=0.001))
                if traj_obj_gt_C:
                    rr.log("object/traj/camera_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[0, 255, 0]], radii=0.001))
                if obj_gs is not None:
                    with torch.no_grad():
                        img_rend, _, _, _ = render_2dgs(
                            obj_gs.gs_params.means, F.normalize(obj_gs.gs_params.quats), torch.exp(obj_gs.gs_params.scales),
                            obj_gs.gs_params.colors, torch.sigmoid(obj_gs.gs_params.opacity),
                            viewmat=torch.from_numpy(geo_tracker.poses[idx]).float().cuda(),
                            K=torch.from_numpy(fd["K"]).float().cuda(), width=fd["image"].shape[1], height=fd["image"].shape[0]
                        )
                        rr.log("object/render", rr.Image(img_rend.permute(1, 2, 0).cpu().numpy().clip(0, 1)))

                    # Log Points
                    gs_pts = obj_gs.gs_params.means.detach().cpu().numpy()
                    gs_colors = obj_gs.gs_params.colors.detach().cpu().numpy()
                    rr.log("object/gs_points", rr.Points3D(gs_pts, colors=gs_colors, radii=0.001))

                # Log Sparse Points from tracker
                sparse_pts = np.array([t['pt3d'] for t in geo_tracker.tracks.values() if idx in t['obs']])
                if len(sparse_pts) > 0:
                    rr.log("object/sparse_points", rr.Points3D(sparse_pts, colors=[255, 255, 0], radii=0.002))

            prev_f = fd
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if mapper is not None:
            print("Stopping GS Mapping...")
            mapper.stop()
            if mapping_thread is not None:
                mapping_thread.join(timeout=2.0)
            print("GS Mapping stopped.")

    if all_dist_errors:
        print("\n" + "="*30)
        print(f"GS Setup {cfg.gs_setup} Results:")
        print(f"Final ATE: {np.mean(all_dist_errors):.4f} m")
        print(f"Final Rot Error: {np.mean(all_rot_errors):.2f} deg")
        print("="*30)

if __name__ == "__main__":
    try:
        cfg = tyro.cli(Config)
        main(cfg)
    except KeyboardInterrupt:
        pass

