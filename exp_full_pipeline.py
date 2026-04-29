import numpy as np
import cv2
import rerun as rr
import torch
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

from gs_dyn_obj.obj_gs import ObjectGS
from gs_dyn_obj.grouped_gs import GaussianSuperPrimitive
from gs_dyn_obj.gs_param import GSParam

from obj_gs_mapping import MappingConfig, start_mapping_process

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
    num_pts: int = 2000
    
    # GS Tracking Parameters
    gs_lr: float = 5e-4
    gs_opt_steps: int = 100
    near_plane: float = 0.01
    far_plane: float = 10.0
    
    no_vis: bool = False
    re_render: bool = False
    
    # Depth source
    use_gt_depth: bool = False
    
    # Grid spacing for geometric tracks
    grid_spacing: int = 12
    ransac_thresh: float = 2.0
    kf_every: int = 5
    max_keyframes: int = 10
    
    # Informed Tracking
    use_informed_filtering: bool = True
    informed_thresh: float = 10.0
    
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


def sample_mask_points(mask, num_pts):
    yy, xx = np.where(mask > 0)
    if len(xx) == 0: return np.array([])
    indices = np.random.choice(len(xx), min(num_pts, len(xx)), replace=False)
    return np.stack([xx[indices], yy[indices]], axis=-1).astype(np.float32)

def main(cfg: Config):
    if not cfg.no_vis:
        rr.init("exp_full_pipeline", spawn=False)
        if cfg.rerun_url:
            rr.connect_grpc(cfg.rerun_url)
        
        # Define Rerun Blueprint for comprehensive visualization
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(
                    rrb.Spatial3DView(name="World-Centric View", origin="/world"),
                    rrb.Spatial3DView(name="Object-Centric View", origin="/object"),
                ),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="GT Image", origin="/input/image"),
                    rrb.Spatial2DView(name="Render", origin="/object/camera_est/image/render"),
                    rrb.Spatial2DView(name="Reprojection View", origin="/object/camera_est/image"),
                    rrb.Spatial2DView(name="Warped View", origin="/opt/warped"),
                    rrb.Spatial2DView(name="Loss Plot", origin="/opt/loss_plot"),
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
            mask_path = data_dir / "obj_masks" / f"{stem}.png"
            depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
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

    # Tracking State: T_CiO (Camera-from-Object)
    poses_CiO_est = {} 
    traj_gt_O = [] # Trajectory of camera in Object frame
    traj_est_O = []
    
    f0 = load_frame_data(cfg.init_frame)
    if f0 is None: return
    H, W = f0["image"].shape[:2]

    # Calculate initial Object-in-Camera pose
    T_C0O_gt = f0["T_CW_gt"] @ f0["T_WO_gt"] if f0["T_WO_gt"] is not None else np.eye(4)
    T_OC0_gt = np.linalg.inv(T_C0O_gt)

    # Stage 1: Geometric Initialization (Path A style)
    print("Stage 1: Geometric Initialization (Path A style)")
    geo_tracker = GeometricTracker(cfg)
    geo_tracker.poses[f0["frame_idx"]] = T_C0O_gt
    geo_tracker.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], f0["depth"], f0["K"], T_C0O_gt)
    geo_tracker.keyframes.append(f0["frame_idx"])
    geo_tracker.K_dict[f0["frame_idx"]] = f0["K"]
    
    prev_f = f0
    for k in range(1, 5):
        fd = load_frame_data(cfg.init_frame + k)
        if fd is None: break
        flow = dis.calc(cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY), None)
        geo_tracker.step(fd["frame_idx"], fd["image"], fd["mask"], fd["depth"], fd["K"], flow)
        prev_f = fd
    
    geo_tracker.keyframes.append(prev_f["frame_idx"])
    geo_tracker.run_ba()
    
    # Update poses_CiO_est from tracker
    for idx, T in geo_tracker.poses.items():
        poses_CiO_est[idx] = T

    # Stage 1 complete. Now initialize GS using refined geometry.
    print("Stage 2: GS Initialization (from refined tracker points)")
    T_CiO_init = geo_tracker.poses[prev_f["frame_idx"]]
    K_init = prev_f["K"]
    h, w = prev_f["image"].shape[:2]
    
    # Use refined tracker points to scale the dense depth map
    ref_depths = []
    obs_depths = []
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
    
    s_map = 1.0
    if len(ref_depths) > 5:
        s_map = np.median(np.array(ref_depths) / np.array(obs_depths))
        print(f"GS Initialization: Refined depth scale = {s_map:.4f} based on {len(ref_depths)} points")
    
    refined_depth = prev_f["depth"] * s_map

    gsp_init = GaussianSuperPrimitive(prev_f["image"], prev_f["mask"], refined_depth, T_CiO_init, K_init)
    # ObjectGS(gs_params, T_W_O, ...). We'll treat Object at this frame as World.
    obj_gs = ObjectGS(gsp_init.gs_params, np.eye(4), obj_scale=1.0)
    
    # Visualization state
    traj_world_gt_C = []
    traj_world_gt_O = []
    traj_world_est_C = []
    traj_world_est_O = []
    traj_obj_est_C = []
    traj_obj_gt_C = []

    # Initialize Mapping Process if requested
    mapper = None
    if cfg.run_mapping:
        print("Starting GS Mapping process...")
        mapping_cfg = MappingConfig(device=cfg.device)
        # We need a copy of GS parameters for the mapping process
        initial_gs_for_mapping = GSParam(
            means=obj_gs.gs_params.means.clone().detach(),
            quats=obj_gs.gs_params.quats.clone().detach(),
            scales=torch.log(obj_gs.gs_params.scales.clone().detach().clamp(1e-6)),
            colors=obj_gs.gs_params.colors.clone().detach(),
            opacity=torch.log(obj_gs.gs_params.opacity.clone().detach().clamp(1e-6, 0.999) / (1 - obj_gs.gs_params.opacity.clone().detach().clamp(1e-6, 0.999)))
        )
        mapper, mapper_proc = start_mapping_process(mapping_cfg, initial_gs_for_mapping)

    def log_frame(fd, T_CiO, obj_gs):
        rr.set_time("frame", sequence=fd["frame_idx"])
        rr.log("input/image", rr.Image(fd["image"]))
        
        # --- 1. World-Centric View ---
        # Absolute GT poses
        T_WC_gt = np.linalg.inv(fd["T_CW_gt"])
        T_WO_gt = fd["T_WO_gt"]
        
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WC_gt[:3, :3], translation=T_WC_gt[:3, 3]))
        rr.log("world/camera_gt/image", rr.Image(fd["image"]))
        
        if T_WO_gt is not None:
            rr.log("world/object_gt", rr.Transform3D(mat3x3=T_WO_gt[:3, :3], translation=T_WO_gt[:3, 3]))

        # Estimated Poses
        T_CiO_full = T_CiO
        T_WO_est = T_WC_gt @ T_CiO_full
        rr.log("world/object_est", rr.Transform3D(mat3x3=T_WO_est[:3, :3], translation=T_WO_est[:3, 3]))
        
        # Log Estimated Camera in World (using GT object as anchor)
        if T_WO_gt is not None:
            T_WC_est = T_WO_gt @ np.linalg.inv(T_CiO_full)
            rr.log("world/camera_est", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
            rr.log("world/camera_est", rr.Transform3D(mat3x3=T_WC_est[:3, :3], translation=T_WC_est[:3, 3]))
            traj_world_est_C.append(T_WC_est[:3, 3])
            rr.log("world/traj/camera_est", rr.LineStrips3D([np.array(traj_world_est_C)], colors=[[255, 0, 0]], radii=0.001))

        # Log points in world (as child of object_est)
        rr.log("world/object_est/points", rr.Points3D(obj_gs.gs_params.means.detach().cpu().numpy(), 
                                                   colors=obj_gs.gs_params.colors.detach().cpu().numpy(), radii=0.002))
        
        # Trajectories in World
        traj_world_gt_C.append(T_WC_gt[:3, 3])
        traj_world_est_O.append(T_WO_est[:3, 3])
        rr.log("world/traj/camera_gt", rr.LineStrips3D([np.array(traj_world_gt_C)], colors=[[0, 255, 0]], radii=0.001))
        rr.log("world/traj/object_est", rr.LineStrips3D([np.array(traj_world_est_O)], colors=[[0, 0, 255]], radii=0.001))
        
        if T_WO_gt is not None:
            traj_world_gt_O.append(T_WO_gt[:3, 3])
            rr.log("world/traj/object_gt", rr.LineStrips3D([np.array(traj_world_gt_O)], colors=[[0, 255, 255]], radii=0.001))

        # --- 2. Object-Centric View ---
        # Fixed Object at origin
        rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        rr.log("object", rr.Transform3D(mat3x3=np.eye(3), translation=np.zeros(3)))
        
        # Estimated Camera in Object frame: T_OC_est = inv(T_CiO)
        T_OC_est = np.linalg.inv(T_CiO)
        rr.log("object/camera_est", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
        rr.log("object/camera_est", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
        rr.log("object/camera_est/image", rr.Image(fd["image"]))

        # Render GS model from estimated pose
        with torch.no_grad():
            img_rend, _, _, _ = render_2dgs(
                obj_gs.gs_params.means, obj_gs.gs_params.quats, obj_gs.gs_params.scales,
                obj_gs.gs_params.colors, obj_gs.gs_params.opacity,
                viewmat=torch.from_numpy(T_CiO).float().cuda(),
                K=torch.from_numpy(fd["K"]).float().cuda(),
                width=W, height=H
            )
            # Flip vertically for Rerun if needed, and ensure contiguous
            img_rend_np = img_rend.permute(1, 2, 0).cpu().numpy().clip(0, 1)
            rr.log("object/camera_est/image/render", rr.Image(np.ascontiguousarray(img_rend_np[::-1])))

        # Reproject points on image
        means_O = obj_gs.gs_params.means.detach().cpu().numpy()
        opacities = obj_gs.gs_params.opacity.detach().cpu().numpy()
        means_C = (T_CiO[:3, :3] @ means_O.T).T + T_CiO[:3, 3]
        
        # Filter points behind camera and low opacity
        valid = (means_C[:, 2] > 0.01) & (opacities.flatten() > 0.1)
        means_C = means_C[valid]
        colors_valid = obj_gs.gs_params.colors.detach().cpu().numpy()[valid]
        
        # Project
        pts2d_homog = (fd["K"] @ means_C.T).T
        pts2d = pts2d_homog[:, :2] / pts2d_homog[:, 2:3]
        
        # Filter points outside image
        valid_uv = (pts2d[:, 0] >= 0) & (pts2d[:, 0] < W) & (pts2d[:, 1] >= 0) & (pts2d[:, 1] < H)
        rr.log("object/camera_est/image/reproj", rr.Points2D(pts2d[valid_uv], colors=colors_valid[valid_uv], radii=1.0))
        
        # Estimated Points (Static in Object frame)
        rr.log("object/points", rr.Points3D(means_O, 
                                         colors=obj_gs.gs_params.colors.detach().cpu().numpy(), radii=0.002))
        
        # Trajectory of Camera in Object frame
        traj_obj_est_C.append(T_OC_est[:3, 3])
        rr.log("object/traj/camera_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[[255, 0, 255]], radii=0.001))

        # GT Camera in Object frame
        if fd["T_WO_gt"] is not None:
            T_CiO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"]
            T_OC_gt = np.linalg.inv(T_CiO_gt)
            rr.log("object/camera_gt", rr.Pinhole(image_from_camera=fd["K"], width=W, height=H))
            rr.log("object/camera_gt", rr.Transform3D(mat3x3=T_OC_gt[:3, :3], translation=T_OC_gt[:3, 3]))
            traj_obj_gt_C.append(T_OC_gt[:3, 3])
            rr.log("object/traj/camera_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[0, 255, 0]], radii=0.001))

    # Stage 3: Hybrid GS + Geometric Tracking
    print("Stage 3: Hybrid GS + Geometric Tracking")
    T_CiO = geo_tracker.poses[prev_f["frame_idx"]]
    
    frames_track = []
    for i in range(5, cfg.n_frames):
        fd = load_frame_data(cfg.init_frame + i)
        if fd is None: break
        frames_track.append(fd)
    
    times_opt = []
    
    for i, f_curr in tqdm(enumerate(frames_track), total=len(frames_track), desc="Stage 3: Tracking"):
        idx = f_curr["frame_idx"]
        
        # 1. GS-LM First (using Motion Model)
        T_prev = geo_tracker.poses.get(idx-1, np.eye(4))
        T_prev_prev = geo_tracker.poses.get(idx-2, T_prev)
        T_MM = T_prev @ np.linalg.inv(T_prev_prev) @ T_prev
        
        obj_gs.T_W_O = T_MM
        
        torch.cuda.synchronize()
        t0 = time.time()
        obj_gs.optimize_wrt_image_lm(np.eye(4), f_curr["image"], f_curr["K"], 
                                   near_plane=cfg.near_plane, far_plane=cfg.far_plane, 
                                   pyramid_levels=[(4, 10), (2, 10), (1, 20)],
                                   damping=10.0,
                                   update_ref=True, re_render=True, rr_vis=not cfg.no_vis)
        torch.cuda.synchronize()
        times_opt.append(time.time() - t0)
        
        T_gs = obj_gs.T_W_O
        
        # 2. Compute Prior Flow for DIS
        # We must align monodepth to the map scale for consistent flow calculation
        s_prev = geo_tracker.estimate_depth_scale(idx-1, prev_f["depth"], prev_f["mask"], prev_f["K"])
        prior_flow = compute_prior_flow(T_gs, T_prev, f_curr["K"], prev_f["depth"] * s_prev)
        
        # 3. Geometric Step (Refinement)
        flow = dis.calc(cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY), 
                        cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), 
                        prior_flow)
        
        # Use GS pose as an informed prior for track filtering and PnP initialization
        geo_tracker.step_informed(idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], T_guess=T_gs, flow_prev_curr=flow)
        T_CiO = geo_tracker.poses[idx]
        
        # Update GS model pose with final refined pose
        obj_gs.T_W_O = T_CiO
        
        # 4. Synchronize Tracker Keyframes
        if i % cfg.kf_every == 0:
            geo_tracker.keyframes.append(idx)
            geo_tracker.add_new_points_from_depth(idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], T_CiO, align=True)
            geo_tracker.run_ba()


        # 4. Push to Mapper
        if mapper is not None:
            mapping_data = {
                "image": f_curr["image"],
                "mask": f_curr["mask"],
                "depth": f_curr["depth"],
                "K": f_curr["K"],
                "T_CiO": T_CiO,
                "frame_idx": idx
            }
            mapper.update(mapping_data)

        if not cfg.no_vis:
            log_frame(f_curr, T_CiO, obj_gs)
        
        prev_f = f_curr
            
    avg_time = np.mean(times_opt) if len(times_opt) > 0 else 0
    print(f"\nTracking Complete.")
    if len(times_opt) > 0:
        print(f"Average Optimization Time: {avg_time*1000:.2f} ms ({1.0/avg_time:.2f} FPS)")
        print(f"Total Tracking Time (Opt only): {np.sum(times_opt):.2f} s")

    if mapper is not None:
        print("Stopping GS Mapping process...")
        mapper.stop()
        mapper_proc.join()

    print("Pipeline complete.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
