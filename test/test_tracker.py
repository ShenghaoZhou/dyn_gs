import os
import sys
import time
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import rerun as rr
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R

# Add project root to sys.path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from BundleGS_better_feat.geometric_tracker import GeometricTracker, GeoTrackerConfig
print(f"[DEBUG] geometric_tracker file: {GeometricTracker.__module__}")
import BundleGS_better_feat.geometric_tracker as geometric_tracker
print(f"[DEBUG] geometric_tracker path: {geometric_tracker.__file__}")
from obj_gs_mapping import GSMapping, MappingConfig, init_gs_from_tracker_points, unproject_depth, d2n_tblr
from gs_dyn_obj.obj_gs import ObjectGS
from gs_dyn_obj.gs_param import GSParam
from pytorch3d.transforms import matrix_to_quaternion, quaternion_multiply

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    device: str = "cuda"
    num_frames: int = 100
    init_frame: int = 0
    stride: int = 1
    n_features: int = 2000
    ransac_thresh: float = 1.0
    informed_thresh: float = 10.0
    max_pose_jump: float = 0.5
    num_opt_steps: int = 100
    mask_loss_weight: float = 20.0
    pyr_levels: int = 2
    gs_type: str = "2d" # "2d" or "3d"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    no_vis: bool = False
    debug: bool = False
    
    # Dynamic specific
    feature_type: str = "orb"
    use_photometric: bool = True
    photometric_mode: str = "lm"
    n_init_frames: int = 5
    num_steps_dyn: int = 150
    densify_every: int = 10
    fix_color: bool = True
    fix_scale: bool = True
    use_ray_dist: bool = True
    multi_view_ncc_weight: float = 1.0
    kf_every: int = 5
    grid_spacing: int = 4
    skip_pnp: bool = False
    do_refine: bool = True
    use_hand_mask: bool = True
    triangulate: bool = True

def load_object_pose_world(data_dir, frame_idx):
    poses_file = Path(data_dir) / "object_poses.txt"
    if not poses_file.exists(): 
        return None
    with open(poses_file, "r") as f: 
        lines = f.readlines()
    if frame_idx >= len(lines): 
        return None
    line = lines[frame_idx].split()
    if len(line) < 8: return None
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])]) # x y z w
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4)
    T_WO[:3, :3] = r_WO
    T_WO[:3, 3] = t_WO
    return T_WO

def load_frame_data(data_dir, frame_idx):
    img_path = data_dir / "images" / f"{frame_idx:06d}.png"
    if not img_path.exists():
        img_path = data_dir / "images" / f"{frame_idx:06d}.jpg"
    
    mask_path = data_dir / "obj_masks" / f"{frame_idx:06d}.png"
    if not mask_path.exists():
        mask_path = data_dir / "model_infer" / f"mask_{frame_idx:05d}.png"
    
    depth_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
    if not depth_path.exists():
        depth_path = data_dir / "depth_dyn" / f"{frame_idx:06d}.npy"
        
    k_path = data_dir / "intrinsics" / f"{frame_idx:06d}.npy"
    extrin_path = data_dir / "extrinsics" / f"{frame_idx:06d}.npy"
    
    if not img_path.exists(): return None
    
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
    if mask is not None and mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
    depth = np.load(depth_path).astype(np.float32) if depth_path.exists() else None
    if depth is not None and depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        
    K = np.load(k_path)
    extrin = np.load(extrin_path)
    
    T_WO_gt = load_object_pose_world(data_dir, frame_idx)
    
    hand_mask_path = data_dir / "hand_masks" / f"{frame_idx:06d}.png"
    hand_mask = cv2.imread(str(hand_mask_path), cv2.IMREAD_GRAYSCALE) if hand_mask_path.exists() else None
    if hand_mask is not None and hand_mask.shape[:2] != (h, w):
        hand_mask = cv2.resize(hand_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                
    return {
        "image": image, "mask": mask, "depth": depth, "K": K, "extrin": extrin,
        "T_WO_gt": T_WO_gt, "frame_idx": frame_idx, "hand_mask": hand_mask
    }

def main():
    cfg = tyro.cli(Config)
    if not cfg.no_vis:
        rr.init("TrackerTest", recording_id="test_tracker")
        rr.connect_grpc(cfg.rerun_url)
        rr.log("object", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)

    data_dir = Path(cfg.data_root) / cfg.clip_id
    device = torch.device(cfg.device)
    
    tracker_cfg = GeoTrackerConfig(
        feature_type=cfg.feature_type,
        n_features=cfg.n_features,
        use_photometric_refinement=cfg.use_photometric,
        photometric_mode=cfg.photometric_mode,
        n_init_frames=cfg.n_init_frames,
        max_pose_jump=cfg.max_pose_jump,
        informed_thresh=cfg.informed_thresh,
        ransac_thresh=cfg.ransac_thresh,
        grid_spacing=cfg.grid_spacing,
        triangulate=cfg.triangulate,
        gs_type=cfg.gs_type
    )
    
    map_cfg = MappingConfig(
        gs_type=cfg.gs_type,
        device=cfg.device,
        num_steps_per_frame=cfg.num_steps_dyn, 
        densify_every=cfg.densify_every,
        mask_loss_weight=cfg.mask_loss_weight,
        fix_color=cfg.fix_color,
        fix_scale=cfg.fix_scale,
        use_ray_dist=cfg.use_ray_dist,
        multi_view_ncc_weight=cfg.multi_view_ncc_weight,
        pyr_levels=cfg.pyr_levels
    )
    
    tracker = GeometricTracker(tracker_cfg)
    poses_wo_est = {}
    
    # 1. Initialize
    print(f"[INFO] Initializing for clip {cfg.clip_id}...")
    f0 = load_frame_data(data_dir, cfg.init_frame)
    if f0 is None:
        print(f"[ERROR] Could not load frame {cfg.init_frame}")
        return
        
    h, w = f0["image"].shape[:2]
    T_WO_gt0 = f0["T_WO_gt"] if f0["T_WO_gt"] is not None else np.eye(4)
    T_C0O_gt0 = f0["extrin"] @ T_WO_gt0
    
    # We initialize at GT pose to test tracking stability
    poses_wo_est[cfg.init_frame] = T_WO_gt0
    tracker.poses[cfg.init_frame] = T_C0O_gt0
    tracker.K_dict[cfg.init_frame] = f0["K"]

    mask_init = f0["mask"] > 0 if f0["mask"] is not None else np.zeros((h, w), dtype=bool)
    if cfg.use_hand_mask and f0["hand_mask"] is not None:
        mask_init = mask_init & (f0["hand_mask"] == 0)
    
    depth_t = torch.from_numpy(f0["depth"]).float().to(device) if f0["depth"] is not None else torch.zeros((h, w), device=device)
    
    print("[INFO] Computing normals...")
    full_pts_c = unproject_depth(depth_t, torch.from_numpy(f0["K"]).float().to(device), h, w)
    normals_c, _ = d2n_tblr(full_pts_c.permute(2, 0, 1).unsqueeze(0))
    normals_c = F.normalize(normals_c[0], dim=0).cpu().numpy()
    
    ys, xs = np.where(mask_init)
    if len(ys) > 0:
        print(f"[INFO] Initializing GS with {len(ys)} points...")
        step = 4
        ys, xs = ys[::step], xs[::step]
        active_points, active_colors, active_normals = [], [], []
        for y, x in zip(ys, xs):
            p_c = full_pts_c[y, x].cpu().numpy()
            p_w = np.linalg.inv(f0["extrin"][:3, :3]) @ (p_c - f0["extrin"][:3, 3])
            p_o = np.linalg.inv(T_WO_gt0[:3, :3]) @ (p_w - T_WO_gt0[:3, 3])
            active_points.append(p_o)
            active_colors.append(f0["image"][y, x] / 255.0)
            active_normals.append(np.linalg.inv(T_WO_gt0[:3, :3]) @ (np.linalg.inv(f0["extrin"][:3, :3]) @ normals_c[:, y, x]))

        gs_params = init_gs_from_tracker_points(np.array(active_points), np.array(active_colors), device, normals=np.array(active_normals), gs_type=cfg.gs_type)
        obj_gs = ObjectGS(gs_params, T_WO_gt0, 1.0)
        
        tracker.add_new_points_from_depth(cfg.init_frame, f0["image"], mask_init, f0["depth"], f0["K"], T_C0O_gt0, align=False)
        
        with torch.no_grad():
            T_C0O_gt0_t = torch.from_numpy(T_C0O_gt0).float().to(device)
            K0_t = torch.from_numpy(f0["K"]).float().to(device)
            render_mode = "3dgs" if cfg.gs_type == "3d" else "normal"
            img_ref, _, _, alpha_ref = obj_gs.gs_params.render(T_C0O_gt0_t, K0_t, width=w, height=h, mode=render_mode)
            obj_gs.update_reference(img_ref, T_C0O_gt0_t, alpha_ref)
        
        mapper = GSMapping(map_cfg, obj_gs.gs_params)
    else:
        print("[WARNING] No object found in first frame mask.")
        return

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    prev_gray = cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY)

    all_dist_errors = []
    all_rot_errors = []
    all_psnrs = []
    traj_obj_est_C = []
    traj_obj_gt_C = []
    
    print("[INFO] Processing sequence...")
    start_time = time.time()
    
    for k in range(1, cfg.num_frames):
        idx = cfg.init_frame + k * cfg.stride
        fd = load_frame_data(data_dir, idx)
        if fd is None: break
        
        loop_start = time.time()
        curr_gray = cv2.cvtColor(fd["image"], cv2.COLOR_RGB2GRAY)
        flow = dis.calc(prev_gray, curr_gray, None)
        prev_gray = curr_gray
        
        # Motion Model (World Space)
        T_WO_prev = poses_wo_est[idx-cfg.stride]
        if k > 1:
            T_WO_pprev = poses_wo_est[idx-2*cfg.stride]
            V = T_WO_prev @ np.linalg.inv(T_WO_pprev)
            T_WO_guess = V @ T_WO_prev
        else:
            T_WO_guess = T_WO_prev.copy()
        
        T_CiO_in = fd["extrin"] @ T_WO_guess
        
        # Photometric Refinement
        if cfg.use_photometric and obj_gs is not None:
            obj_gs.T_W_O = T_WO_guess.copy()
            try:
                T_WO_photo_t, _ = obj_gs.optimize_wrt_image_lm(fd["extrin"], fd["image"], fd["K"], mask=fd["mask"], gs_type=cfg.gs_type)
                T_WO_photo = T_WO_photo_t.detach().cpu().numpy()
                
                diff_t = np.linalg.norm(T_WO_photo[:3, 3] - T_WO_guess[:3, 3])
                diff_R = np.rad2deg(np.arccos(np.clip((np.trace(T_WO_photo[:3, :3] @ T_WO_guess[:3, :3].T) - 1) / 2, -1, 1)))
                if diff_t > 0.5 or diff_R > 20.0:
                    if cfg.debug: print(f"[WARNING] Photometric jump too large ({diff_t:.4f}m, {diff_R:.2f}deg), rejecting.")
                else:
                    if cfg.debug: print(f"[INFO] Photometric refinement adjusted guess: {diff_t:.6f}m, {diff_R:.6f}deg")
                    T_CiO_in = fd["extrin"] @ T_WO_photo
            except Exception as e:
                if cfg.debug: print(f"[ERROR] Photometric refinement failed: {e}")

        # Geometric Tracking
        mask_track = fd["mask"]
        if cfg.use_hand_mask and fd["hand_mask"] is not None:
            mask_track = (mask_track > 0) & (fd["hand_mask"] == 0)
            
        success, n_inliers = tracker.step_informed_with_occlusion(idx, fd["image"], mask_track, fd["depth"], fd["K"], T_CiO_in, flow_prev_curr=flow, skip_pnp=cfg.skip_pnp)
        T_CiO_est = tracker.poses[idx]
        poses_wo_est[idx] = np.linalg.inv(fd["extrin"]) @ T_CiO_est
        

        # Mapping
        if mapper is not None:
            mapper.optimize_frame({
                "image": fd["image"], "mask": fd["mask"], "depth": fd["depth"],
                "K": fd["K"], "T_CiO": T_CiO_est, "frame_idx": idx
            }, num_steps=cfg.num_steps_dyn, frame_count=k)
            
            with torch.no_grad():
                if obj_gs.image_ref is not None:
                    K0 = tracker.K_dict.get(cfg.init_frame)
                    if K0 is not None:
                        T_C0O_t = torch.from_numpy(tracker.poses[cfg.init_frame]).float().to(device)
                        K0_t = torch.from_numpy(K0).float().to(device)
                        render_mode = "3dgs" if cfg.gs_type == "3d" else "normal"
                        img_ref, _, _, alpha_ref = obj_gs.gs_params.render(T_C0O_t, K0_t, width=w, height=h, mode=render_mode)
                        obj_gs.update_reference(img_ref, T_C0O_t, alpha_ref)
                
                new_points = obj_gs.gs_params.means.detach().cpu().numpy()
                tracker.update_object_points(new_points)
        
        # Local BA
        if k % cfg.kf_every == 0:
            tracker.keyframes.append(idx)
            if len(tracker.keyframes) > 2:
                print(f"[INFO] Running Local BA at frame {idx}")
                tracker.run_ba()
                T_CiO_est = tracker.poses[idx]
                poses_wo_est[idx] = np.linalg.inv(fd["extrin"]) @ T_CiO_est
            
        # Evaluation
        if fd["T_WO_gt"] is not None:
            T_WO_gt = fd["T_WO_gt"]
            T_CiO_gt = fd["extrin"] @ T_WO_gt
            
            dist_err = np.linalg.norm(T_CiO_est[:3, 3] - T_CiO_gt[:3, 3])
            rot_err = np.rad2deg(np.arccos(np.clip((np.trace(T_CiO_est[:3, :3] @ T_CiO_gt[:3, :3].T) - 1) / 2, -1, 1)))
            all_dist_errors.append(dist_err)
            all_rot_errors.append(rot_err)
            
            if cfg.debug:
                T_CiO_gt = fd["extrin"] @ T_WO_gt
                guess_dist = np.linalg.norm(T_CiO_in[:3, 3] - T_CiO_gt[:3, 3])
                guess_rot = np.rad2deg(np.arccos(np.clip((np.trace(T_CiO_in[:3, :3] @ T_CiO_gt[:3, :3].T) - 1) / 2, -1, 1)))
                print(f"[DEBUG] Frame {idx}: n_inliers={n_inliers}")
                print(f"      Guess Err: {guess_dist:.6f}m, {guess_rot:.4f}deg")
                print(f"      Est   Err: {dist_err:.6f}m, {rot_err:.4f}deg")
            
            # Rendering PSNR
            with torch.no_grad():
                T_t = torch.from_numpy(T_CiO_est).float().to(device)
                K_t = torch.from_numpy(fd["K"]).float().to(device)
                render_mode = "3dgs" if cfg.gs_type == "3d" else "normal"
                img_render, _, _, alpha_render = obj_gs.gs_params.render(T_t, K_t, w, h, mode=render_mode)
                
                gt_img_t = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0
                mask_t = torch.from_numpy(fd["mask"] > 0).to(device) if fd["mask"] is not None else torch.ones((h, w), device=device, dtype=torch.bool)
                
                if mask_t.any():
                    mse = torch.mean((img_render[:, mask_t] - gt_img_t[:, mask_t])**2)
                    psnr = -10.0 * torch.log10(mse + 1e-10)
                    all_psnrs.append(psnr.item())
                else:
                    psnr = torch.tensor(0.0)

            fps = 1.0 / (time.time() - loop_start)
            print(f"Frame {idx}: ATE={dist_err:.4f}m, RotErr={rot_err:.2f}deg, PSNR={psnr.item():.2f}dB, FPS={fps:.2f}")

            if not cfg.no_vis:
                rr.set_time("frame_idx", sequence=idx)
                rr.log("input/image", rr.Image(fd["image"]).compress(jpeg_quality=50))
                if fd["mask"] is not None:
                    rr.log("input/mask", rr.Image(fd["mask"]))
                
                img_uint8 = (img_render.permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)
                rr.log("output/render", rr.Image(img_uint8).compress(jpeg_quality=50))
                
                rr.log("output/ate", rr.Scalars(dist_err))
                rr.log("output/psnr", rr.Scalars(psnr.item()))
                
                # Log GS points
                means_np = obj_gs.gs_params.means.detach().cpu().numpy()
                colors_sh = obj_gs.gs_params.colors.detach().cpu().numpy()
                if colors_sh.ndim == 3: colors_sh = colors_sh[:, 0, :]
                colors_rgb = np.clip(colors_sh * 0.28209479177387814 + 0.5, 0, 1)
                rr.log("object/gs_points", rr.Points3D(means_np, colors=colors_rgb, radii=0.004))
                
                # Log Poses
                T_OC_est = np.linalg.inv(T_CiO_est)
                rr.log("object/tracker/camera_est", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
                traj_obj_est_C.append(T_OC_est[:3, 3])
                rr.log("object/tracker/traj_est", rr.LineStrips3D([np.array(traj_obj_est_C)], colors=[[0, 255, 0]], radii=0.003))

                if fd["T_WO_gt"] is not None:
                    T_OC_gt = np.linalg.inv(T_CiO_gt)
                    rr.log("object/camera_gt", rr.Transform3D(mat3x3=T_OC_gt[:3, :3], translation=T_OC_gt[:3, 3]))
                    traj_obj_gt_C.append(T_OC_gt[:3, 3])
                    rr.log("object/traj_gt", rr.LineStrips3D([np.array(traj_obj_gt_C)], colors=[[255, 255, 255]], radii=0.003))

                rr.log("world/camera", rr.Transform3D(mat3x3=fd["extrin"][:3, :3].T, translation=-fd["extrin"][:3, :3].T @ fd["extrin"][:3, 3]))
                rr.log("world/obj_gt", rr.Transform3D(mat3x3=T_WO_gt[:3, :3], translation=T_WO_gt[:3, 3]))
                rr.log("world/obj_est", rr.Transform3D(mat3x3=poses_wo_est[idx][:3, :3], translation=poses_wo_est[idx][:3, 3]))


        if idx % cfg.kf_every == 0:
            tracker.add_new_points_from_depth(idx, fd["image"], fd["mask"], fd["depth"], fd["K"], T_CiO_est, align=True)

    total_time = time.time() - start_time
    avg_fps = (cfg.num_frames - 1) / total_time
    
    print("\n" + "="*40)
    print(f"Tracking Performance Report for {cfg.clip_id}:")
    print(f"  Avg ATE: {np.mean(all_dist_errors):.4f} m")
    print(f"  Avg Rot Error: {np.mean(all_rot_errors):.2f} deg")
    print(f"  Avg PSNR: {np.mean(all_psnrs):.2f} dB")
    print(f"  Avg FPS: {avg_fps:.2f}")
    print("="*40)

if __name__ == "__main__":
    main()
