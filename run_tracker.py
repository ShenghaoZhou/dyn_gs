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

from geometric_tracker import GeometricTracker, GeoTrackerConfig
from obj_gs_mapping import GSMapping, MappingConfig, init_gs_from_tracker_points, unproject_depth, d2n_tblr
from gs_dyn_obj.obj_gs import ObjectGS

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-001849"
    init_frame: int = 0
    n_frames_track: int = 30
    n_frames_eval: int = 5
    device: str = "cuda"
    no_vis: bool = False
    
    # Tracker Params
    feature_type: str = "orb"
    n_features: int = 2000
    ransac_thresh: float = 1.0
    informed_thresh: float = 100.0
    max_pose_jump: float = 10.0
    
    # Photometric Params
    use_photometric: bool = True
    gs_type: str = "2d" # "2d" or "3d"
    
    # Mapping Params
    kf_every: int = 5
    num_steps: int = 150
    densify_every: int = 10
    mask_loss_weight: float = 2.0
    use_pgsr: bool = False
    pyr_levels: int = 2
    n_init_frames: int = 10
    
    # ARAP Rigidity
    use_arap: bool = True
    arap_weight: float = 0.1
    arap_rot_weight: float = 0.05
    arap_k: int = 8
    arap_warmup_steps: int = 30
    use_arap_flow_lifting: bool = True

def load_frame_data(data_dir, frame_idx):
    img_path = data_dir / "images" / f"{frame_idx:06d}.png"
    if not img_path.exists():
        img_path = data_dir / "images" / f"{frame_idx:06d}.jpg"
    mask_path = data_dir / "obj_masks" / f"{frame_idx:06d}.png"
    depth_path = data_dir / "depth_dyn" / f"{frame_idx:06d}.npy"
    if not depth_path.exists():
        depth_path = data_dir / "model_infer" / f"depth_{frame_idx:05d}.npy"
    k_path = data_dir / "intrinsics" / f"{frame_idx:06d}.npy"
    
    if not img_path.exists(): return None
    
    image = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_RGB2BGR) # Correcting RGB/BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
    if mask is not None and mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    depth = np.load(depth_path).astype(np.float32) if depth_path.exists() else None
    if depth is not None and depth.shape[:2] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
    K = np.load(k_path)
    
    # Load GT poses if available for evaluation
    T_CW_gt = np.load(data_dir / "extrinsics" / f"{frame_idx:06d}.npy")
    T_WO_gt = None
    obj_pose_path = data_dir / "object_poses.txt"
    if obj_pose_path.exists():
        with open(obj_pose_path, "r") as f:
            lines = f.readlines()
            if frame_idx < len(lines):
                parts = [float(x) for x in lines[frame_idx].split()]
                # parts[0] is timestamp, parts[1:4] is pos, parts[4:8] is quat (xyzw)
                t = np.array(parts[1:4])
                q = np.array(parts[4:8])
                T_WO_gt = np.eye(4)
                T_WO_gt[:3, :3] = R.from_quat(q).as_matrix()
                T_WO_gt[:3, 3] = t
                
    return {
        "image": image, "mask": mask, "depth": depth, "K": K, 
        "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": frame_idx
    }

def main(cfg: Config):
    data_dir = Path(cfg.data_root) / cfg.clip_id
    device = torch.device(cfg.device)
    
    if not cfg.no_vis:
        rr.init(f"HOT3D_Tracker_{cfg.clip_id}")
        # rr.spawn()
        rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")
        
    tracker_cfg = GeoTrackerConfig(
        feature_type=cfg.feature_type,
        n_features=cfg.n_features,
        use_photometric_refinement=cfg.use_photometric,
        grid_spacing=6,
        ransac_thresh=cfg.ransac_thresh,
        n_init_frames=cfg.n_init_frames,
        max_pose_jump=cfg.max_pose_jump,
        informed_thresh=cfg.informed_thresh,
        use_arap_flow_lifting=cfg.use_arap_flow_lifting
    )
    
    map_cfg = MappingConfig(
        device=cfg.device,
        num_steps_per_frame=cfg.num_steps, 
        pyr_levels=cfg.pyr_levels,
        mask_loss_weight=cfg.mask_loss_weight,
        densify_every=cfg.densify_every,
        use_arap=cfg.use_arap,
        arap_weight=cfg.arap_weight,
        arap_rot_weight=cfg.arap_rot_weight,
        arap_k=cfg.arap_k,
        arap_warmup_steps=cfg.arap_warmup_steps
    )
    
    tracker = GeometricTracker(tracker_cfg)
    poses_wo = {} # idx -> T_WO
    K_dict = {}
    
    # 1. Initialize at Frame 0
    f0 = load_frame_data(data_dir, cfg.init_frame)
    h, w = f0["image"].shape[:2]
    
    T_WO_init = f0["T_WO_gt"] if f0["T_WO_gt"] is not None else np.eye(4)
    T_C0O_init = f0["T_CW_gt"] @ T_WO_init
    
    poses_wo[f0["frame_idx"]] = T_WO_init
    tracker.poses[f0["frame_idx"]] = T_C0O_init
    tracker.K_dict[f0["frame_idx"]] = f0["K"]
    K_dict[f0["frame_idx"]] = f0["K"]
    
    # Initialize GS Model
    print(f"Initializing GS model at Frame 0")
    mask_init = f0["mask"] > 0
    depth_t = torch.from_numpy(f0["depth"]).float().to(device)
    full_pts_c = unproject_depth(depth_t, torch.from_numpy(f0["K"]).float().to(device), h, w)
    normals_c, _ = d2n_tblr(full_pts_c.permute(2, 0, 1).unsqueeze(0))
    normals_c = F.normalize(normals_c[0], dim=0).cpu().numpy()
    
    ys, xs = np.where(mask_init)
    step = 4
    ys, xs = ys[::step], xs[::step]
    
    active_points, active_colors, active_normals = [], [], []
    for y, x in zip(ys, xs):
        p_c = full_pts_c[y, x].cpu().numpy()
        p_w = np.linalg.inv(f0["T_CW_gt"][:3, :3]) @ (p_c - f0["T_CW_gt"][:3, 3])
        p_o = np.linalg.inv(T_WO_init[:3, :3]) @ (p_w - T_WO_init[:3, 3])
        active_points.append(p_o)
        active_colors.append(f0["image"][y, x] / 255.0)
        active_normals.append(np.linalg.inv(T_WO_init[:3, :3]) @ (np.linalg.inv(f0["T_CW_gt"][:3, :3]) @ normals_c[:, y, x]))

    gs_params = init_gs_from_tracker_points(np.array(active_points), np.array(active_colors), device, normals=np.array(active_normals), gs_type=cfg.gs_type)
    obj_gs = ObjectGS(gs_params, T_WO_init, 1.0)
    # Render to get a clean torch reference image
    with torch.no_grad():
        T_C0O_init_t = torch.from_numpy(T_C0O_init).float().to(device)
        K0_t = torch.from_numpy(f0["K"]).float().to(device)
        render_mode = "3dgs" if cfg.gs_type == "3d" else "normal"
        img_ref, _, _, alpha_ref = obj_gs.gs_params.render(
            T_C0O_init_t, K0_t, width=w, height=h,
            mode=render_mode, near_plane=tracker_cfg.near_plane, far_plane=tracker_cfg.far_plane
        )
        obj_gs.update_reference(img_ref, T_C0O_init_t, alpha_ref)
    
    mapper = GSMapping(map_cfg, obj_gs.gs_params)
    
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    prev_gray = cv2.cvtColor(f0["image"], cv2.COLOR_RGB2GRAY)
    
    all_dist_errors = []
    pbar = tqdm(range(1, cfg.n_frames_track), desc="Tracking + Mapping")
    for k in pbar:
        f_curr = load_frame_data(data_dir, cfg.init_frame + k)
        if f_curr is None: break
        idx = f_curr["frame_idx"]
        curr_gray = cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY)
        flow = dis.calc(prev_gray, curr_gray, None)
        prev_gray = curr_gray
        
        # 1. Motion Model (World-from-Object)
        if idx > cfg.init_frame + 1:
            T_WO_prev = poses_wo[idx-1]
            T_WO_pprev = poses_wo[idx-2]
            V = T_WO_prev @ np.linalg.inv(T_WO_pprev)
            T_WO_guess = V @ T_WO_prev
        else:
            T_WO_guess = poses_wo[idx-1].copy()
        
        T_CiO_guess = f_curr["T_CW_gt"] @ T_WO_guess

        # 2. Photometric Refinement
        T_WO_photo_np = T_WO_guess
        if cfg.use_photometric and obj_gs is not None:
            # Note: mapper.gs_params is the same object as obj_gs.gs_params
            T_WO_photo_t, _ = obj_gs.optimize_wrt_image_lm(
                f_curr["T_CW_gt"], f_curr["image"], f_curr["K"], mask=f_curr["mask"],
                near_plane=tracker_cfg.near_plane, far_plane=tracker_cfg.far_plane,
                damping=0.01,
                pyramid_levels=[(4, 10), (2, 10), (1, 20)],
                gs_type=cfg.gs_type
            )
            T_CiO_in = (f_curr["T_CW_gt"] @ T_WO_photo_t.detach().cpu().numpy())
        else:
            T_CiO_in = T_CiO_guess

        # 3. Geometric Tracking
        success, n_inliers = tracker.step_informed_with_occlusion(idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], T_CiO_in, flow_prev_curr=flow)
        
        T_CO = tracker.poses[idx]
        poses_wo[idx] = np.linalg.inv(f_curr["T_CW_gt"]) @ T_CO
        
        # 4. Mapping Update
        mapper.optimize_frame({
            "image": f_curr["image"], "mask": f_curr["mask"], "depth": f_curr["depth"],
            "K": f_curr["K"], "T_CiO": T_CO, "frame_idx": idx
        }, num_steps=cfg.num_steps, frame_count=k)
        
        if idx % cfg.kf_every == 0:
            tracker.keyframes.append(idx)
            # Add points periodically
            tracker.add_new_points_from_depth(idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], T_CO, align=True)
            
        # Logging
        T_CiO_gt_actual = f_curr["T_CW_gt"] @ f_curr["T_WO_gt"] if f_curr["T_WO_gt"] is not None else np.eye(4)
        error = np.linalg.norm(T_CO[:3, 3] - T_CiO_gt_actual[:3, 3])
        all_dist_errors.append(error)
        pbar.set_description(f"Frame {idx}, Inliers: {n_inliers}, ATE: {error:.4f}m")
        
        if not cfg.no_vis:
            rr.set_time("frame_idx", sequence=idx)
            rr.log("object/pose", rr.Transform3D(mat3x3=T_CO[:3, :3], translation=T_CO[:3, 3]))
            rr.log("input/image", rr.Image(f_curr["image"]))

    print(f"\n>>> Final ATE (30 frames): {np.mean(all_dist_errors):.4f}m")
    
    # Evaluation
    print("\n>>> Phase 2: Evaluation")
    eval_psnrs = []
    for i in tqdm(range(cfg.n_frames_eval), desc="Phase 2: Eval"):
        idx = cfg.init_frame + cfg.n_frames_track + i
        fd = load_frame_data(data_dir, idx)
        if fd is None: break
        T_CiO_gt = fd["T_CW_gt"] @ fd["T_WO_gt"]
        with torch.no_grad():
            render_image, _, _, _ = mapper.gs_params.render(
                viewmat=torch.from_numpy(T_CiO_gt).float().to(device), K=torch.from_numpy(fd["K"]).float().to(device), width=w, height=h
            )
            target_image = torch.from_numpy(fd["image"]).float().to(device).permute(2, 0, 1) / 255.0
            mask_t = torch.from_numpy(fd["mask"] > 0).to(device)
            if mask_t.any():
                mse = torch.mean((render_image.clamp(0, 1)[:, mask_t] - target_image[:, mask_t])**2)
                psnr = -10.0 * torch.log10(mse + 1e-10)
                eval_psnrs.append(psnr.item())
    
    if eval_psnrs: print(f"Average PSNR: {np.mean(eval_psnrs):.2f} dB")
    mapper.stop()

if __name__ == "__main__":
    main(tyro.cli(Config))
