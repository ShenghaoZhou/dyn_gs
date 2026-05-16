import numpy as np
import cv2
import torch
import rerun as rr
import rerun.blueprint as rrb
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from scipy.spatial.transform import Rotation as R
import poselib
import random
from geometric_tracker import run_ba, sample_grid_on_mask, interpolate_flow

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    start_idx: int = 0
    window_size: int = 5
    path: str = "C" # "A", "B", or "C"
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    grid_spacing: int = 4
    
    # Tracking Parameters
    ransac_thresh: float = 1.0
    
    # Depth source
    use_gt_depth: bool = False
    no_vis: bool = False
    feature_type: str = "flow" # "flow" or "orb"
    n_features: int = 2000

def load_object_pose_world(data_root, frame_idx):
# ... (rest of the functions remain the same) ...
# (skipping to run_vis modification)
    poses_file = Path(data_root) / "object_poses.txt"
    if not poses_file.exists(): return None
    with open(poses_file, "r") as f:
        lines = f.readlines()
    if frame_idx >= len(lines): return None
    line = lines[frame_idx].split()
    t_WO = np.array([float(line[1]), float(line[2]), float(line[3])])
    q_WO = np.array([float(line[4]), float(line[5]), float(line[6]), float(line[7])])
    r_WO = R.from_quat(q_WO).as_matrix()
    T_WO = np.eye(4)
    T_WO[:3, :3] = r_WO
    T_WO[:3, 3] = t_WO
    return T_WO

def triangulate_linear(P1, P2, pts1, pts2):
    """Linear triangulation for points in 3D."""
    pts3d = []
    for i in range(len(pts1)):
        A = np.zeros((4, 4))
        A[0] = pts1[i, 0] * P1[2, :] - P1[0, :]
        A[1] = pts1[i, 1] * P1[2, :] - P1[1, :]
        A[2] = pts2[i, 0] * P2[2, :] - P2[0, :]
        A[3] = pts2[i, 1] * P2[2, :] - P2[1, :]
        _, _, vh = np.linalg.svd(A)
        p3d = vh[-1, :3] / vh[-1, 3]
        pts3d.append(p3d)
    return np.array(pts3d)

def get_cam_dict(K, W, H):
    return {'model': 'PINHOLE', 'width': W, 'height': H, 'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}

def load_frame(data_dir, idx, use_gt_depth):
    stem = f"{idx:06d}"
    img_path = data_dir / "images" / f"{stem}.png"
    if not img_path.exists(): return None
    img = np.array(cv2.imread(str(img_path))[..., ::-1])
    mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
    
    if use_gt_depth:
        depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
    else:
        depth_path = data_dir / "model_infer" / f"depth_{idx:05d}.npy"
        if not depth_path.exists(): depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
    
    if not depth_path.exists(): return None
    depth = np.load(depth_path)
    if depth.shape != img.shape[:2]:
        depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        
    K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
    T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
    T_WO_gt = load_object_pose_world(data_dir, idx)
    return img, mask, depth, K, T_CW_gt, T_WO_gt

def run_vis(cfg: Config):
    if not cfg.no_vis:
        rr.init("init_window_vis", spawn=False)
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
    
    # Load frames for the window
    frames = []
    for i in range(cfg.window_size):
        idx = cfg.start_idx + i
        fd = load_frame(data_dir, idx, cfg.use_gt_depth)
        if fd is None:
            print(f"Error loading frame {idx}")
            return
        frames.append({"image": fd[0], "mask": fd[1], "depth": fd[2], "K": fd[3], 
                       "T_CW_gt": fd[4], "T_WO_gt": fd[5], "frame_idx": idx})
    
    H, W = frames[0]["image"].shape[:2]
    
    # Ground Truth object-centric trajectory
    # We anchor everything to the first frame's object pose
    T_OC0_gt = np.linalg.inv(frames[0]["T_CW_gt"] @ frames[0]["T_WO_gt"])
    traj_gt_obj = []
    for f in frames:
        T_CO_gt = f["T_CW_gt"] @ f["T_WO_gt"]
        T_OC_gt = np.linalg.inv(T_CO_gt)
        traj_gt_obj.append(T_OC_gt[:3, 3])
    traj_gt_obj = np.array(traj_gt_obj)

    # --- Initial Depth Scale Ratio (GT / Inferred) ---
    f0 = frames[0]
    stem = f"{f0['frame_idx']:06d}"
    gt_depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
    if gt_depth_path.exists():
        depth_gt = np.load(gt_depth_path)
        if depth_gt.shape != f0["image"].shape[:2]:
            depth_gt = cv2.resize(depth_gt, (W, H), interpolation=cv2.INTER_NEAREST)
        
        mask = f0["mask"] > 0
        valid = (depth_gt > 0.01) & (f0["depth"] > 0.01) & mask
        if np.any(valid):
            initial_ratio = np.median(depth_gt[valid] / f0["depth"][valid])
            print(f"Initial Depth Scale Ratio (GT/Inferred): {initial_ratio:.6f}")
            if not cfg.no_vis:
                rr.log("logs/initial_depth_scale_ratio", rr.Scalars(initial_ratio))

    # --- Log GT Points (Static for the window) ---
    if not cfg.no_vis:
        # (Using the depth_gt loaded above if available)
        if gt_depth_path.exists():
            mask_gt = f0["mask"]
            pts_uv = sample_grid_on_mask(mask_gt, cfg.grid_spacing)
            K0_inv = np.linalg.inv(f0["K"])
            
            pts3d_gt_C0 = []
            for uv in pts_uv:
                d = depth_gt[int(round(uv[1])), int(round(uv[0]))]
                if d > 0:
                    pt3d_C0 = (K0_inv @ np.array([uv[0], uv[1], 1.0])) * d
                    pts3d_gt_C0.append(pt3d_C0)
            
            if pts3d_gt_C0:
                pts3d_gt_O = [(T_OC0_gt[:3, :3] @ p + T_OC0_gt[:3, 3]) for p in pts3d_gt_C0]
                rr.log("object/points_gt", rr.Points3D(pts3d_gt_O, colors=[0, 255, 0], radii=0.001), static=True)

    # Tracking / Correspondence finding
    if cfg.feature_type == "orb":
        print(f"Sampling initial points using ORB detection (n={cfg.n_features})...")
        orb = cv2.ORB_create(nfeatures=cfg.n_features)
        gray0 = cv2.cvtColor(frames[0]["image"], cv2.COLOR_RGB2GRAY)
        kpts = orb.detect(gray0, mask=frames[0]["mask"])
        current_pts = np.array([kp.pt for kp in kpts], dtype=np.float32)
    else:
        print(f"Sampling initial points using regular grid (spacing={cfg.grid_spacing})...")
        current_pts = sample_grid_on_mask(frames[0]["mask"], cfg.grid_spacing)

    # Tracking with DIS flow (Always used for correspondences)
    print("Tracking points through window using DIS Optical Flow...")
    tracks = {i: {"obs": {frames[0]["frame_idx"]: p}, "id": i} for i, p in enumerate(current_pts)}
    for i in range(len(frames) - 1):
        f_prev, f_curr = frames[i], frames[i+1]
        flow = dis.calc(cv2.cvtColor(f_prev["image"], cv2.COLOR_RGB2GRAY), cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY), None)
        valid_tids = [tid for tid in tracks if f_prev["frame_idx"] in tracks[tid]["obs"]]
        pts_prev = np.array([tracks[tid]["obs"][f_prev["frame_idx"]] for tid in valid_tids])
        pts_next = pts_prev + interpolate_flow(flow, pts_prev)
        ix, iy = np.round(pts_next[:, 0]).astype(int), np.round(pts_next[:, 1]).astype(int)
        valid = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        for j, tid in enumerate(valid_tids):
            if valid[j] and f_curr["mask"][iy[j], ix[j]] > 0:
                tracks[tid]["obs"][f_curr["frame_idx"]] = pts_next[j]

    full_tracks_orig = {tid: t for tid, t in tracks.items() if len(t["obs"]) == cfg.window_size}
    if len(full_tracks_orig) < 10: 
        print(f"Lack of tracks ({len(full_tracks_orig)})")
        return

    # Initialization according to path
    poses_est = {f["frame_idx"]: np.eye(4) for f in frames}
    tracks_est = {tid: {"obs": t["obs"].copy(), "id": tid} for tid, t in full_tracks_orig.items()}
    
    path_choice = cfg.path.upper()
    if path_choice == "A":
        print("Running Path A: PnP Initialization")
        f0 = frames[0]
        K0_inv = np.linalg.inv(f0["K"])
        for tid, t in tracks_est.items():
            uv = t["obs"][f0["frame_idx"]]
            d = f0["depth"][int(round(uv[1])), int(round(uv[0]))]
            t["pt3d"] = (K0_inv @ np.array([uv[0], uv[1], 1.0])) * d
            
        for i in range(1, cfg.window_size):
            f = frames[i]
            pts2d = np.array([t["obs"][f["frame_idx"]] for t in tracks_est.values()])
            pts3d = np.array([t["pt3d"] for t in tracks_est.values()])
            res, _ = poselib.estimate_absolute_pose(pts2d, pts3d, get_cam_dict(f["K"], W, H), {"max_reproj_error": cfg.ransac_thresh}, None)
            if res is not None:
                T = np.eye(4)
                T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix()
                T[:3, 3] = res.pose.t
                poses_est[f["frame_idx"]] = T
            else:
                poses_est[f["frame_idx"]] = poses_est[frames[i-1]["frame_idx"]]
                
    elif path_choice == "B":
        print("Running Path B: MonoDepth Relative (0-4) Initialization")
        f0, f4 = frames[0], frames[4]
        pts0 = np.array([t["obs"][f0["frame_idx"]] for t in tracks_est.values()])
        pts4 = np.array([t["obs"][f4["frame_idx"]] for t in tracks_est.values()])
        d0 = np.array([f0["depth"][int(round(p[1])), int(round(p[0]))] for p in pts0])
        d4 = np.array([f4["depth"][int(round(p[1])), int(round(p[0]))] for p in pts4])
        
        res_b, _ = poselib.estimate_monodepth_relative_pose(pts0, pts4, d0, d4, get_cam_dict(f0["K"], W, H), get_cam_dict(f4["K"], W, H), {"max_reproj_error": cfg.ransac_thresh})
        if res_b is not None:
            T40 = np.eye(4)
            T40[:3, :3] = R.from_quat([res_b.pose.q[1], res_b.pose.q[2], res_b.pose.q[3], res_b.pose.q[0]]).as_matrix()
            T40[:3, 3] = res_b.pose.t
            poses_est[f4["frame_idx"]] = T40
            pts3d_b = triangulate_linear(f0["K"] @ np.eye(3, 4), f4["K"] @ T40[:3, :], pts0, pts4)
            for i, tid in enumerate(tracks_est.keys()): tracks_est[tid]["pt3d"] = pts3d_b[i]
            
            for i in range(1, 4):
                f = frames[i]
                pts2d = np.array([t["obs"][f["frame_idx"]] for t in tracks_est.values()])
                pts3d_in = np.array([t["pt3d"] for t in tracks_est.values()])
                res_abs, _ = poselib.estimate_absolute_pose(pts2d, pts3d_in, get_cam_dict(f["K"], W, H), {"max_reproj_error": cfg.ransac_thresh}, None)
                if res_abs is not None:
                    T = np.eye(4)
                    T[:3, :3] = R.from_quat([res_abs.pose.q[1], res_abs.pose.q[2], res_abs.pose.q[3], res_abs.pose.q[0]]).as_matrix()
                    T[:3, 3] = res_abs.pose.t
                    poses_est[f["frame_idx"]] = T
                else:
                    poses_est[f["frame_idx"]] = poses_est[frames[i-1]["frame_idx"]]
        else:
            print("Path B failed to find relative pose.")
            return

    elif path_choice == "C":
        print("Running Path C: Multi-MonoDepth Relative Initialization")
        f0 = frames[0]
        pts0 = np.array([t["obs"][f0["frame_idx"]] for t in tracks_est.values()])
        d0 = np.array([f0["depth"][int(round(p[1])), int(round(p[0]))] for p in pts0])
        
        valid_c = True
        for i in range(1, cfg.window_size):
            fi = frames[i]
            ptsi = np.array([t["obs"][fi["frame_idx"]] for t in tracks_est.values()])
            di = np.array([fi["depth"][int(round(p[1])), int(round(p[0]))] for p in ptsi])
            res_c, _ = poselib.estimate_monodepth_relative_pose(pts0, ptsi, d0, di, get_cam_dict(f0["K"], W, H), get_cam_dict(fi["K"], W, H), {"max_reproj_error": cfg.ransac_thresh})
            if res_c is not None:
                Ti0 = np.eye(4)
                Ti0[:3, :3] = R.from_quat([res_c.pose.q[1], res_c.pose.q[2], res_c.pose.q[3], res_c.pose.q[0]]).as_matrix()
                Ti0[:3, 3] = res_c.pose.t
                poses_est[fi["frame_idx"]] = Ti0
            else:
                valid_c = False
                break
        
        if valid_c:
            # Triangulate using Frame 0 and Frame 4 (largest baseline in window)
            T40 = poses_est[frames[4]["frame_idx"]]
            pts4 = np.array([t["obs"][frames[4]["frame_idx"]] for t in tracks_est.values()])
            pts3d_c = triangulate_linear(f0["K"] @ np.eye(3, 4), frames[4]["K"] @ T40[:3, :], pts0, pts4)
            for i, tid in enumerate(tracks_est.keys()): tracks_est[tid]["pt3d"] = pts3d_c[i]
        else:
            print("Path C failed to find relative pose for one or more frames.")
            return
    else:
        print(f"Unknown path choice: {path_choice}")
        return

    # Run Bundle Adjustment
    print("Running Bundle Adjustment...")
    run_ba([f["frame_idx"] for f in frames], poses_est, tracks_est, {f["frame_idx"]: f["K"] for f in frames})

    # Visualization and Error Calculation
    traj_est_obj = []
    for f in frames:
        idx = f["frame_idx"]
        T_CiC0_est = poses_est[idx]
        T_OCi_est = T_OC0_gt @ np.linalg.inv(T_CiC0_est)
        traj_est_obj.append(T_OCi_est[:3, 3])
        
        if not cfg.no_vis:
            rr.set_time("frame", sequence=idx)
            rr.log("input/image", rr.Image(f["image"]))
            rr.log("input/mask", rr.Image(f["mask"]))
            
            # Log Estimated Camera
            rr.log("object/camera_est", rr.Transform3D(mat3x3=T_OCi_est[:3, :3], translation=T_OCi_est[:3, 3]))
            rr.log("object/camera_est", rr.Pinhole(image_from_camera=f["K"], width=W, height=H))
            
            # Log Ground Truth Camera
            T_CO_gt = f["T_CW_gt"] @ f["T_WO_gt"]
            T_OC_gt = np.linalg.inv(T_CO_gt)
            rr.log("object/camera_gt", rr.Transform3D(mat3x3=T_OC_gt[:3, :3], translation=T_OC_gt[:3, 3]))
            
            # Log active points
            pts3d = []
            colors = []
            for tid, t in tracks_est.items():
                if idx in t["obs"]:
                    # Tracks are in C0 frame, transform to Object frame
                    pt_C0 = t["pt3d"]
                    pt_O = (T_OC0_gt[:3, :3] @ pt_C0) + T_OC0_gt[:3, 3]
                    pts3d.append(pt_O)
                    colors.append([(tid * 13) % 256, (tid * 71) % 256, (tid * 113) % 256])
            
            if pts3d:
                rr.log("object/points", rr.Points3D(pts3d, colors=colors, radii=0.002))

            # Log trajectory so far
            rr.log("object/traj/est", rr.LineStrips3D([np.array(traj_est_obj)], colors=[[255, 0, 255]], radii=0.001))
            rr.log("object/traj/gt", rr.LineStrips3D([traj_gt_obj[:len(traj_est_obj)]], colors=[[0, 255, 0]], radii=0.001))

    traj_est_obj = np.array(traj_est_obj)
    errors = np.linalg.norm(traj_est_obj - traj_gt_obj, axis=1)
    ate = np.sqrt(np.mean(errors**2))
    
    print(f"\n--- Results for Path {path_choice} at start_idx {cfg.start_idx} ---")
    
    # Calculate scale ratio (GT / Est)
    f0 = frames[0]
    stem = f"{f0['frame_idx']:06d}"
    gt_depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
    if gt_depth_path.exists():
        depth_gt = np.load(gt_depth_path)
        if depth_gt.shape != f0["image"].shape[:2]:
            depth_gt = cv2.resize(depth_gt, (W, H), interpolation=cv2.INTER_NEAREST)
        
        ratios = []
        K0_inv = np.linalg.inv(f0["K"])
        for tid, t in tracks_est.items():
            if f0["frame_idx"] in t["obs"]:
                uv = t["obs"][f0["frame_idx"]]
                d_gt = depth_gt[int(round(uv[1])), int(round(uv[0]))]
                if d_gt > 0.01:
                    pt_est_C0 = t["pt3d"] # In C0 frame
                    dist_est = np.linalg.norm(pt_est_C0)
                    dist_gt = np.linalg.norm((K0_inv @ np.array([uv[0], uv[1], 1.0])) * d_gt)
                    if dist_est > 1e-6:
                        ratios.append(dist_gt / dist_est)
        
        if ratios:
            scale_ratio = np.median(ratios)
            print(f"Scale Ratio (GT/Est): {scale_ratio:.6f}")
            if not cfg.no_vis:
                rr.log("logs/optimized_scale_ratio", rr.Scalars(scale_ratio))

    print(f"ATE: {ate:.6f} m")
    if not cfg.no_vis:
        rr.log("logs/ate", rr.Scalars(ate))
        
    for i, err in enumerate(errors):
        print(f"Frame {frames[i]['frame_idx']} error: {err:.6f} m")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    run_vis(cfg)
