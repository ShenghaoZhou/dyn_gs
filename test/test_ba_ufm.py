import numpy as np
import cv2
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from gs_dyn_obj.utils.init import unproject_depth
from scipy.spatial.transform import Rotation as R
import pycolmap
import shutil
import sqlite3
import flow_vis
from uniflowmatch.models.ufm import UniFlowMatchClassificationRefinement, UniFlowMatchConfidence

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 35
    n_frames: int = 30
    device: str = "cuda"
    max_features: int = 2000 # Number of points to sample per frame
    min_features: int = 1000 # Threshold to trigger new sampling
    min_track_len: int = 5
    output_dir: str = "outputs/test_ba_ufm"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    global_align: bool = False
    global_first_n: int = -1
    ufm_model: str = "base" # base, refine, base-980, refine-980
    match_stride: int = 1 # Step size for frame-to-frame matching

def umeyama(src, dst):
    """
    Computes Sim(3) transform: dst = s * R * src + t
    """
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    
    s_centered = src - mu_s
    d_centered = dst - mu_d
    
    C = d_centered.T @ s_centered / len(src)
    U, S, Vh = np.linalg.svd(C)
    
    # determinant check for reflection
    d = np.linalg.det(U @ Vh)
    S_mat = np.eye(3)
    if d < 0:
        S_mat[2, 2] = -1
    
    R = U @ S_mat @ Vh
    
    var_s = np.var(src, axis=0).sum()
    s = np.trace(np.diag(S) @ S_mat) / (var_s + 1e-8)
    t = mu_d - s * R @ mu_s
    
    return s, R, t

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

def predict_ufm_correspondences(model, source_image, target_image):
    """Predict correspondences between source and target images."""
    device = next(model.parameters()).device
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            # Input images should be [H, W, 3] and uint8
            result = model.predict_correspondences_batched(
                source_image=torch.from_numpy(source_image).to(device),
                target_image=torch.from_numpy(target_image).to(device),
            )
        # flow: [2, H, W], covisibility: [H, W]
        flow_output = result.flow.flow_output[0].cpu().float()
        covisibility = result.covisibility.mask[0].cpu().float()
    return flow_output, covisibility

def main(cfg: Config):
    # Initialize Rerun
    rr.init("test_ba_ufm", spawn=False)
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)
    
    data_dir = Path(cfg.data_root)
    device = torch.device(cfg.device)

    # 1. Load constants
    init_stem = f"{cfg.init_frame:06d}"
    T_WO_init = load_object_pose_world(cfg.data_root, cfg.init_frame)
    if T_WO_init is None: return

    # Output directories
    out_dir = Path(cfg.output_dir) / data_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    img_tmp_dir = out_dir / "images"
    if img_tmp_dir.exists():
        shutil.rmtree(img_tmp_dir)
    img_tmp_dir.mkdir(parents=True)
    db_path = out_dir / "database.db"
    if db_path.exists():
        db_path.unlink()

    # Load UFM model
    print(f"Loading UFM {cfg.ufm_model} model...")
    model_repo_map = {
        "base"              : "infinity1096/UFM-Base",
        "refine"            : "infinity1096/UFM-Refine",
        "base-980"          : "infinity1096/UFM-Base-980",
        "refine-980"        : "infinity1096/UFM-Refine-980",
    }
    if "base" in cfg.ufm_model:
        model = UniFlowMatchConfidence.from_pretrained(model_repo_map[cfg.ufm_model])
    elif "refine" in cfg.ufm_model:
        model = UniFlowMatchClassificationRefinement.from_pretrained(model_repo_map[cfg.ufm_model])
    else:
        raise ValueError("Invalid model variant")
    model.to(device)
    model.eval()

    # Tracking setup
    tracks = {}
    frame_data = {}
    indices = []
    
    # Track state
    prev_image = None
    prev_pts = None # [N, 2]
    prev_ids = []
    next_id = 0

    print(f"Tracking over {cfg.n_frames} frames with UFM...")
    for i in tqdm(range(cfg.n_frames)):
        frame_idx = cfg.init_frame + i
        stem = f"{frame_idx:06d}"
        img_path = data_dir / "images" / f"{stem}.png"
        if not img_path.exists(): break
            
        image = np.array(cv2.imread(str(img_path))[..., ::-1])
        mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        H, W = image.shape[:2]
        
        depth_path = data_dir / "depth_dyn" / f"{stem}.npy"
        if not depth_path.exists():
            depth_path = data_dir / "depth_cache" / "DA3METRIC-LARGE" / f"{stem}.npy"
        depth = np.load(depth_path)

        T_CW_actual = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        T_WO_curr = load_object_pose_world(cfg.data_root, frame_idx)
        if T_WO_curr is None: break

        indices.append(frame_idx)
        img_name = f"image_{i:06d}.png"
        masked_img = image.copy()
        masked_img[mask == 0] = 0
        cv2.imwrite(str(img_tmp_dir / img_name), cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

        frame_data[frame_idx] = {
            "K": K,
            "image": image,
            "depth": depth,
            "mask": mask,
            "T_CW_actual": T_CW_actual,
            "T_WO_curr": T_WO_curr,
            "img_name": img_name
        }

        # UFM Tracking
        current_pts = []
        current_ids = []

        if prev_image is not None and len(prev_pts) > 0:
            # Predict flow from prev to current
            flow, covis = predict_ufm_correspondences(model, prev_image, image)
            flow_np = flow.permute(1, 2, 0).numpy()
            covis_np = covis.numpy()

            # Visualize flow in Rerun
            rr.set_time("frame", sequence=frame_idx)
            flow_vis_image = flow_vis.flow_to_color(flow_np)
            rr.log("ufm/flow", rr.Image(flow_vis_image))
            rr.log("ufm/covisibility", rr.Image(covis_np))

            # Warp previous points
            for pt, tid in zip(prev_pts, prev_ids):
                x, y = int(round(pt[0])), int(round(pt[1]))
                if 0 <= x < W and 0 <= y < H:
                    # Check covisibility
                    if covis_np[y, x] > 0.5:
                        dx, dy = flow_np[y, x]
                        nx, ny = pt[0] + dx, pt[1] + dy
                        ix, iy = int(round(nx)), int(round(ny))
                        # Check if new point is in mask and within bounds
                        if 0 <= ix < W and 0 <= iy < H and mask[iy, ix] > 0:
                            current_pts.append([nx, ny])
                            current_ids.append(tid)
                            tracks[tid][frame_idx] = np.array([nx, ny], dtype=np.float32)

        # Resample if needed
        if len(current_pts) < cfg.min_features:
            # Create a mask for detector that avoids current points
            det_mask = mask.copy()
            for pt in current_pts:
                cv2.circle(det_mask, (int(round(pt[0])), int(round(pt[1]))), 5, 0, -1)
            
            # Sample new points from the remaining mask
            mask_y, mask_x = np.where(det_mask > 0)
            if len(mask_y) > 0:
                num_to_sample = min(cfg.max_features - len(current_pts), len(mask_y))
                indices_sampled = np.random.choice(len(mask_y), num_to_sample, replace=False)
                new_y = mask_y[indices_sampled]
                new_x = mask_x[indices_sampled]
                
                for nx, ny in zip(new_x, new_y):
                    tid = next_id
                    next_id += 1
                    current_pts.append([float(nx), float(ny)])
                    current_ids.append(tid)
                    tracks[tid] = {frame_idx: np.array([float(nx), float(ny)], dtype=np.float32)}

        prev_image = image.copy()
        prev_pts = np.array(current_pts)
        prev_ids = current_ids

    # Filter tracks
    valid_tracks = {tid: t for tid, t in tracks.items() if len(t) >= cfg.min_track_len}
    print(f"Total tracks: {len(tracks)}, Valid tracks: {len(valid_tracks)}")

    # Populate COLMAP database
    db = pycolmap.Database.open(str(db_path))
    K_ref = frame_data[indices[0]]["K"]
    img_shape = frame_data[indices[0]]["image"].shape
    camera = pycolmap.Camera(model="PINHOLE", width=img_shape[1], height=img_shape[0], 
                             params=[K_ref[0,0], K_ref[1,1], K_ref[0,2], K_ref[1,2]])
    cam_id = db.write_camera(camera)
    
    image_ids = {}
    tid_to_local_idx = {}
    for i, idx in enumerate(indices):
        img_name = frame_data[idx]["img_name"]
        image_id = db.write_image(pycolmap.Image(name=img_name, camera_id=cam_id))
        image_ids[idx] = image_id
        
        frame_tids = sorted([tid for tid in valid_tracks if idx in valid_tracks[tid]])
        tid_to_local_idx[idx] = {tid: j for j, tid in enumerate(frame_tids)}
        kpts = np.array([valid_tracks[tid][idx] for tid in frame_tids], dtype=np.float32)
        kpts_full = np.zeros((len(kpts), 4), dtype=np.float32)
        kpts_full[:, :2] = kpts
        kpts_full[:, 2] = 1.0
        db.write_keypoints(image_id, kpts_full)

    pairs = []
    for i in range(len(indices)):
        for j in range(i + 1, min(i + 5, len(indices))):
            idx1, idx2 = indices[i], indices[j]
            common_tids = sorted(list(set(tid_to_local_idx[idx1].keys()) & set(tid_to_local_idx[idx2].keys())))
            if len(common_tids) < 15: continue
            matches = np.array([[tid_to_local_idx[idx1][tid], tid_to_local_idx[idx2][tid]] for tid in common_tids], dtype=np.uint32)
            db.write_matches(image_ids[idx1], image_ids[idx2], matches)
            pairs.append((frame_data[idx1]["img_name"], frame_data[idx2]["img_name"]))
    db.close()

    pairs_path = out_dir / "pairs.txt"
    with open(pairs_path, "w") as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")

    pycolmap.verify_matches(str(db_path), str(pairs_path))
    colmap_out = out_dir / "reconstruction"
    if colmap_out.exists(): shutil.rmtree(colmap_out)
    colmap_out.mkdir(exist_ok=True)
    reconstructions = pycolmap.incremental_mapping(str(db_path), str(img_tmp_dir), str(colmap_out))
    
    if not reconstructions:
        print("COLMAP failed."); return
    rec = sorted(reconstructions.values(), key=lambda x: x.num_points3D(), reverse=True)[0]
    
    if rec.num_reg_images() < 2:
        print("COLMAP reconstruction too small."); return

    # --- Alignment and Scaling ---
    if not cfg.global_align:
        print("Using first-frame alignment and depth scaling...")
        first_idx = -1
        for idx in indices:
            img_name = frame_data[idx]["img_name"]
            if any(img.name == img_name for img in rec.images.values()):
                first_idx = idx
                break
        
        if first_idx == -1:
            print("No registered images found in COLMAP."); return

        colmap_img0 = [img for img in rec.images.values() if img.name == frame_data[first_idx]["img_name"]][0]
        T_C0O_col = np.eye(4)
        T_C0O_col[:3, :] = colmap_img0.cam_from_world().matrix()
        
        depth_gt = frame_data[first_idx]["depth"]
        scales = []
        for pid, point in rec.points3D.items():
            for track_el in point.track.elements:
                if track_el.image_id == colmap_img0.image_id:
                    p_C0 = T_C0O_col[:3, :3] @ point.xyz + T_C0O_col[:3, 3]
                    z_col = p_C0[2]
                    if z_col <= 0: continue
                    p2d = colmap_img0.points2D[track_el.point2D_idx].xy
                    u, v = int(round(p2d[0])), int(round(p2d[1]))
                    if 0 <= u < depth_gt.shape[1] and 0 <= v < depth_gt.shape[0]:
                        z_gt = depth_gt[v, u]
                        if z_gt > 0.01:
                            scales.append(z_gt / z_col)
                    break
        
        scale_factor = np.median(scales) if scales else 1.0
        print(f"Scale factor (depth-based): {scale_factor}")

        T_CW0 = frame_data[first_idx]["T_CW_actual"]
        T_WO0 = frame_data[first_idx]["T_WO_curr"]
        T_C0O_gt = T_CW0 @ T_WO0
        
        R_col0 = T_C0O_col[:3, :3]
        t_col0 = T_C0O_col[:3, 3]
        R_gt0 = T_C0O_gt[:3, :3]
        t_gt0 = T_C0O_gt[:3, 3]
        
        R_align = R_gt0.T @ R_col0
        t_align = R_gt0.T @ (scale_factor * t_col0 - t_gt0)
    else:
        print("Using global Sim(3) alignment (Umeyama)...")
        traj_col_cam = []
        traj_gt_cam = []
        reg_indices = []
        for idx in indices:
            img_name = frame_data[idx]["img_name"]
            col_img_matches = [img for img in rec.images.values() if img.name == img_name]
            if col_img_matches:
                col_img = col_img_matches[0]
                T_CiO_col = np.eye(4)
                T_CiO_col[:3, :] = col_img.cam_from_world().matrix()
                C_O_col = -T_CiO_col[:3, :3].T @ T_CiO_col[:3, 3]
                T_CiO_gt = frame_data[idx]["T_CW_actual"] @ frame_data[idx]["T_WO_curr"]
                C_O_gt = -T_CiO_gt[:3, :3].T @ T_CiO_gt[:3, 3]
                traj_col_cam.append(C_O_col)
                traj_gt_cam.append(C_O_gt)
                reg_indices.append(idx)
        
        if len(reg_indices) < 3:
            print("Not enough points for Sim(3) alignment."); return
            
        traj_col_cam = np.array(traj_col_cam)
        traj_gt_cam = np.array(traj_gt_cam)
        
        if cfg.global_first_n > 0:
            n_align = min(cfg.global_first_n, len(traj_col_cam))
            scale_factor, R_align, t_align = umeyama(traj_col_cam[:n_align], traj_gt_cam[:n_align])
        else:
            scale_factor, R_align, t_align = umeyama(traj_col_cam, traj_gt_cam)
            
        print(f"Scale factor (Sim3): {scale_factor:.4f}")

    # Transform all COLMAP points to GT-object frame
    xyz_col = []
    rgb_col = []
    for pid, point in rec.points3D.items():
        p_aligned = R_align @ (scale_factor * point.xyz) + t_align
        xyz_col.append(p_aligned)
        rgb_col.append(point.color)
    xyz_col = np.array(xyz_col)
    rgb_col = np.array(rgb_col)

    # --- Visualization ---
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    trajectory_gt = []
    trajectory_col = []

    for i, idx in enumerate(indices):
        rr.set_time("frame", sequence=idx)
        fd = frame_data[idx]
        
        T_CW_actual = fd["T_CW_actual"]
        T_WO_curr = fd["T_WO_curr"]
        T_C_O_curr = T_CW_actual @ T_WO_curr
        T_WC_viz_gt = T_WO_init @ np.linalg.inv(T_C_O_curr)
        
        xyz_C = unproject_depth(torch.from_numpy(fd["depth"]).float().to(device), 
                                torch.from_numpy(fd["K"]).float().to(device), 
                                fd["image"].shape[0], fd["image"].shape[1]).reshape(-1, 3).cpu().numpy()
        mask_flat = fd["mask"].reshape(-1) > 0
        valid = (fd["depth"].reshape(-1)[mask_flat] > 0.01)
        pts_C_masked = xyz_C[mask_flat][valid]
        colors_masked = fd["image"].reshape(-1, 3)[mask_flat][valid]
        pts_W_viz_gt = (T_WC_viz_gt[:3, :3] @ pts_C_masked.T).T + T_WC_viz_gt[:3, 3]
        
        rr.log("world/gt_pc_frame", rr.Points3D(pts_W_viz_gt, colors=colors_masked, radii=0.001))
        rr.log("world/camera_gt", rr.Pinhole(image_from_camera=fd["K"], width=fd["image"].shape[1], height=fd["image"].shape[0]))
        rr.log("world/camera_gt", rr.Transform3D(mat3x3=T_WC_viz_gt[:3, :3], translation=T_WC_viz_gt[:3, 3]))
        rr.log("world/camera_gt/image", rr.Image(fd["image"]))
        
        trajectory_gt.append(T_WC_viz_gt[:3, 3])
        
        img_name = fd["img_name"]
        col_img_matches = [img for img in rec.images.values() if img.name == img_name]
        if col_img_matches:
            col_img = col_img_matches[0]
            T_CiO_col = np.eye(4)
            T_CiO_col[:3, :] = col_img.cam_from_world().matrix()
            
            C_O_col = -T_CiO_col[:3, :3].T @ T_CiO_col[:3, 3]
            C_O_gt_rec = R_align @ (scale_factor * C_O_col) + t_align
            R_CiO_gt_rec = T_CiO_col[:3, :3] @ R_align.T
            
            T_WC_viz_col = np.eye(4)
            T_WC_viz_col[:3, :3] = T_WO_init[:3, :3] @ R_CiO_gt_rec.T
            T_WC_viz_col[:3, 3] = T_WO_init[:3, :3] @ C_O_gt_rec + T_WO_init[:3, 3]
            
            rr.log("world/camera_colmap", rr.Pinhole(image_from_camera=fd["K"], width=fd["image"].shape[1], height=fd["image"].shape[0]))
            rr.log("world/camera_colmap", rr.Transform3D(mat3x3=T_WC_viz_col[:3, :3], translation=T_WC_viz_col[:3, 3]))
            
            trajectory_col.append(T_WC_viz_col[:3, 3])
            
            pts_W_viz_col = (T_WO_init[:3, :3] @ xyz_col.T).T + T_WO_init[:3, 3]
            rr.log("world/colmap_points", rr.Points3D(pts_W_viz_col, colors=rgb_col, radii=0.002))

        if len(trajectory_gt) > 1:
            rr.log("world/trajectories/gt", rr.LineStrips3D([np.array(trajectory_gt)], colors=[[0, 255, 0]], radii=0.002))
        if len(trajectory_col) > 1:
            rr.log("world/trajectories/colmap", rr.LineStrips3D([np.array(trajectory_col)], colors=[[0, 0, 255]], radii=0.002))

    print("Done.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
