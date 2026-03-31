import cv2
import numpy as np
import rerun as rr
from data import HOT3DDataLoader
from pathlib import Path
import tyro
from dataclasses import dataclass
from tqdm import tqdm
import pyceres
import pycolmap
import pycolmap.cost_functions
import trimesh

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    data_seq: str = "clip-001924"
    max_features: int = 1000
    min_features: int = 500
    max_frames: int = 50
    skip_frames: int = 2
    min_track_len: int = 5
    huber_loss: float = 1.0
    output_dir: str = "outputs"
    scale: float = 1.0

def get_t_c_o(frame, obj_pose):
    T_c_w = frame["extrin"]
    T_w_o = obj_pose
    T_c_o = T_c_w @ T_w_o
    return T_c_o

def triangulate_track(track, poses, K):
    f_idxs = list(track.keys())
    f_idxs.sort()
    idx1, idx2 = f_idxs[0], f_idxs[-1]
    
    if idx1 not in poses or idx2 not in poses or idx1 == idx2:
        return None
        
    T1 = poses[idx1]
    T2 = poses[idx2]
    P1 = K @ T1[:3, :]
    P2 = K @ T2[:3, :]
    
    pt1 = track[idx1].reshape(-1, 2).astype(float)
    pt2 = track[idx2].reshape(-1, 2).astype(float)
    
    # Check baseline to avoid ill-conditioned triangulation
    C1 = -T1[:3, :3].T @ T1[:3, 3]
    C2 = -T2[:3, :3].T @ T2[:3, 3]
    if np.linalg.norm(C1 - C2) < 0.01: # Need at least 1cm baseline
        return None

    X4 = cv2.triangulatePoints(P1, P2, pt1.T, pt2.T)
    X3 = X4[:3, 0] / X4[3, 0]
    
    P_c1 = T1[:3, :3] @ X3 + T1[:3, 3]
    P_c2 = T2[:3, :3] @ X3 + T2[:3, 3]
    
    # Check if point is in front of the camera (Z > 0)
    if P_c1[2] <= 0 or P_c2[2] <= 0:
        return None
        
    return X3

def main(cfg: Config):
    data_loader = HOT3DDataLoader(Path(cfg.data_root) / cfg.data_seq)
    rr.init("dynamic_object_ba")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")

    indices = []
    if len(data_loader.dynamic_phases) > 0:
        start, _ = data_loader.dynamic_phases[0]
        end = start + cfg.max_frames * cfg.skip_frames
        indices = list(range(start, min(end, len(data_loader)), cfg.skip_frames))
    else:
        indices = list(range(0, min(len(data_loader), cfg.max_frames * cfg.skip_frames), cfg.skip_frames))

    # Data collection
    orb = cv2.ORB_create(nfeatures=cfg.max_features)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    prev_gray, prev_pts, prev_ids = None, None, []
    next_id = 0
    tracks = {}
    frame_data = {}

    print(f"Tracking over {len(indices)} frames...")
    for idx in tqdm(indices):
        f = data_loader[idx]
        gray = cv2.cvtColor(f["image"], cv2.COLOR_RGB2GRAY)
        mask = f["obj_mask"]
        
        frame_data[idx] = {
            "T_c_o_gt": get_t_c_o(f, data_loader.get_obj_pose(idx)),
            "K": f["K"],
            "image": f["image"],
            "T_c_w": f["extrin"]
        }
        
        current_pts, current_ids = None, []
        if prev_pts is not None and len(prev_pts) > 0:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            status = status.flatten().astype(bool)
            valid = []
            for i, p in enumerate(new_pts):
                if status[i]:
                    x, y = int(round(p[0][0])), int(round(p[0][1]))
                    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0:
                        valid.append(True)
                    else: valid.append(False)
                else: valid.append(False)
            valid = np.array(valid)
            if np.any(valid):
                current_pts = new_pts[valid]
                current_ids = [prev_ids[i] for i in range(len(valid)) if valid[i]]
                for pt, tid in zip(current_pts, current_ids):
                    tracks[tid][idx] = pt.flatten()

        if current_pts is None or len(current_pts) < cfg.min_features:
            det_mask = mask.copy()
            if current_pts is not None:
                for p in current_pts: cv2.circle(det_mask, (int(round(p[0][0])), int(round(p[0][1]))), 5, 0, -1)
            kps = orb.detect(gray, mask=det_mask)
            if kps:
                new_pts_arr = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)
                new_ids = list(range(next_id, next_id + len(kps)))
                next_id += len(kps)
                for pt, tid in zip(new_pts_arr, new_ids):
                    tracks[tid] = {idx: pt.flatten()}
                if current_pts is not None:
                    current_pts = np.vstack([current_pts, new_pts_arr])
                    current_ids.extend(new_ids)
                else:
                    current_pts, current_ids = new_pts_arr, new_ids
        
        prev_gray, prev_pts, prev_ids = gray.copy(), current_pts, current_ids

    valid_tracks = {tid: t for tid, t in tracks.items() if len(t) >= cfg.min_track_len}
    print(f"Triangulating {len(valid_tracks)} tracks...")
    pts3D_init = {}
    for tid, t_obs in valid_tracks.items():
        K = frame_data[min(t_obs.keys())]["K"]
        X = triangulate_track(t_obs, {i: frame_data[i]["T_c_o_gt"] for i in t_obs}, K)
        if X is not None: pts3D_init[tid] = X

    # Bundle Adjustment
    print("Starting Bundle Adjustment...")
    rec = pycolmap.Reconstruction()
    cam_id_map = {}
    for idx in indices:
        K = frame_data[idx]["K"]
        img = frame_data[idx]["image"]
        params = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        found_cid = None
        for cid, cam in rec.cameras.items():
            if np.allclose(cam.params, params):
                found_cid = cid
                break
        if found_cid is None:
            new_cid = len(rec.cameras)
            cam = pycolmap.Camera(model="PINHOLE", width=img.shape[1], height=img.shape[0], params=params)
            cam.camera_id = new_cid
            rec.add_camera(cam)
            cam_id_map[idx] = new_cid
        else: cam_id_map[idx] = found_cid

    # BA Parameters
    pose_params = {} # idx -> (q, t)
    point_params = {} # tid -> xyz
    prob = pyceres.Problem()
    loss = pyceres.HuberLoss(cfg.huber_loss) if cfg.huber_loss > 0 else pyceres.TrivialLoss()

    for idx in indices:
        T = frame_data[idx]["T_c_o_gt"]
        q = np.array(pycolmap.Rotation3d(T[:3, :3]).quat)
        t = np.array(T[:3, 3])
        pose_params[idx] = (q, t)
        prob.add_parameter_block(q, 4)
        prob.set_manifold(q, pyceres.EigenQuaternionManifold())
        prob.add_parameter_block(t, 3)

    for tid, X in pts3D_init.items():
        xyz = np.array(X)
        point_params[tid] = xyz
        prob.add_parameter_block(xyz, 3)

    # Residuals
    for tid, t_obs in valid_tracks.items():
        if tid not in pts3D_init: continue
        xyz = point_params[tid]
        for f_idx, uv in t_obs.items():
            if f_idx not in indices: continue
            q, t = pose_params[f_idx]
            cam = rec.cameras[cam_id_map[f_idx]]
            cost = pycolmap.cost_functions.ReprojErrorCost(cam.model, uv)
            prob.add_residual_block(cost, loss, [q, t, xyz, cam.params])

    # Constraints
    for cam in rec.cameras.values(): prob.set_parameter_block_constant(cam.params)
    first_idx = indices[0]
    prob.set_parameter_block_constant(pose_params[first_idx][0])
    prob.set_parameter_block_constant(pose_params[first_idx][1])

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    options.minimizer_progress_to_stdout = True
    options.max_num_iterations = 100
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    print(summary.BriefReport())

    # Calculate final reprojection errors
    reproj_errors = []
    for tid, t_obs in valid_tracks.items():
        if tid not in point_params: continue
        xyz = point_params[tid]
        for f_idx, uv in t_obs.items():
            if f_idx not in indices: continue
            q, t = pose_params[f_idx]
            cam = rec.cameras[cam_id_map[f_idx]]
            R = pycolmap.Rotation3d(q).matrix()
            P_c = R @ xyz + t
            if P_c[2] > 0:
                uv_proj = cam.img_from_cam(P_c)
                err = np.linalg.norm(uv_proj - uv)
                reproj_errors.append(err)
    
    if reproj_errors:
        print(f"Optimized Mean Reprojection Error: {np.mean(reproj_errors):.4f} pixels")
        print(f"Optimized Median Reprojection Error: {np.median(reproj_errors):.4f} pixels")
    else:
        print("No valid reprojection errors to report.")

    # Results & Visualization
    # Invert tracks for visualization: idx -> list of (tid, uv)
    idx_to_tracks = {idx: [] for idx in indices}
    for tid, t_obs in tracks.items():
        for idx, uv in t_obs.items():
            if idx in idx_to_tracks:
                idx_to_tracks[idx].append((tid, uv))

    rr.log("object", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)), static=True)
    rr.log("object/points/init", rr.Points3D(np.array(list(pts3D_init.values())) * cfg.scale, colors=[(100, 100, 100)], radii=0.002), static=True)
    rr.log("object/points/optimized", rr.Points3D(np.array(list(point_params.values())) * cfg.scale, colors=[(0, 255, 0)], radii=0.003), static=True)

    # World-centric logging
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rr.log("world/object/points", rr.Points3D(np.array(list(point_params.values())) * cfg.scale, colors=[(0, 255, 0)], radii=0.003), static=True)

    # Log GT object mesh
    obj_mesh = data_loader.obj_mesh
    if isinstance(obj_mesh, trimesh.Scene):
        # Merge all geometries into one mesh for simplicity
        obj_mesh = obj_mesh.dump(concatenate=True)
    
    # Extract vertices and faces
    vertices = obj_mesh.vertices
    faces = obj_mesh.faces
    
    # Some meshes might have colors
    mesh_colors = None
    if hasattr(obj_mesh.visual, 'vertex_colors'):
        mesh_colors = obj_mesh.visual.vertex_colors
    
    rr.log("world/object_gt/mesh", rr.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=faces,
        vertex_colors=mesh_colors,
        vertex_normals=obj_mesh.vertex_normals
    ), static=True)

    world_obj_positions = []
    gt_world_obj_positions = []
    for idx in indices:
        rr.set_time("frame_idx", sequence=idx)
        
        # Object-centric
        T_c_o_gt = frame_data[idx]["T_c_o_gt"].copy()
        T_c_o_gt[:3, 3] *= cfg.scale
        rr.log("object/camera_gt", rr.Transform3D(translation=T_c_o_gt[:3, 3], mat3x3=T_c_o_gt[:3, :3], relation=rr.TransformRelation.ChildFromParent))

        q_opt, t_opt = pose_params[idx]
        T_c_o_opt = np.eye(4)
        T_c_o_opt[:3, :] = pycolmap.Rigid3d(pycolmap.Rotation3d(q_opt), t_opt).matrix()
        T_c_o_opt_scaled = T_c_o_opt.copy()
        T_c_o_opt_scaled[:3, 3] *= cfg.scale
        rr.log("object/camera_opt", rr.Transform3D(translation=T_c_o_opt_scaled[:3, 3], mat3x3=T_c_o_opt_scaled[:3, :3], relation=rr.TransformRelation.ChildFromParent))
        
        # World-centric
        T_c_w = frame_data[idx]["T_c_w"]
        T_w_c = np.linalg.inv(T_c_w)
        rr.log("world/camera", rr.Transform3D(translation=T_w_c[:3, 3], mat3x3=T_w_c[:3, :3]))
        
        T_w_o_opt = T_w_c @ T_c_o_opt_scaled
        rr.log("world/object", rr.Transform3D(translation=T_w_o_opt[:3, 3], mat3x3=T_w_o_opt[:3, :3]))
        world_obj_positions.append(T_w_o_opt[:3, 3])

        # GT World-centric
        T_w_o_gt = data_loader.get_obj_pose(idx)
        rr.log("world/object_gt", rr.Transform3D(translation=T_w_o_gt[:3, 3], mat3x3=T_w_o_gt[:3, :3]))
        gt_world_obj_positions.append(T_w_o_gt[:3, 3])
        
        K = frame_data[idx]["K"]
        img = frame_data[idx]["image"]
        rr.log("world/camera/view", rr.Pinhole(resolution=(img.shape[1], img.shape[0]), image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF))
        rr.log("world/camera/view", rr.Image(img).compress(jpeg_quality=50))

        rr.log("object/camera_opt/view", rr.Pinhole(resolution=(img.shape[1], img.shape[0]), image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF))
        rr.log("object/camera_opt/view", rr.Image(img).compress(jpeg_quality=50))

        # Log feature tracks
        if idx in idx_to_tracks:
            tinfo = idx_to_tracks[idx]
            if tinfo:
                tids = [t[0] for t in tinfo]
                uvs = [t[1] for t in tinfo]
                # Green if triangulated, red otherwise
                colors = [(0, 255, 0) if tid in pts3D_init else (255, 0, 0) for tid in tids]
                rr.log("object/camera_opt/view/tracks", rr.Points2D(uvs, colors=colors, radii=2))

    if world_obj_positions:
        rr.log("world/trajectory", rr.LineStrips3D([np.array(world_obj_positions)], colors=[(255, 255, 0)], radii=0.002), static=True)
    
    if gt_world_obj_positions:
        rr.log("world/gt_trajectory", rr.LineStrips3D([np.array(gt_world_obj_positions)], colors=[(255, 0, 0)], radii=0.002), static=True)

    # Save Results
    out_dir = Path(cfg.output_dir) / cfg.data_seq
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Save optimized 3D points
    res_points = np.array(list(point_params.values())) * cfg.scale
    res_tids = np.array(list(point_params.keys()))
    
    # Save optimized poses (q, t) - note: these are unscaled raw BA outputs
    # Scaling is usually applied after loading if needed, or we could save scaled ones.
    # To be consistent with visual, let's save scaled translations.
    res_frame_indices = np.array(list(pose_params.keys()))
    res_pose_vectors = []
    for idx in res_frame_indices:
        q, t = pose_params[idx]
        t_scaled = t * cfg.scale
        res_pose_vectors.append(np.concatenate([q, t_scaled]))
    res_pose_vectors = np.array(res_pose_vectors)
    
    np.savez(out_dir / "reconstruction.npz", 
             points=res_points, 
             tids=res_tids, 
             frame_indices=res_frame_indices,
             pose_vectors=res_pose_vectors)
    
    # Save a PLY file for easy visualization
    if len(res_points) > 0:
        with open(out_dir / "reconstruction.ply", "w") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(res_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for p in res_points:
                f.write(f"{p[0]} {p[1]} {p[2]} 0 255 0\n")
            
    print(f"Final reconstruction saved to {out_dir}")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
