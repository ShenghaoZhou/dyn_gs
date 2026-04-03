import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from data import HOT3DDataLoader
from pathlib import Path
import tyro
from dataclasses import dataclass, field
from tqdm import tqdm
import pyceres
import pycolmap
import pycolmap.cost_functions
import trimesh
from typing import List

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
    scale: List[float] = field(default_factory=lambda: [1.0])
    use_gt_obj_pose: bool = False
    two_view_scale_with_gt: bool = False

def get_t_c_o(frame, obj_pose):
    T_c_w = frame["extrin"]
    T_w_o = obj_pose
    T_c_o = T_c_w @ T_w_o
    return T_c_o

def triangulate_track(pt1, pt2, K, T21):
    P1 = K @ np.eye(3, 4)
    P2 = K @ T21[:3, :]
    
    pt1_reshaped = pt1.reshape(-1, 2).astype(float)
    pt2_reshaped = pt2.reshape(-1, 2).astype(float)
    
    X4 = cv2.triangulatePoints(P1, P2, pt1_reshaped.T, pt2_reshaped.T)
    X3 = X4[:3, 0] / X4[3, 0]
    
    P_c1 = X3
    P_c2 = T21[:3, :3] @ X3 + T21[:3, 3]
    
    # Check if point is in front of the camera (Z > 0)
    if P_c1[2] <= 0 or P_c2[2] <= 0:
        return None
        
    return X3

def main(cfg: Config):
    data_loader = HOT3DDataLoader(Path(cfg.data_root) / cfg.data_seq)
    rr.init("dynamic_object_ba")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")

    # Define and send blueprint
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="World View", origin="world"),
            rrb.Vertical(
                rrb.Spatial3DView(name="Object Centric", origin="object"),
                rrb.Spatial2DView(name="Camera View", origin="world/camera/view"),
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

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

    # Pose Initialization
    T_c_o_init = {idx: frame_data[idx]["T_c_o_gt"] for idx in indices}

    valid_tracks = {tid: t for tid, t in tracks.items() if len(t) >= cfg.min_track_len}
    print(f"Triangulating {len(valid_tracks)} tracks using {'ground truth' if cfg.use_gt_obj_pose else 'essential matrix'} initialization...")
    
    rel_poses = {} # (idx1, idx2) -> T21
    pts3D_init = {}
    
    for tid, t_obs in tqdm(valid_tracks.items(), desc="Triangulating"):
        f_idxs = sorted(t_obs.keys())
        idx1, idx2 = f_idxs[0], f_idxs[-1]
        
        K = frame_data[idx1]["K"]
        pt1 = t_obs[idx1]
        pt2 = t_obs[idx2]
        
        if (idx1, idx2) not in rel_poses:
            if cfg.use_gt_obj_pose:
                T1 = T_c_o_init[idx1]
                T2 = T_c_o_init[idx2]
                
                # Check baseline
                C1 = -T1[:3, :3].T @ T1[:3, 3]
                C2 = -T2[:3, :3].T @ T2[:3, 3]
                if np.linalg.norm(C1 - C2) < 0.01:
                    rel_poses[(idx1, idx2)] = None
                else:
                    rel_poses[(idx1, idx2)] = T2 @ np.linalg.inv(T1)
            else:
                # Find all common tracks between idx1 and idx2 for Essential Matrix
                pts1_all, pts2_all = [], []
                for other_tid, other_obs in valid_tracks.items():
                    if idx1 in other_obs and idx2 in other_obs:
                        pts1_all.append(other_obs[idx1])
                        pts2_all.append(other_obs[idx2])
                
                if len(pts1_all) < 8: # Need enough points for robust E
                    rel_poses[(idx1, idx2)] = None
                else:
                    pts1_all = np.array(pts1_all)
                    pts2_all = np.array(pts2_all)
                    E, mask = cv2.findEssentialMat(pts1_all, pts2_all, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
                    if E is not None and E.shape == (3, 3):
                        _, R, t, mask = cv2.recoverPose(E, pts1_all, pts2_all, K, mask=mask)
                        
                        T21 = np.eye(4)
                        T21[:3, :3] = R
                        
                        if cfg.two_view_scale_with_gt:
                            # Scale t to match GT baseline for initialization consistency
                            T1_gt = T_c_o_init[idx1]
                            T2_gt = T_c_o_init[idx2]
                            C1_gt = -T1_gt[:3, :3].T @ T1_gt[:3, 3]
                            C2_gt = -T2_gt[:3, :3].T @ T2_gt[:3, 3]
                            baseline_gt = np.linalg.norm(C1_gt - C2_gt)
                            T21[:3, 3] = t.flatten() * baseline_gt
                        else:
                            T21[:3, 3] = t.flatten()
                            
                        rel_poses[(idx1, idx2)] = T21
                    else:
                        rel_poses[(idx1, idx2)] = None
        
        T21 = rel_poses[(idx1, idx2)]
        if T21 is not None:
            X_c1 = triangulate_track(pt1, pt2, K, T21)
            if X_c1 is not None:
                # Convert from camera 1 frame to object frame
                T_c1_o = T_c_o_init[idx1]
                X_o = np.linalg.inv(T_c1_o) @ np.append(X_c1, 1)
                pts3D_init[tid] = X_o[:3]

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
        T = T_c_o_init[idx]
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

    # --- Shared World Logging (GT and Camera) ---
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    # Log GT object mesh (unscaled, static relative to its moving parent world/object_gt)
    obj_mesh = data_loader.obj_mesh
    if isinstance(obj_mesh, trimesh.Scene):
        obj_mesh = obj_mesh.dump(concatenate=True)
    rr.log("world/object_gt/mesh", rr.Mesh3D(
        vertex_positions=obj_mesh.vertices,
        triangle_indices=obj_mesh.faces,
        vertex_colors=obj_mesh.visual.vertex_colors if hasattr(obj_mesh.visual, 'vertex_colors') else None,
        vertex_normals=obj_mesh.vertex_normals
    ), static=True)

    # Log mesh to object centric view (static at origin)
    rr.log("object/mesh", rr.Mesh3D(
        vertex_positions=obj_mesh.vertices,
        triangle_indices=obj_mesh.faces,
        vertex_colors=obj_mesh.visual.vertex_colors if hasattr(obj_mesh.visual, 'vertex_colors') else None,
        vertex_normals=obj_mesh.vertex_normals
    ), static=True)

    gt_world_obj_positions = []
    world_cam_positions = []
    
    print("Logging shared world data...")
    for idx in tqdm(indices):
        rr.set_time("frame_idx", sequence=idx)
        
        # Camera pose
        T_c_w = frame_data[idx]["T_c_w"]
        T_w_c = np.linalg.inv(T_c_w)
        rr.log("world/camera", rr.Transform3D(translation=T_w_c[:3, 3], mat3x3=T_w_c[:3, :3], relation=rr.TransformRelation.ParentFromChild))
        world_cam_positions.append(T_w_c[:3, 3])
        
        # GT Object pose
        T_w_o_gt = data_loader.get_obj_pose(idx)
        rr.log("world/object_gt", rr.Transform3D(translation=T_w_o_gt[:3, 3], mat3x3=T_w_o_gt[:3, :3], relation=rr.TransformRelation.ParentFromChild))
        gt_world_obj_positions.append(T_w_o_gt[:3, 3])
        
        # Camera views (the actual image from the camera)
        K = frame_data[idx]["K"]
        img = frame_data[idx]["image"]
        rr.log("world/camera/view", rr.Pinhole(resolution=(img.shape[1], img.shape[0]), image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.1))
        rr.log("world/camera/view", rr.Image(img).compress(jpeg_quality=50))

        # Log feature tracks (2D points are scale-independent)
        if idx in idx_to_tracks:
            tinfo = idx_to_tracks[idx]
            if tinfo:
                tids = [t[0] for t in tinfo]
                uvs = [t[1] for t in tinfo]
                # Green if triangulated, red otherwise
                colors = [(0, 255, 0) if tid in pts3D_init else (255, 0, 0) for tid in tids]
                rr.log("world/camera/view/tracks", rr.Points2D(uvs, colors=colors, radii=2))

    rr.log("world/trajectory/camera", rr.LineStrips3D([np.array(world_cam_positions)], colors=[(0, 0, 255)], radii=0.005), static=True)
    rr.log("world/trajectory/gt", rr.LineStrips3D([np.array(gt_world_obj_positions)], colors=[(255, 0, 0)], radii=0.005), static=True)

    # --- Scale-Specific BA Visualization ---
    for s in cfg.scale:
        s_name = f"s{s:.2f}".replace(".", "_")
        print(f"Logging visualization for scale {s}...")

        # Camera-centric BA results
        rr.log(f"object/points_{s_name}/init", rr.Points3D(np.array(list(pts3D_init.values())) * s, colors=[(100, 100, 100)], radii=0.002), static=True)
        rr.log(f"object/points_{s_name}/optimized", rr.Points3D(np.array(list(point_params.values())) * s, colors=[(0, 255, 0)], radii=0.003), static=True)

        # World-centric points (relative to the moving object frame)
        rr.log(f"world/object_{s_name}/points", rr.Points3D(np.array(list(point_params.values())) * s, colors=[(0, 255, 0)], radii=0.003), static=True)

        world_obj_positions = []
        object_cam_gt_positions = []
        object_cam_opt_positions = []

        for idx in indices:
            rr.set_time("frame_idx", sequence=idx)
            
            # Object-centric camera hypotheses
            T_c_o_gt = frame_data[idx]["T_c_o_gt"].copy()
            T_c_o_gt[:3, 3] *= s
            rr.log(f"object/camera_gt_{s_name}", rr.Transform3D(translation=T_c_o_gt[:3, 3], mat3x3=T_c_o_gt[:3, :3], relation=rr.TransformRelation.ChildFromParent))
            # Wireframe frustum for GT
            rr.log(f"object/camera_gt_{s_name}/frustum", rr.Pinhole(resolution=(img.shape[1], img.shape[0]), image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.05))

            q_opt, t_opt = pose_params[idx]
            T_c_o_opt = np.eye(4)
            T_c_o_opt[:3, :] = pycolmap.Rigid3d(pycolmap.Rotation3d(q_opt), t_opt).matrix()
            T_c_o_opt_scaled = T_c_o_opt.copy()
            T_c_o_opt_scaled[:3, 3] *= s
            rr.log(f"object/camera_opt_{s_name}", rr.Transform3D(translation=T_c_o_opt_scaled[:3, 3], mat3x3=T_c_o_opt_scaled[:3, :3], relation=rr.TransformRelation.ChildFromParent))
            # Wireframe frustum for Opt
            rr.log(f"object/camera_opt_{s_name}/frustum", rr.Pinhole(resolution=(img.shape[1], img.shape[0]), image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.05))
            
            # World-centric object hypotheses
            T_c_w = frame_data[idx]["T_c_w"]
            T_w_c = np.linalg.inv(T_c_w)
            T_w_o_opt = T_w_c @ T_c_o_opt_scaled
            world_obj_positions.append(T_w_o_opt[:3, 3])
            
            # Trajectories for object-centric view
            T_o_c_gt = np.linalg.inv(T_c_o_gt)
            object_cam_gt_positions.append(T_o_c_gt[:3, 3])
            
            T_o_c_opt = np.linalg.inv(T_c_o_opt_scaled)
            object_cam_opt_positions.append(T_o_c_opt[:3, 3])
            
            # Note: world/camera and world/object_gt are logged in the pre-loop.
            # We log the specific world hypothesis for this scale here:
            rr.log(f"world/object_{s_name}", rr.Transform3D(translation=T_w_o_opt[:3, 3], mat3x3=T_w_o_opt[:3, :3], relation=rr.TransformRelation.ParentFromChild))

        if world_obj_positions:
            rr.log(f"world/trajectory/optimized_{s_name}", rr.LineStrips3D([np.array(world_obj_positions)], colors=[(255, 255, 0)], radii=0.004), static=True)

        if object_cam_gt_positions:
            rr.log(f"object/trajectory/camera_gt_{s_name}", rr.LineStrips3D([np.array(object_cam_gt_positions)], colors=[(255, 0, 0)], radii=0.005), static=True)
        if object_cam_opt_positions:
            rr.log(f"object/trajectory/camera_opt_{s_name}", rr.LineStrips3D([np.array(object_cam_opt_positions)], colors=[(0, 0, 255)], radii=0.005), static=True)

        # Save Results per scale
        out_dir = Path(cfg.output_dir) / cfg.data_seq
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save optimized 3D points
        res_points = np.array(list(point_params.values())) * s
        res_tids = np.array(list(point_params.keys()))
        
        res_frame_indices = np.array(list(pose_params.keys()))
        res_pose_vectors = []
        for idx in res_frame_indices:
            q, t = pose_params[idx]
            t_scaled = t * s
            res_pose_vectors.append(np.concatenate([q, t_scaled]))
        res_pose_vectors = np.array(res_pose_vectors)
        
        np.savez(out_dir / f"reconstruction_{s_name}.npz", 
                 points=res_points, 
                 tids=res_tids, 
                 frame_indices=res_frame_indices,
                 pose_vectors=res_pose_vectors)
        
        # Save a PLY file for easy visualization
        if len(res_points) > 0:
            with open(out_dir / f"reconstruction_{s_name}.ply", "w") as f:
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
                
        print(f"Scale {s} reconstruction saved to {out_dir}")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
