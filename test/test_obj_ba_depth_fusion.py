import numpy as np
import cv2
import rerun as rr
import torch
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass, field
from scipy.spatial.transform import Rotation as R
import poselib
import pycolmap
import pycolmap.cost_functions
import pyceres
import shutil
import tempfile
import open3d as o3d
from gs_dyn_obj.utils.init import unproject_depth

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed/clip-003312"
    init_frame: int = 0
    n_frames: int = 50 # Reduced for testing
    grid_spacing: int = 8
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # TSDF Parameters
    voxel_size: float = 0.002
    sdf_trunc: float = 0.01
    
    # Visualization
    point_radii: float = 0.005
    traj_radii: float = 0.002

def umeyama(src, dst):
    """Computes Sim(3) transform: dst = s * R * src + t"""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    s_centered = src - mu_s
    d_centered = dst - mu_d
    C = d_centered.T @ s_centered / len(src)
    U, S, Vh = np.linalg.svd(C)
    d = np.linalg.det(U @ Vh)
    S_mat = np.eye(3)
    if d < 0: S_mat[2, 2] = -1
    R_mat = U @ S_mat @ Vh
    var_s = np.var(src, axis=0).sum()
    s = np.trace(np.diag(S) @ S_mat) / (var_s + 1e-8)
    t = mu_d - s * R_mat @ mu_s
    return s, R_mat, t

def load_object_pose_world(data_root, frame_idx):
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

def sample_grid_on_mask(mask, spacing):
    h, w = mask.shape
    yy, xx = np.mgrid[spacing//2:h:spacing, spacing//2:w:spacing]
    pts = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
    ix, iy = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy, pts = ix[valid], iy[valid], pts[valid]
    return pts[mask[iy, ix] > 0]

def interpolate_flow(flow, pts):
    x, y = pts[:, 0], pts[:, 1]
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = x0 + 1, y0 + 1
    h, w = flow.shape[:2]
    x0, x1 = np.clip(x0, 0, w-1), np.clip(x1, 0, w-1)
    y0, y1 = np.clip(y0, 0, h-1), np.clip(y1, 0, h-1)
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    return (wa[:, None] * flow[y0, x0] + wb[:, None] * flow[y1, x0] + 
            wc[:, None] * flow[y0, x1] + wd[:, None] * flow[y1, x1])

def main(cfg: Config):
    rr.init("test_obj_ba_depth_fusion", spawn=False)
    if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    data_dir = Path(cfg.data_root)
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    
    # --- 1. Load and Track ---
    print("Tracking with DIS Flow...")
    frames = []
    indices = list(range(cfg.init_frame, cfg.init_frame + cfg.n_frames))
    
    for idx in tqdm(indices, desc="Loading Frames"):
        stem = f"{idx:06d}"
        img = np.array(cv2.imread(str(data_dir / "images" / f"{stem}.png"))[..., ::-1])
        mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        T_WO_gt = load_object_pose_world(cfg.data_root, idx)
        frames.append({"image": img, "mask": mask, "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "idx": idx})

    # Tracks: tid -> {f_idx: uv}
    tracks = {}
    next_tid = 0
    active_tids = [] # list of tids currently being tracked
    
    kf_every = 5
    
    for i in tqdm(range(len(frames)), desc="Tracking"):
        f_curr = frames[i]
        curr_idx = f_curr["idx"]
        
        # 1. Advect existing tracks
        if i > 0:
            f_prev = frames[i-1]
            gray_prev = cv2.cvtColor(f_prev["image"], cv2.COLOR_RGB2GRAY)
            gray_curr = cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY)
            flow = dis.calc(gray_prev, gray_curr, None)
            
            new_active_tids = []
            for tid in active_tids:
                uv_prev = tracks[tid][f_prev["idx"]]
                delta = interpolate_flow(flow, uv_prev[None])[0]
                uv_curr = uv_prev + delta
                
                ix, iy = int(round(uv_curr[0])), int(round(uv_curr[1]))
                h, w = f_curr["image"].shape[:2]
                if 0 <= ix < w and 0 <= iy < h and f_curr["mask"][iy, ix] > 0:
                    tracks[tid][curr_idx] = uv_curr
                    new_active_tids.append(tid)
            active_tids = new_active_tids

        # 2. Re-seed new points periodically
        if i % kf_every == 0:
            pts_new = sample_grid_on_mask(f_curr["mask"], cfg.grid_spacing)
            # Avoid adding points too close to existing ones
            if active_tids:
                existing_pts = np.array([tracks[tid][curr_idx] for tid in active_tids])
                # Simple distance check
                from scipy.spatial import cKDTree
                tree = cKDTree(existing_pts)
                dist, _ = tree.query(pts_new)
                pts_new = pts_new[dist > cfg.grid_spacing * 0.7]
            
            for pt in pts_new:
                tracks[next_tid] = {curr_idx: pt}
                active_tids.append(next_tid)
                next_tid += 1

    # --- 2. COLMAP BA ---
    print("Running COLMAP BA...")
    f0 = frames[0]
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        db_path = tmp_path / "colmap.db"
        img_tmp_dir = tmp_path / "images"
        img_tmp_dir.mkdir()
        
        db = pycolmap.Database.open(str(db_path))
        K0 = f0["K"]
        H, W = f0["image"].shape[:2]
        camera = pycolmap.Camera(model="PINHOLE", width=W, height=H, 
                                 params=[K0[0,0], K0[1,1], K0[0,2], K0[1,2]])
        camera.has_prior_focal_length = True
        cam_id = db.write_camera(camera)
        
        image_ids = {}
        for f in frames:
            img_name = f"image_{f['idx']:06d}.png"
            cv2.imwrite(str(img_tmp_dir / img_name), cv2.cvtColor(f["image"], cv2.COLOR_RGB2BGR))
            image_id = db.write_image(pycolmap.Image(name=img_name, camera_id=cam_id))
            image_ids[f["idx"]] = image_id
            
        # Write Keypoints
        for f in frames:
            f_idx = f["idx"]
            frame_tids = sorted([tid for tid, t in tracks.items() if f_idx in t])
            keypoints = np.array([tracks[tid][f_idx] for tid in frame_tids], dtype=np.float32)
            kpts_full = np.zeros((len(keypoints), 4), dtype=np.float32)
            kpts_full[:, :2] = keypoints
            kpts_full[:, 2] = 1.0
            db.write_keypoints(image_ids[f_idx], kpts_full)
            
        # Write Matches (All pairs for small sequences)
        pairs = []
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)): 
                idx1, idx2 = frames[i]["idx"], frames[j]["idx"]
                # Find common tracks between frame i and frame j
                common_tids = [tid for tid, t in tracks.items() if idx1 in t and idx2 in t]
                
                if len(common_tids) >= 6: # Lowered from 8
                    # Map tids to local indices
                    frame_tids1 = sorted([tid for tid, t in tracks.items() if idx1 in t])
                    frame_tids2 = sorted([tid for tid, t in tracks.items() if idx2 in t])
                    tid_to_idx1 = {tid: k for k, tid in enumerate(frame_tids1)}
                    tid_to_idx2 = {tid: k for k, tid in enumerate(frame_tids2)}
                    
                    matches = np.array([[tid_to_idx1[tid], tid_to_idx2[tid]] for tid in common_tids], dtype=np.uint32)
                    db.write_matches(image_ids[idx1], image_ids[idx2], matches)
                    pairs.append((f"image_{idx1:06d}.png", f"image_{idx2:06d}.png"))
        
        db.close()
        
        pairs_path = tmp_path / "pairs.txt"
        with open(pairs_path, "w") as f_pairs:
            for p1, p2 in pairs: f_pairs.write(f"{p1} {p2}\n")
            
        pycolmap.verify_matches(str(db_path), str(pairs_path))
        colmap_out = tmp_path / "reconstruction"
        colmap_out.mkdir()
        
        # Incremental mapping options to disable intrinsic refinement
        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.ba_refine_focal_length = False
        mapper_options.ba_refine_principal_point = False
        mapper_options.ba_refine_extra_params = False
        mapper_options.mapper.abs_pose_refine_focal_length = False
        mapper_options.mapper.abs_pose_refine_extra_params = False
        # Set the camera as constant during mapping/BA
        mapper_options.mapper.constant_cameras = {cam_id}
        mapper_options.constant_cameras = {cam_id}

        # Incremental mapping
        reconstructions = pycolmap.incremental_mapping(str(db_path), str(img_tmp_dir), str(colmap_out), options=mapper_options)
        
        if not reconstructions:
            print("COLMAP failed.")
            return
        
        # Select largest model
        rec = sorted(reconstructions.values(), key=lambda x: x.num_points3D(), reverse=True)[0]
        if rec.num_reg_images() < 2:
            print("COLMAP failed to register enough images.")
            return
        
    # --- 3. Scale Estimation and Pose Extraction ---
    # --- 3. Scale Estimation and Alignment ---
    print("Estimating Scale and Aligning...")
    name_to_img = {img.name: img for img in rec.images.values()}
    
    f0 = frames[0]
    img0_name = f"image_{f0['idx']:06d}.png"
    if img0_name not in name_to_img:
        print("First frame not registered, cannot estimate scale.")
        return
    
    colmap_img0 = name_to_img[img0_name]
    T_C0W_col = np.eye(4)
    T_C0W_col[:3, :4] = colmap_img0.cam_from_world().matrix()
    
    # Load first frame's depth (model or GT) for scale estimation
    depth0_path = data_dir / "depth_dyn" / f"{f0['idx']:06d}.npy"
    if not depth0_path.exists():
        depth0_path = data_dir / "model_infer" / f"depth_{f0['idx']:05d}.npy"
    
    if depth0_path.exists():
        depth0 = np.load(depth0_path)
        if depth0.shape != (H, W):
            depth0 = cv2.resize(depth0, (W, H), interpolation=cv2.INTER_NEAREST)
        
        scales = []
        for p2d in colmap_img0.points2D:
            if not p2d.has_point3D(): continue
            pt3d = rec.points3D[p2d.point3D_id].xyz
            pt_c = (T_C0W_col[:3, :3] @ pt3d) + T_C0W_col[:3, 3]
            u, v = int(round(p2d.xy[0])), int(round(p2d.xy[1]))
            if 0 <= u < W and 0 <= v < H:
                d_ref = depth0[v, u]
                if d_ref > 0.01:
                    scales.append(d_ref / pt_c[2])
        
        scale_factor = np.median(scales) if scales else 1.0
        print(f"Scale factor: {scale_factor:.4f}")
    else:
        scale_factor = 1.0
        print("No depth map found for scale estimation, using scale 1.0.")

    # Align COLMAP world to Object frame using the first frame
    # We want to find T_OW (Object from World) such that P_O = T_OW * P_W
    # At frame 0: P_C = T_CO0_gt * P_O  => P_O = T_OC0_gt * P_C
    # Also P_C = T_C0W_col_scaled * P_W
    # So P_O = T_OC0_gt * T_C0W_col_scaled * P_W
    # Thus T_OW = T_OC0_gt * T_C0W_col_scaled
    
    T_CO0_gt = f0["T_CW_gt"] @ f0["T_WO_gt"]
    T_OC0_gt = np.linalg.inv(T_CO0_gt)
    
    # Scale the reconstruction first
    rec.transform(pycolmap.Sim3d(scale_factor, np.eye(3), np.zeros(3)))
    
    # Re-extract scaled T_C0W
    T_C0W_col_scaled = np.eye(4)
    T_C0W_col_scaled[:3, :4] = colmap_img0.cam_from_world().matrix()
    
    T_OW = T_OC0_gt @ T_C0W_col_scaled
    R_OW = T_OW[:3, :3]
    t_OW = T_OW[:3, 3]
    
    # Apply rotation and translation to align with Object frame
    rec.transform(pycolmap.Sim3d(1.0, R_OW, t_OW))


    # --- 4. Depth Reprojection and TSDF Fusion ---
    # --- 4. Depth Reprojection and TSDF Fusion ---
    print("TSDF Fusion...")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.voxel_size,
        sdf_trunc=cfg.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    valid_frames = []
    for f in frames:
        img_name = f"image_{f['idx']:06d}.png"
        if img_name in name_to_img:
            valid_frames.append(f)

    for f in tqdm(valid_frames, desc="Fusing"):
        img_name = f"image_{f['idx']:06d}.png"
        colmap_img = name_to_img[img_name]
        T_CO = np.eye(4) # Now this is Object-to-Camera because we aligned World to Object
        T_CO[:3, :4] = colmap_img.cam_from_world().matrix()
        
        # SfM-estimated intrinsics
        colmap_cam = rec.cameras[colmap_img.camera_id]
        K_sfm = np.eye(3)
        K_sfm[0,0] = colmap_cam.params[0]
        K_sfm[1,1] = colmap_cam.params[1]
        K_sfm[0,2] = colmap_cam.params[2]
        K_sfm[1,2] = colmap_cam.params[3]

        # --- Generate sfm_depths by projecting refined 3D points ---
        # We can project all points or just the ones observed in this frame.
        # Projecting all points provides a denser "model-based" depth map.
        sfm_depths = np.zeros((H, W), dtype=np.float32)
        
        # Method: Project all points in the reconstruction
        for pid, pt3d in rec.points3D.items():
            pt_o = pt3d.xyz
            pt_c = (T_CO[:3, :3] @ pt_o) + T_CO[:3, 3]
            if pt_c[2] <= 0.01: continue
            
            u_homo = K_sfm @ pt_c
            u, v = u_homo[0] / u_homo[2], u_homo[1] / u_homo[2]
            iu, iv = int(round(u)), int(round(v))
            
            if 0 <= iu < W and 0 <= iv < H:
                # Basic Z-buffer
                if sfm_depths[iv, iu] == 0 or pt_c[2] < sfm_depths[iv, iu]:
                    sfm_depths[iv, iu] = pt_c[2]
        
        # Optional: Bloat the sparse points slightly to improve TSDF coverage
        # kernel = np.ones((3,3), np.uint8)
        # sfm_depths = cv2.dilate(sfm_depths, kernel) # This might smear depth, be careful
        
        # Integrate into TSDF
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, K_sfm[0,0], K_sfm[1,1], K_sfm[0,2], K_sfm[1,2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(f["image"]),
            o3d.geometry.Image(sfm_depths),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False
        )
        # Open3D integrate expects world-to-camera (T_CO) extrinsic matrix
        volume.integrate(rgbd, intrinsic, T_CO)

        # Visualization in Rerun (Object-centric)
        T_OC_est = np.linalg.inv(T_CO)
        rr.set_time("frame", sequence=f["idx"])
        rr.log("object/camera", rr.Pinhole(image_from_camera=K_sfm, width=W, height=H))
        rr.log("object/camera", rr.Transform3D(mat3x3=T_OC_est[:3, :3], translation=T_OC_est[:3, 3]))
        rr.log("object/camera/image", rr.Image(f["image"]))
        rr.log("object/camera/sfm_depth", rr.DepthImage(sfm_depths, meter=1.0))

        # Log GT Camera for comparison (Object-centric)
        T_CO_gt = f["T_CW_gt"] @ f["T_WO_gt"]
        T_OC_gt = np.linalg.inv(T_CO_gt)
        rr.log("object/gt_camera", rr.Pinhole(image_from_camera=f["K"], width=W, height=H))
        rr.log("object/gt_camera", rr.Transform3D(mat3x3=T_OC_gt[:3, :3], translation=T_OC_gt[:3, 3]))


    # --- 5. Final Extraction and Visualization ---
    print("Extracting Mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # Log to Rerun (Object-centric)
    vertices = np.asarray(mesh.vertices)
    rr.log("object/mesh", rr.Mesh3D(
        vertex_positions=vertices,
        triangle_indices=np.asarray(mesh.triangles),
        vertex_colors=np.asarray(mesh.vertex_colors),
        vertex_normals=np.asarray(mesh.vertex_normals)
    ), static=True)

    # Log Points with RGB Colors
    colmap_pts = []
    colmap_colors = []
    for p in rec.points3D.values():
        colmap_pts.append(p.xyz)
        colmap_colors.append(p.color)
    
    if len(colmap_pts) > 0:
        rr.log("object/points", rr.Points3D(
            np.array(colmap_pts), 
            colors=np.array(colmap_colors), 
            radii=cfg.point_radii
        ), static=True)

    # Log GT Trajectory in Object frame
    gt_traj_obj = []
    for f in frames:
        T_CO_gt = f["T_CW_gt"] @ f["T_WO_gt"]
        T_OC_gt = np.linalg.inv(T_CO_gt)
        gt_traj_obj.append(T_OC_gt[:3, 3])
    rr.log("object/gt_traj", rr.LineStrips3D([np.array(gt_traj_obj)], colors=[[0, 255, 0]], radii=cfg.traj_radii), static=True)

    print("Done.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
