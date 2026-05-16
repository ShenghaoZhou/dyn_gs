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
    n_frames: int = 50
    kf_every: int = 5
    max_keyframes: int = 10
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # DIS Flow Parameters
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    grid_spacing: int = 4
    
    # ORB Parameters
    max_orb_features: int = 1000
    use_klt: bool = True
    
    # Tracking Parameters
    ransac_thresh: float = 1.0
    min_tracks: int = 200
    
    # Depth source
    use_gt_depth: bool = False
    
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

def sample_grid_on_mask(mask, spacing):
    h, w = mask.shape
    yy, xx = np.mgrid[spacing//2:h:spacing, spacing//2:w:spacing]
    pts = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
    ix, iy = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy, pts = ix[valid], iy[valid], pts[valid]
    return pts[mask[iy, ix] > 0]

def run_ba(frame_indices, poses, tracks, K_dict, fix_first=True):
    if len(frame_indices) < 2: return
    prob = pyceres.Problem()
    loss = pyceres.HuberLoss(1.0)
    pose_params = {}
    for idx in frame_indices:
        T = poses[idx]
        q = R.from_matrix(T[:3, :3]).as_quat()
        q_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
        t = T[:3, 3].copy().astype(np.float64)
        pose_params[idx] = (q_wxyz, t)
    
    track_params = {}
    relevant_tracks = []
    for tid, track in tracks.items():
        win_obs = {f_idx: uv for f_idx, uv in track['obs'].items() if f_idx in frame_indices}
        if len(win_obs) >= 2:
            track_params[tid] = track['pt3d'].copy().astype(np.float64)
            relevant_tracks.append((tid, win_obs))
    
    if not relevant_tracks: return
    
    for tid, win_obs in relevant_tracks:
        pt3d = track_params[tid]
        for f_idx, uv in win_obs.items():
            K = K_dict[f_idx]
            cam_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float64)
            q_wxyz, t = pose_params[f_idx]
            cost = pycolmap.cost_functions.ReprojErrorCost('PINHOLE', uv.astype(np.float64))
            prob.add_residual_block(cost, loss, [q_wxyz, t, pt3d, cam_params])
            prob.set_parameter_block_constant(cam_params)
    
    if fix_first:
        ref_idx = frame_indices[0]
        prob.set_parameter_block_constant(pose_params[ref_idx][0])
        prob.set_parameter_block_constant(pose_params[ref_idx][1])
    
    quat_manifold = pyceres.EigenQuaternionManifold()
    for idx in frame_indices:
        q_wxyz, t = pose_params[idx]
        if not prob.is_parameter_block_constant(q_wxyz):
            prob.set_manifold(q_wxyz, quat_manifold)
            
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
    options.max_num_iterations = 50
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    
    for idx, (q_wxyz, t) in pose_params.items():
        T = np.eye(4)
        T[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
        T[:3, 3] = t
        poses[idx] = T
    for tid, _ in relevant_tracks:
        tracks[tid]['pt3d'] = track_params[tid]

class PathProcessor:
    def __init__(self, name, color, cfg):
        self.name = name
        self.color = color
        self.cfg = cfg
        self.poses = {} # f_idx -> T_CiC0
        self.tracks = {} # tid -> {'obs': {f_idx: uv}, 'pt3d': xyz}
        self.keyframes = [] # list of frame_idx
        self.next_tid = 0
        self.path_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        
        self.orb = cv2.ORB_create(nfeatures=cfg.max_orb_features)
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def add_new_points_from_depth(self, frame_idx, image, mask, depth, K, T_CiC0, align=False):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        kps = self.orb.detect(gray, mask=mask)
        pts2d = np.array([kp.pt for kp in kps], dtype=np.float32)
        
        if align and self.tracks:
            z_est = []
            z_raw = []
            for tid, t in self.tracks.items():
                if frame_idx in t['obs']:
                    uv = t['obs'][frame_idx]
                    pt_Ci = (T_CiC0[:3, :3] @ t['pt3d']) + T_CiC0[:3, 3]
                    z_est.append(pt_Ci[2])
                    z_raw.append(depth[int(round(uv[1])), int(round(uv[0]))])
            
            z_est = np.array(z_est)
            z_raw = np.array(z_raw)
            valid = (z_raw > 0.01) & (z_est > 0.01)
            if np.sum(valid) > 10:
                A = np.stack([z_raw[valid], np.ones_like(z_raw[valid])], axis=1)
                res = np.linalg.lstsq(A, z_est[valid], rcond=None)[0]
                s, b = res[0], res[1]
                print(f"[{self.name}] Depth Alignment: s={s:.4f}, b={b:.4f}")
                depth = s * depth + b

        K_inv = np.linalg.inv(K)
        T_C0Ci = np.linalg.inv(T_CiC0)
        for uv in pts2d:
            d = depth[int(round(uv[1])), int(round(uv[0]))]
            if d <= 0.01: continue
            pt_Ci = (K_inv @ np.array([uv[0], uv[1], 1.0])) * d
            pt_C0 = (T_C0Ci[:3, :3] @ pt_Ci) + T_C0Ci[:3, 3]
            self.tracks[self.next_tid] = {'obs': {frame_idx: uv}, 'pt3d': pt_C0}
            self.next_tid += 1

    def step(self, frame_idx, image, mask, depth, K, flow_prev_curr=None, gray_prev=None, gray_curr=None):
        if flow_prev_curr is not None:
            prev_idx = frame_idx - 1
            valid_tids = [tid for tid, t in self.tracks.items() if prev_idx in t['obs']]
            if valid_tids:
                pts_prev = np.array([self.tracks[tid]['obs'][prev_idx] for tid in valid_tids], dtype=np.float32)
                pts_next_dis = (pts_prev + interpolate_flow(flow_prev_curr, pts_prev)).astype(np.float32)
                
                if self.cfg.use_klt and gray_prev is not None and gray_curr is not None:
                    pts_prev_klt = pts_prev.reshape(-1, 1, 2)
                    pts_next_init = pts_next_dis.reshape(-1, 1, 2)
                    pts_next_klt, status, _ = cv2.calcOpticalFlowPyrLK(
                        gray_prev, gray_curr, pts_prev_klt, pts_next_init, 
                        flags=cv2.OPTFLOW_USE_INITIAL_FLOW, **self.lk_params
                    )
                    pts_next = pts_next_klt.reshape(-1, 2)
                    status = status.flatten().astype(bool)
                else:
                    pts_next = pts_next_dis
                    status = np.ones(len(pts_next), dtype=bool)

                ix, iy = np.round(pts_next[:, 0]).astype(int), np.round(pts_next[:, 1]).astype(int)
                mask_valid = (ix >= 0) & (ix < image.shape[1]) & (iy >= 0) & (iy < image.shape[0])
                for j, tid in enumerate(valid_tids):
                    if status[j] and mask_valid[j] and mask[iy[j], ix[j]] > 0:
                        self.tracks[tid]['obs'][frame_idx] = pts_next[j]
        
        pts2d, pts3d = [], []
        active_tids = []
        for tid, t in self.tracks.items():
            if frame_idx in t['obs']:
                pts2d.append(t['obs'][frame_idx])
                pts3d.append(t['pt3d'])
                active_tids.append(tid)
        
        if len(pts2d) < 10:
            self.poses[frame_idx] = self.poses.get(frame_idx-1, np.eye(4)).copy()
            return False
        
        cam_dict = {'model': 'PINHOLE', 'width': image.shape[1], 'height': image.shape[0], 'params': [K[0,0], K[1,1], K[0,2], K[1,2]]}
        res, info = poselib.estimate_absolute_pose(np.array(pts2d), np.array(pts3d), cam_dict, {'max_reproj_error': self.cfg.ransac_thresh}, None)
        T = np.eye(4)
        if res:
            T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix()
            T[:3, 3] = res.pose.t
        self.poses[frame_idx] = T
        return True

    def run_kf_ba(self, K_dict):
        kf_indices = self.keyframes[-self.cfg.max_keyframes:]
        run_ba(kf_indices, self.poses, self.tracks, K_dict)

def main(cfg: Config):
    rr.init("test_pnp_seq_orb_fusion", spawn=False)
    if cfg.rerun_url: rr.connect_grpc(cfg.rerun_url)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    data_dir = Path(cfg.data_root)
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    
    def load_frame(idx):
        stem = f"{idx:06d}"
        img_path = data_dir / "images" / f"{stem}.png"
        if not img_path.exists(): return None
        img = np.array(cv2.imread(str(img_path))[..., ::-1])
        mask = np.array(cv2.imread(str(data_dir / "obj_masks" / f"{stem}.png"), cv2.IMREAD_GRAYSCALE))
        if cfg.use_gt_depth:
            depth = np.load(data_dir / "depth_dyn" / f"{stem}.npy")
        else:
            d_path = data_dir / "model_infer" / f"depth_{idx:05d}.npy"
            if not d_path.exists(): d_path = data_dir / "depth_dyn" / f"{stem}.npy"
            depth = np.load(d_path)
        if depth.shape != img.shape[:2]:
            depth = cv2.resize(depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        K = np.load(data_dir / "intrinsics" / f"{stem}.npy")
        T_CW_gt = np.load(data_dir / "extrinsics" / f"{stem}.npy")
        T_WO_gt = load_object_pose_world(cfg.data_root, idx)
        return {"image": img, "mask": mask, "depth": depth, "K": K, "T_CW_gt": T_CW_gt, "T_WO_gt": T_WO_gt, "frame_idx": idx}

    frame_cache = {}
    def get_frame(idx):
        if idx not in frame_cache:
            frame_cache[idx] = load_frame(idx)
        return frame_cache[idx]

    f0_data = get_frame(cfg.init_frame)
    H, W = f0_data["image"].shape[:2]
    T_WO0 = f0_data["T_WO_gt"]
    
    path_a = PathProcessor("Path A (Depth)", [0, 0, 255], cfg)
    K_dict = {}
    frames_buffer = []
    
    # Trackers for fusion points (Dense Grid)
    # tid -> {'obs': {f_idx: uv}, 'pt3d': xyz}
    dense_tracks = {}
    next_dense_tid = 0

    print("Initialization...")
    for i in range(cfg.kf_every):
        fd = get_frame(cfg.init_frame + i)
        frames_buffer.append(fd)
        K_dict[fd["frame_idx"]] = fd["K"]

    f0 = frames_buffer[0]
    path_a.poses[f0["frame_idx"]] = np.eye(4)
    path_a.add_new_points_from_depth(f0["frame_idx"], f0["image"], f0["mask"], f0["depth"], f0["K"], path_a.poses[f0["frame_idx"]])
    path_a.keyframes.append(f0["frame_idx"])
    
    # Dense seed f0
    K_inv0 = np.linalg.inv(f0["K"])
    dense_pts0 = sample_grid_on_mask(f0["mask"], cfg.grid_spacing)
    for uv in dense_pts0:
        d = f0["depth"][int(round(uv[1])), int(round(uv[0]))]
        if d > 0.01:
            pt_c = (K_inv0 @ np.array([uv[0], uv[1], 1.0])) * d
            dense_tracks[next_dense_tid] = {'obs': {f0["frame_idx"]: uv}, 'pt3d': pt_c}
            next_dense_tid += 1

    for i in range(1, cfg.kf_every):
        f_prev, f_curr = frames_buffer[i-1], frames_buffer[i]
        gray_prev = cv2.cvtColor(f_prev["image"], cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY)
        flow = dis.calc(gray_prev, gray_curr, None)
        path_a.step(f_curr["frame_idx"], f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow, gray_prev, gray_curr)
        
        # Advect dense tracks
        valid_dense = [tid for tid in dense_tracks if f_prev["frame_idx"] in dense_tracks[tid]['obs']]
        for tid in valid_dense:
            uv_p = dense_tracks[tid]['obs'][f_prev["frame_idx"]]
            delta = interpolate_flow(flow, uv_p[None])[0]
            uv_c = uv_p + delta
            ix, iy = np.round(uv_c[0]).astype(int), np.round(uv_c[1]).astype(int)
            if 0 <= ix < W and 0 <= iy < H and f_curr["mask"][iy, ix] > 0:
                dense_tracks[tid]['obs'][f_curr["frame_idx"]] = uv_c

    path_a.run_kf_ba(K_dict)

    # --- Main Sequential Loop ---
    prev_f = frames_buffer[-1]
    for i in tqdm(range(cfg.kf_every, cfg.n_frames), desc="Tracking"):
        curr_idx = cfg.init_frame + i
        f_curr = get_frame(curr_idx)
        if f_curr is None: break
        K_dict[curr_idx] = f_curr["K"]
        
        gray_prev = cv2.cvtColor(prev_f["image"], cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(f_curr["image"], cv2.COLOR_RGB2GRAY)
        flow = dis.calc(gray_prev, gray_curr, None)
        
        path_a.step(curr_idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], flow, gray_prev, gray_curr)
        
        # Advect dense tracks
        valid_dense = [tid for tid in dense_tracks if (curr_idx - 1) in dense_tracks[tid]['obs']]
        for tid in valid_dense:
            uv_p = dense_tracks[tid]['obs'][curr_idx - 1]
            delta = interpolate_flow(flow, uv_p[None])[0]
            uv_c = uv_p + delta
            ix, iy = np.round(uv_c[0]).astype(int), np.round(uv_c[1]).astype(int)
            if 0 <= ix < W and 0 <= iy < H and f_curr["mask"][iy, ix] > 0:
                dense_tracks[tid]['obs'][curr_idx] = uv_c

        if i % cfg.kf_every == 0:
            path_a.keyframes.append(curr_idx)
            path_a.add_new_points_from_depth(curr_idx, f_curr["image"], f_curr["mask"], f_curr["depth"], f_curr["K"], path_a.poses[curr_idx], align=True)
            path_a.run_kf_ba(K_dict)
            
            # Re-seed dense points
            K_inv = np.linalg.inv(f_curr["K"])
            T_C0Ci = np.linalg.inv(path_a.poses[curr_idx])
            pts_new = sample_grid_on_mask(f_curr["mask"], cfg.grid_spacing)
            for uv in pts_new:
                d = f_curr["depth"][int(round(uv[1])), int(round(uv[0]))]
                if d > 0.01:
                    pt_ci = (K_inv @ np.array([uv[0], uv[1], 1.0])) * d
                    pt_c0 = (T_C0Ci[:3, :3] @ pt_ci) + T_C0Ci[:3, 3]
                    dense_tracks[next_dense_tid] = {'obs': {curr_idx: uv}, 'pt3d': pt_c0}
                    next_dense_tid += 1

        # Rerun Viz (World Space)
        rr.set_time("frame", sequence=curr_idx)
        T_OC0_gt = np.linalg.inv(f0_data["T_CW_gt"] @ f0_data["T_WO_gt"])
        T_WCi_est = T_WO0 @ T_OC0_gt @ np.linalg.inv(path_a.poses[curr_idx])
        rr.log("world/camera", rr.Pinhole(image_from_camera=f_curr["K"], width=W, height=H))
        rr.log("world/camera", rr.Transform3D(mat3x3=T_WCi_est[:3, :3], translation=T_WCi_est[:3, 3]))
        rr.log("world/camera/image", rr.Image(f_curr["image"]))
        
        prev_f = f_curr

    # --- 3. TSDF Fusion ---
    print("TSDF Fusion...")
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.voxel_size,
        sdf_trunc=cfg.sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    # Collect all SfM-like 3D points in C0 frame
    sfm_pts_c0 = np.array([t['pt3d'] for t in path_a.tracks.values()])
    dense_pts_c0 = np.array([t['pt3d'] for t in dense_tracks.values()])
    all_pts_c0 = np.vstack([sfm_pts_c0, dense_pts_c0]) if len(dense_pts_c0) > 0 else sfm_pts_c0
    
    T_OC0_gt = np.linalg.inv(f0_data["T_CW_gt"] @ f0_data["T_WO_gt"])
    
    # Process each frame for fusion
    for i in tqdm(range(cfg.n_frames), desc="Fusion"):
        curr_idx = cfg.init_frame + i
        f = get_frame(curr_idx)
        if f is None or curr_idx not in path_a.poses: continue
        
        T_CiC0 = path_a.poses[curr_idx]
        T_C0Ci = np.linalg.inv(T_CiC0)
        
        # Project all 3D points into current frame
        sfm_depths = np.zeros((H, W), dtype=np.float32)
        pts_ci = (T_CiC0[:3, :3] @ all_pts_c0.T).T + T_CiC0[:3, 3]
        
        # Projection
        u_homo = (f["K"] @ pts_ci.T).T
        u = u_homo[:, 0] / u_homo[:, 2]
        v = u_homo[:, 1] / u_homo[:, 2]
        z = pts_ci[:, 2]
        
        mask = (z > 0.01) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v, z = u[mask], v[mask], z[mask]
        iu, iv = np.round(u).astype(int), np.round(v).astype(int)
        
        # Z-buffer
        # Since we have many points, we can use a more efficient way to Z-buffer
        # This is a simple loop-based version, could be optimized with numpy if needed
        # But for ~50k-100k points it's okay.
        # Actually, let's use a faster way:
        sort_idx = np.argsort(-z) # Back to front
        sfm_depths[iv[sort_idx], iu[sort_idx]] = z[sort_idx]
        
        # TSDF Integration
        # Open3D integrate expects world-to-camera extrinsic. 
        # Here our "world" for fusion is C0. So we pass T_CiC0.
        intrinsic = o3d.camera.PinholeCameraIntrinsic(W, H, f["K"][0,0], f["K"][1,1], f["K"][0,2], f["K"][1,2])
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(f["image"]),
            o3d.geometry.Image(sfm_depths),
            depth_scale=1.0,
            depth_trunc=2.0,
            convert_rgb_to_intensity=False
        )
        volume.integrate(rgbd, intrinsic, T_CiC0)
        
        # Viz refined depth
        rr.set_time("frame", sequence=curr_idx)
        rr.log("world/camera/sfm_depth", rr.DepthImage(sfm_depths, meter=1.0))

    print("Extracting Mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # Mesh is in C0 frame. Align to world for visualization.
    # P_world = T_WO0 @ T_OC0_gt @ P_C0
    T_WC0 = T_WO0 @ T_OC0_gt
    vertices = np.asarray(mesh.vertices)
    vertices_world = (T_WC0[:3, :3] @ vertices.T).T + T_WC0[:3, 3]
    
    rr.log("world/mesh", rr.Mesh3D(
        vertex_positions=vertices_world,
        triangle_indices=np.asarray(mesh.triangles),
        vertex_colors=np.asarray(mesh.vertex_colors),
        vertex_normals=(T_WC0[:3, :3] @ np.asarray(mesh.vertex_normals).T).T
    ), static=True)

    # Log Points
    pts_world = (T_WC0[:3, :3] @ all_pts_c0.T).T + T_WC0[:3, 3]
    rr.log("world/points/sfm", rr.Points3D(pts_world, radii=cfg.point_radii, colors=[[0, 255, 255]] * len(pts_world)), static=True)

    print("Done.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
