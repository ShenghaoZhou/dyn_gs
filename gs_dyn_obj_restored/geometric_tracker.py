import numpy as np
import cv2
import poselib
import pyceres
import pycolmap
import pycolmap.cost_functions
from scipy.spatial.transform import Rotation as R

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
    f_p = (wa[:, None] * flow[y0, x0] + wb[:, None] * flow[y1, x0] + 
           wc[:, None] * flow[y0, x1] + wd[:, None] * flow[y1, x1])
    return f_p

def sample_grid_on_mask(mask, spacing):
    h, w = mask.shape
    yy, xx = np.mgrid[spacing//2:h:spacing, spacing//2:w:spacing]
    pts = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
    ix, iy = np.round(pts[:, 0]).astype(int), np.round(pts[:, 1]).astype(int)
    valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    ix, iy, pts = ix[valid], iy[valid], pts[valid]
    return pts[mask[iy, ix] > 0]
    
def compute_prior_flow(T_curr, T_prev, K, depth_prev):
    h, w = depth_prev.shape
    yy, xx = np.mgrid[0:h, 0:w]
    K_inv = np.linalg.inv(K)
    
    # Back-project to 3D in prev camera frame
    pts2d = np.stack([xx, yy, np.ones_like(xx)], axis=-1) # (h, w, 3)
    pts3d_prev = (pts2d @ K_inv.T) * depth_prev[..., None]
    
    # Relative transformation: T_curr_prev = T_curr @ inv(T_prev)
    T_curr_prev = T_curr @ np.linalg.inv(T_prev)
    
    # Transform to current camera frame
    R_mat = T_curr_prev[:3, :3]
    t_vec = T_curr_prev[:3, 3]
    pts3d_curr = (pts3d_prev @ R_mat.T) + t_vec
    
    # Project back to 2D
    pts2d_curr_hom = pts3d_curr @ K.T
    z_curr = pts2d_curr_hom[..., 2:3]
    z_curr[z_curr < 1e-6] = 1e-6 # Avoid division by zero
    uv_curr = pts2d_curr_hom[..., :2] / z_curr
    
    # Displacements
    flow = (uv_curr - np.stack([xx, yy], axis=-1)).astype(np.float32)
    # Mask out zero depth
    flow[depth_prev <= 0.01] = 0
    return flow


def apply_anms(kpts, n_to_keep):
    """
    Optimized Adaptive Non-Maximal Suppression (ANMS) using Brown's algorithm.
    """
    if len(kpts) <= n_to_keep:
        return kpts
        
    # Sort keypoints by response descending
    kpts = sorted(kpts, key=lambda x: x.response, reverse=True)
    pts = np.array([kp.pt for kp in kpts])
    scores = np.array([kp.response for kp in kpts])
    
    radii = np.full(len(kpts), np.inf)
    
    # Pre-calculate squared distances for all pairs is too memory intensive (N^2)
    # But we can do it in chunks or use a more efficient approach
    # For 5000 points, N^2 is 25M, which is ~100MB in float32. Let's try broadcasting.
    
    for i in range(1, len(kpts)):
        # Only look at keypoints with significantly higher response (j < i)
        # Condition: score_i < 0.9 * score_j
        valid_j = scores[:i] > (kpts[i].response / 0.9)
        if np.any(valid_j):
            dists_sq = np.sum((pts[:i][valid_j] - pts[i])**2, axis=1)
            radii[i] = np.min(dists_sq)
                    
    indices = np.argsort(-radii)
    return [kpts[i] for i in indices[:n_to_keep]]

def detect_features_on_mask(image, mask, nfeatures=2000, feature_type="orb"):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    if feature_type == "orb":
        # Detect significantly more candidates to allow for ANMS
        detector = cv2.ORB_create(nfeatures=nfeatures * 5, fastThreshold=5)
        kpts = detector.detect(gray, mask=mask)
    elif feature_type == "gftt":
        # GFTT has built-in minDistance, but we can also use ANMS for better control
        # We detect many and then suppress
        detector = cv2.GFTTDetector_create(maxCorners=nfeatures * 5, minDistance=1, qualityLevel=0.001)
        kpts = detector.detect(gray, mask=mask)
    else:
        return np.array([], dtype=np.float32).reshape(0, 2)
        
    if not kpts:
        return np.array([], dtype=np.float32).reshape(0, 2)
    
    # Apply ANMS
    selected_kpts = apply_anms(kpts, nfeatures)
    
    pts = np.array([kp.pt for kp in selected_kpts], dtype=np.float32)
    return pts

def run_ba(frame_indices, poses, tracks, K_dict, fix_first=True):
    if len(frame_indices) < 2: return
    prob = pyceres.Problem()
    loss = pyceres.HuberLoss(1.0)
    pose_params = {}
    for idx in frame_indices:
        T = poses[idx]
        q = R.from_matrix(T[:3, :3]).as_quat().astype(np.float64)
        q_wxyz = np.array([q[3], q[0], q[1], q[2]]) # Convert xyzw to wxyz for pycolmap/ceres
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
        if prob.has_parameter_block(pose_params[ref_idx][0]):
            prob.set_parameter_block_constant(pose_params[ref_idx][0])
        if prob.has_parameter_block(pose_params[ref_idx][1]):
            prob.set_parameter_block_constant(pose_params[ref_idx][1])
    
    quat_manifold = pyceres.EigenQuaternionManifold()
    for idx in frame_indices:
        q_wxyz, t = pose_params[idx]
        if prob.has_parameter_block(q_wxyz):
            if not prob.is_parameter_block_constant(q_wxyz):
                prob.set_manifold(q_wxyz, quat_manifold)
            
    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
    options.max_num_iterations = 20
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    
    for idx, (q_wxyz, t) in pose_params.items():
        T = np.eye(4)
        T[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
        T[:3, 3] = t
        poses[idx] = T
    for tid, _ in relevant_tracks:
        tracks[tid]['pt3d'] = track_params[tid]

def match_projections(frame_idx, image, mask, K, T_CiO, tracks, window=5):
    """
    Search for existing 3D points that are visible in the current frame but not currently tracked.
    Uses patch matching or simple proximity if we have a good pose prior.
    """
    h, w = image.shape[:2]
    P = K @ T_CiO[:3, :]
    
    # Identify points not tracked in current frame
    untracked_tids = [tid for tid, t in tracks.items() if frame_idx not in t['obs']]
    if not untracked_tids: return 0
    
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
    
    matched_count = 0
    for tid in untracked_tids:
        t = tracks[tid]
        p_hom = P @ np.append(t['pt3d'], 1.0)
        if p_hom[2] <= 0.01: continue
        
        uv_proj = p_hom[:2] / p_hom[2]
        ix, iy = int(round(uv_proj[0])), int(round(uv_proj[1]))
        
        found = False
        for dy in range(-window, window + 1):
            for dx in range(-window, window + 1):
                ix_s, iy_s = ix + dx, iy + dy
                if 0 <= ix_s < w and 0 <= iy_s < h:
                    if mask[iy_s, ix_s] > 0:
                        t['obs'][frame_idx] = np.array([uv_proj[0] + dx, uv_proj[1] + dy], dtype=np.float32)
                        matched_count += 1
                        found = True
                        break
            if found: break
            
    return matched_count

class GeometricTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.poses = {} # f_idx -> T_CiO
        self.tracks = {} # tid -> {'obs': {f_idx: uv}, 'pt3d': xyz_O}
        self.keyframes = []
        self.next_tid = 0
        self.K_dict = {}
        self.last_scale = 1.0
        self.triangulated_tids = set()

    def estimate_depth_scale(self, frame_idx, depth, mask, K):
        if len(self.tracks) < 10:
            return self.last_scale
            
        ref_depths = []
        obs_depths = []
        T_CiO = self.poses.get(frame_idx)
        if T_CiO is None: return self.last_scale

        for tid, t in self.tracks.items():
            if len(t['obs']) >= 2:
                p_O = t['pt3d']
                p_Ci = T_CiO[:3, :3] @ p_O + T_CiO[:3, 3]
                if p_Ci[2] <= 0.01: continue
                
                uv_proj = (K @ p_Ci)
                u, v = uv_proj[0] / uv_proj[2], uv_proj[1] / uv_proj[2]
                
                ix, iy = int(round(u)), int(round(v))
                if 0 <= ix < depth.shape[1] and 0 <= iy < depth.shape[0] and mask[iy, ix] > 0:
                    d_new = depth[iy, ix]
                    if d_new > 0.01:
                        ref_depths.append(p_Ci[2])
                        obs_depths.append(d_new)
        
        if len(ref_depths) > 10:
            # Use RANSAC for robust scale estimation
            ref = np.array(ref_depths)
            obs = np.array(obs_depths)
            scales = ref / obs
            
            # Simple RANSAC-like filtering: keep values within 10% of median
            med = np.median(scales)
            valid = np.abs(scales - med) < 0.1 * med
            if np.sum(valid) > 5:
                self.last_scale = np.mean(scales[valid])
            else:
                self.last_scale = med
        return self.last_scale

    def update_object_points(self, new_points):
        """
        Update the 3D positions of existing tracks using a new point cloud (e.g. from GS).
        Uses nearest neighbor search to find corresponding points.
        """
        if len(self.tracks) == 0 or len(new_points) == 0:
            return
            
        from scipy.spatial import KDTree
        tree = KDTree(new_points)
        
        updated_count = 0
        for tid, t in self.tracks.items():
            dist, idx = tree.query(t['pt3d'])
            if dist < 0.05: # Only update if a close point exists (5cm)
                t['pt3d'] = new_points[idx].copy()
                updated_count += 1
        
        if updated_count > 0:
            print(f"[GeoTracker] Updated {updated_count} track points from GS model")


    def add_new_points_from_depth(self, frame_idx, image, mask, depth, K, T_CiO, align=False, spacing=None):
        mask_work = mask.copy()
        # Avoid adding points where we already have active tracks
        active_uvs = [t['obs'][frame_idx] for t in self.tracks.values() if frame_idx in t['obs']]
        for uv in active_uvs:
            cv2.circle(mask_work, (int(round(uv[0])), int(round(uv[1]))), 5, 0, -1)

        feature_type = getattr(self.cfg, "feature_type", "grid")
        n_to_detect = getattr(self.cfg, "n_features", 500)
        
        if feature_type in ["orb", "gftt"]:
            pts2d = detect_features_on_mask(image, mask_work, nfeatures=n_to_detect, feature_type=feature_type)
        else:
            spacing = spacing if spacing is not None else self.cfg.grid_spacing
            pts2d = sample_grid_on_mask(mask_work, spacing)
            
        best_s, best_b = 1.0, 0.0
        if align:
            if getattr(self.cfg, "align_depth", False):
                # Robust least squares scale/bias alignment
                ref_depths = []
                obs_depths = []
                T_CiO = self.poses.get(frame_idx)
                if T_CiO is not None:
                    for tid, t in self.tracks.items():
                        if len(t['obs']) >= 2:
                            p_O = t['pt3d']
                            p_Ci = T_CiO[:3, :3] @ p_O + T_CiO[:3, 3]
                            if p_Ci[2] <= 0.01: continue
                            uv_proj = (K @ p_Ci)
                            u, v = uv_proj[0] / uv_proj[2], uv_proj[1] / uv_proj[2]
                            ix, iy = int(round(u)), int(round(v))
                            if 0 <= ix < depth.shape[1] and 0 <= iy < depth.shape[0] and mask[iy, ix] > 0:
                                d_obs = depth[iy, ix]
                                if d_obs > 0.01:
                                    ref_depths.append(p_Ci[2])
                                    obs_depths.append(d_obs)
                    
                    if len(ref_depths) > 20:
                        obs = np.array(obs_depths)
                        ref = np.array(ref_depths)
                        scales = ref / (obs + 1e-6)
                        med = np.median(scales)
                        valid = np.abs(scales - med) < 0.2 * med
                        
                        if np.sum(valid) > 20:
                            if getattr(self.cfg, "align_with_bias", True):
                                A = np.stack([obs[valid], np.ones_like(obs[valid])], axis=-1)
                                B = ref[valid]
                                try:
                                    res, _, _, _ = np.linalg.lstsq(A, B, rcond=None)
                                    best_s, best_b = res[0], res[1]
                                except:
                                    best_s, best_b = med, 0.0
                            else:
                                best_s, best_b = med, 0.0
                                
                            # Robustness checks
                            if best_s <= 0.05 or best_s > 20.0:
                                print(f"[GeoTracker] Warning: Crazy scale {best_s:.4f}, falling back to median {med:.4f}")
                                best_s, best_b = med, 0.0
                            
                            # Final clamp to be extra safe
                            best_s = np.clip(best_s, 0.1, 10.0)
                            print(f"[GeoTracker] Depth alignment frame {frame_idx}: s={best_s:.4f}, b={best_b:.4f}")
                        else:
                            best_s, best_b = med, 0.0
                            best_s = np.clip(best_s, 0.1, 10.0)
            else:
                best_s = self.estimate_depth_scale(frame_idx, depth, mask, K)
        
        depth = best_s * depth + best_b
        
        K_inv = np.linalg.inv(K)
        T_OCi = np.linalg.inv(T_CiO)
        for uv in pts2d:
            d = depth[int(round(uv[1])), int(round(uv[0]))]
            if d <= 0.01: continue
            pt_Ci = (K_inv @ np.array([uv[0], uv[1], 1.0])) * d
            pt_O = (T_OCi[:3, :3] @ pt_Ci) + T_OCi[:3, 3]
            self.tracks[self.next_tid] = {'obs': {frame_idx: uv}, 'pt3d': pt_O}
            self.next_tid += 1

    def triangulate_tracks(self, frame_idx, thresh):
        """Perform multi-view triangulation for tracks that have reached the threshold length."""
        for tid, t in self.tracks.items():
            if len(t['obs']) >= thresh and tid not in self.triangulated_tids:
                self.triangulate_single_track(tid)
                self.triangulated_tids.add(tid)

    def triangulate_single_track(self, tid):
        track = self.tracks[tid]
        obs = track['obs']
        if len(obs) < 2: return
        
        # Check baseline
        f_indices = sorted(list(obs.keys()))
        t1 = self.poses[f_indices[0]][:3, 3]
        t2 = self.poses[f_indices[-1]][:3, 3]
        baseline = np.linalg.norm(t1 - t2)
        if baseline < 0.005: # 5mm baseline minimum
            return

        prob = pyceres.Problem()
        loss = pyceres.HuberLoss(1.0)
        pt3d = track['pt3d'].copy().astype(np.float64)
        
        pose_cache = {} # f_idx -> (q, t, K)
        
        for f_idx, uv in obs.items():
            if f_idx not in self.poses: continue
            
            if f_idx not in pose_cache:
                T = self.poses[f_idx]
                q_xyzw = R.from_matrix(T[:3, :3]).as_quat().astype(np.float64)
                t_vec = T[:3, 3].astype(np.float64)
                K = self.K_dict[f_idx]
                cam_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float64)
                pose_cache[f_idx] = (q_xyzw, t_vec, cam_params)
            
            q_xyzw, t_vec, cam_params = pose_cache[f_idx]
            cost = pycolmap.cost_functions.ReprojErrorCost('PINHOLE', uv.astype(np.float64))
            prob.add_residual_block(cost, loss, [q_xyzw, t_vec, pt3d, cam_params])
            prob.set_parameter_block_constant(q_xyzw)
            prob.set_parameter_block_constant(t_vec)
            prob.set_parameter_block_constant(cam_params)
            
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        options.max_num_iterations = 10
        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        
        if summary.final_cost < summary.initial_cost:
            track['pt3d'] = pt3d
    
    def refine_pose(self, frame_idx, K, tids_pnp=None, inlier_mask=None):
        pts2d, pts3d = [], []
        if tids_pnp is not None and inlier_mask is not None:
            for i in range(len(tids_pnp)):
                if inlier_mask[i]:
                    tid = tids_pnp[i]
                    t = self.tracks[tid]
                    pts2d.append(t['obs'][frame_idx])
                    pts3d.append(t['pt3d'])
        else:
            # Fallback to all observations (original behavior)
            for tid, t in self.tracks.items():
                if frame_idx in t['obs']:
                    pts2d.append(t['obs'][frame_idx])
                    pts3d.append(t['pt3d'])

        if len(pts2d) < 10: return

        prob = pyceres.Problem()
        loss = pyceres.HuberLoss(1.0)
        T = self.poses[frame_idx].copy()
        q = R.from_matrix(T[:3, :3]).as_quat().astype(np.float64)
        q_wxyz = np.array([q[3], q[0], q[1], q[2]])
        t_vec = T[:3, 3].copy().astype(np.float64)
        cam_params = np.array([K[0,0], K[1,1], K[0,2], K[1,2]], dtype=np.float64)
        
        pts3d_copy = [p.copy().astype(np.float64) for p in pts3d]

        for i in range(len(pts2d)):
            cost = pycolmap.cost_functions.ReprojErrorCost('PINHOLE', pts2d[i].astype(np.float64))
            prob.add_residual_block(cost, loss, [q_wxyz, t_vec, pts3d_copy[i], cam_params])
            prob.set_parameter_block_constant(pts3d_copy[i])
        
        prob.set_parameter_block_constant(cam_params)
        prob.set_manifold(q_wxyz, pyceres.EigenQuaternionManifold())
        
        options = pyceres.SolverOptions()
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        options.max_num_iterations = 10
        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        
        if summary.final_cost < summary.initial_cost:
            T_refined = np.eye(4)
            T_refined[:3, :3] = R.from_quat([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]]).as_matrix()
            T_refined[:3, 3] = t_vec
            self.poses[frame_idx] = T_refined

    def step(self, frame_idx, image, mask, depth, K, flow_prev_curr=None):
        # Default behavior: Constant-velocity motion model guess
        T_prev = self.poses.get(frame_idx-1, np.eye(4))
        T_prev_prev = self.poses.get(frame_idx-2, T_prev)
        T_guess = T_prev @ np.linalg.inv(T_prev_prev) @ T_prev
        return self.step_informed(frame_idx, image, mask, depth, K, T_guess, flow_prev_curr=flow_prev_curr)

    def step_informed(self, frame_idx, image, mask, depth, K, T_guess, flow_prev_curr=None, skip_pnp=False):
        self.K_dict[frame_idx] = K
        self.poses[frame_idx] = T_guess

        if flow_prev_curr is not None:
            prev_idx = frame_idx - 1
            K_curr = K
            T_guess_curr = T_guess
            
            for tid, t in self.tracks.items():
                if prev_idx in t['obs']:
                    uv_prev = t['obs'][prev_idx]
                    delta = interpolate_flow(flow_prev_curr, uv_prev[None])[0]
                    uv_curr = uv_prev + delta
                    
                    # Basic boundary and mask check
                    ix, iy = int(round(uv_curr[0])), int(round(uv_curr[1]))
                    if not (0 <= ix < image.shape[1] and 0 <= iy < image.shape[0] and mask[iy, ix] > 0):
                        continue
                        
                    # Informed filtering: if T_guess is available, check if flow matches projection
                    if T_guess_curr is not None and getattr(self.cfg, "use_informed_filtering", True):
                        p_O = t['pt3d']
                        p_Ci = T_guess_curr[:3, :3] @ p_O + T_guess_curr[:3, 3]
                        if p_Ci[2] > 0.01:
                            uv_proj_hom = K_curr @ p_Ci
                            uv_proj = uv_proj_hom[:2] / uv_proj_hom[2]
                            dist = np.linalg.norm(uv_curr - uv_proj)
                            # If flow is too far from guess projection, it's likely drifted
                            # We use a relatively loose threshold (e.g. 10 pixels) to allow for guess error
                            if dist > getattr(self.cfg, "informed_thresh", 10.0):
                                continue
                    
                    t['obs'][frame_idx] = uv_curr
            
            # Debug info
            n_flow = sum(1 for t in self.tracks.values() if frame_idx in t['obs'])
            # print(f"[GeoTracker] Frame {frame_idx}: {n_flow} tracks survived flow/mask/filtering")
        
        pts2d, pts3d = [], []
        tids = []
        for tid, t in self.tracks.items():
            if frame_idx in t['obs']:
                pts2d.append(t['obs'][frame_idx])
                pts3d.append(t['pt3d'])
                tids.append(tid)
        
        if len(pts2d) < 10:
            active_at_prev = sum(1 for t in self.tracks.values() if (frame_idx-1) in t['obs'])
            print(f"[GeoTracker] Lost tracking at frame {frame_idx}. Active at prev: {active_at_prev}, survived filtering: {len(pts2d)}")
            self.poses[frame_idx] = self.poses.get(frame_idx-1, np.eye(4)).copy()
            return False, 0
        
        pts2d = np.array(pts2d)
        pts3d = np.array(pts3d)
        
        # Subset selection for PnP
        max_pnp_points = getattr(self.cfg, "max_pnp_points", -1)
        if 0 < max_pnp_points < len(pts2d):
            indices = np.random.choice(len(pts2d), max_pnp_points, replace=False)
            pts2d_pnp = pts2d[indices]
            pts3d_pnp = pts3d[indices]
        else:
            pts2d_pnp = pts2d
            pts3d_pnp = pts3d
            
        cam_dict = {'model': 'PINHOLE', 'width': image.shape[1], 'height': image.shape[0], 'params': [K[0,0], K[1,1], K[0,2], K[1,2]]}
        
        initial_pose = None
        if T_guess is not None:
            import poselib
            initial_pose = poselib.CameraPose()
            q = R.from_matrix(T_guess[:3, :3]).as_quat()
            initial_pose.q = np.array([q[3], q[0], q[1], q[2]])
            initial_pose.t = T_guess[:3, 3]

        if not skip_pnp:
            res, info = poselib.estimate_absolute_pose(
                pts2d_pnp.astype(np.float64), pts3d_pnp, cam_dict, 
                {'max_reproj_error': self.cfg.ransac_thresh, 'min_iterations': 100, 'max_iterations': 1000}, initial_pose
            )
            
            # Correctly associate inliers with track IDs if a subset was used
            tids_pnp = np.array(tids)[indices] if (0 < max_pnp_points < len(pts2d)) else np.array(tids)
            
            T = np.eye(4)
            if res is not None:
                T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix()
                T[:3, 3] = res.pose.t
                
                # diff_t = np.linalg.norm(T[:3, 3] - T_guess[:3, 3])
                # diff_R = np.linalg.norm(R.from_matrix(T[:3, :3]).as_rotvec() - R.from_matrix(T_guess[:3, :3]).as_rotvec())
                # print(f"[GeoTracker] Frame {frame_idx}: PnP moved {diff_t:.4f}m, {np.degrees(diff_R):.2f}deg from guess")
            else:
                T = T_guess.copy()
                print(f"[GeoTracker] PnP FAILED at frame {frame_idx}, using informed guess.")
        else:
            print(f"[GeoTracker] Frame {frame_idx}: Skipping PnP, using informed guess directly.")
            T = T_guess.copy()
            res = None
            info = None
            tids_pnp = None
            
        self.poses[frame_idx] = T
        
        # Per-frame Motion-only Refinement
        # Note: refine_pose uses self.poses[frame_idx] as initial value.
        if getattr(self.cfg, "do_refine", True):
            inliers = info.get('inliers') if info is not None else None
            self.refine_pose(frame_idx, K, tids_pnp=tids_pnp, inlier_mask=inliers)
            
            # diff_t = np.linalg.norm(T_refined[:3, 3] - T[:3, 3])
            # print(f"[GeoTracker] Frame {frame_idx}: Refine moved {diff_t:.4f}m")
        
        if info is not None and 'inliers' in info:
            # Re-evaluate all points with the new T to prune outliers
            P = K @ T[:3, :]
            to_remove = []
            for tid, t in self.tracks.items():
                if frame_idx in t['obs']:
                    uv = t['obs'][frame_idx]
                    p_hom = P @ np.append(t['pt3d'], 1.0)
                    if p_hom[2] <= 0: # Behind camera
                        del t['obs'][frame_idx]
                        if len(t['obs']) < 2: to_remove.append(tid)
                        continue
                        
                    uv_proj = p_hom[:2] / p_hom[2]
                    err = np.linalg.norm(uv - uv_proj)
                    if err > self.cfg.ransac_thresh * 2.0: 
                        del t['obs'][frame_idx]
                        if len(t['obs']) < 2:
                            to_remove.append(tid)
            for tid in to_remove:
                if tid in self.tracks: del self.tracks[tid]
            
            n_inliers = len(info.get('inliers', [])) if info is not None else 0
            return True, n_inliers
            
        return True, 0
            
    def step_informed_with_occlusion(self, frame_idx, image, mask, depth, K, T_guess, flow_prev_curr=None, skip_pnp=False):
        """
        Tracking step with explicit depth-based occlusion handling and pose jump protection.
        """
        self.K_dict[frame_idx] = K
        self.poses[frame_idx] = T_guess

        if flow_prev_curr is not None:
            prev_idx = frame_idx - 1
            for tid, t in self.tracks.items():
                if prev_idx in t['obs']:
                    uv_prev = t['obs'][prev_idx]
                    delta = interpolate_flow(flow_prev_curr, uv_prev[None])[0]
                    uv_curr = uv_prev + delta
                    
                    ix, iy = int(round(uv_curr[0])), int(round(uv_curr[1]))
                    # 1. Mask Check
                    if not (0 <= ix < image.shape[1] and 0 <= iy < image.shape[0] and mask[iy, ix] > 0):
                        continue
                        
                    # 2. Depth Occlusion Check (Crucial for BundleSDF-like behavior)
                    p_O = t['pt3d']
                    p_Ci = T_guess[:3, :3] @ p_O + T_guess[:3, 3]
                    if p_Ci[2] > 0.01:
                        # Depth Consistency
                        if depth is not None and getattr(self.cfg, "use_occlusion_check", True):
                            d_obs = depth[iy, ix]
                            if d_obs > 0.01 and d_obs < p_Ci[2] - getattr(self.cfg, "occlusion_margin", 0.05):
                                continue # Occluded by something closer
                        
                        # Flow consistency with guess
                        if getattr(self.cfg, "use_informed_filtering", True):
                            uv_proj_hom = K @ p_Ci
                            uv_proj = uv_proj_hom[:2] / uv_proj_hom[2]
                            dist = np.linalg.norm(uv_curr - uv_proj)
                            if dist > getattr(self.cfg, "informed_thresh", 10.0):
                                continue
                    
                    t['obs'][frame_idx] = uv_curr
        
        # 3. Match Projections (Recover lost tracks)
        if getattr(self.cfg, "use_match_projections", True):
            self.match_projections(frame_idx, image, mask, K, T_guess)
        
        pts2d, pts3d, tids = [], [], []
        for tid, t in self.tracks.items():
            if frame_idx in t['obs']:
                pts2d.append(t['obs'][frame_idx])
                pts3d.append(t['pt3d'])
                tids.append(tid)
        
        if len(pts2d) < getattr(self.cfg, "min_pnp_inliers", 15):
            print(f"[GeoTracker] Insufficient tracks at frame {frame_idx} (found {len(pts2d)})")
            self.poses[frame_idx] = T_guess.copy()
            return False, 0
        
        pts2d_np = np.array(pts2d)
        pts3d_np = np.array(pts3d)
        cam_dict = {'model': 'PINHOLE', 'width': image.shape[1], 'height': image.shape[0], 'params': [K[0,0], K[1,1], K[0,2], K[1,2]]}
        
        initial_pose = None
        if T_guess is not None:
            initial_pose = poselib.CameraPose()
            q = R.from_matrix(T_guess[:3, :3]).as_quat()
            initial_pose.q = np.array([q[3], q[0], q[1], q[2]])
            initial_pose.t = T_guess[:3, 3]

        n_inliers = 0
        if not skip_pnp:
            res, info = poselib.estimate_absolute_pose(
                pts2d_np.astype(np.float64), pts3d_np, cam_dict, 
                {'max_reproj_error': self.cfg.ransac_thresh, 'min_iterations': 100, 'max_iterations': 1000}, initial_pose
            )
            
            if res is not None:
                n_inliers = len(info.get('inliers', []))
                T = np.eye(4)
                T[:3, :3] = R.from_quat([res.pose.q[1], res.pose.q[2], res.pose.q[3], res.pose.q[0]]).as_matrix()
                T[:3, 3] = res.pose.t
            else:
                T = T_guess.copy()
        else:
            T = T_guess.copy()
            info = None
            
        self.poses[frame_idx] = T
        
        if getattr(self.cfg, "do_refine", True) and n_inliers > 0:
            self.refine_pose(frame_idx, K, tids_pnp=tids, inlier_mask=info.get('inliers'))
            
        return True, n_inliers

    def run_ba(self):
        if len(self.keyframes) < 2: return
        kf_indices = self.keyframes[-self.cfg.max_keyframes:]
        # Before BA, try to triangulate points that have enough observations
        if getattr(self.cfg, "triangulate", False):
            self.triangulate_tracks(kf_indices[-1], getattr(self.cfg, "triangulate_thresh", 5))
            
        run_ba(kf_indices, self.poses, self.tracks, self.K_dict)

    def match_projections(self, frame_idx, image, mask, K, T_CiO):
        return match_projections(frame_idx, image, mask, K, T_CiO, self.tracks)
