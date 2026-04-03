import cv2
import numpy as np
import rerun as rr
from data import HOT3DDataLoader
from pathlib import Path
import tyro
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    data_seq: str = "clip-002793"
    max_features: int = 1000
    min_features: int = 200
    depth_model: str = "DA3-GIANT-1.1"

def main(cfg: Config):
    data_loader = HOT3DDataLoader(Path(cfg.data_root) / cfg.data_seq, depth_model=cfg.depth_model)
    rr.init("dynamic_object_tracking")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")

    # ORB detector
    orb = cv2.ORB_create(nfeatures=cfg.max_features)
    
    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    prev_gray = None
    prev_image = None
    prev_pts = None
    prev_frame = None
    
    # Pose of current camera in object frame (Starting frame's camera frame)
    T_O_C = np.eye(4)
    accumulated_pts = np.zeros((0, 3))
    
    # Initialize object frame as static identity
    rr.log("object", rr.Transform3D(translation=[0, 0, 0], mat3x3=np.eye(3)), static=True)

    print(f"Processing sequence {cfg.data_seq}...")
    
    for idx in tqdm(range(len(data_loader))):
        rr.set_time("frame_idx", sequence=idx)
        frame = data_loader[idx]
        image = frame["image"]
        K = frame["K"]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = frame["obj_mask"]
        H, W = gray.shape
        
        # Log the image and mask
        rr.log("image", rr.Image(image).compress(jpeg_quality=50))
        rr.log("image/mask", rr.Image(mask))

        vis_current_pts = None
        vis_prev_pts = None
        current_pts = None
        
        if prev_pts is None or len(prev_pts) == 0:
            # First frame or needs reset
            keypoints = orb.detect(gray, mask=mask)
            if keypoints:
                current_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
            else:
                current_pts = None
        else:
            # Track existing features from prev_pts to gray
            new_pts, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            
            # Filter by status and mask
            if new_pts is not None:
                status = status.flatten()
                mask_at_new = []
                for p in new_pts:
                    x, y = int(round(p[0][0])), int(round(p[0][1]))
                    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0:
                        mask_at_new.append(True)
                    else:
                        mask_at_new.append(False)
                
                mask_at_new = np.array(mask_at_new)
                valid = (status == 1) & mask_at_new
                
                if np.any(valid):
                    vis_current_pts = new_pts[valid].reshape(-1, 2)
                    vis_prev_pts = prev_pts[valid].reshape(-1, 2)
                
                good_new = new_pts[valid]
                
                if len(good_new) < cfg.min_features:
                    # Redetect for the NEXT frame
                    keypoints = orb.detect(gray, mask=mask)
                    if keypoints:
                        current_pts = np.array([kp.pt for kp in keypoints], dtype=np.float32).reshape(-1, 1, 2)
                    else:
                        current_pts = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
                else:
                    current_pts = good_new.reshape(-1, 1, 2)
            
            if idx % 10 == 0:
                tracked_cnt = len(vis_current_pts) if vis_current_pts is not None else 0
                next_cnt = len(current_pts) if current_pts is not None else 0
                print(f"Frame {idx}: Tracked {tracked_cnt}, ready for next: {next_cnt}")

        # Geometric reconstruction logic
        if vis_current_pts is not None and vis_prev_pts is not None and prev_image is not None:
            # 1. Essential Matrix Decomposition
            E, mask_E = cv2.findEssentialMat(vis_prev_pts, vis_current_pts, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            if E is not None and E.shape == (3, 3):
                mask_E = mask_E.flatten().astype(bool)
                if mask_E.sum() < 5:
                    pass # Skip if not enough inliers
                else:
                    num_inliers, R, t, mask_pose = cv2.recoverPose(E, vis_prev_pts[mask_E], vis_current_pts[mask_E], K)
                    
                    mask_pose = mask_pose.flatten().astype(bool)
                    pts_p = vis_prev_pts[mask_E][mask_pose]
                    pts_c = vis_current_pts[mask_E][mask_pose]
                    
                    if len(pts_p) >= 5:
                        # 2. Scale estimation using depth (if available)
                        scale = 1.0
                        if frame.get("depth") is not None and prev_frame.get("depth") is not None:
                            scales = []
                            K_inv = np.linalg.inv(K)
                            for i in range(len(pts_p)):
                                ux_p, uy_p = int(round(pts_p[i][0])), int(round(pts_p[i][1]))
                                ux_c, uy_c = int(round(pts_c[i][0])), int(round(pts_c[i][1]))
                                
                                if (0 <= ux_p < W and 0 <= uy_p < H and 
                                    0 <= ux_c < W and 0 <= uy_c < H):
                                    d_p = prev_frame["depth"][uy_p, ux_p]
                                    d_c = frame["depth"][uy_c, ux_c]
                                    if d_p > 0 and d_c > 0:
                                        X_p = d_p * K_inv @ np.array([pts_p[i][0], pts_p[i][1], 1.0])
                                        X_c = d_c * K_inv @ np.array([pts_c[i][0], pts_c[i][1], 1.0])
                                        alpha = np.dot(X_c - R @ X_p, t.flatten())
                                        if alpha > 0: scales.append(alpha)
                            if scales: scale = np.median(scales)

                        # 3. Accumulated camera pose in object frame
                        T_rel = np.eye(4)
                        T_rel[:3, :3] = R
                        T_rel[:3, 3] = (t * scale).flatten()
                        
                        T_O_C_prev = T_O_C.copy()
                        T_O_C = T_O_C @ np.linalg.inv(T_rel)
                        
                        # 4. Triangulate Points
                        P1 = K @ np.eye(3, 4)
                        P2 = K @ T_rel[:3, :]
                        pts4D = cv2.triangulatePoints(P1, P2, pts_p.T, pts_c.T)
                        pts3D_prev = (pts4D[:3, :] / pts4D[3, :]).T
                        
                        pts3D_prev_homo = np.concatenate([pts3D_prev, np.ones((len(pts3D_prev), 1))], axis=-1)
                        pts3D_obj = (T_O_C_prev @ pts3D_prev_homo.T).T[:, :3]

                        # Accumulate points
                        accumulated_pts = np.vstack([accumulated_pts, pts3D_obj])

                        # Log results to Rerun
                        # Note: Parent is 'object', Child is 'camera'. T_O_C is ParentFromChild.
                        rr.log("object/camera", rr.Transform3D(translation=T_O_C[:3, 3], mat3x3=T_O_C[:3, :3], 
                                                             relation=rr.TransformRelation.ParentFromChild))
                        rr.log("object/camera/image", rr.Pinhole(resolution=(W, H), image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF))
                        
                        # Green for new points, Red for accumulated
                        rr.log("object/points/new", rr.Points3D(pts3D_obj, colors=[(0, 255, 0)], radii=0.005))
                        rr.log("object/points/accumulated", rr.Points3D(accumulated_pts, colors=[(255, 0, 0)], radii=0.002))

            # Logging matches for visualization (existing logic)
            kp1 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in vis_prev_pts]
            kp2 = [cv2.KeyPoint(float(p[0]), float(p[1]), 1.0) for p in vis_current_pts]
            matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]
            
            match_img = cv2.drawMatches(prev_image, kp1, image, kp2, matches, None, 
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            
            rr.log("matches_view", rr.Image(match_img).compress(jpeg_quality=50))
        
        # Save images for next iteration
        prev_gray = gray.copy()
        prev_image = image.copy()
        prev_pts = current_pts
        prev_frame = frame

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
