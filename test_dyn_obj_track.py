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

def main(cfg: Config):
    data_loader = HOT3DDataLoader(Path(cfg.data_root) / cfg.data_seq)
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
    
    print(f"Processing sequence {cfg.data_seq}...")
    
    for idx in tqdm(range(len(data_loader))):
        rr.set_time("frame_idx", sequence=idx)
        frame = data_loader[idx]
        image = frame["image"]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mask = frame["obj_mask"]
        
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

        # Log tracked points
        if vis_current_pts is not None and vis_prev_pts is not None and prev_image is not None:
            # Use cv2.drawMatches to visualize the matches between prev_image and image
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

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
