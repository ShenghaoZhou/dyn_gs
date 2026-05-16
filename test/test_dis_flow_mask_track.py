import cv2
import numpy as np
from data import HOT3DDataLoader
from pathlib import Path
import tyro
from dataclasses import dataclass
from tqdm import tqdm
import rerun as rr

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    data_seq: str = "clip-001924"
    max_frames: int = 100
    skip_frames: int = 1
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    dis_preset: int = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
    point_radius: float = 1.0

def interpolate_flow(flow, pts):
    """Bilinear interpolation of flow at sub-pixel point positions."""
    x = pts[:, 0]
    y = pts[:, 1]
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    
    h, w = flow.shape[:2]
    x0, x1 = np.clip(x0, 0, w-1), np.clip(x1, 0, w-1)
    y0, y1 = np.clip(y0, 0, h-1), np.clip(y1, 0, h-1)
    
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    
    # Apply bilinear interpolation
    f_p = (wa[:, None] * flow[y0, x0] + 
           wb[:, None] * flow[y1, x0] + 
           wc[:, None] * flow[y0, x1] + 
           wd[:, None] * flow[y1, x1])
    return f_p

def main(cfg: Config):
    # Initialize Rerun
    rr.init("dis_flow_mask_track", spawn=False)
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)
    
    data_loader = HOT3DDataLoader(Path(cfg.data_root) / cfg.data_seq)
    
    # Identify the start of the dynamic phase
    start_idx = 0
    if len(data_loader.dynamic_phases) > 0:
        start_idx, _ = data_loader.dynamic_phases[0]
    
    end_idx = start_idx + cfg.max_frames * cfg.skip_frames
    indices = list(range(start_idx, min(end_idx, len(data_loader)), cfg.skip_frames))

    # DIS Optical Flow setup
    dis = cv2.DISOpticalFlow_create(cfg.dis_preset)
    
    prev_gray = None
    current_pts = None # Stores all pixels initialized from the first mask
    point_colors = None
    
    print(f"Tracking mask pixels for {len(indices)} frames in sequence {cfg.data_seq}...")
    
    for idx_local, idx in enumerate(tqdm(indices)):
        f = data_loader[idx]
        img = f["image"]
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = f["obj_mask"]
        
        rr.set_time("frame_idx", sequence=idx_local)
        
        if idx_local == 0:
            # 1. Initialize points from all pixels in the mask
            yy, xx = np.where(mask > 0)
            if len(xx) == 0:
                print(f"Warning: No mask found at frame {idx}. Tracking might be empty.")
                current_pts = np.zeros((0, 2), dtype=np.float32)
                point_colors = np.zeros((0, 3), dtype=np.uint8)
            else:
                current_pts = np.stack([xx, yy], axis=-1).astype(np.float32)
                # Pick a color for the tracked mask (e.g., green or original colors)
                # Let's use the original colors from the first frame for a "painting" effect
                point_colors = img[yy, xx]
            # Log initial state
            overlay_img = img.copy()
            if len(current_pts) > 0:
                ix, iy = np.round(current_pts[:, 0]).astype(int), np.round(current_pts[:, 1]).astype(int)
                mask_color = np.array([0, 255, 0], dtype=np.uint8)
                alpha = 0.5
                overlay_img[iy, ix] = (overlay_img[iy, ix] * (1 - alpha) + mask_color * alpha).astype(np.uint8)

            rr.log("image", rr.Image(img))
            rr.log("image/overlay", rr.Image(overlay_img))
            if len(current_pts) > 0:
                rr.log("tracked_mask_pts", rr.Points2D(current_pts, colors=point_colors, radii=cfg.point_radius))
        else:
            # 2. Compute Dense Flow from prev_gray to gray
            flow = dis.calc(prev_gray, gray, None)
            
            # 3. Advect points using the computed flow
            if len(current_pts) > 0:
                delta = interpolate_flow(flow, current_pts)
                current_pts = current_pts + delta
                
                # 4. Filter points that stay within image bounds
                h, w = gray.shape
                valid = (current_pts[:, 0] >= 0) & (current_pts[:, 0] < w) & \
                        (current_pts[:, 1] >= 0) & (current_pts[:, 1] < h)
                
                current_pts = current_pts[valid]
                point_colors = point_colors[valid]
            
            # 5. Visualization
            # Create a semi-transparent overlay
            overlay_img = img.copy()
            if len(current_pts) > 0:
                ix, iy = np.round(current_pts[:, 0]).astype(int), np.round(current_pts[:, 1]).astype(int)
                # Ensure within bounds for the overlay image
                h_img, w_img = img.shape[:2]
                valid_overlay = (ix >= 0) & (ix < w_img) & (iy >= 0) & (iy < h_img)
                ix, iy = ix[valid_overlay], iy[valid_overlay]
                
                # Apply a semi-transparent green tint to tracked pixels
                mask_color = np.array([0, 255, 0], dtype=np.uint8)
                alpha = 0.5
                overlay_img[iy, ix] = (overlay_img[iy, ix] * (1 - alpha) + mask_color * alpha).astype(np.uint8)

            rr.log("image", rr.Image(img))
            rr.log("image/overlay", rr.Image(overlay_img))
            if len(current_pts) > 0:
                rr.log("tracked_mask_pts", rr.Points2D(current_pts, colors=point_colors, radii=cfg.point_radius))
            
            # Optional: Log the mask movement as a separate "prediction" or overlay
            # We can also create a mask image from the points
            # tracked_mask_img = np.zeros_like(mask)
            # ix, iy = np.round(current_pts[:, 0]).astype(int), np.round(current_pts[:, 1]).astype(int)
            # tracked_mask_img[iy, ix] = 255
            # rr.log("tracked_mask_img", rr.Image(tracked_mask_img))

        prev_gray = gray.copy()

    print("Tracking completed.")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
