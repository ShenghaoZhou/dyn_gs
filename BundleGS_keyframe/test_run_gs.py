import os
import sys
import cv2
import numpy as np
import argparse
import yaml
import logging

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from BundleGS.bundlesdf_gs import BundleSdfGS
from BundleSDF.bundlesdf import YcbineoatReader # Reuse reader from BundleSDF

def run_one_video(video_dir, out_folder, stride=1):
    os.makedirs(out_folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    
    # Minimal config for now
    cfg_track_dir = os.path.join(out_folder, 'config_track.yml')
    with open(cfg_track_dir, 'w') as f:
        yaml.dump({'SPDLOG': 2}, f)
        
    tracker = BundleSdfGS(cfg_track_dir=cfg_track_dir)
    
    reader = YcbineoatReader(video_dir=video_dir, shorter_side=480)
    
    for i in range(0, len(reader.color_files), stride):
        color = cv2.imread(reader.color_files[i])
        depth = reader.get_depth(i)
        mask = reader.get_mask(i)
        
        H, W = depth.shape[:2]
        color = cv2.resize(color, (W, H), interpolation=cv2.INTER_NEAREST)
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        
        # BundleSDF color is RGB for tracker
        color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        
        id_str = reader.id_strs[i]
        K = reader.K
        
        logging.info(f"Processing frame {i} ({id_str})")
        pose = tracker.run(color_rgb, depth, K, id_str, mask=mask)
        
        # Save pose
        pose_file = os.path.join(out_folder, f"{id_str}.txt")
        np.savetxt(pose_file, pose)
        
    tracker.on_finish()
    logging.info("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', type=str, required=True)
    parser.add_argument('--out_folder', type=str, default="output_gs")
    parser.add_argument('--stride', type=int, default=1)
    args = parser.parse_args()
    
    run_one_video(args.video_dir, args.out_folder, args.stride)
