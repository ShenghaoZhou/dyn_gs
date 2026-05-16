import numpy as np
import cv2
from pathlib import Path
from run_full_system_debug import load_frame_data_v2

data_dir = Path("data/hot3d_clips_processed/clip-003312")

poses_wo = {}
for i in range(25):
    fd = load_frame_data_v2(data_dir, i)
    T_WO = fd["T_WO_gt"]
    poses_wo[i] = T_WO
    if i > 1:
        T_WO_prev = poses_wo[i-1]
        T_WO_guess = (T_WO_prev @ np.linalg.inv(poses_wo[i-2])) @ T_WO_prev
        err = np.linalg.norm(T_WO_guess[:3, 3] - T_WO[:3, 3])
        print(f"Frame {i} World CV error: {err:.4f}m")
        
        # Camera CV error
        T_CW = fd["extrin"]
        T_CO = T_CW @ T_WO
        T_CO_prev = load_frame_data_v2(data_dir, i-1)["extrin"] @ poses_wo[i-1]
        T_CO_pprev = load_frame_data_v2(data_dir, i-2)["extrin"] @ poses_wo[i-2]
        T_CO_guess = (T_CO_prev @ np.linalg.inv(T_CO_pprev)) @ T_CO_prev
        err_c = np.linalg.norm(T_CO_guess[:3, 3] - T_CO[:3, 3])
        print(f"Frame {i} Camera CV error: {err_c:.4f}m")
