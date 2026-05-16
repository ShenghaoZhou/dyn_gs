import sys
from pathlib import Path
import tyro
import torch
import torch.multiprocessing as mp

from run_full_system_debug import GlobalConfig, HighFidelityGSMapping, load_frame_data_v2, init_gs_from_tracker_points, unproject_depth
from obj_gs_mapping import MappingConfig
import numpy as np

def main():
    cfg = GlobalConfig()
    data_dir = Path(cfg.data_root) / cfg.clip_id
    f0 = load_frame_data_v2(data_dir, cfg.init_frame)
    if f0 is None:
        print("Data not found")
        return
        
    device = torch.device(cfg.device)
    h, w = f0["image"].shape[:2]
    mask_init = f0["mask"] > 0
    depth_t = torch.from_numpy(f0["depth"]).float().to(device)
    K0_t = torch.from_numpy(f0["K"]).float().to(device)
    full_pts_c = unproject_depth(depth_t, K0_t, h, w)
    
    ys, xs = np.where(mask_init)
    active_points = []
    active_colors = []
    T_WO_gt0 = f0["T_WO_gt"]
    T_CW0 = f0["extrin"]
    for y, x in zip(ys, xs):
        p_c = full_pts_c[y, x].cpu().numpy()
        p_w = np.linalg.inv(T_CW0[:3, :3]) @ (p_c - T_CW0[:3, 3])
        p_o = np.linalg.inv(T_WO_gt0[:3, :3]) @ (p_w - T_WO_gt0[:3, 3]) if T_WO_gt0 is not None else p_w
        active_points.append(p_o)
        active_colors.append(f0["image"][y, x] / 255.0)

    gs_params = init_gs_from_tracker_points(np.array(active_points), np.array(active_colors), device, gs_type=cfg.gs_type)
    
    map_cfg = MappingConfig(gs_type=cfg.gs_type, device=cfg.device)
    mapper = HighFidelityGSMapping(map_cfg, gs_params)
    print("Before run():", gs_params.means.requires_grad)
    
    # simulate what HighFidelityGSMapping.run() does initially
    mapper.device = torch.device(mapper.cfg.device)
    mapper.gs_params.to(mapper.device)
    mapper.setup_optimizer()
    
    print("After setup_optimizer():", mapper.gs_params.means.requires_grad)
    
if __name__ == "__main__":
    main()
