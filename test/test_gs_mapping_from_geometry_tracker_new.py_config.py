@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    clip_id: str = "clip-003312"
    init_frame: int = 0
    n_frames: int = 50
    device: str = "cuda"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    
    # Tracker Parameters
    feature_type: str = "grid" 
    n_features: int = 1200
    ransac_thresh: float = 1.0
    kf_disparity_thresh: float = 20.0
    kf_min_interval: int = 8
    kf_max_interval: int = 15
    kf_overlap_thresh: float = 0.6
    max_keyframes: int = 20
    triangulate: bool = True
    triangulate_thresh: int = 3
    triangulate_parallax_thresh: float = 15.0
    do_refine: bool = True
    use_gt_depth: bool = False
    use_informed_filtering: bool = True
    skip_pnp: bool = False
    informed_thresh: float = 20.0 
    guess_type: str = "GT-Cam+CV-Obj"
    
    # Mapping Parameters
    num_steps_per_frame: int = 200
    lr_means: float = 1e-3
    lr_quats: float = 1e-3
    lr_scales: float = 5e-3
    lr_colors: float = 2.5e-3
    lr_opacity: float = 5e-2
    
    pyr_levels: int = 2
    pyr_interval: int = 30
    
    # PGSR Loss Weights
    lambda_dssim: float = 0.5
    single_view_weight: float = 0.015
    multi_view_ncc_weight: float = 0.5
    multi_view_geo_weight: float = 10.0
    multi_view_photo_weight: float = 10.0
    scale_loss_weight: float = 100.0
    
    # PGSR Multi-view Parameters
    multi_view_min_dis: float = 0.02 # Min distance to be considered a neighbor
    multi_view_max_dis: float = 0.1 # Max distance for virtual cam noise
    multi_view_patch_size: int = 3
    multi_view_sample_num: int = 8000
    multi_view_pixel_noise_th: float = 1.0
    use_virtul_cam: bool = True
    virtul_cam_prob: float = 0.5
    
    near_plane: float = 0.01
    far_plane: float = 100.0
    
    # Densification & Pruning
    densify_from_tracker: bool = True
    densify_from_depth: bool = False 
    prune_opacity_th: float = 0.01
    prune_screen_size_th: float = 100.0
    
    # TSDF Parameters
    run_tsdf: bool = True
    tsdf_voxel_size: float = 0.005
    tsdf_margin: float = 0.02
