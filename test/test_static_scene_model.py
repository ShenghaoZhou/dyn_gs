import torch
import numpy as np
import cv2
import rerun as rr
import sys
from pathlib import Path
from tqdm import tqdm
import tyro
from dataclasses import dataclass
from PIL import Image
import time
from collections import defaultdict

# Add project roots and lib paths to sys.path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

lib_map_anything = project_root / "lib" / "map-anything"
if str(lib_map_anything) not in sys.path:
    sys.path.append(str(lib_map_anything))

lib_efficient_tam = project_root / "lib" / "EfficientTAM"
if str(lib_efficient_tam) not in sys.path:
    sys.path.append(str(lib_efficient_tam))

# Imports from libraries
from data import HOT3DDataLoader
from mapanything.models.mapanything import MapAnything
from kv_tracker_map_anything import KVTrackerMapAnything
from mapanything.utils.image import preprocess_inputs
from gs_scene.scene_model import SceneModel
from old_ref.keyframe_window import KeyFrameWindow

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    data_seq: str = "clip-002793"
    mapanything_model: str = "facebook/map-anything-v1"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"
    num_frames: int = None
    chunk_size: int = 8
    num_opt_steps: int = 5
    use_anchors: bool = False
    use_exposure: bool = False
    pyr_levels: int = 2
    anchor_dist_threshold: float = 2.0
    use_guided_mvs: bool = False

def prepare_mapanything_view(frame_data, idx):
    """Convert HOT3D frame data to MapAnything view format."""
    # HOT3D extrinsics is T_c_w. MapAnything expects T_w_c (camera2world)
    T_cw = frame_data["extrin"]
    T_wc = np.linalg.inv(T_cw)
    
    view = {
        "img": torch.from_numpy(frame_data["image"]),
        "intrinsics": torch.from_numpy(frame_data["K"].astype(np.float32)),
        "camera_poses": torch.from_numpy(T_wc.astype(np.float32)),
        "is_metric_scale": torch.tensor([True]),
        "idx": idx
    }
    return view

def main(cfg: Config):
    # Enable TF32 for better performance on Ampere+ GPUs
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # 1. Initialize Rerun
    rr.init("test_static_scene_model")
    try:
        rr.connect_grpc(cfg.rerun_url)
    except Exception as e:
        print(f"Failed to connect to Rerun: {e}")

    # 2. Load Data
    data_path = Path(cfg.data_root) / cfg.data_seq
    print(f"Loading data from {data_path}...")
    loader = HOT3DDataLoader(data_path)
    num_frames = cfg.num_frames if cfg.num_frames is not None else len(loader)
    
    # 3. Initialize Models
    print("Initializing MapAnything...")
    ma_model = MapAnything.from_pretrained(cfg.mapanything_model).to(cfg.device).eval()
    ma_tracker = KVTrackerMapAnything(ma_model)

    print("Initializing SceneModel...")
    # Initialize with first frame's resolution
    h, w = loader[0]["image"].shape[:2]
    scene_model = SceneModel(
        width=w, 
        height=h, 
        num_steps=cfg.num_opt_steps,
        use_anchors=cfg.use_anchors,
        use_exposure=cfg.use_exposure,
        pyr_levels=cfg.pyr_levels,
        anchor_dist_threshold=cfg.anchor_dist_threshold,
        use_guided_mvs=cfg.use_guided_mvs
    )

    keyframe_window = KeyFrameWindow(cfg.chunk_size, rr_log=True)

    # 4. Prepare MapAnything views
    print("Preparing MapAnything views...")
    ma_views_raw = []
    for i in range(num_frames):
        ma_views_raw.append(prepare_mapanything_view(loader[i], i))
    
    ma_views_processed = preprocess_inputs(
        ma_views_raw,
        size=252, # Standard size for MapAnything
        norm_type=ma_model.encoder.data_norm_type,
        patch_size=ma_model.encoder.patch_size,
        verbose=False
    )
    del ma_views_raw

    # 5. Process Sequence
    print("Running online GS scene building...")
    with torch.no_grad():
        for i in tqdm(range(num_frames)):
            rr.set_time("frame_idx", sequence=i)
            frame_data = loader[i]
            img_np = frame_data["image"]
            
            # MapAnything Tracking/Mapping
            ma_view = ma_views_processed[i]
            # Ensure tensors are on the correct device
            for k, v in ma_view.items():
                if isinstance(v, torch.Tensor):
                    ma_view[k] = v.to(cfg.device)
                    if ma_view[k].is_floating_point():
                        ma_view[k] = ma_view[k].to(torch.float32)

            if i == 0:
                ma_out = ma_tracker.mapping_step([ma_view])[0]
            else:
                ma_out = ma_tracker.tracking_step(ma_view)[0]
            
            # Extract depth
            if "pts3d_cam" in ma_out:
                pts_cam = ma_out["pts3d_cam"][0].cpu().numpy()
                depth = pts_cam[..., 2]
                # Resize depth to original image size
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                depth = np.zeros((h, w))

            # Check if Keyframe
            frame_for_window = {
                "image": img_np,
                "extrin": frame_data["extrin"],
                "K": frame_data["K"],
                "depth": depth,
                "obj_mask": frame_data["obj_mask"],
                "hand_mask": frame_data["hand_mask"]
            }

            if keyframe_window.try_add_frame(frame_for_window, scene_model):
                print(f"Adding keyframe at frame {i}")
                
                # Update GS model
                # Use hand and object mask to exclude dynamic parts from the static scene
                obj_mask = frame_data.get("obj_mask")
                hand_mask = frame_data.get("hand_mask")
                
                mask = np.zeros((h, w), dtype=bool)
                if obj_mask is not None:
                    mask = np.logical_or(mask, obj_mask > 0)
                if hand_mask is not None:
                    mask = np.logical_or(mask, hand_mask > 0)
                
                static_mask = ~mask
                
                # Debug mask counts
                print(f"Frame {i}: mask (objects) count: {np.sum(mask)}, static_mask (bg) count: {np.sum(static_mask)}")
                
                # We need to switch back to train mode for scene_model.update since it involves optimization
                with torch.enable_grad():
                    scene_model.update(img_np, depth, frame_data["extrin"], frame_data["K"], mask=static_mask)
                
                print(f"Number of Gaussians: {len(scene_model.xyz)}")
                
                # Log GS model
                with torch.no_grad():
                    xyz = scene_model.xyz.detach().cpu().numpy()
                    colors = scene_model.colors.detach().cpu().numpy().squeeze()
                    if len(xyz) > 0:
                        rr.log("scene/gs_points", rr.Points3D(xyz, colors=colors))

            # Visualization
            rr.log("input/image", rr.Image(img_np).compress(jpeg_quality=50))
            rr.log("output/depth", rr.DepthImage(depth))
            
            # Render from current view
            if scene_model.xyz.shape[0] > 0:
                with torch.no_grad():
                    extrin = frame_data["extrin"]
                    K = frame_data["K"]
                    # Use extrin.T because SceneModel.render expects transposed W2C
                    view_mat = torch.from_numpy(extrin).float().cuda()
                    
                    fx, fy = K[0, 0], K[1, 1]
                    fov_x = 2 * np.arctan(w / (2 * fx))
                    fov_y = 2 * np.arctan(h / (2 * fy))
                    
                    render_pkg = scene_model.render(w, h, view_mat, fov_x=fov_x, fov_y=fov_y)
                    rendered_img_tensor = render_pkg["render"]
                    
                    # Apply exposure compensation for visualization/PSNR if enabled
                    if cfg.use_exposure and len(scene_model.keyframes) > 0:
                        # Use the latest keyframe's exposure as an approximation for the current view
                        last_kf = scene_model.keyframes[-1]
                        rendered_img_tensor = last_kf.apply_exposure(rendered_img_tensor)

                    rendered_img = rendered_img_tensor.permute(1, 2, 0).cpu().numpy()
                    
                    # Compute PSNR
                    obs_img = img_np.astype(np.float32) / 255.0
                    mse = np.mean((rendered_img - obs_img) ** 2)
                    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else 100
                    print(f"Frame {i}: PSNR: {psnr:.2f}")
                    # Using rr.Scalars archetype for Rerun 0.31.3
                    rr.log("output/psnr", rr.Scalars(psnr))
                    
                    rendered_img_uint8 = (np.clip(rendered_img, 0, 1) * 255).astype(np.uint8)
                    rr.log("output/rendered_gs", rr.Image(rendered_img_uint8).compress(jpeg_quality=50))

    print("Done!")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
