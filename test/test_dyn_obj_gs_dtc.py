import numpy as np
import cv2
import rerun as rr
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from gs_dyn_obj.grouped_gs import GaussianSuperPrimitive, GaussianSuperPrimitive3D
from gs_dyn_obj.utils.vis import vis_2dgs_rerun
import tyro
from dataclasses import dataclass
import rerun.blueprint as rrb
import matplotlib.pyplot as plt

def depth_to_rgb(depth, min_val=0.0, max_val=0.6):
    """Consistent colormapping for depth visualization."""
    depth_norm = np.clip((depth - min_val) / (max_val - min_val), 0, 1)
    # Using 'magma' or 'viridis' for high contrast
    colormap = plt.get_cmap("magma")
    rgb = (colormap(depth_norm)[..., :3] * 255).astype(np.uint8)
    # Ensure background (depth=0) is black
    rgb[depth == 0] = 0
    return rgb

@dataclass
class Config:
    data_root: str = "data/dtc_sample"
    near_plane: float = 0.01
    far_plane: float = 10.0
    device: str = "cuda"
    backend: str = "2dgs" # "2dgs" or "3dgs"
    renderer_backend: str = "inria" # "inria" or "gsplat"

def main(cfg: Config):
    rr.init("dyn_obj_gs_dtc")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")

    # Define Rerun Blueprint for UI layout
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Spatial3DView(origin="/world", name="3D Reconstruction"),
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/image", name="GT RGB"),
                rrb.Spatial2DView(origin="/rendered_gs", name="GS Render"),
                name="Color Comparison"
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/depth_gt", name="GT Depth"),
                rrb.Spatial2DView(origin="/rendered_depth", name="GS Depth"),
                name="Depth Comparison"
            ),
            rrb.Horizontal(
                rrb.Spatial2DView(origin="/mask", name="Object Mask"),
                rrb.Spatial2DView(origin="/rendered_normal", name="GS Normal"),
                name="Mask and Normals"
            )
        ),
        collapse_panels=True
    )
    rr.send_blueprint(blueprint)
    
    data_dir = Path(cfg.data_root)
    
    # Load intrinsics (fixed across frames in DTC sample)
    K = np.load(data_dir / "intrinsics.npy")
    
    # Load and log GT point cloud for the scene (if available)
    points_gt_path = data_dir / "points.npy"
    if points_gt_path.exists():
        points_gt = np.load(points_gt_path)
        rr.log("world/gt_points", rr.Points3D(points_gt, colors=[200, 200, 200]), static=True)
        print(f"Logged {len(points_gt)} world GT points.")
    
    # Get all frames and sort numerically by timestamp/filename
    image_paths = sorted((data_dir / "image").glob("*.jpg"), key=lambda p: int(p.stem))
    
    # Prepare GS object
    dyn_obj = None
    
    print(f"Total frames: {len(image_paths)}")
    
    for idx, img_path in enumerate(tqdm(image_paths)):
        stem = img_path.stem
        rr.set_time("frame_idx", sequence=idx)
        rr.set_time("timestamp", sequence=int(stem))
        
        # Load frame data
        image = np.array(cv2.imread(str(img_path))[..., ::-1])
        mask_path = data_dir / "mask" / f"{stem}.png"
        mask = np.array(cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE))
        
        depth_path = data_dir / "obj_depth_gt" / f"{stem}.npy"
        depth_gt = np.load(depth_path)
        
        pose_path = data_dir / "pose" / f"{stem}.npy"
        T_W_C = np.load(pose_path) # Dataset provides camera-to-world pose
        extrin = np.linalg.inv(T_W_C) # Convert to world-to-camera extrinsics
        
        # Log basic observation
        rr.log("image", rr.Image(image).compress(jpeg_quality=50))
        rr.log("mask", rr.Image(mask))
        rr.log("depth_gt", rr.Image(depth_to_rgb(depth_gt)).compress(jpeg_quality=50))

        # Log camera pose in 3D
        # Using T_W_C (Camera from Parent)
        rr.log("world/camera", rr.Transform3D(
            translation=T_W_C[:3, 3],
            mat3x3=T_W_C[:3, :3],
            relation=rr.TransformRelation.ParentFromChild
        ))
        rr.log("world/camera/image", rr.Pinhole(
            resolution=(image.shape[1], image.shape[0]),
            image_from_camera=K,
            camera_xyz=rr.ViewCoordinates.RDF # Right-Down-Forward
        ))
        rr.log("world/camera/image", rr.Image(image).compress(jpeg_quality=50))

        # Initial initialization from the first valid mask/depth
        if dyn_obj is None:
            if mask.sum() > 0 and depth_gt.sum() > 0:
                # Following approach in test_dyn_obj_track_gs.py (line 107)
                # dyn_obj = GaussianSuperPrimitive(image, obj_mask, depth, extrin, K)
                if cfg.backend == "2dgs":
                    dyn_obj = GaussianSuperPrimitive(image, mask, depth_gt, extrin, K)
                else:
                    dyn_obj = GaussianSuperPrimitive3D.from_depth(image, mask, depth_gt, extrin, K)
                print(f"Initialized Dynamic Object ({cfg.backend}) with {len(dyn_obj.gs_params.means)} Gaussians at frame {idx}.")
                
                # Visualize initial GS in 3D as a point cloud
                # We log it in 'world' space
                from gs_dyn_obj.utils.vis import vis_3dgs_rerun
                if cfg.backend == "2dgs":
                    vis_2dgs_rerun(dyn_obj.gs_params, name="world/gs_points", points_only=True)
                else:
                    vis_3dgs_rerun(dyn_obj.gs_params, name="world/gs_points", points_only=True)
            else:
                continue
                
        if dyn_obj is not None:
            # Render the initialized GS using GT camera pose of the current frame
            extrin_torch = torch.from_numpy(extrin).float().to(cfg.device)
            K_torch = torch.from_numpy(K).float().to(cfg.device)
            
            # Render using the GSParam.render method
            render_mode = "normal" if cfg.backend == "2dgs" else "3dgs"
            render_pkg = dyn_obj.gs_params.render(
                extrin_torch, K_torch, 
                width=image.shape[1], height=image.shape[0],
                near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                bg=torch.zeros(3, device=cfg.device),
                mode=render_mode,
                backend=cfg.renderer_backend
            )
            
            # render_pkg contains (render_image, render_depth, render_normal)
            rendered_image = render_pkg[0].permute(1, 2, 0).cpu().detach().numpy()
            rendered_image = (rendered_image.clip(0, 1) * 255).astype(np.uint8)
            rr.log("rendered_gs", rr.Image(rendered_image).compress(jpeg_quality=50))
            
            rendered_depth = render_pkg[1].squeeze(0).cpu().detach().numpy()
            rr.log("rendered_depth", rr.Image(depth_to_rgb(rendered_depth)).compress(jpeg_quality=50))
            
            # Optional: log rendered normals
            rendered_normal = render_pkg[2].permute(1, 2, 0).cpu().detach().numpy()
            # Convert normals from [-1, 1] to [0, 255] for visualization
            rendered_normal = ((rendered_normal + 1.0) / 2.0 * 255.0).astype(np.uint8)
            rr.log("rendered_normal", rr.Image(rendered_normal).compress(jpeg_quality=50))

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
