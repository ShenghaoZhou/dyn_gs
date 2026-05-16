import rerun as rr
from dataclasses import dataclass
import tyro
from data import HOT3DDataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    data_seq: str = "clip-003363"
    depth_model: str = "DA3-GIANT-1.1"
    depth_path: str = "/media/shzhou/RPNG_FLASH_2/dynamic_gs/grouped_gs/outputs/depth_save/depth"


def rr_log_camera(image, K, extrin, name, static=False, color=None):
    rr.log(f"{name}", rr.Transform3D(
        translation=extrin[:3, 3], mat3x3=extrin[:3, :3],
        relation=rr.TransformRelation.ChildFromParent
    ), static=static)
    if color is None:
        rr.log(f"{name}/image", rr.Pinhole(
            resolution=(image.shape[1], image.shape[0]),
            image_from_camera=K,
            camera_xyz=rr.ViewCoordinates.RDF,
        ), static=static)
    else:
        rr.log(f"{name}/image", rr.Pinhole(
            resolution=(image.shape[1], image.shape[0]),
            image_from_camera=K,
            camera_xyz=rr.ViewCoordinates.RDF,
            color=color,
        ), static=static)
    rr.log(f"{name}/image", rr.Image(
        image).compress(jpeg_quality=30), static=static)


def unproject_depth(depth, K, obj_mask=None):
    H, W = depth.shape
    v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    
    if obj_mask is not None:
        mask = (obj_mask > 0) & (depth > 0)
    else:
        mask = depth > 0
        
    u = u[mask]
    v = v[mask]
    z = depth[mask]
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    pts = np.stack([x, y, z], axis=-1)
    return pts


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    data_loader = HOT3DDataLoader(Path(cfg.data_root) / cfg.data_seq, depth_model=cfg.depth_model)
    rr.init("dynamic_object_depth")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")

    points = data_loader.points3D
    mesh = data_loader.obj_mesh

    rr.log("world/static_background", rr.Points3D(
        points
    ), static=True)

    # Log mesh once as static
    rr.log("world/obj_mesh", rr.Mesh3D(vertex_positions=mesh.vertices,
                                triangle_indices=mesh.faces,
                                vertex_normals=mesh.vertex_normals), static=True)
    
    rr.log("object/obj_mesh", rr.Mesh3D(vertex_positions=mesh.vertices,
                                     triangle_indices=mesh.faces,
                                     vertex_normals=mesh.vertex_normals), static=True)

    cam_positions_world = []
    obj_positions_world = []
    cam_positions_obj = []

    for idx in tqdm(range(len(data_loader))):
        rr.set_time("frame_idx", sequence=idx)
        
        # World space poses
        pose_wo = data_loader.get_obj_pose(idx)
        obj_pos_world = pose_wo[:3, 3]
        obj_positions_world.append(obj_pos_world)

        frame = data_loader[idx]
        T_cw = frame["extrin"]
        T_wc = np.linalg.inv(T_cw)
        cam_pos_world = T_wc[:3, 3]
        cam_positions_world.append(cam_pos_world)

        # Object space pose: T_co = T_cw * T_wo
        T_co = T_cw @ pose_wo
        T_oc = np.linalg.inv(T_co)
        cam_pos_obj = T_oc[:3, 3]
        cam_positions_obj.append(cam_pos_obj)

        # Depth unprojection
        if frame.get("depth") is not None:
            # Mask depth with object mask to only get object points
            obj_pts_cam = unproject_depth(frame["depth"], frame["K"], obj_mask=frame["obj_mask"])
            
            # Transform to object frame: P_o = T_oc * P_c = (T_cw @ T_wo)^-1 * P_c
            # Actually, P_o = T_oc @ P_c_homogeneous
            if obj_pts_cam.shape[0] > 0:
                obj_pts_cam_homo = np.concatenate([obj_pts_cam, np.ones((obj_pts_cam.shape[0], 1))], axis=-1)
                obj_pts_obj = (T_oc @ obj_pts_cam_homo.T).T[:, :3]
                
                # Transform to world frame: P_w = T_wc @ P_c
                obj_pts_world = (T_wc @ obj_pts_cam_homo.T).T[:, :3]
            else:
                obj_pts_obj = np.zeros((0, 3))
                obj_pts_world = np.zeros((0, 3))
        else:
            obj_pts_obj = np.zeros((0, 3))
            obj_pts_world = np.zeros((0, 3))

        # --- World space view ---
        rr.log("world/obj_mesh", rr.Transform3D(
            translation=pose_wo[:3, 3], mat3x3=pose_wo[:3, :3]))
        
        rr.log("world/obj_points", rr.Points3D(obj_pts_world, colors=[(0, 255, 0)], radii=0.002))

        rr_log_camera(frame["image"], frame["K"],
                          T_cw,
                          name=f"world/cam",
                          color=(255, 0, 0))
        
        rr.log("world/trajectory/camera", rr.LineStrips3D([cam_positions_world], colors=[(255, 0, 0)], radii=0.01))
        rr.log("world/trajectory/object", rr.LineStrips3D([obj_positions_world], colors=[(0, 255, 0)], radii=0.01))

        # --- Object-centric view ---
        # Object is at the origin in this space
        rr.log("object/obj_points", rr.Points3D(obj_pts_obj, colors=[(0, 255, 0)], radii=0.002))

        rr_log_camera(frame["image"], frame["K"],
                          T_co,
                          name=f"object/cam",
                          color=(255, 0, 0))
        
        rr.log("object/trajectory/camera", rr.LineStrips3D([cam_positions_obj], colors=[(255, 0, 0)], radii=0.01))
