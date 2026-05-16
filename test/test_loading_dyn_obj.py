import rerun 
from dataclasses import dataclass
import tyro 
from data import HOT3DDataLoader
from pathlib import Path
import rerun as rr 
import rerun.blueprint as rrb
from tqdm import tqdm 
import numpy as np

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    # data_seq: str = "clip-002793"
    data_seq: str = "clip-003312"


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





if __name__ == "__main__":
    cfg = tyro.cli(Config)
    data_loader = HOT3DDataLoader(Path(cfg.data_root) / cfg.data_seq)
    rr.init("dynamic_object_tracking")
    rr.connect_grpc("rerun+http://128.175.109.248:9876/proxy")

    # Define Rerun Blueprint for side-by-side visualization
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(
                name="World View",
                origin="world",
            ),
            rrb.Spatial3DView(
                name="Object-Centric View",
                origin="object",
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    points = data_loader.points3D
    mesh = data_loader.obj_mesh

    rr.log("world/static_background", rr.Points3D(
        points
    ), static=True)

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

        # --- World space view ---
        rr.log("world/obj_mesh", rr.Transform3D(
            translation=pose_wo[:3, 3], mat3x3=pose_wo[:3, :3]))
        rr.log("world/obj_mesh", rr.Mesh3D(vertex_positions=mesh.vertices,
                                    triangle_indices=mesh.faces,
                                    vertex_normals=mesh.vertex_normals),)
        
        rr_log_camera(frame["image"], frame["K"],
                          T_cw,
                          name=f"world/cam",
                          color=(255, 0, 0))
        
        rr.log("world/trajectory/camera", rr.LineStrips3D([cam_positions_world], colors=[(255, 0, 0)], radii=0.01))
        rr.log("world/trajectory/object", rr.LineStrips3D([obj_positions_world], colors=[(0, 255, 0)], radii=0.01))

        # --- Object-centric view ---
        # Object is at the origin in this space
        rr.log("object/obj_mesh", rr.Mesh3D(vertex_positions=mesh.vertices,
                                         triangle_indices=mesh.faces,
                                         vertex_normals=mesh.vertex_normals),)
        
        rr_log_camera(frame["image"], frame["K"],
                          T_co,
                          name=f"object/cam",
                          color=(255, 0, 0))
        
        rr.log("object/trajectory/camera", rr.LineStrips3D([cam_positions_obj], colors=[(255, 0, 0)], radii=0.01))

    





