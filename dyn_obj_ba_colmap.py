import cv2
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from data import HOT3DDataLoader
from pathlib import Path
import tyro
from dataclasses import dataclass, field
from tqdm import tqdm
import pycolmap
import trimesh
from typing import List, Dict
import sqlite3
import shutil

@dataclass
class Config:
    data_root: str = "data/hot3d_clips_processed"
    data_seq: str = "clip-001924"
    max_features: int = 1500
    min_features: int = 800
    max_frames: int = 60
    skip_frames: int = 2
    min_track_len: int = 10
    output_dir: str = "outputs/dynBA_colmap"
    rerun_url: str = "rerun+http://128.175.109.248:9876/proxy"

def get_t_c_o(frame, obj_pose):
    T_c_w = frame["extrin"]
    T_w_o = obj_pose
    T_c_o = T_c_w @ T_w_o
    return T_c_o

def main(cfg: Config):
    data_loader = HOT3DDataLoader(Path(cfg.data_root) / cfg.data_seq)
    rr.init("dyn_obj_ba_colmap")
    if cfg.rerun_url:
        rr.connect_grpc(cfg.rerun_url)

    # Define and send blueprint
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="World View", origin="world"),
            rrb.Vertical(
                rrb.Spatial3DView(name="Object Centric", origin="object"),
                rrb.Spatial2DView(name="Camera View", origin="world/camera/view"),
            ),
        ),
        collapse_panels=True,
    )
    rr.send_blueprint(blueprint)

    start_idx = 0
    if len(data_loader.dynamic_phases) > 0:
        start_idx, _ = data_loader.dynamic_phases[0]
    
    end_idx = start_idx + cfg.max_frames * cfg.skip_frames
    indices = list(range(start_idx, min(end_idx, len(data_loader)), cfg.skip_frames))

    # Output directories
    out_dir = Path(cfg.output_dir) / cfg.data_seq
    out_dir.mkdir(parents=True, exist_ok=True)
    img_tmp_dir = out_dir / "images"
    if img_tmp_dir.exists():
        shutil.rmtree(img_tmp_dir)
    img_tmp_dir.mkdir(parents=True)
    db_path = out_dir / "database.db"
    if db_path.exists():
        db_path.unlink()

    # KLT Tracking logic from test_dyn_obj_ba.py
    orb = cv2.ORB_create(nfeatures=cfg.max_features)
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    prev_gray, prev_pts, prev_ids = None, None, []
    next_id = 0
    tracks = {}
    frame_data = {}

    print(f"Tracking over {len(indices)} frames...")
    for idx_local, idx in enumerate(tqdm(indices)):
        f = data_loader[idx]
        gray = cv2.cvtColor(f["image"], cv2.COLOR_RGB2GRAY)
        mask = f["obj_mask"]
        
        # Save image for COLMAP
        img_name = f"image_{idx_local:06d}.png"
        # Mask out everything outside the object for cleaner SfM if desired.
        # But KLT tracks are already on the object, so we can use full image or masked.
        # Let's use masked image to be safe.
        masked_img = f["image"].copy()
        masked_img[mask == 0] = 0
        cv2.imwrite(str(img_tmp_dir / img_name), cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))

        frame_data[idx] = {
            "T_c_o_gt": get_t_c_o(f, data_loader.get_obj_pose(idx)),
            "K": f["K"],
            "image": f["image"],
            "T_c_w": f["extrin"],
            "img_name": img_name
        }
        
        current_pts, current_ids = None, []
        if prev_pts is not None and len(prev_pts) > 0:
            new_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, **lk_params)
            status = status.flatten().astype(bool)
            valid = []
            for i, p in enumerate(new_pts):
                if status[i]:
                    x, y = int(round(p[0][0])), int(round(p[0][1]))
                    if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0] and mask[y, x] > 0:
                        valid.append(True)
                    else: valid.append(False)
                else: valid.append(False)
            valid = np.array(valid)
            if np.any(valid):
                current_pts = new_pts[valid]
                current_ids = [prev_ids[i] for i in range(len(valid)) if valid[i]]
                for pt, tid in zip(current_pts, current_ids):
                    tracks[tid][idx] = pt.flatten()

        if current_pts is None or len(current_pts) < cfg.min_features:
            det_mask = mask.copy()
            if current_pts is not None:
                for p in current_pts: cv2.circle(det_mask, (int(round(p[0][0])), int(round(p[0][1]))), 5, 0, -1)
            kps = orb.detect(gray, mask=det_mask)
            if kps:
                new_pts_arr = np.array([kp.pt for kp in kps], dtype=np.float32).reshape(-1, 1, 2)
                new_ids = list(range(next_id, next_id + len(kps)))
                next_id += len(kps)
                for pt, tid in zip(new_pts_arr, new_ids):
                    tracks[tid] = {idx: pt.flatten()}
                if current_pts is not None:
                    current_pts = np.vstack([current_pts, new_pts_arr])
                    current_ids.extend(new_ids)
                else:
                    current_pts, current_ids = new_pts_arr, new_ids
        
        prev_gray, prev_pts, prev_ids = gray.copy(), current_pts, current_ids

    # Filter tracks
    valid_tracks = {tid: t for tid, t in tracks.items() if len(t) >= cfg.min_track_len}
    print(f"Total tracks: {len(tracks)}, Valid tracks: {len(valid_tracks)}")

    # Populate COLMAP database
    image_name_to_id = {}
    db = pycolmap.Database.open(str(db_path))
    
    # Camera
    K = frame_data[indices[0]]["K"]
    img_shape = frame_data[indices[0]]["image"].shape
    camera = pycolmap.Camera(
        model="PINHOLE",
        width=img_shape[1],
        height=img_shape[0],
        params=[K[0,0], K[1,1], K[0,2], K[1,2]]
    )
    cam_id = db.write_camera(camera)
    
    # Images and Keypoints
    image_ids = {} # idx -> image_id
    tid_to_local_idx = {} # idx -> {tid: local_idx}
    
    print("Writing images and keypoints to database...")
    for idx_local, idx in enumerate(tqdm(indices)):
        img_name = frame_data[idx]["img_name"]
        image = pycolmap.Image(name=img_name, camera_id=cam_id)
        image_id = db.write_image(image)
        image_ids[idx] = image_id
        image_name_to_id[img_name] = image_id
        
        # Get all valid tracks in this image
        frame_tids = sorted([tid for tid in valid_tracks if idx in valid_tracks[tid]])
        tid_to_local_idx[idx] = {tid: i for i, tid in enumerate(frame_tids)}
        
        kpts = np.array([valid_tracks[tid][idx] for tid in frame_tids], dtype=np.float32)
        # Add scale and orientation placeholders
        kpts_full = np.zeros((len(kpts), 4), dtype=np.float32)
        kpts_full[:, :2] = kpts
        kpts_full[:, 2] = 1.0 # scale
        db.write_keypoints(image_id, kpts_full)

    # Matches
    print("Writing matches to database...")
    pairs = []
    for i in range(len(indices)):
        for j in range(i + 1, min(i + 5, len(indices))): # Sequential + short skip
            idx1, idx2 = indices[i], indices[j]
            common_tids = sorted(list(set(tid_to_local_idx[idx1].keys()) & set(tid_to_local_idx[idx2].keys())))
            
            if len(common_tids) < 15:
                continue
            
            matches = np.array([[tid_to_local_idx[idx1][tid], tid_to_local_idx[idx2][tid]] for tid in common_tids], dtype=np.uint32)
            db.write_matches(image_ids[idx1], image_ids[idx2], matches)
            pairs.append((frame_data[idx1]["img_name"], frame_data[idx2]["img_name"]))

    db.close()

    # Geometric Verification
    print("Verifying matches...")
    pairs_path = out_dir / "pairs.txt"
    with open(pairs_path, "w") as f:
        for p1, p2 in pairs:
            f.write(f"{p1} {p2}\n")
            
    pycolmap.verify_matches(str(db_path), str(pairs_path))

    # Incremental Mapping
    print("Starting incremental mapping...")
    colmap_out = out_dir / "reconstruction"
    if colmap_out.exists():
        shutil.rmtree(colmap_out)
    colmap_out.mkdir(exist_ok=True)
    
    # Run mapping
    reconstructions = pycolmap.incremental_mapping(str(db_path), str(img_tmp_dir), str(colmap_out))
    
    if not reconstructions:
        print("COLMAP failed to reconstruct any model.")
        return

    # Select largest model
    rec = sorted(reconstructions.values(), key=lambda x: x.num_points3D(), reverse=True)[0]

    print(f"Reconstruction complete: {rec.num_reg_images()} images, {rec.num_points3D()} points.")

    # Visualization
    print("Logging to Rerun...")
    # Common world setup
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    
    obj_mesh = data_loader.obj_mesh
    if isinstance(obj_mesh, trimesh.Scene):
        obj_mesh = obj_mesh.dump(concatenate=True)
    
    rr.log("object/mesh", rr.Mesh3D(
        vertex_positions=obj_mesh.vertices,
        triangle_indices=obj_mesh.faces,
        vertex_colors=obj_mesh.visual.vertex_colors if hasattr(obj_mesh.visual, 'vertex_colors') else None,
        vertex_normals=obj_mesh.vertex_normals
    ), static=True)

    # Reconstructed points
    xyzs = []
    rgbs = []
    for pid, point in rec.points3D.items():
        xyzs.append(point.xyz)
        rgbs.append(point.color)
    
    if xyzs:
        rr.log("object/points_reconstructed", rr.Points3D(xyzs, colors=rgbs, radii=0.003), static=True)

    # Log Camera and GT Object poses
    for idx_local, idx in enumerate(indices):
        rr.set_time("frame_idx", sequence=idx)
        
        # Camera in world
        T_c_w = frame_data[idx]["T_c_w"]
        T_w_c = np.linalg.inv(T_c_w)
        rr.log("world/camera", rr.Transform3D(translation=T_w_c[:3, 3], mat3x3=T_w_c[:3, :3], relation=rr.TransformRelation.ParentFromChild))
        
        # GT Object in world
        T_w_o_gt = data_loader.get_obj_pose(idx)
        rr.log("world/object_gt", rr.Transform3D(translation=T_w_o_gt[:3, 3], mat3x3=T_w_o_gt[:3, :3], relation=rr.TransformRelation.ParentFromChild))
        
        # Camera view
        K = frame_data[idx]["K"]
        img = frame_data[idx]["image"]
        rr.log("world/camera/view", rr.Pinhole(resolution=(img.shape[1], img.shape[0]), image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.1))
        rr.log("world/camera/view", rr.Image(img).compress(jpeg_quality=50))

        # Reconstructed pose for this image
        img_name = frame_data[idx]["img_name"]
        # Find image in reconstruction
        colmap_img = None
        for r_img in rec.images.values():
            if r_img.name == img_name:
                colmap_img = r_img
                break
        
        if colmap_img is not None:
            # T_c_o_rec (where c is camera at frame idx, o is COLMAP object frame)
            # Note: in COLMAP, images are cam_from_world. 
            # Here "world" in COLMAP means our object-centric frame.
            T_c_o_rec = np.eye(4)
            T_c_o_rec[:3, :] = colmap_img.cam_from_world().matrix()
            
            # Log object centric camera
            rr.log("object/camera_rec", rr.Transform3D(translation=T_c_o_rec[:3, 3], mat3x3=T_c_o_rec[:3, :3], relation=rr.TransformRelation.ChildFromParent))
            rr.log("object/camera_rec/frustum", rr.Pinhole(resolution=(img.shape[1], img.shape[0]), image_from_camera=K, camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.05))

            # Transform COLMAP world to actual world hypothesis
            # T_w_o_rec = T_w_c @ T_c_o_rec^-1
            T_w_o_rec = T_w_c @ np.linalg.inv(T_c_o_rec)
            rr.log("world/object_reconstructed", rr.Transform3D(translation=T_w_o_rec[:3, 3], mat3x3=T_w_o_rec[:3, :3], relation=rr.TransformRelation.ParentFromChild))
            
            # Log reconstructed points in world via current object pose
            rr.log("world/object_reconstructed/points", rr.Points3D(xyzs, colors=rgbs, radii=0.003))

    print(f"Results saved to {out_dir}")

if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
