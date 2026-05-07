import numpy as np
from PIL import Image
import json
from pathlib import Path
from tqdm import tqdm
import open3d as o3d
import trimesh
from evo.tools.file_interface import read_tum_trajectory_file 

class HOT3DDataLoader:
    def __init__(self, clip_path, depth_model=None):
        self.processed_data_dir = Path(clip_path)
        self.depth_model = depth_model

        self.num_frames = len(list(self.processed_data_dir.joinpath(
            "images").glob("*.png")))

        # self.points3D = np.load(
        #     self.processed_data_dir / "points" / "points.npy")
        pcd = o3d.io.read_point_cloud(
            str(self.processed_data_dir / "points3D.ply"))
        self.points3D = np.asarray(pcd.points)
        with open(self.processed_data_dir / "split" / "phase_frame_index.txt", "r") as file:
            phases = [tuple(map(int, line.strip().split(","))) for line in file]
        
        # self.static_phases = [phases[i] for i in range(len(phases)) if i % 2 == 0]
        self.dynamic_phases = [phases[i] for i in range(len(phases)) if i % 2 != 0]

        # for HOT3D, we have GT object mesh and pose'
        with open(self.processed_data_dir / "split" / "dynamic_object_name.txt", "r") as file:
            dyn_obj_name = file.read().strip()
        obj_model_dir = clip_path.parent.parent / "object_models"
        with open(obj_model_dir / "models_info.json", "r") as f:
            obj_model_info = json.load(f)
        tgt_obj_id = -1
        for obj_id in obj_model_info:
            if obj_model_info[obj_id]["name"] == dyn_obj_name:
                tgt_obj_id = int(obj_id)
                break
        assert tgt_obj_id != -1, "Dynamic object name not found in model info." 
        
        self.obj_mesh = trimesh.load_mesh(str(
            obj_model_dir / f"obj_{tgt_obj_id:06d}.glb"))
        self.obj_poses = read_tum_trajectory_file(
            str(self.processed_data_dir / "object_poses.txt")).poses_se3

    def get_obj_pose(self, idx):
        return self.obj_poses[idx]

    def is_dynamic(self, idx):
        for start, end in self.dynamic_phases:
            if idx >= start and idx <= end:
                return True
        return False
    
    def load_image(self, image_path):
        return np.array(Image.open(image_path).convert("RGB"))

    def load_mask(self, mask_path):
        return np.array(Image.open(mask_path).convert("L"))

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        frame_key = f"{idx:06d}"
        extrin = np.load(
            self.processed_data_dir / "extrinsics" / f"{frame_key}.npy"
        )
        K = np.load(
            self.processed_data_dir / "intrinsics" / f"{frame_key}.npy"
        )
        image = self.load_image(
            self.processed_data_dir / "images" / f"{frame_key}.png")
        mask_path = self.processed_data_dir / "model_infer" / f"mask_{idx:05d}.png"
        if not mask_path.exists():
            mask_path = self.processed_data_dir / "obj_masks" / f"{frame_key}.png"
        obj_mask = self.load_mask(mask_path)
        hand_mask = self.load_mask(
            self.processed_data_dir / "hand_masks" / f"{frame_key}.png")
        if self.processed_data_dir.joinpath("pred_twohands").exists():
            hand_arm_mask = self.load_mask(
                self.processed_data_dir / "pred_twohands" / f"{frame_key}.png")
            hand_arm_mask = hand_arm_mask > 0
            hand_mask = hand_mask | hand_arm_mask
        
        frame = {
            "image": image,
            "extrin": extrin,
            "K": K,
            "obj_mask": obj_mask,
            "hand_mask": hand_mask,
            "frame_id": idx
        }

        if self.depth_model == "GT":
            depth_path = self.processed_data_dir / "depth_dyn" / f"{frame_key}.npy"
            if depth_path.exists():
                frame["depth"] = np.load(depth_path)
            else:
                frame["depth"] = None
        elif self.depth_model is not None:
            depth_path = self.processed_data_dir / "depth_cache" / self.depth_model / f"{frame_key}.npy"
            if depth_path.exists():
                frame["depth"] = np.load(depth_path)
            else:
                frame["depth"] = None

        return frame