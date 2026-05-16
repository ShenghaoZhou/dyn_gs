import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class Anchor:
    """
    Represents an anchor that holds Gaussian parameters and associated keyframes.
    """

    def __init__(
        self,
        gaussian_params: Dict[str, Dict[str, torch.Tensor]],
        position: torch.Tensor = None,
    ):
        self.gaussian_params = gaussian_params
        if position is None:
            position = torch.zeros(3, dtype=torch.float32)
        self.position = position
        self.keyframe_ids = []

    def add_keyframe_id(self, keyframe_id: int):
        self.keyframe_ids.append(keyframe_id)

    @property
    def device(self):
        return self.gaussian_params["xyz"]["val"].device

    def to(self, device: str):
        if self.device != device:
            for param in self.gaussian_params.values():
                for key, tensor in param.items():
                    if isinstance(tensor, torch.Tensor):
                        param[key] = tensor.to(device)
        return self

    @classmethod
    def blend(
        cls, cam_centre: torch.Tensor, anchors: List["Anchor"], anchor_overlap: float
    ) -> Tuple[Dict[str, Dict[str, torch.Tensor]], np.ndarray, torch.Tensor]:
        """
        Blend the Gaussian parameters of the closest anchors based on their distance to the camera centre.
        """
        anchor_weights = np.zeros(len(anchors))
        anchor_positions = torch.stack(
            [anchor.position for anchor in anchors], dim=0
        )
        anchor_dists = torch.linalg.vector_norm(
            anchor_positions - cam_centre[None], dim=-1
        )
        
        # Get the two closest anchors
        k = min(2, len(anchors))
        closest_anchors_dist, closest_anchors_ids = torch.topk(
            anchor_dists, k, largest=False
        )

        # Move relevant anchors to GPU
        for i, anchor in enumerate(anchors):
            if i in closest_anchors_ids:
                anchor.to("cuda")
            else:
                pass

        if k == 1:
            gaussian_params = anchors[closest_anchors_ids[0]].gaussian_params
            anchor_weights[closest_anchors_ids[0]] = 1.0
            return gaussian_params, anchor_weights, closest_anchors_ids

        ratio = closest_anchors_dist[0] / (closest_anchors_dist[1] + 1e-8)

        if ratio < (1 - anchor_overlap):
            gaussian_params = anchors[closest_anchors_ids[0]].gaussian_params
            anchor_weights[closest_anchors_ids[0]] = 1.0
            return gaussian_params, anchor_weights, closest_anchors_ids[:1]
        else:
            # Blend the opacities of the two closest anchors
            blending_weights = 1.0 - (ratio - (1.0 - anchor_overlap)) * (0.5 / anchor_overlap)
            blending_weights = torch.clamp(blending_weights, 0.5, 1.0)
            
            params1 = anchors[closest_anchors_ids[0]].gaussian_params
            params2 = anchors[closest_anchors_ids[1]].gaussian_params
            
            gaussian_params = {
                name: {"val": torch.cat([params1[name]["val"], params2[name]["val"]], dim=0)}
                for name in params1
                if name != "opacity"
            }
            
            # Blend opacities using sigmoid/inverse_sigmoid for proper range
            op1 = torch.sigmoid(params1["opacity"]["val"])
            op2 = torch.sigmoid(params2["opacity"]["val"])
            
            blended_opacity = torch.cat([
                inverse_sigmoid(op1 * blending_weights),
                inverse_sigmoid(op2 * (1.0 - blending_weights))
            ], dim=0)
            
            gaussian_params["opacity"] = {"val": blended_opacity}

            anchor_weights[closest_anchors_ids[0]] = blending_weights.item()
            anchor_weights[closest_anchors_ids[1]] = 1.0 - blending_weights.item()

        return gaussian_params, anchor_weights, closest_anchors_ids
