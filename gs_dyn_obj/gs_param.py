import dataclasses
import torch
from pathlib import Path
from .gs_rendering import render_2dgs, render_2dgs_full, render_2dgs_visiblity


@dataclasses.dataclass
class GSParam:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    colors: torch.Tensor
    opacity: torch.Tensor

    def __add__(self, other):
        return GSParam(
            torch.cat((self.means, other.means), dim=0),
            torch.cat((self.quats, other.quats), dim=0),
            torch.cat((self.scales, other.scales), dim=0),
            torch.cat((self.colors, other.colors), dim=0),
            torch.cat((self.opacity, other.opacity), dim=0)
        )

    def __getitem__(self, key):
        return GSParam(
            self.means[key],
            self.quats[key],
            self.scales[key],
            self.colors[key],
            self.opacity[key]
        )

    def render(self, viewmat, K, width, height, mode: str = "normal", near_plane: float = 0.01,
               far_plane: float = 100.0, scaling_modifier: float = 1.0,
               bg=torch.zeros(3)):
        if mode == "normal":
            return render_2dgs(
                self.means, self.quats, self.scales, self.colors, self.opacity,
                viewmat, K, width, height, near_plane, far_plane,
                scaling_modifier, bg
            )
        elif mode == "full":
            return render_2dgs_full(
                self.means, self.quats, self.scales, self.colors, self.opacity,
                viewmat, K, width, height, near_plane, far_plane,
                scaling_modifier, bg
            )
        elif mode == "visibility":
            return render_2dgs_visiblity(
                self.means, self.quats, self.scales, self.colors, self.opacity,
                viewmat, K, width, height, near_plane, far_plane,
                scaling_modifier, bg
            )

    def dump(self, path: Path):
        torch.save({
            'means': self.means,
            'quats': self.quats,
            'scales': self.scales,
            'colors': self.colors,
            'opacity': self.opacity
        }, path)

    def load(path: Path):
        data = torch.load(path)
        return GSParam(
            data['means'],
            data['quats'],
            data['scales'],
            data['colors'],
            data['opacity']
        )

    def clone(self):
        return GSParam(
            self.means.clone(),
            self.quats.clone(),
            self.scales.clone(),
            self.colors.clone(),
            self.opacity.clone()
        )
