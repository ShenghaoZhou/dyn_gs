import dataclasses
import torch
from pathlib import Path
from .gs_rendering import render_2dgs, render_2dgs_full, render_2dgs_visiblity, render_3dgs


@dataclasses.dataclass
class GSParam:
    means: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    colors: torch.Tensor
    opacity: torch.Tensor
    ray_o: torch.Tensor = None
    ray_d: torch.Tensor = None
    ray_dist: torch.Tensor = None

    def __add__(self, other):
        return GSParam(
            torch.cat((self.means, other.means), dim=0),
            torch.cat((self.quats, other.quats), dim=0),
            torch.cat((self.scales, other.scales), dim=0),
            torch.cat((self.colors, other.colors), dim=0),
            torch.cat((self.opacity, other.opacity), dim=0),
            torch.cat((self.ray_o, other.ray_o), dim=0) if self.ray_o is not None and other.ray_o is not None else None,
            torch.cat((self.ray_d, other.ray_d), dim=0) if self.ray_d is not None and other.ray_d is not None else None,
            torch.cat((self.ray_dist, other.ray_dist), dim=0) if self.ray_dist is not None and other.ray_dist is not None else None,
        )

    def __getitem__(self, key):
        return GSParam(
            self.means[key],
            self.quats[key],
            self.scales[key],
            self.colors[key],
            self.opacity[key],
            self.ray_o[key] if self.ray_o is not None else None,
            self.ray_d[key] if self.ray_d is not None else None,
            self.ray_dist[key] if self.ray_dist is not None else None,
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
        elif mode == "3dgs":
            return render_3dgs(
                self.means, self.quats, self.scales, self.colors, self.opacity,
                viewmat, K, width, height, near_plane, far_plane,
                scaling_modifier, bg
            )

    def dump(self, path: Path):
        data = {
            'means': self.means,
            'quats': self.quats,
            'scales': self.scales,
            'colors': self.colors,
            'opacity': self.opacity
        }
        if self.ray_o is not None: data['ray_o'] = self.ray_o
        if self.ray_d is not None: data['ray_d'] = self.ray_d
        if self.ray_dist is not None: data['ray_dist'] = self.ray_dist
        torch.save(data, path)

    def load(path: Path):
        data = torch.load(path)
        return GSParam(
            data['means'],
            data['quats'],
            data['scales'],
            data['colors'],
            data['opacity'],
            data.get('ray_o'),
            data.get('ray_d'),
            data.get('ray_dist')
        )

    def clone(self):
        return GSParam(
            self.means.clone(),
            self.quats.clone(),
            self.scales.clone(),
            self.colors.clone(),
            self.opacity.clone(),
            self.ray_o.clone() if self.ray_o is not None else None,
            self.ray_d.clone() if self.ray_d is not None else None,
            self.ray_dist.clone() if self.ray_dist is not None else None,
        )

    def to(self, device):
        self.means = self.means.to(device)
        self.quats = self.quats.to(device)
        self.scales = self.scales.to(device)
        self.colors = self.colors.to(device)
        self.opacity = self.opacity.to(device)
        if self.ray_o is not None: self.ray_o = self.ray_o.to(device)
        if self.ray_d is not None: self.ray_d = self.ray_d.to(device)
        if self.ray_dist is not None: self.ray_dist = self.ray_dist.to(device)
        return self
