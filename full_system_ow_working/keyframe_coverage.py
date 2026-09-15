"""
Coverage-driven keyframe selection for the object Gaussian mapper.

Why this exists
---------------
The mapper (obj_gs_mapping.GSMapping) previously admitted keyframes on a fixed
cadence -- ``frame_count % kf_every == 0``. That cadence is blind to the view:
when the object moves under the camera, consecutive frames can be visually
near-identical (no new surface to fit) or radically different (a new side of the
object just became visible). A fixed 5-frame step therefore either duplicates a
view the window already has, or lets the most information-rich frames fall
between two keyframes. Multi-view optimization then runs on a window that is
either redundant or already a frame behind, and the shape the mapper converges
to is a compromise of whichever views happened to land on a multiple of 5.

This module replaces the cadence with a measurement taken from the mask alone:
the silhouette's centroid, orientation and shape. Those are the observables of
the view that actually carry surface information, and they are computed from
the GT mask the system already trusts -- not from the tracked pose. That last
point is the whole reason for the module: the pose estimate is precisely the
thing that degrades when the object moves, so a selection criterion built on it
inherits that degradation. A mask-only criterion cannot be fooled by it.

What it does NOT do
-------------------
Silhouette moments are 2D and are blind to out-of-plane rotation of a shape-
symmetric object: a box turned 180 degrees about the image vertical axis has the
same centroid, orientation and eccentricity as before, and reads as redundant.
That limit is inherent to 2D moments, not a coding choice. A 3D coverage metric
(camera-to-object position direction, which is exact) is available as soon as the
pose is trustworthy -- i.e. after the trajectory parameterization makes it so --
and would be a strict upgrade here. It is deliberately not used now, because
"exact given a good pose" is the wrong property while the pose is the variable.

Pure numpy: no torch, no dataset, no GPU. The mapper calls ``update`` once per
frame (not once per optimization step), so the cost is one ``np.argwhere`` on a
640x480 boolean mask -- negligible against 300 render-and-backward steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ViewDescriptor:
    """A view's silhouette summary.

    centroid      2-vector, the mask's centre relative to the image centre, in
                  normalised image coordinates (x right, y down). One unit is
                  one image width / one image height respectively, so the
                  distance between two centroids is directly comparable across
                  frames and resolutions.
    angle_deg     orientation of the principal axis, in (-90, 90]. A silhouette
                  is a line, not an arrow, so the axis is defined up to a 180
                  degree flip and this range covers each axis exactly once.
                  It is NOT mod 90: a needle at -80 degrees and one at +10 are
                  perpendicular, hence the most different views possible, and a
                  mod 90 would fold them together. See _angle_diff_deg.
    area_frac     mask area over image area, in (0, 1].
    ecc           eccentricity of the second moment, in [0, 1): 0 is round,
                  close to 1 is needle-shaped. Shape is nearly scale-free, so
                  it separates "the same view of a smaller object" from "a
                  different view".
    """

    centroid: np.ndarray
    angle_deg: float
    area_frac: float
    ecc: float


def silhouette_descriptor(mask, width: int, height: int) -> Optional[ViewDescriptor]:
    """Silhouette moments of ``mask``.

    ``mask`` is any non-empty 2-D array whose nonzero pixels are the object
    (uint8 labels, boolean, floats -- ``> 0`` is applied). Returns None when the
    mask has no pixels: there is nothing to key on, and the caller should treat
    that frame as not a keyframe rather than crash.

    Centroids are taken relative to the image centre because that is what makes
    them comparable between frames -- a mask that stays put but the camera
    drifting is exactly the case we want to register as movement.
    """
    if mask is None:
        return None
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")

    iy, ix = np.nonzero(mask)
    n = iy.size
    if n == 0:
        return None

    W, H = float(width), float(height)
    x = (ix - W * 0.5) / W
    y = (iy - H * 0.5) / H

    # Centre by the SAMPLE mean before taking moments, so the covariance is of
    # the shape and not of the position -- the position is reported separately
    # as ``centroid``. Centring by the image centre instead would leave the mean
    # offset in the second moment: a uniform mask's sample mean sits a half
    # pixel off the image centre, so sxy would pick up n*cx*cy and the principal
    # axis of an isotropic mask would read as arbitrary rather than 0.
    xc = x - x.mean()
    yc = y - y.mean()
    sxx = float((xc * xc).sum())
    syy = float((yc * yc).sum())
    sxy = float((xc * yc).sum())

    tr = sxx + syy
    disc = float(np.sqrt(max(tr * tr - 4.0 * (sxx * syy - sxy * sxy), 0.0)))

    # 0.5*atan2(2*sxy, sxx-syy) is the principal axis, folded into (-90, 90].
    # The 0.5 does the folding: atan2 is mod 360, so the axis, which is mod 180,
    # lands in a 180-degree range. Do not mod it again -- see ViewDescriptor.
    # When the axis is undefined (sxx == syy and sxy == 0) atan2 returns 0 by
    # IEEE 754, which is a deterministic rather than arbitrary answer; there is
    # no divide-by-zero here to guard against.
    angle_deg = float(np.degrees(0.5 * np.arctan2(2.0 * sxy, sxx - syy)))

    lam1 = 0.5 * (tr + disc)
    ecc = float(np.sqrt(max(1.0 - (0.5 * (tr - disc)) / max(lam1, 1e-12), 0.0))) if lam1 > 1e-12 else 0.0

    return ViewDescriptor(
        centroid=np.array([float(x.mean()), float(y.mean())], dtype=np.float64),
        angle_deg=angle_deg,
        area_frac=float(n / (width * height)),
        ecc=ecc,
    )


def _angle_diff_deg(a: float, b: float) -> float:
    """Shortest difference between two silhouette orientations, in [0, 90].

    The axis is a line, so a and a+180 are the same view and the largest
    difference is 90 degrees (perpendicular). Folding to 180 rather than to 90
    is the whole point: 10 and -80 are perpendicular, and a mod 90 would report
    them as 10 apart.
    """
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def view_distance(a: ViewDescriptor, b: ViewDescriptor,
                  weight_ang: float = 1.0,
                  weight_cent: float = 1.0,
                  weight_shape: float = 1.0,
                  eps: float = 1e-9) -> float:
    """Weighted distance between two views.

    Orientation is scaled by 90 so it lives on the same 0..1 range as "a
    silhouette moved a whole image width"; eccentricity by 1 since it is already
    a unit quantity; area by a log ratio so a 2x and a 5x change do not
    swamp the other two terms by being unbounded on one side.
    """
    d_ang = _angle_diff_deg(a.angle_deg, b.angle_deg) / 90.0
    d_cent = float(np.linalg.norm(a.centroid - b.centroid))
    d_shape = abs(np.log2(max(a.ecc, eps) / max(b.ecc, eps))) if min(a.ecc, b.ecc) > eps else 0.0
    d_area = abs(np.log2(max(a.area_frac, eps) / max(b.area_frac, eps)))
    # Area and eccentricity both describe shape-at-scale; fold them together so
    # they contribute one term at weight_shape rather than two competing ones.
    return weight_ang * d_ang + weight_cent * d_cent + weight_shape * (0.5 * d_area + 0.5 * d_shape)


class KeyframeCoverage:
    """A sliding window of view descriptors, deciding which views earn a keyframe.

    ``update`` returns True when the new view is not covered by the window, and
    in that case the descriptor is admitted. So a caller needs no bookkeeping of
    its own: the return value is its keyframe decision.
    """

    def __init__(self, max_window: int = 20, min_interval: int = 3,
                 angle_deg: float = 15.0,
                 centroid_shift: float = 0.25,
                 shape_ratio: float = 0.5,
                 weight_ang: float = 1.0,
                 weight_cent: float = 1.0,
                 weight_shape: float = 1.0):
        self.max_window = max_window
        self.min_interval = max(1, int(min_interval))
        self.angle_deg = float(angle_deg)
        self.centroid_shift = float(centroid_shift)
        self.shape_ratio = float(shape_ratio)
        self.weight_ang = float(weight_ang)
        self.weight_cent = float(weight_cent)
        self.weight_shape = float(weight_shape)
        self._descs: List[ViewDescriptor] = []
        self._last_kf_frame: Optional[int] = None

    def __len__(self) -> int:
        return len(self._descs)

    def reset(self) -> None:
        """Empty the window. GSMapping.run() re-initialises its state before
        each mapping session, and the interval anchor must go with it or the
        first frame of a second session would be refused as too close to the
        last keyframe of the previous one."""
        self._descs = []
        self._last_kf_frame = None

    def descriptors(self) -> List[ViewDescriptor]:
        return list(self._descs)

    def update(self, mask, width: int, height: int, frame_idx: int) -> bool:
        """Admit this frame as a keyframe? Empty mask and interval violation say no."""
        desc = silhouette_descriptor(mask, width, height)
        if desc is None:
            return False
        if self._last_kf_frame is not None and frame_idx - self._last_kf_frame < self.min_interval:
            return False
        if self._redundant(desc):
            return False
        self._descs.append(desc)
        if len(self._descs) > self.max_window:
            self._descs.pop(0)
        self._last_kf_frame = frame_idx
        return True

    def _redundant(self, desc: ViewDescriptor) -> bool:
        """True if the window already contains a view this close.

        Three separate tests rather than one fused score, so each threshold has a
        meaning an operator can reason about: orientation, silhouette position,
        and shape. Any one being close is enough to call the view redundant --
        the mapper wants a window that spans the object, and a window that
        already has this angle at this position has nothing new to say about it.
        """
        for d in self._descs:
            if _angle_diff_deg(d.angle_deg, desc.angle_deg) < self.angle_deg:
                if float(np.linalg.norm(d.centroid - desc.centroid)) < self.centroid_shift:
                    return True
                if self.shape_ratio > 0 and max(d.area_frac, desc.area_frac) / max(
                        min(d.area_frac, desc.area_frac), 1e-9) < 1.0 + self.shape_ratio:
                    return True
        return False

    def most_diverse_indices(self, desc: ViewDescriptor, limit: Optional[int] = None,
                             max_dis: Optional[float] = None) -> List[Tuple[int, float]]:
        """Window indices furthest from ``desc``, in decreasing order of distance.

        Indices rather than descriptors because the caller (the mapper) holds its
        own ``self.keyframes`` list position-aligned with this window: both are
        appended together and both are capped at window_size with a left pop, so
        descriptor i and keyframe i are the same view. ``max_dis`` bounds how far
        a neighbour may be -- the geometric-consistency loss compares depths,
        which diverge with baseline, so the most informative view is not always
        the most reliable one.
        """
        out: List[Tuple[int, float]] = []
        for i, d in enumerate(self._descs):
            dis = view_distance(d, desc, self.weight_ang, self.weight_cent, self.weight_shape)
            if max_dis is not None and dis > max_dis:
                continue
            out.append((i, dis))
        out.sort(key=lambda t: t[1], reverse=True)
        return out if limit is None else out[:limit]

    def most_diverse(self, desc: ViewDescriptor, max_dis: Optional[float] = None,
                     limit: Optional[int] = None) -> List[ViewDescriptor]:
        """Descriptor-level wrapper over most_diverse_indices, for callers that
        do not hold an aligned list of their own."""
        return [self._descs[i] for i, _ in self.most_diverse_indices(desc, limit, max_dis)]
