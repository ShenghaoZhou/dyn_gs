import torch
import rerun as rr
import numpy as np


def SH2RGB(sh):
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def vis_2dgs_rerun(gs, name, static=False, points_only=False):
    with torch.no_grad():
        if points_only:
            rr.log(
                name,
                rr.Points3D(
                    gs.means.detach().cpu().numpy(),
                    colors=gs.colors.detach().cpu().numpy(),
                ),
                static=static
            )
        else:
            quat_wxyz = gs.quats.detach().cpu().numpy()
            quat_xyzw = np.zeros_like(quat_wxyz)
            quat_xyzw[:, 0] = quat_wxyz[:, 3]
            quat_xyzw[:, 1] = quat_wxyz[:, 0]
            quat_xyzw[:, 2] = quat_wxyz[:, 1]
            quat_xyzw[:, 3] = quat_wxyz[:, 2]
            scales = gs.scales.detach().cpu().numpy()
            scales = np.concatenate(
                [scales, np.zeros((scales.shape[0], 1))], axis=1)
            rr.log(
                name,
                rr.Ellipsoids3D(
                    centers=gs.means.detach().cpu().numpy(),
                    half_sizes=scales,
                    colors=gs.colors.detach().cpu().numpy(),
                    quaternions=quat_xyzw,
                ),
                static=static
            )


def vis_3dgs_rerun(gs, name, static=False, points_only=False):
    with torch.no_grad():
        if points_only:
            rr.log(
                name,
                rr.Points3D(
                    gs.means.detach().cpu().numpy(),
                    colors=SH2RGB(gs.colors.detach().cpu().numpy()),
                ),
                static=static
            )
