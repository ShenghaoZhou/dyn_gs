"""
Bundle adjustment for the geometric tracker.

Two layers, both pyceres factor graphs over C++ cost functions:

1. :func:`bundle_adjust` — reprojection BA over a *bounded sliding window* of
   keyframes, with a gauge pin, robust loss, an acceptance test and logging.
   The forward pass (photometric LM, then PnP) is treated as a *measurement*,
   not as the answer: its estimate enters only as an optional covariance-weighted
   pose prior, and the result is written back only if it strictly improves.

2. :func:`align_to_object_frame` — ties the tracker's keyframe frame to the
   mapper's object frame through ``Point3DAlignmentCost`` with a log-parameterised
   scale block, replacing the silent ``np.clip(s, 0.1, 10.0)`` absorption that
   ``ObjGSCameraMapper.densify`` used to perform on every keyframe depth buffer.

Empirical constraints verified against pyceres 2.6.0 / pycolmap 3.13.0 before this
file was written. Every one of them is a footgun that a "reasonable-looking" rewrite
would step on:

* ``pyceres.CostFunction`` cannot be subclassed from Python. Overriding ``evaluate``
  still resolves to the C++ pure virtual ``Evaluate`` and raises
  ``RuntimeError: Tried to call pure virtual function "Ceres::CostFunction::Evaluate"``
  at solve time. Consequently NO photometric / GS-render residual can live in this
  graph; the torch LM stays a separate solve whose output is fused here as a prior.
* ``ReprojErrorCost`` block order is ``(q[4], t[3], pt3d[3], cam_params[4])``. The
  model string ``'PINHOLE'`` selects the 4-parameter ``(fx, fy, cx, cy)`` camera;
  passing a bare int id instead selects a 3-parameter camera and silently changes
  the layout to ``[4, 3, 3, 3]``.
* ``point2D`` must be a length-2 1-D array (or a Python list). A ``[2, 1]`` column
  vector is rejected with a ``TypeError``.
* ``position_in_world_prior`` must be a length-3 1-D array; ``[1, 3]`` is rejected.
* The covariance overloads whiten the residual by ``Sigma**-0.5``: a position prior
  with ``diag(0.01, 0.01, 1.0)`` turned ``[1, 2, 13]`` into ``[10, 20, 13]``, i.e. the
  cost is a proper Mahalanobis chi-square. A covariance estimate from the LM's
  ``JTJ**-1`` (computed at ``obj_gs.py:632-654``) therefore drops straight in.
* The 6-DoF pose priors lay out the residual as ``[rotation(3), translation(3)]``,
  so a 6x6 covariance must be diagonalised in that same order — rotation first.
* One ``Point3DAlignmentCost`` residual over *free* source points is a pure gauge:
  the solver drives the cost to ~1e-24 with an arbitrary R, t and s. The source
  points must already be constrained by reprojection terms in the same graph.
* Keep the ``EigenQuaternionManifold`` instance alive for the solver's whole
  lifetime. A temporary constructed inline at the ``set_manifold`` call site gets
  garbage-collected before ``solve`` runs and the graph then dereferences a dead
  manifold. Hold it in a local for the duration of the function.
* ``ReprojErrorCost`` ZEROES the residual of any point at ``z <= 0`` in the camera
  frame (a cheirality guard, verified numerically: a finite behind-camera
  projection returns exactly ``[0, 0]``). Such observations silently drop out of
  the cost AND the gradient instead of pulling the pose back -- one more reason
  the acceptance test and the shift guard below are not optional.
* Monocular reprojection BA can NEVER observe scale. Pinning one keyframe fixes
  only 6 of the 7 similarity-gauge dofs; the remaining orbit through the pinned
  camera is ``t_i' = s*t_i - (s-1)*R_i*R_0^T*t_0``, ``p' = s*p + (s-1)*R_0^T*t_0``
  and leaves every reprojection exactly invariant, no matter where the pinned
  camera sits (verified numerically: zero final cost, rotations exact to 1e-6,
  translation error a perfectly linear ramp along the trajectory, invariant to
  the iteration budget). The tracker seeds metric scale from depth, but a
  reprojection-only BA is free to random-walk along this orbit at zero cost --
  bounded per commit only by the shift guard. ``use_pose_priors`` (OFF by
  default) is the metric leash that closes the orbit; ``anchor_pose_prior``
  closes everything except scale. This is also why a naive "BA always helps"
  assumption is dangerous: gauge drift shows up as a *successful* solve.

Block-conversion convention, used throughout: quaternion parameter blocks are
``xyzw`` (Eigen memory order, scalar LAST). Verified numerically against both
``ReprojErrorCost`` and ``Point3DAlignmentCost``, and consistent with
``pycolmap.Rotation3d().quat`` (which returns ``[0,0,0,1]`` for identity) and
with ``EigenQuaternionManifold``. This is exactly scipy's ``Rotation.as_quat()``
layout, so NO reordering is needed anywhere in this file. (COLMAP's *C++ docs*
say wxyz; the legacy ``geometric_tracker.run_ba`` therefore reorders to wxyz --
that is a latent bug that would corrupt poses, harmless only because that
function has no callers. Do not copy it.)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pyceres
import pycolmap
import pycolmap.cost_functions as costs
from scipy.spatial.transform import Rotation as R


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

@dataclass
class BaConfig:
    """Knobs for :func:`bundle_adjust`.

    The defaults are deliberately *additive*: they add a bounded window, an
    acceptance test, a shift guard and logging to the legacy behaviour, and change
    nothing else. Every richer feature is behind a flag so the paired 5-clip A/B in
    ``docs/bundlesdf-gate-ab-findings.md`` stays comparable when this lands.
    """

    # window
    max_keyframes: int = 20            # keyframes (not frames) kept in one graph
    fix_first: bool = True             # hard gauge pin on the oldest keyframe in graph

    # solver
    max_num_iterations: int = 20
    huber_delta: float = 1.0           # pixels, applied to the (optionally whitened) residual
    linear_solver: str = "dense_schur"

    # acceptance
    require_cost_decrease: bool = True
    min_relative_decrease: float = 1e-6
    max_frame_shift: float = 0.05      # m; reject the whole BA if any frame moved further
    max_frame_rot_deg: float = 5.72    # deg, ~= 0.1 rad
    min_points_per_frame: int = 3      # keyframes with fewer in-window observations are excluded
    max_tracks: int = 4000             # cap on points in the graph (largest-baseline tracks kept)

    # weighting / priors (OFF by default: see module docstring)
    use_inlier_cov: bool = False       # whiten reprojection by a per-point 2D noise covariance
    point_noise_sigma: float = 1.5     # pixels, isotropic 2D detection noise
    use_pose_priors: bool = False      # RelativePosePriorCost between consecutive keyframes
    prior_sigma_trans: float = 0.05    # m
    prior_sigma_rot: float = 0.03      # rad
    anchor_pose_prior: bool = False    # soft 6-DoF AbsolutePosePriorCost replacing the hard pin
    anchor_sigma_trans: float = 1e-4
    anchor_sigma_rot: float = 1e-3

    verbose: bool = False


# --------------------------------------------------------------------------- #
# results
# --------------------------------------------------------------------------- #

@dataclass
class BaResult:
    """Everything the caller needs to log, gate or unit-test a BA run."""

    ran: bool
    reason: str = ""
    n_keyframes: int = 0
    n_tracks: int = 0
    n_residuals: int = 0
    n_points_total: int = 0
    initial_cost: float = float("nan")
    final_cost: float = float("nan")
    accepted: bool = False
    reject_reason: str = ""
    max_frame_shift: float = 0.0
    max_frame_rot_deg: float = 0.0
    seconds: float = 0.0

    def summary_line(self) -> str:
        if not self.ran:
            return f"[BA] skipped: {self.reason}"
        verdict = "accepted" if self.accepted else f"REJECTED ({self.reject_reason})"
        return (
            f"[BA] {verdict}: {self.n_keyframes} kfs, {self.n_tracks} pts, "
            f"{self.n_residuals} residuals, cost {self.initial_cost:.4g} -> "
            f"{self.final_cost:.4g}, max shift {self.max_frame_shift * 1000:.1f}mm / "
            f"{self.max_frame_rot_deg:.2f}deg, {self.seconds * 1000:.1f}ms"
        )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _t_to_blocks(T: np.ndarray):
    """4x4 ``T_CiO`` -> (q_xyzw float64, t float64); scipy layout IS the block layout."""
    q_xyzw = R.from_matrix(np.asarray(T, dtype=np.float64)[:3, :3]).as_quat().astype(np.float64)
    t = np.array(T[:3, 3], dtype=np.float64)
    return q_xyzw, t


def _blocks_to_t(q_xyzw: np.ndarray, t: np.ndarray) -> np.ndarray:
    """(q_xyzw, t) -> 4x4 transform."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R.from_quat(np.asarray(q_xyzw, dtype=np.float64)).as_matrix()
    T[:3, 3] = t
    return T


def _cam_params(K: np.ndarray) -> np.ndarray:
    return np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]], dtype=np.float64)


def _linear_solver(name: str):
    table = {
        "dense_schur": pyceres.LinearSolverType.DENSE_SCHUR,
        "dense_qr": pyceres.LinearSolverType.DENSE_QR,
        "sparse_schur": pyceres.LinearSolverType.SPARSE_SCHUR,
    }
    if name not in table:
        raise ValueError(f"unknown linear_solver {name!r}; expected one of {sorted(table)}")
    return table[name]


def _rot_deg_between(T0: np.ndarray, T1: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(
        R.from_matrix(np.asarray(T0)[:3, :3].T @ np.asarray(T1)[:3, :3]).as_rotvec()
    )))


# --------------------------------------------------------------------------- #
# layer 1: reprojection BA over a sliding keyframe window
# --------------------------------------------------------------------------- #

def bundle_adjust(
    frame_indices: Sequence[int],
    poses: Dict[int, np.ndarray],
    tracks: Dict[int, dict],
    K_dict: Dict[int, np.ndarray],
    cfg: Optional[BaConfig] = None,
    verbose: Optional[bool] = None,
) -> BaResult:
    """Run reprojection bundle adjustment and commit only if it improves.

    Parameters
    ----------
    frame_indices
        Keyframe indices to include in this graph, oldest first. Any index past
        ``cfg.max_keyframes`` is dropped from the *front*, so the window is always
        bounded regardless of how long the run lasts.
    poses
        ``{frame_idx: T_CiO (4x4)}``. Replaced in place, entry by entry, only when
        the run is accepted.
    tracks
        ``{tid: {'obs': {frame_idx: uv (len-2 float)}, 'pt3d': xyz (len-3 float)}}``.
        Points are written back the same way.
    K_dict
        ``{frame_idx: K (3x3)}``.
    """
    cfg = cfg or BaConfig()
    t0 = time.perf_counter()
    verbose = cfg.verbose if verbose is None else verbose

    if len(frame_indices) < 2:
        res = BaResult(ran=False, reason=f"need >=2 keyframes, got {len(frame_indices)}",
                       seconds=time.perf_counter() - t0)
        if verbose:
            print(res.summary_line())
        return res

    # Bounded window: drop from the front so the graph cannot grow without limit
    # over a long run. This is the fix for run_ba_python(), which passed
    # self.keyframes whole.
    window = list(frame_indices)[-cfg.max_keyframes:]

    # ---- select tracks visible in >=2 window keyframes ------------------ #
    track_rows: List[tuple] = []
    for tid, track in tracks.items():
        win_obs = {f: uv for f, uv in track["obs"].items() if f in window}
        if len(win_obs) >= 2:
            track_rows.append((tid, win_obs))
    if not track_rows:
        res = BaResult(ran=False, reason="no track observed in >=2 window keyframes",
                       seconds=time.perf_counter() - t0)
        if verbose:
            print(res.summary_line())
        return res

    # Cap the graph: keep the tracks with the widest camera baseline, which are the
    # best-conditioned and the ones that actually constrain the poses.
    if len(track_rows) > cfg.max_tracks:
        def _baseline(row):
            _, win_obs = row
            # Filter BEFORE indexing: poses[f] on a missing key raises KeyError.
            frames_in = [f for f in sorted(win_obs) if f in poses]
            if len(frames_in) < 2:
                return 0.0
            a, b = poses[frames_in[0]][:3, 3], poses[frames_in[-1]][:3, 3]
            return float(np.linalg.norm(a - b))
        track_rows.sort(key=_baseline, reverse=True)
        track_rows = track_rows[:cfg.max_tracks]

    # Per-keyframe observation count, to drop near-blank views from the graph.
    obs_per_frame: Dict[int, int] = {f: 0 for f in window}
    for _, win_obs in track_rows:
        for f in win_obs:
            obs_per_frame[f] += 1
    active_frames = [f for f in window if obs_per_frame[f] >= cfg.min_points_per_frame]
    if len(active_frames) < 2:
        res = BaResult(ran=False,
                       reason=f"only {len(active_frames)} keyframes have >= "
                              f"{cfg.min_points_per_frame} observations",
                       seconds=time.perf_counter() - t0)
        if verbose:
            print(res.summary_line())
        return res
    dropped = [f for f in window if f not in active_frames]
    if verbose and dropped:
        print(f"[BA] excluding {dropped}: fewer than {cfg.min_points_per_frame} observations")

    # ---- build parameter blocks ---------------------------------------- #
    # Copy everything: the inputs stay untouched until an accepted result is
    # committed. This is what makes the acceptance test able to *refuse* a
    # bad solve instead of silently writing it back.
    qblocks: Dict[int, np.ndarray] = {}
    tblocks: Dict[int, np.ndarray] = {}
    for f in active_frames:
        q, t = _t_to_blocks(poses[f])
        qblocks[f] = q
        tblocks[f] = t
    ptblocks: Dict[int, np.ndarray] = {}
    cam_blocks: Dict[int, np.ndarray] = {}
    for tid, _ in track_rows:
        ptblocks[tid] = np.array(tracks[tid]["pt3d"], dtype=np.float64)

    prior_poses = {f: np.array(poses[f], dtype=np.float64) for f in active_frames}

    prob = pyceres.Problem()
    for f in active_frames:
        prob.add_parameter_block(qblocks[f], 4)
        prob.add_parameter_block(tblocks[f], 3)
    for tid in ptblocks:
        prob.add_parameter_block(ptblocks[tid], 3)

    # Hold the manifold for the solver's whole lifetime; see module docstring.
    quat_manifold = pyceres.EigenQuaternionManifold()
    for f in active_frames:
        prob.set_manifold(qblocks[f], quat_manifold)

    loss = pyceres.HuberLoss(cfg.huber_delta)
    cov22 = None
    if cfg.use_inlier_cov:
        cov22 = np.eye(2) * (cfg.point_noise_sigma ** 2)

    n_residuals = 0
    for tid, win_obs in track_rows:
        pt = ptblocks[tid]
        for f, uv in win_obs.items():
            if f not in active_frames or f not in K_dict:
                continue
            if f not in cam_blocks:
                cam_blocks[f] = _cam_params(K_dict[f])
                prob.add_parameter_block(cam_blocks[f], 4)
                prob.set_parameter_block_constant(cam_blocks[f])
            if cov22 is None:
                cost = costs.ReprojErrorCost("PINHOLE", np.asarray(uv, dtype=np.float64).ravel())
            else:
                cost = costs.ReprojErrorCost(
                    "PINHOLE", cov22, np.asarray(uv, dtype=np.float64).ravel()
                )
            prob.add_residual_block(cost, loss, [qblocks[f], tblocks[f], pt, cam_blocks[f]])
            n_residuals += 1

    # ---- gauge pin ------------------------------------------------------ #
    # Pin the OLDEST keyframe that actually entered the graph. If the original
    # frame_indices[0] has no qualifying tracks it was never added, and
    # has_parameter_block() would be False -- in which case pinning it is a no-op
    # and the graph is unscentred, so fall back to the oldest active frame.
    anchor = active_frames[0]
    if cfg.fix_first:
        for q in (qblocks[anchor], tblocks[anchor]):
            prob.set_parameter_block_constant(q)

    # ---- optional priors ------------------------------------------------ #
    if cfg.anchor_pose_prior:
        # Soft 6-DoF anchor: a Mahalanobis pose prior instead of a hard constant
        # block, so the anchor can still be nudged by a large measurement set.
        # A position-only prior would leave the rotation (and scale) gauge open;
        # the full pose prior closes everything except scale. Residual layout is
        # [rotation(3), translation(3)], so the covariance is ordered rot-first.
        cov66 = np.diag(
            [cfg.anchor_sigma_rot ** 2] * 3 + [cfg.anchor_sigma_trans ** 2] * 3
        )
        cost = costs.AbsolutePosePriorCost(
            cov66,
            pycolmap.Rigid3d(
                pycolmap.Rotation3d(prior_poses[anchor][:3, :3]),
                np.asarray(prior_poses[anchor][:3, 3], dtype=np.float64),
            ),
        )
        prob.add_residual_block(cost, loss, [qblocks[anchor], tblocks[anchor]])
        n_residuals += 1

    if cfg.use_pose_priors:
        cov66 = np.diag(
            [cfg.prior_sigma_rot ** 2] * 3 + [cfg.prior_sigma_trans ** 2] * 3
        )
        for fa, fb in zip(active_frames[:-1], active_frames[1:]):
            # Residual is (t_a - t_b) - t_prior, so the prior is the relative
            # translation from the later keyframe to the earlier one.
            t_prior = prior_poses[fa][:3, 3] - prior_poses[fb][:3, 3]
            cost = costs.RelativePosePriorCost(
                cov66,
                pycolmap.Rigid3d(
                    pycolmap.Rotation3d(np.eye(3)),
                    t_prior.astype(np.float64),
                ),
            )
            prob.add_residual_block(
                cost, loss, [qblocks[fa], tblocks[fa], qblocks[fb], tblocks[fb]]
            )
            n_residuals += 1

    # ---- solve ---------------------------------------------------------- #
    options = pyceres.SolverOptions()
    options.linear_solver_type = _linear_solver(cfg.linear_solver)
    options.max_num_iterations = cfg.max_num_iterations
    options.minimizer_progress_to_stdout = False
    summary = pyceres.SolverSummary()
    try:
        pyceres.solve(options, prob, summary)
    except Exception as exc:  # never let a solver failure kill the tracking loop
        res = BaResult(ran=False, reason=f"solve failed: {type(exc).__name__}: {exc}",
                       n_keyframes=len(active_frames), n_tracks=len(ptblocks),
                       seconds=time.perf_counter() - t0)
        print(f"[BA] {res.summary_line()}")
        return res

    res = BaResult(
        ran=True,
        n_keyframes=len(active_frames),
        n_tracks=len(ptblocks),
        n_residuals=n_residuals,
        n_points_total=int(sum(len(w) for _, w in track_rows)),
        initial_cost=float(summary.initial_cost),
        final_cost=float(summary.final_cost),
        seconds=time.perf_counter() - t0,
    )

    if res.n_residuals == 0:
        res.reject_reason = "empty graph after residual construction"
    elif cfg.require_cost_decrease and not (
        res.final_cost <= res.initial_cost - cfg.min_relative_decrease * max(1.0, res.initial_cost)
    ):
        res.reject_reason = (
            f"cost did not decrease ({res.initial_cost:.6f} -> {res.final_cost:.6f})"
        )

    # Shift guard: the solve can still drive cost down while dragging a pose far
    # away (a bad triangulation does exactly that). Measure the worst movement and
    # refuse to commit anything that exceeds the per-frame budget.
    max_shift, max_rot = 0.0, 0.0
    for f in active_frames:
        T_new = _blocks_to_t(qblocks[f], tblocks[f])
        max_shift = max(max_shift, float(np.linalg.norm(T_new[:3, 3] - prior_poses[f][:3, 3])))
        max_rot = max(max_rot, _rot_deg_between(prior_poses[f], T_new))
    res.max_frame_shift = max_shift
    res.max_frame_rot_deg = max_rot

    if not res.reject_reason:
        if max_shift > cfg.max_frame_shift:
            res.reject_reason = (
                f"max frame shift {max_shift * 1000:.1f}mm > {cfg.max_frame_shift * 1000:.1f}mm"
            )
        elif max_rot > cfg.max_frame_rot_deg:
            res.reject_reason = (
                f"max frame rotation {max_rot:.2f}deg > {cfg.max_frame_rot_deg:.2f}deg"
            )

    res.accepted = res.ran and not res.reject_reason

    if res.accepted:
        for f in active_frames:
            poses[f] = _blocks_to_t(qblocks[f], tblocks[f])
        for tid, _ in track_rows:
            tracks[tid]["pt3d"] = np.array(ptblocks[tid], dtype=np.float64)
    elif verbose:
        print(f"[BA] nothing written back: {res.reject_reason}")

    if verbose:
        print(res.summary_line())
    return res


# --------------------------------------------------------------------------- #
# drop-in replacement for geometric_tracker.run_ba
# --------------------------------------------------------------------------- #

def run_ba(
    frame_indices: Sequence[int],
    poses: Dict[int, np.ndarray],
    tracks: Dict[int, dict],
    K_dict: Dict[int, np.ndarray],
    fix_first: bool = True,
    cfg: Optional[BaConfig] = None,
) -> BaResult:
    """Signature-compatible wrapper around :func:`bundle_adjust`.

    The legacy ``geometric_tracker.run_ba`` returned ``None`` and wrote its result
    back unconditionally; this keeps the call sites valid while returning a result
    the caller can actually use to log or gate.
    """
    cfg = cfg or BaConfig(fix_first=fix_first)
    return bundle_adjust(frame_indices, poses, tracks, K_dict, cfg=cfg)


def _declined(reason: str) -> dict:
    """A refusal from :func:`align_to_object_frame` that still carries the full
    key set promised in its docstring.

    The identity transform / s = 1 are the meaningful defaults for the keys this
    solve did not produce: a caller that reads ``s`` first and gates on
    ``accepted`` afterwards (``ObjGSCameraMapper.densify`` does) falls through to
    its "no alignment" branch with s == 1 instead of raising ``KeyError``.
    """
    T = np.eye(4, dtype=np.float64)
    return {"T_to_object": T, "log_s": 0.0, "s": 1.0,
            "accepted": False, "initial_cost": 0.0, "final_cost": 0.0,
            "reason": reason}


# --------------------------------------------------------------------------- #
# layer 2: tracker-frame <-> object-frame alignment with an explicit scale
# --------------------------------------------------------------------------- #

def align_to_object_frame(
    pts_tracker: np.ndarray,
    pts_object: np.ndarray,
    T_og_prior: Optional[np.ndarray] = None,
    log_s0: float = 0.0,
    pin_scale: bool = False,
    pin_pose: bool = False,
    huber_delta: float = 1000.0,
    max_iterations: int = 200,
    verbose: bool = False,
) -> dict:
    """Least-squares similarity transform between two point sets, in pyceres.

    Resolves the scale explicitly instead of absorbing it: the model is

        X_object = exp(log_s) * R @ X_tracker + t

    with ``log_s`` a single 1-D parameter block, so scale is optimised on the
    multiplicative (log) parameter that makes the problem symmetric in ``s`` and
    ``1/s`` and well-conditioned far from 1 -- unlike the ``np.clip(s, 0.1, 10.0)``
    clamp that ``ObjGSCameraMapper.densify`` applied to every keyframe depth buffer.

    Both inputs must be correspondences of equal length (N, 3). N >= 4 non-coplanar
    points are needed; with fewer, or with (near-)coplanar input, the transform is
    not uniquely determined and the caller should skip the alignment. Exception:
    with ``pin_pose=True`` the pose blocks are held constant at the prior and only
    ``log_s`` is optimised, which stays observable even on (near-)coplanar input
    (a free pose trades translation against scale there). That is the mode the
    mapper's depth alignment uses: sensor and rendered depth live in the SAME
    camera frame, so any nonzero R/t is pose error that must not leak into the
    depth buffer. Pinning a prior the data does not actually satisfy is not
    neutral -- with the pose frozen the fit has nowhere to put that pose error
    but the scale (measured s = 2.288 for a true 2.5 against the prior used in
    test_align_pin_pose_keeps_a_nonidentity_prior_fixed, -8.5%). Blaming the
    rotation would be wrong: that prior rotates 0.10/-0.05/0.15 DEGREES, which
    enters only second order (s * theta^2 / 3 ~ 1e-6 there). The -8.5% comes
    from the frozen TRANSLATION prior_t = (0.3, -0.2, 1.1) -- in the closed
    form s* = s_true - <prior_t, R p> / <p, p> that term is 4.95 / 23.67 =
    0.209. Both blocks are pinned below, so either one can do this. The mapper's
    case is safe because sensor and rendered depth share a frame, so the pose
    error held back is exactly the one that would otherwise inflate the scale;
    another caller must not pin an arbitrary prior.

    ``accepted`` is gated: the result is accepted only when the solver produced a
    usable solution that strictly decreased the cost (a perfect prior with no new
    information is rejected, leaving the caller's default in place).

    Known modelling gap, left uncorrected on purpose: in ``pin_pose=True`` mode
    this fits `s * X + t` only, so a depth-buffer offset that is best described as
    an additive bias (`depth_obj = s * depth_track + b`) is absorbed into ``s``
    instead of being estimated separately. The legacy affine fit in
    ``ObjGSCameraMapper.densify`` models both and is strictly more expressive.
    Measured on a synthetic case with a true 1.25x scale and a moderate bias: this
    fit returns s = 1.3575 (+8.6%) and is accepted, leaving ~0.14 m of depth
    error; with a larger bias, +17.1% and ~0.28 m. It trades that bias error for
    rejecting implausible scales outright, which is what the flag exists for.

    Returns ``{'T_to_object': T (4x4), 'log_s': float, 's': float, 'accepted': bool,
    'initial_cost': float, 'final_cost': float}``. Every return path -- including
    the pre-solve refusals -- carries that full key set, with the identity pose and
    ``s = 1`` on refusal, plus a ``reason`` string. Callers may read ``s`` before
    checking ``accepted`` and rely on that.
    """
    pts_tracker = np.asarray(pts_tracker, dtype=np.float64)
    pts_object = np.asarray(pts_object, dtype=np.float64)
    if pts_tracker.shape != pts_object.shape or pts_tracker.ndim != 2 or pts_tracker.shape[1] != 3:
        raise ValueError(f"expected matching (N,3) arrays, got {pts_tracker.shape} vs {pts_object.shape}")
    n = pts_tracker.shape[0]
    if n < 4:
        # Carry the full key set, not just 'accepted': callers read res['s']
        # BEFORE checking res['accepted'] (obj_gs_mapping.densify), so a bare
        # {"accepted": False} here is a KeyError, not a graceful skip.
        return _declined(f"need >=4 correspondences, got {n}")

    centroid = pts_tracker.mean(axis=0)
    spread = float(np.linalg.norm(pts_tracker - centroid))
    if spread <= 1e-8:
        return _declined("source points have zero spread")

    # Start the optimisation from the prior transform when one is available; the
    # prior is a good initial guess and a safeguard against converging to the
    # antipodal quaternion when the point cloud is nearly planar.
    if T_og_prior is None:
        q = pycolmap.Rotation3d(np.eye(3)).quat.astype(np.float64).copy()
        t = np.zeros(3, dtype=np.float64)
    else:
        q, t = _t_to_blocks(T_og_prior)
        t = t.copy()

    prob = pyceres.Problem()
    # Source points are held constant: the transform is what we solve for. Holding
    # them free makes this a pure gauge (see module docstring).
    src_blocks = [np.array(p, dtype=np.float64) for p in pts_tracker]
    log_s = np.array([float(log_s0)])

    for b in src_blocks:
        prob.add_parameter_block(b, 3)
        prob.set_parameter_block_constant(b)
    prob.add_parameter_block(q, 4)
    prob.add_parameter_block(t, 3)
    prob.add_parameter_block(log_s, 1)
    if pin_scale:
        prob.set_parameter_block_constant(log_s)
    if pin_pose:
        # Scale-only fit (see docstring): pose stays at the prior.
        prob.set_parameter_block_constant(q)
        prob.set_parameter_block_constant(t)

    # Keep the manifold alive for the whole solve.
    quat_manifold = pyceres.EigenQuaternionManifold()
    if not pin_pose:
        prob.set_manifold(q, quat_manifold)

    loss = pyceres.HuberLoss(huber_delta)
    for dst, prior in zip(src_blocks, pts_object):
        prob.add_residual_block(
            costs.Point3DAlignmentCost(np.asarray(prior, dtype=np.float64), True),
            loss,
            [dst, q, t, log_s],
        )

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
    options.max_num_iterations = max_iterations
    options.minimizer_progress_to_stdout = False
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)

    T = _blocks_to_t(q, t)
    s = float(np.exp(log_s[0]))
    solved = bool(summary.IsSolutionUsable())
    accepted = solved and summary.final_cost < summary.initial_cost
    if verbose:
        # :.4g, not :.4f -- costs are summed squared scale residuals and sit at
        # ~1e-14 to 1e-1, so .4f rendered every real cost as 0.0000 and the log
        # line said nothing about whether the solve moved. And say WHICH leg
        # refused: solver-refused and solved-but-no-improvement used to read as
        # the same bare "rejected".
        if accepted:
            verdict = "accepted"
        else:
            verdict = "rejected: " + ("solver refused" if not solved
                                      else "solved but cost did not decrease")
        # float() before formatting, as bundle_adjust does: summary.initial_cost
        # and summary.final_cost are pybind11 properties, and a format spec
        # applied to anything other than a plain float is where this line used to
        # silently print 0.0000.
        print(f"[Align] s={s:.4f}, cost {float(summary.initial_cost):.4g} -> "
              f"{float(summary.final_cost):.4g} ({verdict})")

    return {
        "accepted": accepted,
        "T_to_object": T,
        "log_s": float(log_s[0]),
        "s": s,
        "initial_cost": float(summary.initial_cost),
        "final_cost": float(summary.final_cost),
        "n_points": n,
    }
