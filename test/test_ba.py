"""
Self-contained tests for full_system_ow_working/ba.py.

Nothing here touches a dataset, a GPU or a checkpoint: every test synthesises
cameras and points, projects them, and checks that bundle adjustment recovers the
true configuration. Run with:

    python -m pytest test/test_ba.py -q
or:
    python test/test_ba.py
"""

import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "full_system_ow_working"))

import ba  # noqa: E402


# --------------------------------------------------------------------------- #
# synthetic scene helpers
# --------------------------------------------------------------------------- #

def _make_scene(n_keyframes=6, n_points=60, seed=0):
    """Keyframes travelling +x with a mild roll, points in front, exact reprojection."""
    rng = np.random.default_rng(seed)
    f, cx, cy = 250.0, 320.0, 240.0
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

    poses = {}
    for i in range(n_keyframes):
        ang = 0.08 * i  # roll about the optical axis
        T = np.eye(4)
        T[:3, :3] = R.from_euler("zyx", [0.0, 0.0, ang]).as_matrix()
        # Keep cameras off the origin for general geometry. NOTE: this does NOT
        # fix the scale gauge -- monocular reprojection BA never observes scale,
        # wherever the pinned camera sits (see ba.py's module docstring).
        T[:3, 3] = np.array([0.4 * i + 0.35, -0.2, 0.3])
        poses[i] = T

    # Explicit low AND high (uniform's high defaults to 1.0 -- easy trap).
    points = rng.uniform(np.array([-2.0, -2.0, 4.0]), np.array([6.0, 2.0, 8.0]),
                         size=(n_points, 3))

    tracks = {}
    scene_pts = {}
    tid = 0
    for p in points:
        obs = {}
        for i, T in poses.items():
            P = T[:3, :3] @ p + T[:3, 3]
            if P[2] <= 0.01:
                continue
            u = K @ P
            uv = np.array([u[0] / u[2], u[1] / u[2]], dtype=np.float32)
            if not (0 < uv[0] < 2 * cx and 0 < uv[1] < 2 * cy):
                continue
            obs[i] = uv
        if len(obs) >= 2:
            # Seed with a deliberately poor guess so BA has real work to do.
            # The scene truth is returned separately: pt3d is NOT the truth.
            tracks[tid] = {"obs": obs, "pt3d": np.array(p, dtype=np.float64) + 0.05}
            scene_pts[tid] = p
            tid += 1

    return poses, {i: K for i in poses}, tracks, scene_pts


def _pose_error(T0, T1):
    dt = float(np.linalg.norm(T0[:3, 3] - T1[:3, 3]))
    dr = float(np.degrees(np.linalg.norm(
        R.from_matrix(T0[:3, :3].T @ T1[:3, :3]).as_rotvec())))
    return dt, dr


def _umeyama(src, dst):
    """Similarity (s, R, t) mapping src -> dst, both (N, 3)."""
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    Xc, Yc = src - mu_s, dst - mu_d
    cov = (Yc.T @ Xc) / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    Rm = U @ S @ Vt
    s = float(np.sum(D * np.diag(S)) / np.sum(Xc ** 2) * len(src))
    t = mu_d - s * (Rm @ mu_s)
    return s, Rm, t


def _camera_centers(poses):
    return np.array([-poses[i][:3, :3].T @ poses[i][:3, 3] for i in sorted(poses)])


# --------------------------------------------------------------------------- #
# round-trip conventions
# --------------------------------------------------------------------------- #

def test_quaternion_roundtrip_is_exact():
    for seed in range(20):
        T = np.eye(4)
        T[:3, :3] = R.random(random_state=seed).as_matrix()
        T[:3, 3] = np.array([seed - 10, 0.0, -2.0])
        q, t = ba._t_to_blocks(T)
        assert q.shape == (4,) and q.dtype == np.float64
        assert np.allclose(t, T[:3, 3])
        assert np.allclose(ba._blocks_to_t(q, t), T, atol=1e-12)
        # Block convention is xyzw (Eigen layout): identical to scipy's as_quat.
        assert np.allclose(q, R.from_matrix(T[:3, :3]).as_quat())
        # ... and to pycolmap's Rotation3d().quat.
        import pycolmap
        assert np.allclose(q, pycolmap.Rotation3d(T[:3, :3]).quat) or np.allclose(
            q, -pycolmap.Rotation3d(T[:3, :3]).quat)


def test_cam_params_layout():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    assert np.allclose(ba._cam_params(K), [500.0, 500.0, 320.0, 240.0])


def test_linear_solver_unknown_raises():
    with pytest.raises(ValueError, match="unknown linear_solver"):
        ba._linear_solver("conj_grad")


# --------------------------------------------------------------------------- #
# bundle adjustment
# --------------------------------------------------------------------------- #

def test_rejects_when_too_few_keyframes():
    poses, K_dict, tracks, scene_pts = _make_scene(n_keyframes=1)
    res = ba.bundle_adjust([0], poses, tracks, K_dict)
    assert not res.ran and "need >=2 keyframes" in res.reason


def test_recovers_perturbed_poses_and_points():
    poses, K_dict, tracks, scene_pts = _make_scene(seed=1)
    true_poses = {k: v.copy() for k, v in poses.items()}
    # The scene truth is what the observations were generated from -- NOT the
    # deliberately offset pt3d seeds.
    true_points = scene_pts

    rng = np.random.default_rng(7)
    for i in list(poses):
        if i == 0:
            continue  # anchor
        poses[i][:3, 3] += rng.normal(0, 0.004, size=3)
        poses[i][:3, :3] = (
            R.from_rotvec(rng.normal(0, 0.004, size=3)).as_matrix()
            @ poses[i][:3, :3]
        )
    for t in tracks:
        tracks[t]["pt3d"] = true_points[t] + rng.normal(0, 0.004, size=3)

    res = ba.bundle_adjust([0, 1, 2, 3, 4, 5], poses, tracks, K_dict)
    assert res.ran, res.reason
    assert res.accepted, res.reject_reason
    assert res.final_cost < res.initial_cost
    assert res.n_residuals > 0

    # Anchor must not move at all.
    assert np.allclose(poses[0], true_poses[0], atol=1e-9)

    # Rotations live off the scale-gauge orbit: recovered exactly, in raw space.
    for i in (1, 2, 3, 4, 5):
        _, dr = _pose_error(true_poses[i], poses[i])
        assert dr < 0.08, f"frame {i}: rotation error {dr:.4f}deg"

    # Translations and points are only observable up to the 1-DoF scale gauge
    # that survives the pin, so assert AFTER a similarity alignment (Umeyama).
    # With exact observations the aligned recovery should be near-machine precision.
    s, Rm, tv = _umeyama(_camera_centers(poses), _camera_centers(true_poses))
    assert abs(s - 1.0) < 0.05, f"scale drift {s}"

    rec_c = s * (Rm @ _camera_centers(poses).T).T + tv
    cerr = np.linalg.norm(rec_c - _camera_centers(true_poses), axis=1)
    assert cerr.max() < 1e-3, f"aligned camera-center error {cerr.max():.6f}"

    rec_pts = np.array([tracks[t]["pt3d"] for t in sorted(tracks)])
    true_pts = np.array([true_points[t] for t in sorted(tracks)])
    aligned = s * (Rm @ rec_pts.T).T + tv
    perr = np.linalg.norm(aligned - true_pts, axis=1)
    assert perr.mean() < 1e-3, f"aligned mean point error {perr.mean():.6f}"
    assert perr.max() < 3e-3, f"aligned worst point error {perr.max():.6f}"


def test_window_is_bounded_from_the_front():
    """10 keyframes requested, window of 4: only the LAST 4 may enter the graph,
    and poses outside the window must be left exactly as they were."""
    poses, K_dict, tracks, scene_pts = _make_scene(n_keyframes=10, seed=2)
    outside = {k: poses[k].copy() for k in range(6)}
    cfg = ba.BaConfig(max_keyframes=4)
    res = ba.bundle_adjust(list(range(10)), poses, tracks, K_dict, cfg=cfg, verbose=True)
    assert res.ran, res.reason
    assert 2 <= res.n_keyframes <= 4
    for k, T in outside.items():
        assert np.array_equal(poses[k], T), f"pose {k} outside the window was modified"


def test_refuses_to_commit_a_worse_solve():
    """A garbage initial guess that the solver cannot improve must not be written back."""
    poses, K_dict, tracks, scene_pts = _make_scene(seed=3)

    # Inject an enormous, unsolvable perturbation.
    for i in poses:
        if i == 0:
            continue
        poses[i][:3, 3] += np.array([3.0, -2.0, 1.5])
        poses[i][:3, :3] = (R.from_rotvec([1.2, -0.9, 0.7]).as_matrix()
                            @ poses[i][:3, :3])

    # Snapshot AFTER the perturbation: a rejected BA must leave the inputs
    # exactly as they were handed in (perturbed), not restore the pre-perturbation state.
    handed_in_poses = {k: v.copy() for k, v in poses.items()}
    handed_in_points = {t: v["pt3d"].copy() for t, v in tracks.items()}

    cfg = ba.BaConfig(require_cost_decrease=True, max_num_iterations=20)
    res = ba.bundle_adjust([0, 1, 2, 3, 4, 5], poses, tracks, K_dict, cfg=cfg, verbose=True)
    assert res.ran
    # Either the solver failed to decrease (rejected on cost) or the shift guard
    # caught the large movement -- both are correct refusals. Nothing may change.
    if not res.accepted:
        for k in poses:
            assert np.allclose(poses[k], handed_in_poses[k])
        for t in tracks:
            assert np.allclose(tracks[t]["pt3d"], handed_in_points[t])
        assert res.reject_reason, "a refusal must carry a reason"


def test_shift_guard_rejects_large_motion():
    """With the shift budget at zero, any non-trivial move is refused and nothing is written."""
    poses, K_dict, tracks, scene_pts = _make_scene(seed=4)
    rng = np.random.default_rng(11)
    for i in poses:
        if i == 0:
            continue
        poses[i][:3, 3] += rng.normal(0, 0.01, size=3)
    # Snapshot AFTER the perturbation: a rejected BA must leave the perturbed
    # inputs exactly as they were handed in.
    original_poses = {k: v.copy() for k, v in poses.items()}

    cfg = ba.BaConfig(max_frame_shift=0.0, max_frame_rot_deg=0.0)
    res = ba.bundle_adjust([0, 1, 2, 3, 4, 5], poses, tracks, K_dict, cfg=cfg, verbose=True)
    assert res.ran
    assert not res.accepted
    assert "shift" in res.reject_reason or "rotation" in res.reject_reason
    for k in poses:
        assert np.allclose(poses[k], original_poses[k])


def test_pose_priors_are_available_and_harden_the_graph():
    """Covariance-weighted pose priors must build and solve without error, and
    they are the metric leash that closes the scale-gauge orbit: with them the
    RAW translations stay near the seed noise instead of ramping."""
    poses, K_dict, tracks, scene_pts = _make_scene(seed=5)
    true_poses = {k: v.copy() for k, v in poses.items()}
    rng = np.random.default_rng(21)
    for i in poses:
        if i == 0:
            continue
        poses[i][:3, 3] += rng.normal(0, 0.004, size=3)

    cfg = ba.BaConfig(use_pose_priors=True, prior_sigma_trans=0.05, prior_sigma_rot=0.03)
    res = ba.bundle_adjust([0, 1, 2, 3, 4, 5], poses, tracks, K_dict, cfg=cfg, verbose=True)
    assert res.ran
    assert res.accepted, res.reject_reason
    for i in (1, 2, 3, 4, 5):
        dt, _ = _pose_error(true_poses[i], poses[i])
        # The leash is loose on purpose (5cm): the gauge walk is BOUNDED by the
        # prior, not eliminated -- contrast the un-pinned ramp, which is only
        # bounded by the shift guard. 3*sigma is the meaningful budget.
        assert dt < 3 * cfg.prior_sigma_trans, (
            f"frame {i}: raw translation drift {dt:.6f} beyond the prior leash")


def test_inlier_covariance_weighting_solves():
    poses, K_dict, tracks, scene_pts = _make_scene(seed=6)
    cfg = ba.BaConfig(use_inlier_cov=True, point_noise_sigma=1.5)
    res = ba.bundle_adjust([0, 1, 2, 3, 4, 5], poses, tracks, K_dict, cfg=cfg, verbose=True)
    assert res.ran and res.accepted, res.reject_reason


def test_anchor_pose_prior_replaces_the_hard_pin():
    """With the hard pin off and a soft 6-DoF pose prior on, the anchor may move
    a little, but the rotation gauge is closed and the solve must not run away.
    (A position-only soft anchor leaves the rotation gauge open: the solve then
    drifts along it at zero cost and the shift guard correctly rejects.)"""
    poses, K_dict, tracks, scene_pts = _make_scene(seed=8)
    true_anchor = poses[0].copy()
    rng = np.random.default_rng(13)
    for i in poses:
        poses[i][:3, 3] += rng.normal(0, 0.004, size=3)

    cfg = ba.BaConfig(fix_first=False, anchor_pose_prior=True,
                      anchor_sigma_trans=1e-2, anchor_sigma_rot=1e-2)
    res = ba.bundle_adjust([0, 1, 2, 3, 4, 5], poses, tracks, K_dict, cfg=cfg, verbose=True)
    assert res.ran and res.accepted, res.reject_reason
    dt, dr = _pose_error(true_anchor, poses[0])
    assert dt < 2e-2, f"soft anchor moved {dt:.5f} m"
    assert dr < 2.0, f"soft anchor rotated {dr:.3f} deg"


def test_run_ba_wrapper_returns_a_result():
    """The legacy signature must keep working and now return something usable."""
    poses, K_dict, tracks, scene_pts = _make_scene(seed=9)
    out = ba.run_ba([0, 1, 2, 3, 4, 5], poses, tracks, K_dict)
    assert isinstance(out, ba.BaResult)
    assert out.ran and out.accepted, out.reject_reason
    out2 = ba.run_ba([0, 1, 2, 3, 4, 5], poses, tracks, K_dict, fix_first=False)
    assert isinstance(out2, ba.BaResult)


def test_points_out_of_the_window_are_ignored():
    poses, K_dict, tracks, scene_pts = _make_scene(seed=10)
    # A track observed only at frame 0 must not enter a window that starts at frame 2.
    tid = max(tracks) + 1
    tracks[tid] = {"obs": {0: np.array([100.0, 100.0], np.float32)},
                   "pt3d": np.array([1.0, 1.0, 1.0])}
    res = ba.bundle_adjust([2, 3, 4, 5], poses, tracks, K_dict, verbose=True)
    assert res.ran
    assert res.n_tracks <= sum(1 for t in tracks.values()
                               if len({f for f in t["obs"] if f in [2, 3, 4, 5]}) >= 2)


# --------------------------------------------------------------------------- #
# frame alignment with explicit scale
# --------------------------------------------------------------------------- #

def test_align_recovers_rotation_translation_and_scale():
    rng = np.random.default_rng(0)
    src = rng.uniform([-1, 0, -1], size=(50, 3)) + np.array([0.0, 0.0, -2.0])

    R_true = R.from_euler("zyx", [0.3, -0.2, 0.5])
    s_true = 2.37
    t_true = np.array([5.0, -3.0, 2.0])
    dst = (s_true * (R_true.as_matrix() @ src.T)).T + t_true

    out = ba.align_to_object_frame(src, dst, log_s0=0.0)
    assert out["accepted"]
    assert abs(out["s"] - s_true) < 1e-6, f"s={out['s']} vs {s_true}"
    dt = float(np.linalg.norm(out["T_to_object"][:3, 3] - t_true))
    assert dt < 1e-6, f"t error {dt}"
    R_hat = R.from_matrix(out["T_to_object"][:3, :3])
    dr = float(np.degrees(np.linalg.norm((R_hat.inv() * R_true).as_rotvec())))
    assert dr < 1e-3, f"rotation error {dr} deg"
    assert out["final_cost"] < 1e-18


def test_align_uses_the_prior_as_a_start():
    src = np.array([[0.0, 0.0, -2.0], [1.0, 0.0, -2.0], [0.0, 1.0, -2.0], [1.0, 1.0, -2.0]])
    s_true = 0.41
    dst = s_true * src + np.array([9.0, 9.0, 9.0])
    T_prior = np.eye(4)
    T_prior[:3, 3] = np.array([9.0, 9.0, 9.0])
    out = ba.align_to_object_frame(src, dst, T_og_prior=T_prior, log_s0=0.0)
    assert out["accepted"]
    assert abs(out["s"] - s_true) < 1e-6


def test_align_rejects_too_few_correspondences():
    out = ba.align_to_object_frame(np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 1.0, 1.0]]))
    assert not out["accepted"] and "need >=4" in out["reason"]


def test_align_rejects_degenerate_input():
    src = np.array([[0.0, 0.0, 0.0]] * 10)
    dst = src + 1.0
    out = ba.align_to_object_frame(src, dst)
    assert not out["accepted"] and "zero spread" in out["reason"]


def test_align_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="expected matching"):
        ba.align_to_object_frame(np.zeros((5, 3)), np.zeros((4, 3)))


def test_align_recovers_scale_far_from_one():
    """The point of doing this in log space: s away from 1 must not be clipped."""
    rng = np.random.default_rng(3)
    src = rng.uniform([-0.5, -0.5, -0.5], size=(40, 3))
    for s_true in (0.13, 0.3, 3.5, 8.2):
        dst = s_true * src + np.array([1.0, 2.0, 3.0])
        out = ba.align_to_object_frame(src, dst)
        assert out["accepted"]
        assert abs(out["s"] - s_true) / s_true < 1e-6, f"scale {s_true} -> {out['s']}"


def test_align_pin_pose_recovers_scale_only():
    """Mapper depth-alignment mode: pose pinned at identity, scale only."""
    rng = np.random.default_rng(5)
    src = rng.uniform(np.array([-2., -2., 3.]), np.array([2., 2., 6.]), size=(200, 3))
    s_true = 1.7
    dst = s_true * src + rng.normal(scale=1e-3, size=src.shape)
    out = ba.align_to_object_frame(src, dst, pin_pose=True, huber_delta=0.05)
    assert out["accepted"]
    assert abs(out["s"] - s_true) < 1e-2, f"s={out['s']} vs {s_true}"
    np.testing.assert_allclose(out["T_to_object"][:3, :3], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(out["T_to_object"][:3, 3], np.zeros(3), atol=1e-9)


def test_align_pin_pose_stays_observable_on_planar_points():
    """A fronto-parallel plane is the degenerate case for a free-pose fit
    (translation along the normal trades against scale); with the pose pinned,
    the 1-dof scale fit -- exactly what a depth buffer can express -- stays
    observable. This is the mode ObjGSCameraMapper.densify uses."""
    rng = np.random.default_rng(7)
    xy = rng.uniform(-1.0, 1.0, size=(200, 2))
    src = np.stack([xy[:, 0], xy[:, 1], np.full(200, 5.0)], axis=-1)
    s_true = 1.6
    dst = s_true * src + rng.normal(scale=1e-3, size=src.shape)
    out = ba.align_to_object_frame(src, dst, pin_pose=True, huber_delta=0.05)
    assert out["accepted"]
    assert abs(out["s"] - s_true) < 1e-2, f"s={out['s']} vs {s_true}"


def test_align_pin_pose_keeps_a_nonidentity_prior_fixed():
    """pin_pose pins at the PRIOR, not at the identity: the mapper passes the
    last committed object pose, which is generally neither. With data generated
    in that pose the fit must recover the scale exactly and hold both pose
    blocks at the prior. The identity-prior tests above cannot see a regression
    here -- e.g. optimising the pose blocks anyway, or dropping them from the
    cost -- because an identity prior makes "held at identity" and "solved to
    identity" indistinguishable.

    The converse is the trap this mode guards against leaking elsewhere: pinning
    a WRONG prior is not neutral, the fit absorbs the prior's pose error into the
    scale. Reproduced and asserted at the bottom, which is what the -8.5% quoted
    in ba.align_to_object_frame's docstring refers to.
    """
    rng = np.random.default_rng(19)
    src = rng.uniform(np.array([-2., -2., 3.]), np.array([2., 2., 6.]), size=(200, 3))

    prior_R = R.from_euler("xyz", [0.10, -0.05, 0.15], degrees=True)
    prior_t = np.array([0.3, -0.2, 1.1])
    prior = np.eye(4)
    prior[:3, :3] = prior_R.as_matrix()
    prior[:3, 3] = prior_t

    # dst = s * R @ src + t, the model this solve actually fits (rows are points,
    # so R @ x becomes x @ R.T).
    s_true = 2.5
    dst = s_true * (src @ prior_R.as_matrix().T) + prior_t
    dst = dst + rng.normal(scale=1e-4, size=src.shape)

    out = ba.align_to_object_frame(src, dst, T_og_prior=prior, pin_pose=True,
                                   huber_delta=0.05)
    assert out["accepted"]
    # The fit is a 1-dof least squares in s, so the injected 1e-4 noise moves it
    # by ~2e-6. 1e-3 is far tighter than the scale errors this gate is meant to
    # reject, and far looser than the -8.5% a wrong prior induces.
    assert abs(out["s"] - s_true) < 1e-3, f"s={out['s']} vs {s_true}"
    # Held constant, not merely close: with pin_pose the pose blocks are never
    # optimised, so the answer is exact up to the block<->matrix roundtrip.
    np.testing.assert_allclose(out["T_to_object"][:3, :3], prior[:3, :3], atol=1e-12)
    np.testing.assert_allclose(out["T_to_object"][:3, 3], prior_t, atol=1e-12)

    # The converse, pinned rather than remembered: the same point set generated
    # at the IDENTITY pose, fitted with the non-identity prior above. pin_pose
    # freezes both blocks, so the prior's error has nowhere to go but s.
    dst_wrong = s_true * src
    wrong = ba.align_to_object_frame(src, dst_wrong, T_og_prior=prior,
                                     pin_pose=True, huber_delta=0.05)
    assert wrong["accepted"]
    # Closed form: s* = s_true - <prior_t, R p> / <p, p>. For this box
    # mean(|p|^2) = 23.67 and prior_t . mean(p) = 4.95, so s* = 2.5 - 0.209 =
    # 2.291 (measured 2.288, -8.5%). The rotation term is second order --
    # s*theta^2/3 ~ 1e-6 at 0.1 deg -- so this -8.5% is the frozen TRANSLATION,
    # not the 0.10/-0.05/0.15 deg rotation the earlier wording blamed.
    assert abs(wrong["s"] - 2.29) < 0.02, f"s={wrong['s']}"


def test_align_rejects_when_there_is_nothing_to_improve():
    """Acceptance gate: identical point sets at the identity prior give zero
    initial cost, so no strict decrease is possible and the fit must be
    rejected -- the caller keeps its default (s=1) instead of committing a
    no-information solve."""
    rng = np.random.default_rng(11)
    src = rng.uniform(np.array([-2., -2., 3.]), np.array([2., 2., 6.]), size=(50, 3))
    out = ba.align_to_object_frame(src, src.copy(), pin_pose=True)
    assert not out["accepted"]
    assert out["s"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
