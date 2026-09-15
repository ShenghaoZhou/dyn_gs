"""
Self-contained tests for the Phase-0 geometric-tracker estimator fixes in
full_system_ow_working/geometric_tracker.py, plus the caller-side covariance
helpers in full_system_ow_working/_gs().py.

Nothing here touches a dataset, a GPU or a checkpoint, and no RANSAC solve is
ever run: every PnP result is a hand-built 4x4 pose, and
poselib.estimate_absolute_pose / poselib.refine_absolute_pose are replaced with
monkeypatches that count their calls and return the pose the test chose. The
tracker's own poselib import is the same module object this file patches, so the
spy is what the tracker actually reads. Run with:

    python -m pytest test/test_phase0_gates.py -q
or:
    python test/test_phase0_gates.py

Every test builds its own tracker with the real
GeometricTracker(GeoTrackerConfig(...)) constructor, and every Phase-0 flag is
exercised both ways so removing the corresponding implementation line makes the
test red.
"""

import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "full_system_ow_working"))

import poselib  # noqa: E402
import geometric_tracker as gt  # noqa: E402
from geometric_tracker import GeometricTracker, GeoTrackerConfig  # noqa: E402

try:
    import bundlesdf_gs  # noqa: E402
except Exception as _gs_exc:  # pragma: no cover - construction guard
    bundlesdf_gs = _gs_exc  # keeps the GS-free tests runnable without the stack


def _gs():
    """bundlesdf_gs, or a skip: the caller-side helpers live there, but pulling
    it in needs the torch/GS stack."""
    if isinstance(bundlesdf_gs, Exception):
        pytest.skip(f"BundleSdfGS needs the torch/GS stack: {bundlesdf_gs}")
    return bundlesdf_gs


# --------------------------------------------------------------------------- #
# synthetic scene constants
# --------------------------------------------------------------------------- #

W, H = 640, 480
FRAME = 7  # any index; avoids the tracker's frame-1 debug print
K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
T_GUESS = np.eye(4)
N_TRACKS = 40          # >= min_pnp_inliers(20) and >= 20 (no re-detection churn)
S = 0.01               # 10 mm / 0.01 rad innovation sigma for the chi2 tests
CHI2_LIM = float(chi2_dist.ppf(0.95, 6))   # 12.591587243743977


def _camera():
    """Black image (zero features), full mask, constant depth -- only used for
    cam_dict dimensions, for pruning, and by the <20-track re-detection path."""
    image = np.zeros((H, W, 3), dtype=np.uint8)
    mask = np.ones((H, W), dtype=np.uint8)
    depth = np.full((H, W), 3.0, dtype=np.float32)
    return image, mask, depth


def _tracker(**overrides):
    return GeometricTracker(GeoTrackerConfig(**overrides))


def _seed_tracks(tracker, n, fit_guess=True, seed=0):
    """Plant n tracks observed at FRAME.

    fit_guess=True  -> obs are the exact (float32) projection of pt3d under
                       T_GUESS, so the tracker's guess-inlier count is n: the
                       'guess fits' case, where the guess wins the guess-vs-PnP
                       vote and the jump gate is never reached.
    fit_guess=False -> obs are pinned 200+ px from every possible projection, so
                       the guess scores 0 inliers and PnP always wins that vote:
                       the 'only the gate can stop this' case.
    """
    rng = np.random.default_rng(seed)
    for tid in range(n):
        if fit_guess:
            pt3d = np.array(rng.uniform([-1.5, -1.5, 4.0], [1.5, 1.5, 8.0]))
            p = K @ pt3d
            uv = np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float32)
        else:
            pt3d = np.array([rng.uniform(-20.0, -15.0),
                             rng.uniform(-20.0, -15.0),
                             rng.uniform(5.0, 15.0)])
            uv = np.array([5.0, 5.0], dtype=np.float32)
        tracker.tracks[tid] = {"obs": {FRAME: uv}, "pt3d": pt3d}
    tracker.next_tid = n


def _tpnp(dt=(0.0, 0.0, 0.0), rotvec=(0.0, 0.0, 0.0)):
    """A synthetic PnP result, as a camera-to-world 4x4 offset from T_GUESS."""
    T = np.eye(4)
    T[:3, :3] = Rot.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    T[:3, 3] = T_GUESS[:3, 3] + np.asarray(dt, dtype=np.float64)
    return T


def _inlier_info(n_candidates, n_true):
    """Mimics poselib's real contract (probe-verified): info['inliers'] is a
    BOOLEAN mask over ALL candidates, and the true count is in info['num_inliers'].
    len(mask) is therefore the candidate count -- the Phase-0 bug being tested."""
    mask = np.zeros(n_candidates, dtype=np.bool_)
    mask[:n_true] = True
    assert mask.dtype == np.bool_ and len(mask) == n_candidates
    assert int(mask.sum()) == n_true
    if n_true != n_candidates:
        assert len(mask) != n_true, "this fixture would not exhibit the bug"
    return {"inliers": mask, "num_inliers": n_true}


class _FakePose:
    """The only two attributes the tracker reads off a poselib pose."""

    def __init__(self, q_wxyz, t):
        self.q = np.asarray(q_wxyz, dtype=np.float64)
        self.t = np.asarray(t, dtype=np.float64)


class _FakeRes:
    def __init__(self, T_Ci):
        q = Rot.from_matrix(T_Ci[:3, :3]).as_quat()  # xyzw
        self.pose = _FakePose([q[3], q[0], q[1], q[2]], T_Ci[:3, 3])


_UNSET = object()  # distinguishes "res not specified" from an explicit res=None


def _fake_pnp(T_pnp=None, res=_UNSET, info=None):
    """Calls (and their n_points) are recorded; the pose is whatever the test chose.
    Pass res=None to simulate a poselib failure."""
    calls = []

    def fn(pts2d, pts3d, cam_dict, params, initial_pose):
        calls.append(len(pts2d))
        return (_FakeRes(T_pnp) if res is _UNSET else res), info

    return fn, calls


def _fake_refine(T_refined=None):
    """Default: echo the incoming pose, so refinement is an exact no-op and the
    committed pose is exactly what step_informed_with_occlusion decided. With
    T_refined given, the caller controls the refined pose instead."""

    calls = []

    def fn(pts2d, pts3d, pose_in, cam_dict, params):
        calls.append(len(pts2d))
        if T_refined is None:
            q = np.asarray(pose_in.q, dtype=np.float64).copy()
            t = np.asarray(pose_in.t, dtype=np.float64).copy()
        else:
            q_xyzw = Rot.from_matrix(T_refined[:3, :3]).as_quat()
            q, t = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]), T_refined[:3, 3]
        return _FakePose(q, t), None

    return fn, calls


def _run_step(tracker, monkeypatch, T_pnp=None, *, res=_UNSET, info=None,
              skip_pnp=False, motion_cov=None):
    """Drive exactly one step_informed_with_occlusion with fully synthetic PnP."""
    pnp, pnp_calls = _fake_pnp(T_pnp, res=res, info=info)
    refine, refine_calls = _fake_refine()
    monkeypatch.setattr(poselib, "estimate_absolute_pose", pnp)
    monkeypatch.setattr(poselib, "refine_absolute_pose", refine)
    image, mask, depth = _camera()
    ret = tracker.step_informed_with_occlusion(
        FRAME, image, mask, depth, K, T_GUESS.copy(),
        skip_pnp=skip_pnp, motion_cov=motion_cov)
    return ret, pnp_calls, refine_calls


def _freeze_asserts(tracker, note):
    """A veto must commit T_GUESS, never the PnP result."""
    assert np.allclose(tracker.poses[FRAME], T_GUESS, atol=1e-9), (
        f"{note}: committed pose is not the frozen guess")


# --------------------------------------------------------------------------- #
# 1. honest inlier accounting
# --------------------------------------------------------------------------- #

def test_honest_inlier_count_replaces_candidate_count(monkeypatch):
    """len(info['inliers']) is the CANDIDATE count; num_inliers is the truth.
    At min_pnp_inliers=20 both numbers happen to be accepted, so this test pins
    the reported count rather than a decision."""
    info = _inlier_info(N_TRACKS, 30)

    legacy = _tracker(honest_pnp_inliers=False)
    _seed_tracks(legacy, N_TRACKS, fit_guess=False)
    ret, pnp_calls, _ = _run_step(legacy, monkeypatch, T_pnp=_tpnp(), info=info)
    assert pnp_calls == [N_TRACKS]
    assert ret == (True, N_TRACKS), (
        f"honest_pnp_inliers off must report the candidate count {N_TRACKS}, got {ret}")

    honest = _tracker(honest_pnp_inliers=True)
    _seed_tracks(honest, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(honest, monkeypatch, T_pnp=_tpnp(), info=info)
    assert ret == (True, 30), (
        f"honest_pnp_inliers on must report the true count 30, got {ret}")


def test_honest_inlier_count_changes_the_acceptance_decision(monkeypatch):
    """The consequence the counting fix exists for: 15 real inliers of 50
    candidates PASS the 20-inlier gate in legacy mode and FAIL in honest mode."""
    info = _inlier_info(N_TRACKS, 15)
    T_pnp = _tpnp(dt=(0.05, 0.0, 0.0))  # inside max_pose_jump, so only the count decides

    legacy = _tracker(honest_pnp_inliers=False)
    _seed_tracks(legacy, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(legacy, monkeypatch, T_pnp=T_pnp, info=info)
    assert ret == (True, N_TRACKS), (
        f"legacy: {len(info['inliers'])} candidates must clear min_pnp_inliers=20, got {ret}")
    assert np.allclose(legacy.poses[FRAME], T_pnp, atol=1e-9), (
        "legacy mode must have committed the PnP result")

    honest = _tracker(honest_pnp_inliers=True)
    _seed_tracks(honest, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(honest, monkeypatch, T_pnp=T_pnp, info=info)
    assert ret[0] is True, "honest_pnp_success is off here: the legacy True is expected"
    assert ret[1] == 0, (
        f"honest mode: 15 inliers < 20 must veto, and the veto reports the guess "
        f"count (0 here) instead of the discarded honest count, got {ret}")
    _freeze_asserts(honest, "honest low-inlier veto")
    assert np.linalg.norm(honest.poses[FRAME][:3, 3] - T_pnp[:3, 3]) > 0.04, (
        "the vetoed frame must not keep the PnP translation")


# --------------------------------------------------------------------------- #
# 2-3. soft PnP attempt floor
# --------------------------------------------------------------------------- #

def test_soft_floor_attempts_pnp_below_min_inliers(monkeypatch):
    """8 tracks: below min_pnp_inliers=20 but above pnp_attempt_floor=6. Off: the
    cliff -- PnP is never attempted. On: the slope -- PnP runs and the
    post-PnP inlier gate decides acceptance. Asserts the CALL COUNT, not the
    return value, so removing the attempt_floor branch turns this red."""
    info = _inlier_info(8, 8)

    off = _tracker(soft_pnp_floor=False)
    _seed_tracks(off, 8, fit_guess=True)
    ret, pnp_calls, _ = _run_step(off, monkeypatch, T_pnp=_tpnp(), info=info)
    assert pnp_calls == [], "soft_pnp_floor off must not attempt PnP below min_pnp_inliers"
    assert ret == (False, 0)
    _freeze_asserts(off, "pre-PnP bail")

    on = _tracker(soft_pnp_floor=True)
    _seed_tracks(on, 8, fit_guess=True)
    ret, pnp_calls, _ = _run_step(on, monkeypatch, T_pnp=_tpnp(), info=info)
    assert pnp_calls == [8], f"soft_pnp_floor on must attempt PnP at 8 tracks, got {pnp_calls}"
    assert ret[0] is True, f"honest_pnp_success off: expected True, got {ret}"
    # 8 < min_pnp_inliers still vetoes the PnP; the veto reports the guess count,
    # which is 8 here because the guess is a perfect fit.
    assert ret[1] == 8, f"post-PnP gate must report 8, got {ret}"
    _freeze_asserts(on, "post-PnP inlier veto")


def test_sub_floor_starvation_still_bails(monkeypatch):
    """3 tracks is below pnp_attempt_floor=6: the soft floor is a slope, not a
    hole. PnP is still skipped and the frame is refused."""
    off = _tracker(soft_pnp_floor=True)
    _seed_tracks(off, 3, fit_guess=True)
    ret, pnp_calls, _ = _run_step(off, monkeypatch, T_pnp=_tpnp(), info=_inlier_info(3, 3))
    assert pnp_calls == [], f"below the attempt floor PnP must not run, got {pnp_calls}"
    assert ret == (False, 0)
    _freeze_asserts(off, "sub-floor bail")


# --------------------------------------------------------------------------- #
# 4-6. honest success accounting
# --------------------------------------------------------------------------- #

def test_jump_veto_freezes_guess_and_honest_success_rejects(monkeypatch):
    """0.2 m PnP jump against max_pose_jump=0.1, with 40 inliers so ONLY the jump
    gate trips. The committed pose must be the frozen guess, and honest_pnp_success
    must report failure so the caller cannot bank the refined guess as a PnP
    measurement."""
    T_pnp = _tpnp(dt=(0.2, 0.0, 0.0))
    info = _inlier_info(N_TRACKS, N_TRACKS)

    legacy = _tracker(honest_pnp_success=False)
    _seed_tracks(legacy, N_TRACKS, fit_guess=False)
    ret, pnp_calls, refine_calls = _run_step(legacy, monkeypatch, T_pnp=T_pnp, info=info)
    assert pnp_calls == [N_TRACKS]
    assert refine_calls == [N_TRACKS], "the frozen guess is still refined before it is stored"
    assert ret == (True, 0), (
        f"honest_pnp_success off reports success for a vetoed frame, got {ret}")
    _freeze_asserts(legacy, "jump veto")
    assert np.linalg.norm(legacy.poses[FRAME][:3, 3] - T_pnp[:3, 3]) > 0.19, (
        "the committed pose must not be the PnP result")

    honest = _tracker(honest_pnp_success=True)
    _seed_tracks(honest, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(honest, monkeypatch, T_pnp=T_pnp, info=info)
    assert ret == (False, 0), f"honest_pnp_success on must report failure, got {ret}"
    _freeze_asserts(honest, "jump veto")


def test_low_inlier_veto_is_honest(monkeypatch):
    """20 inliers is the acceptance floor; 19 must be refused. Pose frozen either
    way, but only honest_pnp_success says so to the caller."""
    T_pnp = _tpnp()  # identical to the guess: no jump, so only the count can trip
    info = _inlier_info(N_TRACKS, 19)

    legacy = _tracker(honest_pnp_inliers=True, honest_pnp_success=False)
    _seed_tracks(legacy, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(legacy, monkeypatch, T_pnp=T_pnp, info=info)
    assert ret[0] is True, f"legacy reports success for a low-inlier veto, got {ret}"
    assert ret[1] == 0, f"veto must report the guess inlier count, got {ret}"
    _freeze_asserts(legacy, "low-inlier veto")

    honest = _tracker(honest_pnp_inliers=True, honest_pnp_success=True)
    _seed_tracks(honest, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(honest, monkeypatch, T_pnp=T_pnp, info=info)
    assert ret == (False, 0), f"honest_pnp_success must report the veto, got {ret}"
    _freeze_asserts(honest, "low-inlier veto")


def test_pnp_failure_path_is_honest(monkeypatch):
    """res is None: poselib found nothing. n_inliers stays 0 and the guess is
    committed; only honest_pnp_success reports it as a failure."""
    for honest in (False, True):
        tr = _tracker(honest_pnp_success=honest)
        _seed_tracks(tr, N_TRACKS, fit_guess=False)
        ret, pnp_calls, refine_calls = _run_step(tr, monkeypatch, res=None, info=None)
        assert pnp_calls == [N_TRACKS]
        # The legacy path refines whatever got committed: tids_pnp is passed with a
        # None mask, so refine falls back to every track observed at this frame --
        # an inlier mask that does not exist at all, not merely a suspect one.
        # honest_pnp_success=True knows the committed pose is a frozen guess, not a
        # measurement, so that refinement must not run (contract fix 2).
        assert (refine_calls == [] if honest else refine_calls == [N_TRACKS]), (
            f"honest_pnp_success={honest}: refine_calls={refine_calls}")
        _freeze_asserts(tr, "PnP failure")
        if honest:
            assert ret == (False, 0), (
                f"honest_pnp_success must report a missing PnP result as failure, got {ret}")
        else:
            assert ret == (True, 0), (
                f"legacy reports success for a missing PnP result, got {ret}")


def test_skip_pnp_returns_true_without_pnp(monkeypatch):
    """Regression guard for the pnp_ok scoping: skip_pnp returns before any PnP
    work, so an early return placed before pnp_ok is assigned would NameError and
    a widened default would report failure here."""
    for honest in (False, True):
        tr = _tracker(honest_pnp_success=honest)
        _seed_tracks(tr, N_TRACKS, fit_guess=False)
        ret, pnp_calls, refine_calls = _run_step(tr, monkeypatch, skip_pnp=True)
        assert pnp_calls == [], "skip_pnp must not call PnP"
        assert refine_calls == [], "skip_pnp must not refine"
        assert ret == (True, 0), f"skip_pnp must still return (True, 0), got {ret}"
        _freeze_asserts(tr, "skip_pnp")


# --------------------------------------------------------------------------- #
# 7-8. chi-square innovation gate
# --------------------------------------------------------------------------- #

def _motion_cov():
    return np.diag([S * S] * 3 + [S * S] * 3)


def test_chi2_gate_accepts_small_and_vetoes_large_translation(monkeypatch):
    """Hand-computed 6-dof statistic against a tight diagonal covariance
    (s_t = s_r = 0.01): 5 mm -> chi2 = 0.25, 5 cm -> chi2 = 25, against a
    df=6 95% limit of 12.591587243743977. The test computes the numbers itself
    instead of trusting the implementation, and captures the tracker's own d
    vector to prove it is laid out rotvec-first, then translation."""
    assert abs(CHI2_LIM - 12.591587243743977) < 1e-9, f"df=6 95% limit {CHI2_LIM}"
    assert abs(CHI2_LIM - 12.592) < 0.001, "documented as 12.592"

    chi2_5mm = (0.005 ** 2) / (S ** 2)
    chi2_5cm = (0.05 ** 2) / (S ** 2)
    assert abs(chi2_5mm - 0.25) < 1e-12
    assert abs(chi2_5cm - 25.0) < 1e-12
    assert chi2_5mm < CHI2_LIM < chi2_5cm

    info = _inlier_info(N_TRACKS, N_TRACKS)

    # --- accepted: 5 mm translation, rotation identical -------------------
    tr = _tracker(use_chi2_gate=True, honest_pnp_success=True)
    _seed_tracks(tr, N_TRACKS, fit_guess=False)
    orig_solve = gt.np.linalg.solve
    seen_d = []

    def spy(a, b, *args, **kwargs):
        a_, b_ = np.asarray(a), np.asarray(b)
        if a_.shape == (6, 6) and b_.shape == (6,):
            seen_d.append(b_.copy())
        return orig_solve(a, b, *args, **kwargs)

    monkeypatch.setattr(gt.np.linalg, "solve", spy)
    T_pnp = _tpnp(dt=(0.005, 0.0, 0.0))
    ret, _, _ = _run_step(tr, monkeypatch, T_pnp=T_pnp, info=info, motion_cov=_motion_cov())
    assert ret == (True, N_TRACKS), f"5 mm must be accepted at chi2={chi2_5mm}, got {ret}"
    assert np.allclose(tr.poses[FRAME], T_pnp, atol=1e-9), (
        f"5 mm must commit the PnP result, got t={tr.poses[FRAME][:3, 3]}")

    assert len(seen_d) == 1, f"the chi2 branch must have built exactly one 6-dof d, got {len(seen_d)}"
    d_hand = np.concatenate([Rot.from_matrix(T_GUESS[:3, :3].T @ T_pnp[:3, :3]).as_rotvec(),
                             T_pnp[:3, 3] - T_GUESS[:3, 3]])
    np.testing.assert_allclose(seen_d[0], d_hand, atol=1e-12)
    assert abs(float(seen_d[0] @ np.linalg.solve(_motion_cov(), seen_d[0])) - chi2_5mm) < 1e-12, (
        "tracker's d must reproduce the hand-computed chi2 for a translation-only step")

    # --- vetoed: 5 cm translation -----------------------------------------
    tr = _tracker(use_chi2_gate=True, honest_pnp_success=True)
    _seed_tracks(tr, N_TRACKS, fit_guess=False)
    T_pnp = _tpnp(dt=(0.05, 0.0, 0.0))
    ret, _, _ = _run_step(tr, monkeypatch, T_pnp=T_pnp, info=info, motion_cov=_motion_cov())
    assert ret == (False, 0), f"5 cm must be vetoed at chi2={chi2_5cm}, got {ret}"
    _freeze_asserts(tr, "chi2 veto")


def test_chi2_gate_treats_rotation_the_same_way(monkeypatch):
    """Rotation lives in the first three dofs of d: 0.005 rad -> chi2=0.25 is
    accepted, 0.05 rad -> chi2=25 is vetoed. Same magnitudes as the translation
    case above, same decision, so the gate is not translation-only."""
    info = _inlier_info(N_TRACKS, N_TRACKS)
    rot_5mm = float(np.linalg.norm(Rot.from_rotvec([0.0, 0.0, 0.005]).as_rotvec()))
    rot_5cm = float(np.linalg.norm(Rot.from_rotvec([0.0, 0.0, 0.05]).as_rotvec()))
    assert abs(rot_5mm ** 2 / S ** 2 - 0.25) < 1e-12
    assert abs(rot_5cm ** 2 / S ** 2 - 25.0) < 1e-12

    tr = _tracker(use_chi2_gate=True, honest_pnp_success=True)
    _seed_tracks(tr, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(tr, monkeypatch, T_pnp=_tpnp(rotvec=(0.0, 0.0, 0.005)),
                          info=info, motion_cov=_motion_cov())
    assert ret == (True, N_TRACKS), f"0.005 rad must be accepted, got {ret}"

    tr = _tracker(use_chi2_gate=True, honest_pnp_success=True)
    _seed_tracks(tr, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(tr, monkeypatch, T_pnp=_tpnp(rotvec=(0.0, 0.0, 0.05)),
                          info=info, motion_cov=_motion_cov())
    assert ret == (False, 0), f"0.05 rad must be vetoed, got {ret}"
    _freeze_asserts(tr, "rotation chi2 veto")


def test_abs_jump_floor_fires_despite_a_huge_covariance(monkeypatch):
    """motion_cov = 1e6*I makes chi2 = 0.6^2/1e6 = 3.6e-7, so no chi2 could ever
    trip. abs_jump_floor (0.5 m) is a hard ceiling that fires regardless."""
    info = _inlier_info(N_TRACKS, N_TRACKS)
    cov = np.eye(6) * 1e6
    chi2 = 0.6 ** 2 / 1e6
    assert chi2 < 1e-6 < CHI2_LIM

    tr = _tracker(use_chi2_gate=True, honest_pnp_success=True)
    _seed_tracks(tr, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(tr, monkeypatch, T_pnp=_tpnp(dt=(0.6, 0.0, 0.0)),
                          info=info, motion_cov=cov)
    assert ret == (False, 0), f"0.6 m must be vetoed by abs_jump_floor, got {ret}"
    _freeze_asserts(tr, "abs_jump_floor veto")

    # Just under the floor the chi2 must decide: 0.4 m at 1e6*I is accepted.
    tr = _tracker(use_chi2_gate=True, honest_pnp_success=True)
    _seed_tracks(tr, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(tr, monkeypatch, T_pnp=_tpnp(dt=(0.4, 0.0, 0.0)),
                          info=info, motion_cov=cov)
    assert ret == (True, N_TRACKS), f"0.4 m must clear abs_jump_floor, got {ret}"


def test_chi2_gate_falls_back_to_constant_jump_without_covariance(monkeypatch):
    """motion_cov=None with use_chi2_gate on is the pre-history state
    (chi2_min_history not yet reached): it must take the legacy constant-jump
    branch rather than crash."""
    info = _inlier_info(N_TRACKS, N_TRACKS)

    tr = _tracker(use_chi2_gate=True, honest_pnp_success=True, max_pose_jump=0.1)
    _seed_tracks(tr, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(tr, monkeypatch, T_pnp=_tpnp(dt=(0.2, 0.0, 0.0)), info=info)
    assert ret == (False, 0), f"0.2 m must trip max_pose_jump=0.1, got {ret}"
    _freeze_asserts(tr, "legacy jump veto")

    tr = _tracker(use_chi2_gate=True, honest_pnp_success=True, max_pose_jump=0.1)
    _seed_tracks(tr, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(tr, monkeypatch, T_pnp=_tpnp(dt=(0.05, 0.0, 0.0)), info=info)
    assert ret == (True, N_TRACKS), f"0.05 m must pass max_pose_jump=0.1, got {ret}"


def test_guess_preference_bypasses_the_innovation_gate(monkeypatch):
    """DOCUMENTED GAP (not fixed here): the chi2 gate lives inside the
    'PnP beat the guess' branch. When the guess already fits all points within
    ransac_thresh, n_inliers_guess > 0.95 * n_inliers short-circuits to the
    guess and the innovation gate is never evaluated -- a frame whose guess is
    self-consistent can therefore clear it at any magnitude, and the caller-side
    gate in bundlesdf_gs sees a zero innovation because the committed pose IS the
    guess. Here the PnP result is 0.6 m away (chi2 = 3600 >> 12.59) and still
    accepted."""
    cov = _motion_cov()
    info = _inlier_info(N_TRACKS, N_TRACKS)
    T_pnp = _tpnp(dt=(0.6, 0.0, 0.0))
    chi2_that_never_gets_tested = 0.6 ** 2 / (S ** 2)
    assert chi2_that_never_gets_tested > CHI2_LIM

    tr = _tracker(use_chi2_gate=True, honest_pnp_success=True)
    _seed_tracks(tr, N_TRACKS, fit_guess=True)  # the guess is a perfect fit
    ret, _, _ = _run_step(tr, monkeypatch, T_pnp=T_pnp, info=info, motion_cov=cov)
    assert ret == (True, N_TRACKS), "current behaviour: accepted despite a 3600 chi2 PnP"
    assert np.allclose(tr.poses[FRAME], T_GUESS, atol=1e-9), (
        "the guess is committed, not the PnP result")


# --------------------------------------------------------------------------- #
# 10. refine_pose degenerate-set guard and keyword-only W/H
# --------------------------------------------------------------------------- #

def test_refine_is_skipped_on_a_vetoed_pnp(monkeypatch):
    """A vetoed PnP commits the FROZEN guess, so refining it would re-solve that
    guess with the PnP's inlier mask and overwrite the freeze: the "frozen
    guess" was never actually frozen. honest_pnp_success=True makes the tracker
    know the pose is not a measurement, so refine must not run at all. Off, the
    legacy behaviour (refine whatever got committed) is preserved."""
    T_pnp = _tpnp(dt=(0.2, 0.0, 0.0))  # trips max_pose_jump=0.1
    info = _inlier_info(N_TRACKS, N_TRACKS)

    honest = _tracker(honest_pnp_success=True)
    _seed_tracks(honest, N_TRACKS, fit_guess=False)
    ret, _, refine_calls = _run_step(honest, monkeypatch, T_pnp=T_pnp, info=info)
    assert ret == (False, 0)
    assert refine_calls == [], (
        f"a vetoed PnP must not be refined, got {refine_calls} refine call(s)")
    _freeze_asserts(honest, "vetoed refine skip")

    legacy = _tracker(honest_pnp_success=False)
    _seed_tracks(legacy, N_TRACKS, fit_guess=False)
    ret, _, refine_calls = _run_step(legacy, monkeypatch, T_pnp=T_pnp, info=info)
    assert refine_calls == [N_TRACKS], (
        "honest_pnp_success off must still refine the committed pose")
    _freeze_asserts(legacy, "legacy refine")


def test_refine_still_runs_on_an_accepted_pnp(monkeypatch):
    """The veto skip must not suppress refinement of a pose the PnP did produce:
    a small, in-gate PnP result must still be refined."""
    T_pnp = _tpnp(dt=(0.005, 0.0, 0.0))  # well inside every threshold
    info = _inlier_info(N_TRACKS, N_TRACKS)
    T_refined = _tpnp(dt=(0.005, 0.0, 0.0), rotvec=(0.0, 0.0, 0.05))

    tr = _tracker(honest_pnp_success=True, honest_pnp_inliers=True)
    _seed_tracks(tr, N_TRACKS, fit_guess=False)
    pnp, pnp_calls = _fake_pnp(T_pnp, info=info)
    refine, refine_calls = _fake_refine(T_refined)
    monkeypatch.setattr(poselib, "estimate_absolute_pose", pnp)
    monkeypatch.setattr(poselib, "refine_absolute_pose", refine)
    image, mask, depth = _camera()
    ret = tr.step_informed_with_occlusion(FRAME, image, mask, depth, K,
                                          T_GUESS.copy(), motion_cov=None)
    assert pnp_calls == [N_TRACKS]
    assert ret[0] is True, f"an in-gate PnP must be accepted, got {ret}"
    assert refine_calls == [N_TRACKS], (
        f"an accepted PnP must be refined once, got {refine_calls}")
    np.testing.assert_allclose(tr.poses[FRAME], T_refined, atol=1e-9)


def test_refine_pose_skips_degenerate_inlier_sets(monkeypatch):
    """0-5 inliers must leave the stored pose bit-for-bit untouched and must not
    call poselib.refine_absolute_pose."""
    tr = _tracker()
    _seed_tracks(tr, 20, fit_guess=True)
    tr.poses[FRAME] = T_GUESS.copy()
    T_before = tr.poses[FRAME].copy()
    tids = np.arange(20)

    for n_sel in (0, 1, 3, 5):
        refine, calls = _fake_refine()
        monkeypatch.setattr(poselib, "refine_absolute_pose", refine)
        mask = np.zeros(20, dtype=np.bool_)
        mask[:n_sel] = True
        T_snap = tr.poses[FRAME].copy()
        tr.refine_pose(FRAME, K, tids_pnp=tids, inlier_mask=mask, W=W, H=H)
        assert calls == [], f"{n_sel} inliers must not reach poselib, got {calls}"
        assert np.array_equal(tr.poses[FRAME], T_snap), (
            f"{n_sel} inliers must leave the pose bit-identical")
    assert np.array_equal(tr.poses[FRAME], T_before)


def test_refine_pose_runs_at_six_and_requires_W_H(monkeypatch):
    """>= 6 inliers must actually refine (and move the pose), and W/H must now be
    keyword-only required arguments."""
    T_refined = _tpnp(dt=(0.0, 0.0, 0.02), rotvec=(0.0, 0.0, 0.1))
    tr = _tracker()
    _seed_tracks(tr, 20, fit_guess=True)
    tr.poses[FRAME] = T_GUESS.copy()
    T_before = tr.poses[FRAME].copy()
    tids = np.arange(20)

    refine, calls = _fake_refine(T_refined)
    monkeypatch.setattr(poselib, "refine_absolute_pose", refine)
    mask = np.zeros(20, dtype=np.bool_)
    mask[:8] = True
    tr.refine_pose(FRAME, K, tids_pnp=tids, inlier_mask=mask, W=W, H=H)
    assert calls == [8], f"refine must be called once with the 8 inliers, got {calls}"
    assert np.allclose(tr.poses[FRAME], T_refined, atol=1e-9), "refine must commit its result"
    assert not np.array_equal(tr.poses[FRAME], T_before), "the pose must have moved"

    with pytest.raises(TypeError):
        tr.refine_pose(FRAME, K, tids_pnp=tids, inlier_mask=np.ones(20, np.bool_))


def test_refine_pose_guard_covers_the_no_mask_path(monkeypatch):
    """With no inlier mask the fallback path uses every track observed at the
    frame; fewer than 6 of those is also degenerate."""
    tr = _tracker()
    _seed_tracks(tr, 4, fit_guess=True)
    tr.poses[FRAME] = T_GUESS.copy()
    T_before = tr.poses[FRAME].copy()

    refine, calls = _fake_refine()
    monkeypatch.setattr(poselib, "refine_absolute_pose", refine)
    tr.refine_pose(FRAME, K, W=W, H=H)
    assert calls == [], f"4 observations must not reach poselib, got {calls}"
    assert np.array_equal(tr.poses[FRAME], T_before)


# --------------------------------------------------------------------------- #
# 11. caller-side covariance helper
# --------------------------------------------------------------------------- #

def test_caller_side_motion_covariance(monkeypatch):
    """_motion_covariance: floors, the 3*scale*median rule, the rot-first 6x6
    diagonal, the chi2_min_history gate, and the 50-step window."""
    try:
        import bundlesdf_gs
    except Exception as exc:  # pragma: no cover - construction guard
        pytest.skip(f"BundleSdfGS needs the torch/GS stack: {exc}")

    def build():
        cfg = _gs().GeoTrackerConfig(
            chi2_min_history=5, motion_sigma_floor_t=0.005,
            motion_sigma_floor_r_deg=0.5, motion_sigma_scale=3.0)
        return _gs().BundleSdfGS(cfg, _gs().MappingConfig(),
                                        use_multiprocessing=False)

    bs = build()
    assert bs._motion_covariance() is None, "empty history must not yield a covariance"
    for _ in range(4):
        bs.accepted_step_t.append(0.01)
        bs.accepted_step_r.append(1.0)
    assert bs._motion_covariance() is None, "4 < chi2_min_history=5 must stay None"

    bs.accepted_step_t.append(0.01)
    bs.accepted_step_r.append(1.0)
    s_t = max(0.005, 3.0 * float(np.median(bs.accepted_step_t)))     # 0.03
    s_r = np.deg2rad(max(0.5, 3.0 * float(np.median(bs.accepted_step_r))))  # deg2rad(3)
    assert abs(s_t - 0.03) < 1e-12 and abs(s_r - np.deg2rad(3.0)) < 1e-12
    cov = bs._motion_covariance()
    expected = np.diag([s_r ** 2] * 3 + [s_t ** 2] * 3)
    np.testing.assert_allclose(cov, expected, atol=1e-15)
    assert cov.shape == (6, 6)
    assert np.allclose(cov[:3, :3], s_r ** 2 * np.eye(3)), "rotations must come first"
    assert np.allclose(cov[3:, 3:], s_t ** 2 * np.eye(3)), "translations must come last"

    # Observed motion far below the floors: the floors win.
    bs2 = build()
    for _ in range(5):
        bs2.accepted_step_t.append(1e-6)
        bs2.accepted_step_r.append(1e-3)
    cov2 = bs2._motion_covariance()
    np.testing.assert_allclose(
        cov2, np.diag([np.deg2rad(0.5) ** 2] * 3 + [0.005 ** 2] * 3), atol=1e-18)

    # The 50-step window: a burst of big steps must age out.
    bs3 = build()
    for _ in range(5):
        bs3.accepted_step_t.append(0.01)
        bs3.accepted_step_r.append(1.0)
    assert abs(bs3._motion_covariance()[3, 3] - s_t ** 2) < 1e-15
    for _ in range(45):
        bs3.accepted_step_t.append(100.0)
        bs3.accepted_step_r.append(100.0)
    assert abs(bs3._motion_covariance()[3, 3] - (3.0 * 100.0) ** 2) < 1e-6
    for _ in range(45):
        bs3.accepted_step_t.append(0.01)
        bs3.accepted_step_r.append(1.0)
    assert abs(bs3._motion_covariance()[3, 3] - s_t ** 2) < 1e-15, (
        "the 50-step window must forget the 100 m steps")


def test_caller_side_innovation_chi2_layout(monkeypatch):
    """_innovation_chi2 must score d = [rotvec(R_a^T R_b) (rad), t_b - t_a (m)]
    against the same diagonal the tracker's gate uses. Mixing the rotation and
    translation blocks up gives a completely different number, so this catches a
    layout mismatch between caller and tracker."""
    try:
        import bundlesdf_gs
    except Exception as exc:  # pragma: no cover - construction guard
        pytest.skip(f"BundleSdfGS needs the torch/GS stack: {exc}")

    bs = _gs().BundleSdfGS(
        _gs().GeoTrackerConfig(), _gs().MappingConfig(),
        use_multiprocessing=False)

    T_a = np.eye(4)
    T_b = np.eye(4)
    T_b[:3, :3] = Rot.from_rotvec([0.0, 0.0, 0.2]).as_matrix()
    T_b[:3, 3] = [0.01, 0.02, 0.03]
    cov = np.diag([1.0, 1.0, 1.0, 1e-4, 1e-4, 1e-4])

    rot_sq = 0.04
    tr_sq = 1e-4 + 4e-4 + 9e-4
    expected = rot_sq / 1.0 + tr_sq / 1e-4      # 14.04
    swapped = tr_sq / 1.0 + rot_sq / 1e-4       # 400.0014

    got = bs._innovation_chi2(T_a, T_b, cov)
    assert abs(got - expected) < 1e-9, f"rot-first chi2 {got} != {expected}"
    assert abs(got - swapped) > 100.0, (
        f"the rotation and translation blocks must not be swapped: "
        f"got {got}, rot-first {expected}, swapped {swapped}")
    assert abs(bs._innovation_chi2(T_a, T_a, cov)) < 1e-15, "zero motion must score zero"

    # Symmetry of the statistic, and an independent hand-computation of the
    # tracker's own expression for the same pair of poses.
    d = np.concatenate([Rot.from_matrix(T_a[:3, :3].T @ T_b[:3, :3]).as_rotvec(),
                        T_b[:3, 3] - T_a[:3, 3]])
    assert abs(got - float(d @ np.linalg.solve(cov, d))) < 1e-9


# --------------------------------------------------------------------------- #
# 12. carrying the tracker's own verdict to the caller
# --------------------------------------------------------------------------- #

def test_tracker_veto_is_invisible_to_the_caller_gate(monkeypatch):
    """The double-gate blind spot this carry fixes: the tracker gates the RAW
    PnP result against T_guess; the caller gates the COMMITTED pose. After an
    internal veto the committed pose IS T_guess, so the caller's chi2 and jump
    are both exactly 0 and would ACCEPT the frame. Asserting the 0.0 here pins
    the very asymmetry that makes the carry necessary."""
    cov = _motion_cov()
    T_pnp = _tpnp(dt=(0.05, 0.0, 0.0))  # chi2 = 25 >> 12.59
    info = _inlier_info(N_TRACKS, N_TRACKS)

    tr = _tracker(use_chi2_gate=True, honest_pnp_success=True)
    _seed_tracks(tr, N_TRACKS, fit_guess=False)
    ret, _, refine_calls = _run_step(tr, monkeypatch, T_pnp=T_pnp, info=info, motion_cov=cov)
    assert ret == (False, 0), f"the chi2 veto must be reported as failure, got {ret}"
    _freeze_asserts(tr, "caller-blind chi2 veto")

    gate = tr.last_pnp_gate
    assert gate["vetoed"] is True
    assert gate["chi2"] > CHI2_LIM

    committed = tr.poses[FRAME]
    d = np.concatenate([Rot.from_matrix(T_GUESS[:3, :3].T @ committed[:3, :3]).as_rotvec(),
                        committed[:3, 3] - T_GUESS[:3, 3]])
    naive_chi2 = float(d @ np.linalg.solve(cov, d))
    naive_jump = float(np.linalg.norm(committed[:3, 3] - T_GUESS[:3, 3]))
    assert naive_chi2 < 1e-12, f"the caller must see zero innovation, got {naive_chi2}"
    assert naive_jump < 1e-12, f"the caller must see zero jump, got {naive_jump}"

    def _caller_gate(tracker):
        """The BundleSdfGS Stage-2 test, including the carry."""
        chi2 = float(d @ np.linalg.solve(cov, d))
        gate_tripped = chi2 > CHI2_LIM or naive_jump > 0.5
        carried, reason = _gs().BundleSdfGS._pnp_gate_carry(tracker)
        if carried:
            gate_tripped = True
        return gate_tripped, reason

    gate_tripped, reason = _caller_gate(tr)
    assert gate_tripped is True, "the carried veto must reject the frame"
    assert reason is not None and "inside the tracker" in reason
    assert refine_calls == [], "a vetoed PnP must not be refined either"


def test_pnp_gate_carry_is_idle_without_a_veto(monkeypatch):
    """An accepted (in-gate) PnP records chi2 but no veto: the carry must be a
    no-op there, or every accepted frame would be rejected."""
    tr = _tracker(use_chi2_gate=True, honest_pnp_success=True)
    _seed_tracks(tr, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(tr, monkeypatch, T_pnp=_tpnp(dt=(0.005, 0.0, 0.0)),
                          info=_inlier_info(N_TRACKS, N_TRACKS), motion_cov=_motion_cov())
    assert ret[0] is True
    assert tr.last_pnp_gate["chi2"] < CHI2_LIM
    np.testing.assert_allclose(tr.poses[FRAME], _tpnp(dt=(0.005, 0.0, 0.0)), atol=1e-6)
    carried, reason = _gs().BundleSdfGS._pnp_gate_carry(tr)
    assert carried is False and reason is None

    # A tracker without chi2 bookkeeping at all must not raise.
    bare = _tracker()
    assert _gs().BundleSdfGS._pnp_gate_carry(bare) == (False, None)


def test_caller_gate_carries_a_legacy_veto_too(monkeypatch):
    """The legacy constant-jump branch sets vetoed=True with chi2 == 0.0 (never
    computed: motion_cov was None). The carry must still read `vetoed`, not
    `chi2`, or these frames stay invisible -- they are equally invisible to a
    chi2-based caller test, since the committed pose is the guess again."""
    info = _inlier_info(N_TRACKS, N_TRACKS)
    tr = _tracker(use_chi2_gate=True, honest_pnp_success=True, max_pose_jump=0.1)
    _seed_tracks(tr, N_TRACKS, fit_guess=False)
    ret, _, _ = _run_step(tr, monkeypatch, T_pnp=_tpnp(dt=(0.2, 0.0, 0.0)), info=info)
    assert ret == (False, 0)
    assert tr.last_pnp_gate["chi2"] == 0.0, "the legacy branch never computes chi2"
    assert tr.last_pnp_gate["vetoed"] is True
    carried, reason = _gs().BundleSdfGS._pnp_gate_carry(tr)
    assert carried is True, "a legacy jump veto is still a veto"
    assert "inside the tracker" in reason


# --------------------------------------------------------------------------- #
# 13. gap-normalised motion covariance
# --------------------------------------------------------------------------- #

def test_accepted_step_gaps_alignment_and_fallback():
    """accepted_step_t[k] is the displacement from accepted_rel[k] to rel[k+1]:
    the last len(steps) pairs of consecutive rel entries, clipped at 1 because a
    duplicated frame index would zero out the normalisation."""
    # 3 step records, 4 rel entries: the record count is one below, and the
    # requested 2 steps are the LAST 2 pairs.
    rel = [0, 3, 4, 9]
    gaps = _gs().BundleSdfGS._accepted_step_gaps(2, rel)
    np.testing.assert_allclose(gaps, [1.0, 5.0])

    # An exact fit still resolves: 3 rel entries are 2 records, enough for 2 steps.
    np.testing.assert_allclose(_gs().BundleSdfGS._accepted_step_gaps(2, rel[:-1]),
                               [3.0, 1.0])
    # Fewer RECORDS than the requested history (not entries): the caller must fall
    # back to the raw medians rather than guess at the alignment.
    assert _gs().BundleSdfGS._accepted_step_gaps(2, rel[:-2]) is None
    assert _gs().BundleSdfGS._accepted_step_gaps(5, rel) is None

    # A duplicated frame index is clipped to 1, never 0 (no division by zero).
    np.testing.assert_allclose(_gs().BundleSdfGS._accepted_step_gaps(1, [7, 7]), [1.0])


def test_motion_covariance_normalises_by_the_frame_gap():
    """The statistics are per ACCEPTED step, not per frame, while the innovation
    the gate tests is a ONE-frame increment. One 5 cm step across 1 frame and
    one across 5 frames must give sigmas of 0.05 and 0.01 m, not 0.05 twice."""
    def build():
        return _gs().BundleSdfGS(
            _gs().GeoTrackerConfig(chi2_min_history=1, motion_sigma_floor_t=1e-12,
                                   motion_sigma_floor_r_deg=1e-12, motion_sigma_scale=1.0),
            _gs().MappingConfig(), use_multiprocessing=False)

    bs = build()
    for gap in (1, 2, 5):
        bs.accepted_step_t.clear(); bs.accepted_step_r.clear(); bs.accepted_rel.clear()
        bs.accepted_rel.append(0)
        bs.accepted_step_t.append(0.05)
        bs.accepted_step_r.append(5.0)
        bs.accepted_rel.append(gap)
        cov = bs._motion_covariance()
        np.testing.assert_allclose(cov[3:, 3:], (0.05 / gap) ** 2 * np.eye(3), atol=1e-22)
        np.testing.assert_allclose(cov[:3, :3], (np.deg2rad(5.0 / gap)) ** 2 * np.eye(3), atol=1e-26)

    # The direction the fix claims: an identical accepted-step history spread
    # over more frames must give a TIGHTER sigma, not a looser one.
    bs2 = build()
    for frame_gaps, expected_t in (([1, 1, 1, 1, 1], 0.05), ([5, 5, 5, 5, 5], 0.01)):
        bs2.accepted_step_t.clear(); bs2.accepted_step_r.clear(); bs2.accepted_rel.clear()
        bs2.accepted_rel.append(0)
        nxt = 0
        for g in frame_gaps:
            nxt += g
            bs2.accepted_step_t.append(0.05)
            bs2.accepted_step_r.append(5.0)
            bs2.accepted_rel.append(nxt)
        cov = bs2._motion_covariance()
        assert abs(cov[3, 3] - expected_t ** 2) < 1e-18, (
            f"gaps={frame_gaps}: sigma_t must be {expected_t}, got {cov[3, 3] ** 0.5}")


def test_motion_covariance_falls_back_when_alignment_broken():
    """If the accepted_rel bookkeeping ever gets out of step with the step
    records, the raw medians must be used rather than the wrong gaps."""
    bs = _gs().BundleSdfGS(
        _gs().GeoTrackerConfig(chi2_min_history=2, motion_sigma_floor_t=1e-9,
                                      motion_sigma_floor_r_deg=1e-9, motion_sigma_scale=1.0),
        _gs().MappingConfig(), use_multiprocessing=False)
    for _ in range(5):
        bs.accepted_step_t.append(0.05)
        bs.accepted_step_r.append(5.0)
    # Only one rel entry: 0 records, so the gaps helper refuses.
    bs.accepted_rel.append(0)
    cov = bs._motion_covariance()
    np.testing.assert_allclose(cov[3:, 3:], 0.05 ** 2 * np.eye(3), atol=1e-18)


# --------------------------------------------------------------------------- #
# 14. ba._declined: the pre-solve refusals carry the full key set
# --------------------------------------------------------------------------- #

def test_declined_alignment_returns_carry_the_full_key_set():
    """align_to_object_frame's two pre-solve refusals used to return only
    {'accepted': False} while ObjGSCameraMapper.densify reads res['s'] BEFORE
    checking res['accepted'] -- a KeyError instead of a graceful skip."""
    import ba
    keys = {"T_to_object", "log_s", "s", "accepted", "initial_cost", "final_cost", "reason"}

    cases = [
        (np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
         np.array([[1.0, 1.0, 1.0]], dtype=np.float64), "too few"),
        (np.zeros((10, 3), dtype=np.float64),
         np.ones((10, 3), dtype=np.float64), "zero spread"),
    ]
    for pts_tracker, pts_object, why in cases:
        out = ba.align_to_object_frame(pts_tracker, pts_object)
        missing = keys - set(out)
        assert not missing, f"{why}: the refused result is missing keys {missing}"
        assert out["accepted"] is False
        assert out["s"] == 1.0, "a refusal must mean no scale correction"
        assert out["log_s"] == 0.0
        np.testing.assert_allclose(out["T_to_object"], np.eye(4))
        assert out["reason"], f"{why}: the refusal must explain itself"


# --------------------------------------------------------------------------- #
# the chi2 knobs must be reachable from the runners, not just from the tracker
# --------------------------------------------------------------------------- #

RUNNER_FILES = [
    "run_full_system_bundlegs_ow.py",
    "run_full_system_bundlegs_ow_3dgs.py",
    "run_full_system_bundlegs_ow_dense_densification.py",
]
# The gating flags must be in here too: they are the ones an operator flips, so
# the test is worthless if deleting a forwarding line or defaulting one to True
# both come back green.
TRACKER_KNOBS = ("honest_pnp_inliers", "honest_pnp_success", "soft_pnp_floor",
                 "min_pnp_inliers", "pnp_attempt_floor", "use_chi2_gate",
                 "chi2_quantile", "chi2_min_history", "abs_jump_floor",
                 "motion_sigma_floor_t", "motion_sigma_floor_r_deg",
                 "motion_sigma_scale", "debug",
                 # Bundle adjustment (ba.py). run_ba_on_keyframe was already wired,
                 # but not these two: enabling BA left the window size and the log
                 # level pinned at GeoTrackerConfig's values, so the operator had
                 # no dial for either.
                 "ba_max_keyframes", "ba_verbose",
                 # The triangulate call BA's pre-step reads from tracker_cfg.
                 "triangulate", "triangulate_thresh",
                 # P0-5 and the GS<->tracker point sync. Both are live reads that
                 # no runner forwarded.
                 "occlusion_margin", "update_tracker_points_from_gs")


def _class_field_defaults(path, class_name):
    """{field: default} for a dataclass, read with ast.

    The runner modules pull dataset loaders and rerun servers at import time, so
    parsing them is the only way to ask what they declare without booting the
    pipeline. Defaults that are not literals (a name, a call) cannot be read this
    way and are dropped -- the caller must treat a missing key as "unreadable",
    not as "undeclared".
    """
    import ast
    with open(path) as fh:
        tree = ast.parse(fh.read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    try:
                        out[stmt.target.id] = ast.literal_eval(stmt.value)
                    except (ValueError, TypeError):
                        pass
    return out


def _class_names(path, class_name):
    """Names of every class the file defines that is called class_name."""
    import ast
    with open(path) as fh:
        tree = ast.parse(fh.read())
    return [n.name for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef) and n.name == class_name]


def _geo_tracker_config_keywords(path):
    """{kwarg name} passed at every `GeoTrackerConfig(...)` call site.

    A substring search for "cfg.<knob>" would be satisfied by a comment, so the
    forward check has to ask the call itself which keywords it actually sends.
    Names only -- a hardcoded constant reaches the tracker too. Use
    _geo_tracker_config_cfg_sourced for the value as well.
    """
    import ast
    with open(path) as fh:
        tree = ast.parse(fh.read())
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "GeoTrackerConfig":
            continue
        for kw in node.keywords:
            if kw.arg:
                out.add(kw.arg)
    return out


def _geo_tracker_config_cfg_sourced(path):
    """{kwarg name} whose value at a `GeoTrackerConfig(...)` call site is
    literally `cfg.<same name>`.

    A knob can be forwarded and still be inert: `pnp_attempt_floor=99` reaches
    the tracker but no longer reads the CLI value, so the operator's dial does
    nothing. Comparing the GlobalConfig default against the tracker default
    cannot catch that either, because the mismatch is at the call site. Only the
    value's form can, and a keyword NAME check passes `99` through.
    """
    import ast
    with open(path) as fh:
        tree = ast.parse(fh.read())
    out = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = getattr(func, "id", None) or getattr(func, "attr", None)
        if name != "GeoTrackerConfig":
            continue
        for kw in node.keywords:
            v = kw.value
            if (isinstance(v, ast.Attribute) and v.attr == kw.arg
                    and isinstance(v.value, ast.Name) and v.value.id == "cfg"):
                out.add(kw.arg)
    return out


@pytest.mark.parametrize("runner", RUNNER_FILES)
def test_tracker_knobs_are_reachable_from_every_runner(runner):
    """Each knob must reach the tracker AND default to the value GeoTrackerConfig
    already ships.

    The first part is a wiring bug that used to be live: the knobs existed on
    GeoTrackerConfig and the docs told the operator to tune them, but no runner
    forwarded them, so `use_chi2_gate=True` turned the gate on fully hardcoded.
    `min_pnp_inliers` was the same bug one step narrower: it had always been on
    GeoTrackerConfig at 20, so geometric_tracker's `getattr(cfg, ..., 20)`
    fallback never fired -- but no runner declared or forwarded it either, so
    there was still no dial. That mattered because soft_pnp_floor softens only
    the pre-PnP ATTEMPT floor, so PnP could run on 6 points and still be vetoed
    by this acceptance gate no matter what the operator set.
    The second is the discipline that keeps the paired 5-clip A/B baselines
    reproducible -- flipping use_chi2_gate alone must be bit-identical to before,
    which only holds if the runner defaults equal the tracker defaults.
    """
    import dataclasses

    cfgs = _gs()
    # f.default raises ValueError on a default_factory field, so collect the
    # literal defaults only -- a factory knob simply cannot be read this way.
    tracker_defaults = {f.name: f.default
                        for f in dataclasses.fields(cfgs.GeoTrackerConfig)
                        if f.default_factory is dataclasses.MISSING}
    path = os.path.join(os.path.dirname(__file__), os.pardir,
                        "full_system_ow_working", runner)
    # _class_field_defaults merges every same-named class into one dict, so a
    # second GlobalConfig could satisfy this test while the one actually used is
    # ignored. Name it explicitly first.
    gc = _class_names(path, "GlobalConfig")
    assert len(gc) == 1, (
        f"{runner}: expected exactly one GlobalConfig, found {gc} -- this test "
        f"cannot tell which one it read")
    declared = _class_field_defaults(path, "GlobalConfig")
    forwarded = _geo_tracker_config_keywords(path)
    cfg_sourced = _geo_tracker_config_cfg_sourced(path)

    for knob in TRACKER_KNOBS:
        assert knob in tracker_defaults, (
            f"{runner}: tracker knob {knob} is no longer on GeoTrackerConfig -- "
            f"is it now a dead test entry?")
        if knob not in declared:
            pytest.fail(
                f"{runner}: GlobalConfig does not declare {knob} (or its default "
                f"is not a literal, which this test cannot read; class definitions "
                f"found: {_class_names(path, 'GlobalConfig')})")
        assert declared[knob] == tracker_defaults[knob], (
            f"{runner}: GlobalConfig.{knob}={declared[knob]!r} drifted from "
            f"GeoTrackerConfig.{knob}={tracker_defaults[knob]!r}")
        assert knob in forwarded, (
            f"{runner}: {knob} is declared in GlobalConfig but never forwarded "
            f"to GeoTrackerConfig")
        assert knob in cfg_sourced, (
            f"{runner}: {knob} reaches GeoTrackerConfig but not as cfg.{knob} -- "
            f"a hardcoded value at the call site leaves the CLI knob inert")


def test_geo_tracker_config_declares_no_field_twice():
    """A repeated annotated field in a dataclass body is silently shadowed, not an
    error: dataclasses builds fields from cls.__annotations__, which is a dict and
    therefore dedupes. So `min_pnp_inliers: int = 20` written twice in
    GeoTrackerConfig constructed fine and reported one field, leaving the first
    declaration (and its comment) dead. dataclasses.fields() cannot reveal this --
    it has already deduped -- so the check has to read the class body itself.
    """
    import ast

    path = os.path.join(os.path.dirname(__file__), os.pardir,
                        "full_system_ow_working", "bundlesdf_gs.py")
    with open(path) as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "GeoTrackerConfig":
            targets = [s.target.id for s in node.body
                       if isinstance(s, ast.AnnAssign)
                       and isinstance(s.target, ast.Name)]
            dups = sorted({t for t in targets if targets.count(t) > 1})
            assert not dups, (
                f"GeoTrackerConfig declares {dups} twice; dataclasses shadows the "
                f"second silently, so the earlier declaration is dead code")
