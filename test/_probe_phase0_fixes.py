"""Temporary probe: verifies the Phase-0 gate fixes land where they should.

Run:  python test/_probe_phase0_fixes.py
Not a pytest module on purpose -- it monkeypatches poselib and prints verdicts.
"""
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "full_system_ow_working"))

import poselib  # noqa: E402
import ba  # noqa: E402
import geometric_tracker as gt  # noqa: E402
import bundlesdf_gs  # noqa: E402  pulls the torch/GS stack; needed for _pnp_gate_carry

BundleSdfGS = bundlesdf_gs.BundleSdfGS
from scipy.spatial.transform import Rotation as R  # noqa: E402
from scipy.stats import chi2 as chi2_dist  # noqa: E402

W, H = 640, 480
F, CX, CY = 500.0, 320.0, 240.0
K = np.array([[F, 0.0, CX], [0.0, F, CY], [0.0, 0.0, 1.0]], dtype=np.float32)

FRAME = 5
N = 30
rng = np.random.default_rng(0)
PTS3D = rng.uniform(-1.0, 1.0, size=(N, 3)) + np.array([0.0, 0.0, 2.0])
# Projected with the IDENTITY pose, so identity is the true pose for these tracks.
_H = K @ PTS3D.T
PTS2D = ((_H[:2] / _H[2:]).T).astype(np.float32)

T_GUESS = np.eye(4)
T_GUESS[:3, 3] = np.array([0.5, 0.0, 0.0])  # wrong: the guess is 500mm off
T_PNP = np.eye(4)
T_PNP[:3, 3] = np.zeros(3)                   # poselib "recovers" the true pose
MOTION_COV = np.diag([0.01 ** 2] * 3 + [0.01 ** 2] * 3)

CALLS = {"refine": 0}


def _fake_estimate_absolute_pose(pts2d, pts3d, cam_dict, params, initial_pose):
    # poselib returns an Image wrapper whose .pose is the CameraPose; .q is wxyz.
    # CameraPose itself has no .pose, so the wrapper must be a plain namespace.
    pose = poselib.CameraPose()
    pose.q = np.array([1.0, 0.0, 0.0, 0.0])          # wxyz identity
    pose.t = np.array([0.0, 0.0, 0.0])
    res = SimpleNamespace(pose=pose)
    return res, {"num_inliers": N, "inliers": np.ones(N, dtype=bool),
                 "inlier_ratio": 1.0, "iterations": 1}


def _fake_refine_absolute_pose(pts2d, pts3d, pose_in, cam_dict, params):
    CALLS["refine"] += 1
    pose = poselib.CameraPose()
    pose.q = np.array([1.0, 0.0, 0.0, 0.0])
    pose.t = np.array([0.1, 0.0, 0.0])               # an unvalidated 100mm move
    return pose, {}


poselib.estimate_absolute_pose = _fake_estimate_absolute_pose
poselib.refine_absolute_pose = _fake_refine_absolute_pose


def _tracker(honest_success, do_refine=True):
    t = gt.GeometricTracker(gt.GeoTrackerConfig(
        do_refine=do_refine, min_pnp_inliers=20, use_chi2_gate=True,
        honest_pnp_inliers=True, honest_pnp_success=honest_success))
    for k in range(N):
        t.tracks[k] = {"obs": {FRAME: PTS2D[k].copy()}, "pt3d": PTS3D[k].copy()}
    return t


def _caller_gate(t, honest_success):
    """Replicates the BundleSdfGS Stage-2 gate, both with and without the
    carry-forward of the tracker's own verdict."""
    success, n_inliers = t.last_result
    committed = t.poses[FRAME]
    jump = float(np.linalg.norm(committed[:3, 3] - T_GUESS[:3, 3]))
    dR = R.from_matrix(T_GUESS[:3, :3].T @ committed[:3, :3]).as_rotvec()
    d = np.concatenate([dR, committed[:3, 3] - T_GUESS[:3, 3]])
    chi2 = float(d @ np.linalg.solve(MOTION_COV, d))
    lim = float(chi2_dist.ppf(0.95, 6))
    naive = bool((chi2 > lim) or (jump > 0.5))
    tg = t.last_pnp_gate
    # Use the real carry helper, not a local copy: a fork here would let the probe
    # pass while the caller-side fix rotted. Coerce to python bool: `chi2 > lim` is
    # an np.bool_, and `np.bool_(False) is False` is False, so identity checks on
    # the raw expression would spuriously fail.
    carried_flag, _reason = BundleSdfGS._pnp_gate_carry(t)
    carried = bool(naive or carried_flag)
    rejected = bool((not success) or (n_inliers < 20) or carried)
    naive_rejected = bool((not success) or (n_inliers < 20) or naive)
    return {"chi2": chi2, "jump": jump, "naive_tripped": naive,
            "carried_tripped": carried, "naive_rejected": naive_rejected,
            "rejected": rejected, "tracker_vetoed": tg["vetoed"]}


def _run(honest_success):
    t = _tracker(honest_success)
    image = np.zeros((H, W, 3), np.uint8)
    mask = np.ones((H, W), np.uint8)
    depth = np.full((H, W), 2.0, np.float32)
    success, n_inliers = t.step_informed_with_occlusion(
        FRAME, image, mask, depth, K, T_GUESS, motion_cov=MOTION_COV)
    t.last_result = (success, n_inliers)
    return t, success, n_inliers


def main():
    ok = True
    lim = float(chi2_dist.ppf(0.95, 6))

    # --- 1. carry-forward: the naive caller gate is blind to a tracker veto ---
    for honest in (False, True):
        CALLS["refine"] = 0
        t, success, n_inliers = _run(honest)
        g = _caller_gate(t, honest)
        frozen = np.array_equal(t.poses[FRAME], T_GUESS)
        chi2_big = g["chi2"] < 1e-9 and g["naive_tripped"] is False
        print(f"\n[honest_pnp_success={honest}] success={success} n_inliers={n_inliers} "
              f"refine_calls={CALLS['refine']}")
        print(f"  tracker chi2={t.last_pnp_gate['chi2']:.1f} vs limit {lim:.3f}, "
              f"vetoed={t.last_pnp_gate['vetoed']}")
        print(f"  caller chi2={g['chi2']:.2e} jump={g['jump']:.2e} "
              f"naive_tripped={g['naive_tripped']} carried_tripped={g['carried_tripped']}")
        print(f"  naive_rejected={g['naive_rejected']}  rejected(with carry)={g['rejected']}")

        # The freeze only holds when refine is skipped. Under honest_pnp_success the
        # committed pose IS the guess; off, the legacy refine overwrites it, which is
        # finding 2 being deliberately preserved for the A/B baseline.
        ok &= (frozen if honest else not frozen)
        ok &= t.last_pnp_gate["vetoed"] is True
        ok &= t.last_pnp_gate["chi2"] > lim
        # The blind spot requires the committed pose to still BE T_GUESS, which is
        # only true when refine was skipped -- i.e. only under honest_pnp_success
        # =True. With the flag off the (legacy) refine overwrites the frozen guess
        # with an unvalidated re-solve, so the caller's gate trips on THAT drift
        # instead: finding 2 accidentally masks finding 1. The fake moves 100 mm;
        # a refine that drifted less than the jump floor would be just as blind.
        if honest:
            ok &= g["naive_tripped"] is False
            ok &= chi2_big
            # success=False already rejects the frame in the full caller predicate,
            # so what is blind is the GATE, not the whole rejection.
            ok &= g["naive_rejected"] is True
        else:
            ok &= g["naive_tripped"] is True
            ok &= g["naive_rejected"] is True
        # The carry fires in both cases and is what rejects the frame when the
        # success flag and the inlier leg both stay quiet (honest_pnp_success off
        # and the guess carrying >= min_pnp_inliers inliers).
        ok &= g["carried_tripped"] is True
        ok &= g["rejected"] is True

    # --- 2. refine_pose must not overwrite a vetoed (frozen) pose ---
    CALLS["refine"] = 0
    t_on, _, _ = _run(True)
    refine_on_calls = CALLS["refine"]
    CALLS["refine"] = 0
    t_off, _, _ = _run(False)
    refine_off_calls = CALLS["refine"]
    print(f"\n[refine on a vetoed frame] honest_pnp_success=True  -> refine calls={refine_on_calls}, "
          f"pose unchanged={np.array_equal(t_on.poses[FRAME], T_GUESS)}")
    print(f"[refine on a vetoed frame] honest_pnp_success=False -> refine calls={refine_off_calls}, "
          f"pose moved={not np.array_equal(t_off.poses[FRAME], T_GUESS)}")
    ok &= refine_on_calls == 0
    ok &= np.array_equal(t_on.poses[FRAME], T_GUESS)
    ok &= refine_off_calls == 1
    ok &= not np.array_equal(t_off.poses[FRAME], T_GUESS)

    # --- 3. default-off path: chi2 bookkeeping is inert, nothing else changes ---
    t_d = _tracker(False, do_refine=False)
    for k in range(N):
        t_d.tracks[k] = {"obs": {FRAME: PTS2D[k].copy()}, "pt3d": PTS3D[k].copy()}
    t_d.cfg.use_chi2_gate = False
    t_d.cfg.max_pose_jump = 0.2          # legacy constant veto still fires (jump 0.5)
    s, ni = t_d.step_informed_with_occlusion(FRAME, np.zeros((H, W, 3), np.uint8),
                                              np.ones((H, W), np.uint8),
                                              np.full((H, W), 2.0, np.float32),
                                              K, T_GUESS, motion_cov=None)
    print(f"\n[default-off] use_chi2_gate=False -> success={s} "
          f"vetoed={t_d.last_pnp_gate['vetoed']} chi2={t_d.last_pnp_gate['chi2']:.1f}")
    ok &= t_d.last_pnp_gate["vetoed"] is True
    ok &= t_d.last_pnp_gate["chi2"] == 0.0   # never computed when motion_cov is None

    # --- 4. ba._declined: the early returns carry the full key set ---
    for src, dst, why in ((np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 1.0, 1.0]]), "n<4"),
                          (np.zeros((10, 3)), np.ones((10, 3)), "zero spread")):
        out = ba.align_to_object_frame(src, dst)
        keys = {"T_to_object", "log_s", "s", "accepted", "initial_cost", "final_cost"}
        full = keys <= set(out)
        print(f"\n[ba early return: {why}] accepted={out['accepted']} keys_ok={full} "
              f"s={out['s']} reason={out['reason']!r}")
        ok &= full and out["accepted"] is False and out["s"] == 1.0

    print(f"\n{'ALL PROBE CHECKS PASSED' if ok else 'PROBE FAILURE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
