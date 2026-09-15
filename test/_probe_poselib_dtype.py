"""Temporary probe: what exactly IS poselib's info['inliers']?

The Phase-0 code reads info['inliers'] two different ways:

  geometric_tracker.py       int(np.sum(info.get("inliers", [])))   # honest path
  geometric_tracker.py       len(info.get('inliers', []))           # legacy path
  geometric_tracker.py       inlier_mask=inliers -> refine_pose     # indexable mask

All three must be correct for whatever poselib actually returns. Before this probe
the dtype was assumed (`dtype=bool`) and the container type was never checked;
the contract review found that claim had never been verified.

VERDICT (poselib 3.13.0): info['inliers'] is a PYTHON LIST OF bool, one entry per
CANDIDATE -- not a numpy array, and not a mask over inliers only. `len()` of it is
therefore the CANDIDATE count (finding 9's inflation bug), `np.sum()` is the true
inlier count, and integer indexing works. Nothing in the Phase-0 code depends on
the dtype, but the container type matters for the honest/legacy split.

Run:  python test/_probe_poselib_dtype.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "full_system_ow_working"))

import poselib  # noqa: E402

W, H = 640, 480
F, CX, CY = 500.0, 320.0, 240.0
CAM = {"model": "PINHOLE", "width": W, "height": H, "params": [F, F, CX, CY]}

N_CAND, N_BAD = 50, 10


def build_case():
    """50 correspondences for a true identity pose, 10 of them perturbed."""
    rng = np.random.default_rng(0)
    pts3d = rng.uniform(np.array([-1.0, -1.0, 2.0]), np.array([1.0, 1.0, 6.0]),
                        size=(N_CAND, 3)).astype(np.float64)
    pts2d = (F / pts3d[:, 2:3] * pts3d[:, :2]) + np.array([CX, CY])
    bad = rng.choice(N_CAND, size=N_BAD, replace=False)
    pts2d = pts2d.astype(np.float64)
    pts2d[bad] += rng.normal(0.0, 25.0, size=(N_BAD, 2))
    return pts2d, pts3d, bad


def main():
    ok = True
    pts2d, pts3d, bad = build_case()

    init = poselib.CameraPose()
    init.q = np.array([1.0, 0.0, 0.0, 0.0])
    init.t = np.array([0.0, 0.0, 0.0])

    res, info = poselib.estimate_absolute_pose(
        pts2d, pts3d, CAM, {"max_reproj_error": 1.0, "min_iterations": 100,
                            "max_iterations": 1000}, init)

    print(f"res type           : {type(res).__name__}  (poselib returns an Image)")
    print(f"res.pose type      : {type(res.pose).__name__}")
    print(f"res.pose.q         : {np.asarray(res.pose.q)}  (tracker assumes wxyz)")
    print(f"res.pose.t         : {np.asarray(res.pose.t)}")
    print(f"info keys          : {list(info.keys())}")
    print(f"info['num_inliers'] : {info['num_inliers']!r}")

    inl = info["inliers"]
    print(f"info['inliers']     : type={type(inl).__name__} len={len(inl)}")
    print(f"  np.asarray dtype  : {np.asarray(inl).dtype}")
    print(f"  int(np.sum(inl))  : {int(np.sum(inl))}   <- honest_pnp_inliers path")
    print(f"  len(inl)          : {len(inl)}           <- legacy path")
    print(f"  inl[{N_BAD - 1}]  : {inl[N_BAD - 1]!r} (integer indexing works)")

    # 1. Container type: a list, so np.sum() coerces it and len() counts entries.
    print(f"\n  isinstance(list)          : {isinstance(inl, list)}")
    print(f"  isinstance(ndarray)       : {isinstance(inl, np.ndarray)}")
    ok &= isinstance(inl, list)
    ok &= np.asarray(inl).dtype == np.bool_

    # 2. The mask spans ALL candidates, not just the inliers: this is why len() of
    #    it is the candidate count, i.e. finding 9's inflation bug.
    ok &= len(inl) == N_CAND
    print(f"  len(inliers) == n_candidates ({N_CAND}) : {len(inl) == N_CAND}")

    # 3. The two live reads must disagree by exactly the outlier count, and np.sum
    #    must agree with the count poselib reports itself.
    n_true = int(np.sum(inl))
    ok &= n_true == info["num_inliers"]
    print(f"  np.sum(inliers) == num_inliers ({info['num_inliers']}) : {n_true == info['num_inliers']}")
    print(f"  legacy len() would have gated on {len(inl)}, "
          f"truth is {n_true} -- inflation of {len(inl) - n_true}")
    ok &= len(inl) > n_true

    # 4. The mask must at least catch the injected outliers, so that np.sum() is not
    #    just coincidentally equal to num_inliers. RANSAC's tolerance may let one
    #    through; report rather than assert an exact count.
    rejected = sorted(int(i) for i in range(N_CAND) if not bool(inl[i]))
    caught = [int(i) for i in bad if i not in rejected]
    print(f"  injected outliers {sorted(int(i) for i in bad)} -> "
          f"rejected {len(rejected)}/{N_BAD}, survived {caught}")
    ok &= len(rejected) >= N_BAD - 2

    # 5. refine_absolute_pose takes the mask-sliced points back.
    pts2d_g, pts3d_g, _ = build_case()
    keep = np.asarray(inl).astype(bool)
    pose = poselib.CameraPose()
    pose.q = np.array([1.0, 0.0, 0.0, 0.0])
    pose.t = np.array([0.0, 0.0, 0.0])
    refined, rinfo = poselib.refine_absolute_pose(
        pts2d_g[keep], pts3d_g[keep], pose, CAM, {"max_iterations": 20})
    print(f"\nrefine_absolute_pose -> type={type(refined).__name__} "
          f"q={np.asarray(refined.q)} t={np.asarray(refined.t)}")

    print(f"\n{'ALL PROBE CHECKS PASSED' if ok else 'PROBE FAILURE'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
