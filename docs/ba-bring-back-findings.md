# Bringing back BA: what landed, and the empirical findings it stands on

Date: 2026-09-10. Scope: `full_system_ow_working/` (the authoritative tree).

This note records (1) what was built, (2) the pyceres/pycolmap facts that were
verified *numerically, in this environment* before being trusted — including two
"facts" an earlier investigation got wrong — and (3) the protocol for turning the
feature on. Everything in §2 is also encoded in `full_system_ow_working/ba.py`'s
module docstring, next to the code that depends on it.

## 1. What landed

| Piece | Where | State |
|---|---|---|
| Bounded-window, acceptance-gated reprojection BA | `full_system_ow_working/ba.py` (`bundle_adjust`, `run_ba`) | New, tested |
| Tracker↔object-frame alignment with an explicit log-scale block | `ba.py` (`align_to_object_frame`) | New, tested |
| Mapper depth alignment via pose-pinned log-scale fit | `obj_gs_mapping.py` (`align_depth_sim3` branch of `densify`), runner flag `align_depth_sim3_mapper` | Wired, **default OFF** |
| BA on keyframe commit | `bundlesdf_gs.py:run_ba_python`, hook right before `T_CiO` is composed | Wired, **default OFF** (`run_ba_on_keyframe`) |
| Runner flag | `run_full_system_bundlegs_ow.py` GlobalConfig `run_ba_on_keyframe` → `GeoTrackerConfig` | Wired, default OFF |
| Dead-code repairs in the legacy tracker | `geometric_tracker.py` (P0-1a/b, P0-2, P0-3) | Landed, behavior-neutral |
| Phase-0: honest PnP inlier counting | `geometric_tracker.py` (`step_informed_with_occlusion`), flag `honest_pnp_inliers` | Wired, **default OFF** |
| Phase-0: honest PnP success accounting | idem, flag `honest_pnp_success` | Wired, **default OFF** |
| Phase-0: soft PnP attempt floor | idem, flag `soft_pnp_floor` (+ `pnp_attempt_floor=6`) | Wired, **default OFF** |
| Phase-0: chi-square innovation gate | tracker gate + `bundlesdf_gs.py` (`_motion_covariance`, `_innovation_chi2`), flag `use_chi2_gate` | Wired, **default OFF** |
| Phase-0: `<6`-point guard in `refine_pose` | `geometric_tracker.py` (P0-4) | Landed, **un-flagged** (see §3) |
| Contract fixes 1-7 (see §4): gate carry-forward, refine skip on veto, honest rejection log, gap-normalised covariance, variant-runner wiring, `ba._declined` | `geometric_tracker.py`, `bundlesdf_gs.py`, `ba.py`, both `..._3dgs`/`_dense_densification` runners | Landed; all default-off-safe |
| Tracker knobs reachable from the runners (see §7) | all three runners' `GlobalConfig` → `tracker_cfg`: the four flags (`honest_pnp_inliers`, `honest_pnp_success`, `soft_pnp_floor`, `use_chi2_gate` — already forwarded, now test-guarded) plus 15 that were not: `min_pnp_inliers`, `pnp_attempt_floor`, `chi2_quantile`, `chi2_min_history`, `abs_jump_floor`, `motion_sigma_floor_t`, `motion_sigma_floor_r_deg`, `motion_sigma_scale`, `debug`, `ba_max_keyframes`, `ba_verbose`, `triangulate`, `triangulate_thresh`, `occlusion_margin`, `update_tracker_points_from_gs`. 19 knobs total, all already on the live `GeoTrackerConfig` | Wired; defaults equal `GeoTrackerConfig`'s, so the A/B baselines are unchanged |
| Unit tests | `test/test_ba.py` — 23 tests; `test/test_phase0_gates.py` — 29 functions / 31 tests (the runner-knob test is parametrised over the three runners) | 54 collected, synthetic only, no dataset/GPU |

The defaults are deliberately *additive*: window bound, acceptance test, shift
guard and logging ON; pose priors, inlier covariance weighting and the soft
anchor OFF. With `run_ba_on_keyframe=False` (the default) the live path is
bit-identical to before, so the paired 5-clip A/B baselines in
`docs/bundlesdf-gate-ab-findings.md` stay reproducible. The four Phase-0
flags are the same discipline on the *tracker* side: each is one knob for one
paired A/B, so `honest_pnp_inliers` can be separated from `honest_pnp_success`
even though they were found in the same code path.

### The keyframe-commit hook, and why it sits there

`run_ba_python()` is called after the keyframe decision and **before** `T_CiO`
is composed (the `if is_kf and self.tracker_cfg.run_ba_on_keyframe` block in
`BundleSdfGS.run()`, immediately above `T_CiO = self.tracker.poses[self.cnt] @
self.poses[0]`), so the render, `add_new_points_from_depth`
and the mapper enqueue all see the refined pose when the solve is accepted.
A refused solve writes nothing anywhere (copy-then-commit). Accepted solves
also re-sync `self.poses[f]` for in-window keyframes, which the
keyframe-redundancy check reads every frame. The mapper's already-enqueued
frames keep their pre-BA poses; that loop closes only via
`update_tracker_points_from_gs` on the next mapping update.

## 2. Empirical findings (verified against pyceres 2.6.0 / pycolmap 3.13.0 / poselib)

1. **Quaternion parameter blocks are `xyzw`** (Eigen memory order, scalar last)
   for both `ReprojErrorCost` and `Point3DAlignmentCost`.
   `pycolmap.Rotation3d().quat` returns `xyzw` too — identical to scipy's
   `Rotation.as_quat()`, so no reordering is ever needed. COLMAP's C++ *docs*
   say wxyz, and the legacy `geometric_tracker.run_ba` converted accordingly:
   that made it an **inert no-op** (initial == final cost, blocks bit-identical;
   independently re-verified by a code scout). The fix landed in the legacy
   function as well, but nothing calls it; use `ba.py`.
2. **`ReprojErrorCost` zeroes residuals for points at z ≤ 0** in the camera
   frame (cheirality guard). A pose flip can therefore make observations
   *silently vanish* from the cost instead of pulling back — one reason the
   acceptance test and shift guard are not optional.
3. **Monocular reprojection BA can never observe scale.** Pinning one keyframe
   fixes 6 of the 7 similarity-gauge dofs; the remaining orbit
   (`t_i' = s·t_i − (s−1)·R_iR_0ᵀt_0`, `p' = s·p + (s−1)·R_0ᵀt_0`) leaves every
   reprojection exactly invariant. Verified: zero final cost, rotations exact
   to 1e-6°, translation error a perfectly linear ramp, invariant to the
   iteration budget. Production meaning: the tracker seeds metric scale from
   depth, but a reprojection-only BA can random-walk along this orbit at zero
   cost — a *successful-looking* solve is not evidence of correctness. The
   shift guard bounds the walk per commit; `use_pose_priors` (currently OFF) is
   the metric leash that closes the orbit. **Watch Umeyama s when enabling
   `run_ba_on_keyframe`** — the A/B data says it already ranges 0.17–2.71.
4. **A position-only soft anchor is not enough.** With the hard pin off and
   only `AbsolutePosePositionPriorCost` on the anchor, the rotation gauge stays
   open and the solve drifts along it (measured: 1.1 m / 2.5° at zero cost) —
   the shift guard correctly rejects, but the solve is wasted. The soft anchor
   is therefore the full 6-DoF `AbsolutePosePriorCost` (`anchor_pose_prior`,
   residual layout `[rotation(3), translation(3)]`, covariance ordered
   rot-first), which closes everything except scale.
5. **`pyceres.CostFunction` cannot be subclassed from Python.** Overriding
   `evaluate` still resolves to the C++ pure virtual `Evaluate` (raises at
   `pyceres.solve`; a scout reproduced a segfault variant three ways). No
   photometric / GS-render residual can live in a pyceres graph, so the torch
   LM (`obj_gs.py:optimize_wrt_image_lm`) stays a separate solve. The fusion
   channel is covariance-weighted priors: the LM already forms `JᵀJ`
   (`obj_gs.py:632-654`), and the prior costs' covariance overloads whiten by
   `Σ^-1/2`, so `Σ = (JᵀJ)⁻¹` drops straight in.
6. **`Point3DAlignmentCost(prior, use_log_scale=True)`** has blocks
   `[pt_a(3), q(4), t(3), log_s(1)]` and residual
   `exp(log_s)·R(q)@pt_a + t − prior`. Log-parameterised scale is symmetric in
   `s` and `1/s` and well-conditioned far from 1 — the principled replacement
   for the silent `np.clip(s, 0.1, 10.0)` depth absorption in
   `ObjGSCameraMapper.densify`, now wired behind `align_depth_sim3` (default
   OFF). The wiring fits **scale only**: sensor and rendered depth live in the
   same camera frame, so the pose blocks are pinned at the identity
   (`pin_pose=True`) — a nonzero R/t there would be pose error leaking into
   the depth buffer, and a free pose would trade translation against scale on
   near-planar depth maps. Implausible results are *rejected* (logged), never
   clamped; the acceptance gate inside `align_to_object_frame` (usable solve
   AND strict cost decrease) also rejects no-information fits, so an
   already-aligned frame keeps s=1 by default rather than committing noise.
   With *free* source points the cost is a pure gauge (solver reaches ~1e-24
   with arbitrary R/t/s); source points must be held constant or constrained
   by reprojection terms.
7. **Manifold lifetime**: the `EigenQuaternionManifold` instance must be held
   in a local for the solver's whole lifetime; a temporary constructed inline
   at `set_manifold` is GC'd and the graph dereferences a dead manifold.
8. API shapes that reject reasonable-looking inputs: `point2D` must be 1-D
   length-2 (a `[2,1]` column raises `TypeError`); position priors need 1-D
   length-3; `model='PINHOLE'` selects the 4-param `(fx,fy,cx,cy)` camera while
   a bare int id silently selects a 3-param camera and changes the block
   layout.
9. **`poselib.estimate_absolute_pose`'s `info['inliers']` is a boolean mask
   over *all candidates*, not a list of inlier indices** (probe: 50 candidates
   with 10 injected outliers → `num_inliers=40`, `len(info['inliers'])=50`,
   `dtype=bool`). The tracker read `len(info.get('inliers', []))` as the
   inlier count, so a 50-candidate solve with 3 true inliers reported 50 and
   sailed through the `min_pnp_inliers=20` gate. The verified-correct key is
   `num_inliers`. Note the same array *is* used correctly when passed as
   `inlier_mask` into `refine_pose` — boolean-mask semantics there — which is
   why the bug was invisible: one wrong use, one right use, same object.
   Landed as `honest_pnp_inliers`.
10. **A vetoed PnP still returned `success=True` and committed the frozen
    guess** (`geometric_tracker.py`, `step_informed_with_occlusion`). When the
    jump veto fired the method committed `T_guess` and returned `True`, so the
    caller could accept the frame whenever `n_inliers_guess >= 20` — accepting
    the *refined velocity guess* as if it were a PnP measurement, appending it
    to `accepted_rel` and adding points at that pose. Every path except the
    pre-PnP starvation bail returned `True`. Landed as `honest_pnp_success`,
    which clears `pnp_ok` on all four failure paths.
11. **The pre-PnP `min_pnp_inliers` guard was a cliff, not a slope.** On
    clip-003318, 66 of 116 rejections were 19-correspondence frames where PnP
    was never attempted, so the reported inlier count was a sentinel, not a
    measurement. Lowering the *attempt* floor to `pnp_attempt_floor=6` and
    letting the post-PnP inlier gate decide acceptance recovers the information
    without loosening acceptance. Landed as `soft_pnp_floor`.
    Caveat recorded in §7: `min_pnp_inliers` had *always* been on the live
    `GeoTrackerConfig` at 20 — the `getattr(self.cfg, "min_pnp_inliers", 20)`
    fallback in `geometric_tracker` never fired — so the coupling was not to a
    dead fallback but to a real field that no runner declared or forwarded.
    `soft_pnp_floor` could soften the attempt floor with no way to soften
    acceptance, because the acceptance value had no dial. It is now declared and
    tunable.
12. **Chi-square innovation gate**: innovation `d = [rotvec(R_guessᵀR_pnp)
    (rad), t_pnp − t_guess (m)]`, `chi2 = dᵀΣ⁻¹d` against
    `chi2.ppf(0.95, 6)` = 12.592, with `Σ = diag([s_r²]×3 + [s_t²]×3)` from
    the last 50 *accepted* steps (`s = max(floor, 3·median)`; `s_r` in
    radians). The gate replaces the constant `max_pose_jump` veto with
    statistics that adapt to the object's actual velocity, and keeps an
    `abs_jump_floor` (0.5 m) hard ceiling that fires regardless of how loose
    the covariance gets (§7 corrects both halves: the floor is a leg of that
    branch only, and at shipped defaults the chi² term binds first by ~10×). It
    stays inactive until `chi2_min_history=5` accepted steps exist, at which
    point the caller falls back to the legacy constant gate. The
    photometric-refinement sanity gate keeps its own adaptive limit
    and is unchanged.

## 3. Repairs to the legacy tracker (behavior-neutral, dead paths only)

- **P0-1a/b** — removed the `wxyz` conversions in `run_ba` (the module-level
  `run_ba(frame_indices, ...)` in `geometric_tracker.py`, both the pose-read and
  the pose-write site). The function has zero callers, so
  nothing in the live path changes; it is no longer a trap for future callers.
- **P0-2** — `triangulate_single_track` now requires
  `summary.IsSolutionUsable()` before accepting a solve.
- **P0-3** — the `triangulated_tids` latch is set only when a solve actually
  ran. Previously it latched even on the `baseline < 5mm` bail-out, permanently
  barring those tracks from triangulation.
- **P0-5** — `occlusion_margin` declared in the live `GeoTrackerConfig`
  (it was read only via a `getattr(self.cfg, ..., 0.05)` fallback, so the
  0.05 always won), and in this pass given a runner dial too — otherwise it
  would sit on `GeoTrackerConfig` tunable in-process only, which is the same
  defect as `min_pnp_inliers`. Default unchanged.
- **P0-4** — `refine_pose` now takes `W`/`H` as keyword-only required
  parameters and bails when the inlier set has fewer than 6 points. The only
  live caller already passed both, so this is an interface tightening plus a
  real safety fix: `poselib.refine_absolute_pose` cannot refine a 0–5 point
  set, and committing its unvalidated output would have moved the pose on
  exactly the frames that are already starved. Unlike the rest of the
  Phase-0 work this is **un-flagged**, because the old behaviour was to trust
  an output the solver had never been asked to produce at all. Worth watching
  in the first A/B run.

Phase-0 live-path fixes now landed behind flags (all awaiting their own A/B):
`honest_pnp_inliers`, `honest_pnp_success`, `soft_pnp_floor`, `use_chi2_gate`
— findings 9–12. `align_depth_sim3_mapper` sits alongside them.

Still deferred: re-validation of the `refine_pose` output by the same gate
(the refined pose is still committed without a jump check of its own — but
since the contract fixes that path is now only reachable for poses the PnP
actually produced, so the unvalidated step is bounded by the PnP result it
refined rather than by whatever was frozen in place), selective
motion-model priors on PnP-rejected frames, and the `src/dynfg` package from
the design review (§5) — design only, the implementation deliberately stayed
in-tree behind the equivalent `run_ba_on_keyframe` flag.

## 4. Contract review of the Phase-0 gates (and what it found)

The Phase-0 flags were then handed to an independent contract review: given the
tracker/caller boundary as built, is each gate scored against the right pose,
and does the accounting the gate depends on actually measure what it claims?
Seven findings, all dispositioned:

| # | Finding | Disposition |
|---|---|---|
| 1 | **Double-gate blind spot.** The tracker gates the *raw* PnP result against `T_guess`; the caller gates the *committed* pose against `T_guess`. After an internal chi² veto the committed pose **is** `T_guess`, so the caller's chi² and jump are `0.0` → it ACCEPTS the frame, banks it in `accepted_rel`, and appends a near-zero step to the very statistics `_motion_covariance` is built from: the gate tightens on the strength of the jump it just rejected. The `0.0` requires refine to be skipped — only under `honest_pnp_success` (measured counterexample: with the flag off, `chi2=1.6e3`, `jump=0.40 m` against a 0.5 m floor). | **Fixed** — `last_pnp_gate` (set inside the tracker, including `vetoed`) is carried to the caller by `BundleSdfGS._pnp_gate_carry` and OR-ed into `gate_tripped`. Extracted as a static method so it is unit-testable without the GS stack. |
| 2 | **The frozen guess was not frozen.** `refine_pose` ran unconditionally after a veto, re-solving `T_guess` with the vetoed PnP's inlier mask (possibly below `min_pnp_inliers`) and overwriting `self.poses[frame_idx]` with that unvalidated output. | **Fixed** — `refine_allowed = pnp_ok or not honest_pnp_success`, so refinement only touches a pose the PnP actually produced. Flag-gated: with `honest_pnp_success` off the legacy "refine whatever got committed" is preserved for the A/B. |
| 3 | **The rejection log lied.** A gate trip printed `Rejecting PnP (inliers: ... < 20)` regardless of the actual cause, so the log could not distinguish an inlier failure from a jump/chi² failure. | **Fixed** — `gate_reasons` records the leg(s) that actually fired (`chi2=… > …`, `jump=…m > abs_floor=…`, `tracker chi2=… > … (PnP vetoed inside the tracker)`) and are composed into the printed `reasons` alongside the independent `success=False` / inlier-count legs. |
| 4 | **Covariance built from the wrong unit.** `accepted_step_t[k]` is the displacement from `accepted_rel[k]` to `accepted_rel[k+1]` — per *accepted* step, not per frame. After a run of rejections each record spans several frames, so `median` inflated with the object's own motion over that span: sigma grew and the gate disabled itself exactly when it was needed most. | **Fixed** — `_accepted_step_gaps` recovers the frame spans (`len(accepted_step_t) == len(accepted_rel) − 1`, aligned by construction) and the medians are computed over `step / gap`, clipped at 1 so a duplicated frame index cannot divide by zero. If the bookkeeping ever desyncs the helper returns `None` and the raw medians are used — a deliberate fallback to the pre-fix behaviour rather than a wrong number. |
| 5 | **The variant runners could not run the A/B.** `run_full_system_bundlegs_ow_3dgs.py` and `..._dense_densification.py` had none of the Phase-0 flags, so the paired 5-clip runs were only reproducible on the base runner. | **Fixed** — all Phase-0 flags plus `gate_kf_commit`, `run_ba_on_keyframe`, `align_depth_sim3_mapper` declared on `GlobalConfig`, forwarded into `tracker_cfg` / `map_cfg`, identical in both files. |
| 6 | **`KeyError` on a graceful refusal.** `align_to_object_frame`'s two pre-solve refusals (`n < 4`, zero source spread) returned `{"accepted": False}` only, while `ObjGSCameraMapper.densify` reads `res["s"]` **before** checking `res["accepted"]`. | **Fixed** — `_declined(reason)` returns the full promised key set with identity `T_to_object`, `log_s = 0.0`, `s = 1.0` and a `reason`, so every return path honours the docstring and callers may read `s` first. Fixed on the producer side: every caller is covered, not just this one. |
| 7 | **Overstated comment.** `info['inliers']` was described as `dtype=bool` probe-verified; the probe recorded `len == n_candidates` and `num_inliers`, not the dtype. | **Settled** — `test/_probe_poselib_dtype.py` now settles it directly: `info['inliers']` is a **python `list` of bool, one entry per *candidate*** (`isinstance(list)` True, `isinstance(ndarray)` False, `len == 50 == n_candidates`, `int(np.sum(inl)) == 41 == info['num_inliers']`, integer indexing works). So the legacy `len()` was the *candidate* count and the 20-inlier gate measured how many points PnP was asked about; both live reads (`np.sum(mask)`, `mask[i]`) are dtype-agnostic, so the container type is what splits the honest/legacy paths, not any dtype assumption. RANSAC rejected 9 of 10 injected outliers, so the count is a real measurement. |

Two gaps are documented and deliberately **not** closed, because closing them
changes default-off behaviour:

- **The carry-forward lives only in the chi² branch.** With `use_chi2_gate` off
  the caller trips on `jump > _jump_limit()`; a veto by the tracker's constant
  `max_pose_jump` threshold there also freezes the guess to `T_guess` and is
  equally invisible — under `honest_pnp_success`, which is what skips the refine
  that would otherwise overwrite the frozen guess. Carrying it through would
  change the default-off path the
  paired baselines are pinned to. `_pnp_gate_carry` reads `vetoed`, not `chi2`,
  so the same helper catches those frames the moment the carry is wired into the
  `else` branch (`test_caller_gate_carries_a_legacy_veto_too` pins that the
  helper does read them, with `chi2 == 0.0`).
- **The guess-preference shortcut bypasses the innovation gate.** `n_inliers_guess
  > 0.95 * n_inliers` short-circuits to the guess before the chi² test, so a
  frame whose guess is self-consistent clears it at any magnitude — no chi² and
  no `abs_jump_floor` check either (§7 has the measured case). The caller is
  blind to it the same way as in finding 1: the committed pose is `T_guess`, so
  its chi² and jump are `0.0` — and `_pnp_gate_carry` reports nothing, because
  no gate tripped inside the tracker. That blind spot only holds while the
  committed pose really *is* `T_guess`. Under `honest_pnp_success` the refine
  skip keeps it there; with that flag off the legacy refine overwrites the guess
  with an unvalidated re-solve and the caller trips on the drift *refine*
  introduced instead — finding 2 accidentally masking finding 1, exactly as it
  does for the double-gate blind spot. `test_guess_preference_bypasses_the_innovation_gate`
  pins the current behaviour as a documented gap.

Worth knowing for anyone wiring the flags: `do_refine` exists **only** on the
standalone `GeoTrackerConfig` in `geometric_tracker.py` (it is what the tests
use); the live `GeoTrackerConfig` in `bundlesdf_gs.py` has no such field, so
the live path always refines via `getattr(cfg, "do_refine", True)` and the
variant runners must not pass it.

## 5. Reconciliation with the independent design review

A five-scout + synthesis design pass (workflow `wf_c96f1b56-76e`) converged on
the same graph semantics and the same three primary repairs, and independently
re-verified findings 1, 3 and 5 (it chose a new `src/dynfg` package behind a
`use_dynfg` flag; the implementation here keeps the code in the authoritative
tree behind the equivalent `run_ba_on_keyframe` flag — same graph, same
gating discipline). Its "seam stitch prior" (tying the window's oldest frame
to the last committed out-of-window pose instead of hard-pinning it) is the
job `anchor_pose_prior` does; switching `fix_first` off and the soft anchor on
is the config change if re-baselining shows up in the A/B.

## 6. Protocol for enabling the new flags

1. Same code, same seed, same 5 clips as `docs/bundlesdf-gate-ab-findings.md`;
   the only difference between the two runs is the flag.
2. Read the `[BA]` lines and the `on_finish` tally (`BA: N runs, A accepted,
   R rejected`). A high reject rate means the triangulation upstream is
   feeding the graph garbage — loosen triangulation, not the guards.
3. Primary metrics: scale-free ATE/travel (comparable across the baseline),
   plus **Umeyama s** — finding 3 says s is where BA-visible damage will show
   up first. If s degrades while ATE improves, enable `use_pose_priors`
   (the metric leash) and re-run before drawing conclusions.
4. `align_depth_sim3_mapper` is a separate A/B under the same protocol (it
   changes the mapper's depth buffer, not the tracker). Watch the
   `[Mapper] Sim3 depth alignment rejected` rate in addition to the metrics:
   a high reject rate means render and sensor depth disagree by more than
   scale, which no depth alignment can fix.

### Enabling the Phase-0 tracker flags

`honest_pnp_inliers`, `honest_pnp_success`, `soft_pnp_floor` and
`use_chi2_gate` are each one flag for one paired run, same protocol. Turn them
on in this order, because each one changes what the log lines *mean* for the
next:

1. **`honest_pnp_inliers` first.** It changes the numbers already printed, not
   the decisions that follow. Expect the PnP-rejection rate to jump — the
   `min_pnp_inliers` gate was reading a candidate count, so every clip gets a
   new baseline of true inliers. Do not read the jump as a regression; it is
   the measurement finally working.
2. **`honest_pnp_success` second.** Now `success=False` on vetoed frames, so
   the caller's keyframe-commit gating and `add_new_points_from_depth` see a
   frame that is *not* a measurement. This is the flag that changes state
   downstream, not just accounting. It is also the switch that makes the
   contract fix 2 real: `refine_allowed = pnp_ok or not honest_pnp_success`
   means a vetoed frame stops being re-refined with the vetoed PnP's inlier
   mask, so the frozen guess actually stays frozen instead of being
   overwritten by an unvalidated re-solve of itself. With the flag off that
   overwrite still happens, which is why the legacy path is unchanged.
3. **`soft_pnp_floor` third.** It converts the starvation cliff into attempts.
   Compare the `[GeoTracker] Insufficient tracks` count before/after: that
   count should fall, and the inlier-gate rejections should rise by roughly the
   same amount. With this flag on, `min_pnp_inliers` is the dial that decides
   how many of those new attempts get accepted — it used to be fixed at 20, so
   an attempt with 6–19 correspondences could never be accepted (§7).
4. **`use_chi2_gate` last**, and only after enough accepted steps that
   `chi2_min_history` is satisfied — before that the caller silently falls back
   to the legacy constant gate. The six numeric knobs (`chi2_quantile`,
   `chi2_min_history`, `abs_jump_floor`, `motion_sigma_floor_t`,
   `motion_sigma_floor_r_deg`, `motion_sigma_scale`, plus `pnp_attempt_floor`)
   are now declared on every runner's `GlobalConfig` and forwarded into
   `tracker_cfg`; each defaults to what `GeoTrackerConfig` already ships, so
   flipping `use_chi2_gate` alone is unchanged. Before this wiring they were
   tracker-only and the step "tune `abs_jump_floor`" was not executable without
   editing `bundlesdf_gs.py`. `test_tracker_knobs_are_reachable_from_every_runner`
   pins four halves — the knobs still exist on `GeoTrackerConfig`, they are
   forwarded at a real `GeoTrackerConfig(...)` call site, the forwarding value
   is `cfg.<knob>` rather than a hardcoded constant, and the runner defaults
   have not drifted. The flags themselves (`use_chi2_gate`,
   `honest_pnp_inliers`, `honest_pnp_success`, `soft_pnp_floor`) are tracked
   too — before this pass deleting `use_chi2_gate`'s forwarding line left the
   test green. Before the per-knob loop it asserts exactly one `GlobalConfig`
   per runner: the reader merges every same-named class into one dict, so a
   second one could have satisfied the test while the class the runner actually
   uses was ignored.
   **Do not tune `abs_jump_floor` first, the framing used to say the opposite:**
   at shipped defaults it is dominated by the chi² term by roughly 10×. A single
   translation axis has ceiling `sqrt(chi2.ppf(0.95, 6)) * s_t` = `3.5485 * s_t`,
   and the `3 ×` inside `s_t = max(motion_sigma_floor_t, 3 * med_t)` matters —
   the median is not the sigma. At shipped defaults `med_t` is 4.6 mm, so
   `s_t = 13.8 mm` and the ceiling is 49 mm against a 500 mm floor: a 10.2×
   margin, and `abs_jump_floor` cannot bind. Ceilings against `s_t`: 13.8 mm →
   49, 30 → 107, 60 → 213, 141 → 500, 150 → 532 mm. So the floor first binds
   once `s_t > 500 / 3.5485 = 141 mm` — that means setting
   `motion_sigma_floor_t` above ~141 mm (3 × the median stays at 13.8 mm), not
   50 mm as an earlier draft of this step said. Rotation caps at 6.4°
   (`3.5485 * 1.8 deg`, `med_r` 0.60 deg). Note `med_t` is gap-normalised, so on
   a clip with rejection runs the effective `s_t` is lower still. Editing
   `abs_jump_floor` produces no observable change unless `s_t` is already above
   ~141 mm — tune the sigma floors first, and only reach for `abs_jump_floor`
   once `s_t` has been widened that far, or the chi² quantile raised.
5. `P0-4`'s `<6`-point `refine_pose` guard is the one un-flagged behaviour
   change: watch for the pose of starved frames to stop drifting, and confirm
   the `[GeoTracker]` refine path is not silently skipped on frames that should
   be refining.

## 7. Second-round findings: what landed, and what stayed documented only

A second contract pass over the same boundary (workflow `wf_14a0ae66-38e`) turned
up sixteen more items; a third pass over the wiring (workflow
`wf_5a4fc4ae-d8d`) added two more. Eight are fixed in code, the rest are
recorded because fixing them changes default-off behaviour or is out of scope
for this pass.

**Fixed**

- **The chi² knobs were unreachable from the runner.** They existed on
  `GeoTrackerConfig` and this document told the operator to tune them, but no
  runner forwarded them: enabling `use_chi2_gate` turned the gate on fully
  hardcoded. All three runners' `GlobalConfig` now declare
  `pnp_attempt_floor`, `chi2_quantile`, `chi2_min_history`, `abs_jump_floor`,
  `motion_sigma_floor_t`, `motion_sigma_floor_r_deg`, `motion_sigma_scale` and
  forward each into `tracker_cfg`. Every default equals what `GeoTrackerConfig`
  already ships, so `use_chi2_gate=True` with none of them touched is
  bit-identical to before.
- **`min_pnp_inliers` had no runner dial.** It had *always* been on the live
  `GeoTrackerConfig` at 20 — `geometric_tracker`'s
  `getattr(self.cfg, "min_pnp_inliers", 20)` fallback never fired — but no runner
  declared or forwarded it, so every entry point ran at 20. That was the same
  wiring bug as the chi² knobs, one step narrower. It made `soft_pnp_floor`
  half-broken by construction: `soft_pnp_floor` softens only the pre-PnP
  *attempt* floor, so PnP could run on 6 correspondences and still be vetoed by
  the post-PnP *acceptance* gate no matter what the operator set. All three
  runners' `GlobalConfig` now declare it and forward it into `tracker_cfg`.
  Default 20 everywhere, so nothing changes until someone sets it.
- **The BA surface had a switch but no dial.** `run_ba_on_keyframe` was
  forwarded, but not `ba_max_keyframes` or `ba_verbose`, which `run_ba_python`
  reads to size the window and set the log level: enabling BA turned it on with
  a pinned 20-keyframe window and no way to shrink it. The same gap covered
  `triangulate` / `triangulate_thresh`, which the same method reads for its
  pre-step, `occlusion_margin` (P0-5 above), and
  `update_tracker_points_from_gs`, the loop closure that resyncs the tracker's
  points to the mapper's GS means after BA. All six are now declared and
  forwarded in all three runners, each default matching `GeoTrackerConfig`
  (`20` / `True` / `True` / `0.05` / `0.05` / `True`), so they are inert until
  someone sets one.
- **A duplicate-field regression introduced and caught by this pass.** Repairing
  the above, the field was re-declared lower in `GeoTrackerConfig`, believing it
  absent. Dataclasses do not error on a repeated annotated field — they build
  fields from `cls.__annotations__`, a dict that dedupes — so it constructed,
  reported one field, and silently left the original declaration and its comment
  dead. `dataclasses.fields()` cannot expose this because it has already deduped.
  The duplicate is gone, and
  `test_geo_tracker_config_declares_no_field_twice` reads the class body with
  `ast` and fails on any repeated annotation target, so the class of defect is
  guarded rather than just fixed.
- **`--debug` was a no-op for the tracker.** All three runners declared
  `GlobalConfig.debug` and forwarded nothing, so the flag existed but never
  reached `GeoTrackerConfig.debug`. Both sides default to `False` and the
  tracker reads it only in two `print` statements (`geometric_tracker.py`, the
  depth-alignment and guess-preference branches), so this is default-neutral —
  the only effect is that `--debug` now actually produces the two lines it
  advertised. Forwarded as `debug=cfg.debug` in all three runners and added to
  `TRACKER_KNOBS`, so the wiring test guards it too.
- `test_tracker_knobs_are_reachable_from_every_runner`
  (formerly `test_gate_knobs_are_reachable_from_every_runner`) parses the three
  runners with `ast` (importing them would boot the dataset loaders) and asserts
  four things: each knob still exists on `GeoTrackerConfig` (so a deleted knob
  fails as "dead test entry" rather than as a silent skip), the knob appears as
  a keyword at an actual `GeoTrackerConfig(...)` call site, that keyword's
  value is literally `cfg.<knob>`, and the runner's default has not drifted from
  the tracker's. Two of those halves are repairs to earlier drafts of this same
  test: the call-site check used to be a substring search for `cfg.<knob>` in the
  file, which a comment could satisfy — the exact failure mode of the bug it
  guards against — and the keyword check looked at the *name* only, so
  `pnp_attempt_floor=99` passed as reachable while the CLI dial did nothing
  (both verified by mutation: those edits escaped the test). The list also now
  covers the flags themselves (`use_chi2_gate`,
  `honest_pnp_inliers`, `honest_pnp_success`, `soft_pnp_floor`), not just the
  numeric knobs — mutation-tested too: flipping `use_chi2_gate` to `True` and
  deleting its forwarding line both left the test green before this pass, which
  is a worthless guard for the one flag an operator actually flips. It now
  asserts exactly one `GlobalConfig` per runner before reading it: the reader
  merges every same-named class into one dict, so a second one could have
  satisfied the test while the class the runner actually uses was ignored.
- **The `[Align]` log rendered real costs as `0.0000`.** `initial_cost` was
  formatted `:.4f` while costs are summed squared scale residuals in the
  1e-14 to 1e-1 range, so the line said nothing about whether the solve moved;
  and `accepted=False` conflated a solver refusal with a solve that refused to
  decrease the cost. It now prints `:.4g` for both costs and names the leg that
  refused (`solver refused` vs `solved but cost did not decrease`). The
  bundle-adjacency costs are pybind11 properties, so each is passed through
  `float()` before formatting — a format spec on anything else is how the line
  used to come out `0.0000`. `BaResult.summary_line` had the same `:.4f` on its
  cost pair and is fixed identically; it is print-only (five call sites, no
  tests), so this is cosmetic and cannot change a verdict.

**Documented, not fixed**

- **`abs_jump_floor` is effectively dead at shipped defaults.** It is dominated
  by the chi² term by roughly 10×: one translation axis has ceiling
  `sqrt(chi2.ppf(0.95, 6)) * s_t` = `3.5485 * s_t`, where
  `s_t = max(motion_sigma_floor_t, 3 * med_t)`. At shipped defaults `med_t` is
  4.6 mm, so `s_t = 13.8 mm` and the ceiling is 49 mm against a 500 mm floor —
  a 10.2× margin. It first binds at `s_t > 500 / 3.5485 = 141 mm`, which means
  setting `motion_sigma_floor_t` above ~141 mm (3 × the median alone stays at
  13.8 mm). Rotation caps at 6.4° (`3.5485 * 1.8 deg`). `med_t` is
  gap-normalised, so on a clip with rejection runs the effective `s_t` is lower
  still. It is retained as a second, independent leg rather than deleted,
  because the chi² term is only as trustworthy as `_motion_covariance`, and a
  covariance built from contaminated accepted steps is precisely the failure the
  absolute floor is there to catch.
- **The tracker's internal gate is not self-sufficient.** `refine_pose` runs
  *after* the tracker's chi² test, so an accepted PnP can be pushed past
  `abs_jump_floor` by LM with no second internal check. Not a safety hole in the
  live path — the caller re-gates the committed pose in every branch, including
  the refine branch — but anyone consuming `tracker.poses` directly, without the
  caller's Stage-2 gate, is unprotected by `abs_jump_floor`.
- **`abs_jump_floor` was described as "a hard sanity ceiling that fires
  regardless of covariance".** False for one branch: the guess-preference
  shortcut (`n_inliers_guess > 0.95 * n_inliers`) keeps `T` at the guess and
  skips the whole block — no chi², no floor check — so a self-consistent guess
  clears the tracker at any magnitude. Measured on a 60-guess-inlier /
  30-PnP-inlier case: the guess was preferred and the committed refined pose
  sat 0.900 m from `T_guess` at chi² 4253. The caller's gate catches it, but
  the tracker's own docstring was wrong. The comment in `geometric_tracker.py`
  now says what is true: it is a covariance-independent leg of *that* branch
  only, and `motion_cov=None` falls through to the constant `max_pose_jump`
  veto.
- **The two variant runners advertise a photometric-jump knob that does not
  exist.** `run_full_system_bundlegs_ow_3dgs.py` and
  `..._dense_densification.py` both declare `max_photo_jump_t = 0.5` and
  `max_photo_jump_R = 20.0` in `GlobalConfig` and forward neither, so the
  tracker runs at `GeoTrackerConfig`'s 0.05 m / 3.0° while the CLI advertises
  values 10× and ~7× looser. Same bug class as the knobs fixed above, and more
  misleading because the declared value differs from the live one — the knob
  looks like it took effect. The main runner does not have this gap: it declares
  0.05 / 3.0 (matching the tracker) and forwards both. Pre-existing, and the
  two variants differ from each other only in `gs_type` and
  `densify_error_threshold`, so they carry the same gap. Left alone: forwarding
  them would change those variants' live photometric gate and needs its own A/B,
  while deleting them removes a control. Recorded rather than guessed at.
- **The pose-pinned Sim3 fit drops the additive bias the legacy affine fit
  models.** `X_object = s * R @ X_tracker + t` has no depth-offset term `b`, so
  a depth disagreement best described as `depth_obj = s * depth_track + b` is
  absorbed into `s`. Measured on a synthetic case with a true 1.25× scale and a
  moderate bias: s = 1.3575 (+8.6%), accepted, ~0.14 m of residual depth error;
  at a larger bias, +17.1% and ~0.28 m. The flag-gated replacement is strictly
  less expressive than the `np.clip(s, 0.1, 10.0)` path it replaces, in exchange
  for rejecting implausible scales instead of clamping them. Now stated in
  `align_to_object_frame`'s docstring.
- **Pinning a wrong prior is not neutral.** With `pin_pose=True` the pose blocks
  are frozen, so a prior the data does not satisfy has nowhere to go but the
  scale: measured s = 2.288 for a true 2.5 against the test's prior (−8.5%).
  Correcting an earlier misattribution, that −8.5% is **not** the rotation: the
  prior rotates only 0.10/−0.05/0.15 **degrees**, which enters second order
  (`s·θ²/3 ≈ 1e-6`), and both pose blocks are frozen. The driver is the pinned
  translation `prior_t = (0.3, −0.2, 1.1)` — in the closed form
  `s* = s_true − ⟨prior_t, Rp⟩ / ⟨p, p⟩` that is `4.95 / 23.67 = 0.209`, i.e.
  `s* = 2.291` against the measured 2.288. The mapper's use is safe because
  sensor and rendered depth share a frame, so the pose error held back is
  exactly the one that would otherwise inflate the scale — but the prior must be
  a prior for *this* point set. Recorded in the docstring and pinned by
  `test_align_pin_pose_keeps_a_nonidentity_prior_fixed`, which generates the
  data in the prior's pose and asserts both the exact scale and that neither
  pose block moves off the prior (atol 1e-12), and which also reproduces the
  wrong-prior case above as an assertion rather than leaving 2.288 as a
  remembered number.
- **A singular `motion_cov` raises an uncaught `LinAlgError`** at both call
  sites. Unreachable at shipped defaults — `s_t = max(0.005, …)` and
  `s_r = deg2rad(max(0.5, …))` are both strictly positive, so
  `_motion_covariance` returns a diagonal with positive entries — but reachable
  if an operator zeroes both sigma floors on a frozen clip. Every guard is
  `motion_cov is not None`, never invertibility. Left as-is: catching it would
  have to choose between rejecting the frame and running ungated, and neither is
  the right default for a path that cannot currently occur.
- **`from scipy.stats import chi2` adds ~0.17 s per process**
  (0.066 s bare → 0.101 s with `scipy.spatial` → 0.273 s with `scipy.stats`,
  cumulative), and each multiprocess worker pays it independently. It cannot
  change any metric, but it does change wall-clock, so paired A/B runs should be
  compared on the metrics, not elapsed time.
- **The wiring test cannot see a runner swap config sources.**
  `geometric_tracker.py` defines its own reduced `GeoTrackerConfig` (ends at
  `honest_pnp_*`/`soft_pnp_floor`/`pnp_attempt_floor`/`abs_jump_floor`; no
  `chi2_quantile`, `chi2_min_history` or sigma floors), while the test reads its
  baseline from `bundlesdf_gs.GeoTrackerConfig`. All three runners currently
  import from `bundlesdf_gs`, so this is inert. Not guarded: a swap would raise
  `TypeError` on the unknown kwargs at worker startup, which is loud and
  unmissable, so a silent-regression assertion would be paying for a check that
  the runtime already provides.
- **Two pre-existing items, no action taken.** `float(cam.K[…])` inside the
  mapper's lock forces four CUDA synchronisations per densification — no
  correctness impact, pre-existing style. And a `+inf` render depth passes the
  overlap mask: only the ±20 % ratio filter keeps non-finite values out
  (715 of 716 such pixels survived), which works but is incidental rather than
  explicit.
