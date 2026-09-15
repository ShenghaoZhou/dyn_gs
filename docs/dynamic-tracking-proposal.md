# Building a Good Dynamic Object Tracking System on Top of `dyn_gs`

Date: 2026-09-10

Scope: this is a design proposal, not an implementation. Every claim about current
behaviour is checked against the working copy in `full_system_ow_working/` unless
marked otherwise; five plausible-sounding things I verified do **not** exist or are
misplaced are called out in §11 so nobody builds on them.

---

## 1. The diagnosis: this is a pose optimizer with oracle perception and an oracle camera

The single most important thing to see before adding features: the system today does
not track dynamic objects. It *refines a single object pose* under four external
assumptions, and one of them is a fifth of the problem.

| Sub-problem | What a "dynamic object tracking system" needs | Current state |
|---|---|---|
| Perception | detect + segment instances | **oracle** — `model_infer/mask_*.png`, one class, no IDs |
| Association | instance continuity, multi-object, ID stability | **absent** — `self.obj_gs` is one global, no lifecycle |
| Camera pose | estimate `T_CW` | **oracle** — read from `extrinsics/{i:06d}.npy` |
| Object pose | estimate `T_WO` | ✅ the only thing actually measured |
| Appearance | per-object model | ✅ the strongest part of the repo |

See `full_system_ow_working/run_full_system_bundlegs_ow.py:569-574` — the tracker is
called with `T_CW=fd["extrin"]` and `T_WO_init=T_WO_gt` (GT on frame 0) — and
`load_frame_data_v2()` reads masks from `model_infer/` with fallbacks to
`dyn_obj_masked_infer/` and `obj_masks/`. The GT `object_poses.txt` is even parsed
inline (`:310-322`). Every number reported by this repo is therefore a measurement of
*one* sub-problem in isolation.

That framing is not a criticism — it's the plan. The repo already has a strong pose+
model core; it needs the four things around it. The proposal below is ordered so each
phase removes exactly one oracle.

**Naming caveat, because it will matter when reading the code:** despite
`BundleSdfGS` in the class name there is no SDF anywhere in the pipeline — the model
is Gaussians only. The "Bundle" half was, until recently, also absent from the *live*
path — but the full story is more interesting than "no bundle adjustment": a real,
complete pyceres reprojection BA existed at
`full_system_ow_working/geometric_tracker.py:159-217` (`run_ba`) with **zero callers**
in the working tree — orphaned, not absent — and on pycolmap 3.13 it was additionally
an *inert no-op*, because it stored the quaternion block as `wxyz` while
`ReprojErrorCost` reads the block as `xyzw` (Eigen memory layout). Both directions of
the wrong conversion were self-consistent, so it failed silently: initial cost equal
to final cost, every parameter block bit-identical. This has now been corrected
in place (dead-code-neutral), and a proper BA lives in `full_system_ow_working/ba.py`:
bounded sliding window, gauge pin, Huber loss, covariance-weighted reprojection,
cost-decrease acceptance test plus a translation/rotation shift guard with
copy-then-commit semantics (a refused solve writes nothing), and structured
`BaResult` logging — wired into the keyframe commit behind the default-OFF
`run_ba_on_keyframe` flag so the paired A/B baselines stay reproducible. Two-stage
refinement (photometric LM, then geometric PnP) remains the live default, and it is
open-loop between stages: the PnP result is never fed back into the photometric
solve. (Why the photometric residual can't simply join the pyceres graph:
`pyceres.CostFunction` cannot be subclassed from Python — the C++ pure virtual
`Evaluate` is unbound — so the torch LM must stay a separate solve whose output
enters BA only as a covariance-weighted prior.)

---

## 2. Phase 0 — Fix the estimator, and make failures measurable

Do this first. Everything else is unmeasurable until §9 exists, and the documented
blocker sits here.

### 2.1 The primary failure is upstream of every threshold

`docs/bundlesdf-gate-ab-findings.md` and the comment block at
`run_full_system_bundlegs_ow.py:207-218` both converge on: the dominant error is
depth-buffer scale error that **tracks PnP rejection rate**, and in the bad clips PnP
doesn't reject at all — it never runs.

The mechanism is in `full_system_ow_working/geometric_tracker.py:720-723`:

```python
if len(pts2d) < getattr(self.cfg, "min_pnp_inliers", 20):
    print(f"[GeoTracker] Insufficient tracks at frame {frame_idx} (found {len(pts2d)})")
    self.poses[frame_idx] = T_guess.copy()
    return False, 0
```

19 tracks and below → bail → 0 inliers → the pose is frozen at the motion-model guess
→ drift accumulates unopposed for as long as track count stays low. No amount of
threshold tuning fixes this, because no threshold is evaluated.

The findings doc quantifies this: on clip-003318, **66 of 116 rejections are this
sentinel path**, and the GS map had pruned 363 → 13 Gaussians over four keyframes
right before it. That is a starvation cliff, not a threshold problem — and a cliff,
not a slope: the tracker jumps from "full PnP" to "no measurement" at a 1-point
boundary. Softer floors would convert most of those sentinels into usable-but-weak
estimates instead of no measurement at all.

**Fix P1 — never let the track buffer starve.** The object GS *already contains* ~10k
object-space means, which is a strictly better 3D point source than the initial depth
cloud. Add a top-up path: when active observations fall below `min_pnp_inliers`, warp
the GS means through the guess, keep those within the predicted mask, and run PnP on
that instead of returning. The map should be a *fallback measurement*, not the sole
measurement — and this also breaks the circularity in P2.

**Fix P2 — break the self-referential map.** `update_object_points`
(`geometric_tracker.py:338-357`) does a KDTree nearest search with a 5 cm cutoff and
overwrites **every** track that has a nearby point:

```python
dist, idx = tree.query(t['pt3d'])
if dist < 0.05:
    t['pt3d'] = new_points[idx].copy()
```

The map was fitted to the object at the (already drifting) pose, then handed back to
the tracker as its measurement. That is a tautology with a 5 cm tolerance, not a
correction. Keep the original depth-unprojected cloud as the reference, store per-track
corrections in object space with a covariance, and update only tracks that have *new
depth observations*. Never overwrite from the renderer.

### 2.2 PnP has no outlier defense

`geometric_tracker.py:742-745` is a single solver call seeded by the guess:

```python
res, info = poselib.estimate_absolute_pose(
    pts2d_np.astype(np.float64), pts3d_np, cam_dict,
    {'max_reproj_error': self.cfg.ransac_thresh, 'min_iterations': 100, 'max_iterations': 1000},
    initial_pose)
```

No RANSAC loop in the tracker, and unlike `step_informed` (which prunes observations
after the fact at `:641-661`), `step_informed_with_occlusion` does no post-PnP pruning
at all. The jump gate at `:761-777` is the *only* outlier defense in the pipeline,
which is why it exists and why it's being blamed for everything.

**Fix P3.** RANSAC on a 5-point init, then Huber-weighted linear refinement on the
inliers, then re-projection pruning (port the `:641-661` block into the loop path).
`test/test_ba_vs_e5p1_seq.py` already exists as a BA-vs-5-point-1-sequence harness.
Once outliers are handled structurally, the jump gate can shrink to a sanity limit
instead of carrying the load.

### 2.3 `success` is always `True`

`geometric_tracker.py:786-790`:

```python
if getattr(self.cfg, "do_refine", True) and not skip_pnp:
    inliers = info.get('inliers') if info is not None else None
    self.refine_pose(...)
return True, n_inliers
```

When `res is None` or the jump gate fires, the function returns `True` with the guess
in `self.poses[frame_idx]`. The caller's `not success or n_inliers < min_pnp_inliers`
check is only saved by the inlier count. **Fix P4** — return `(accepted, n_inliers,
T)` where `accepted` is genuinely False when the guess was reused. `pnp_rejections`
(`bundlesdf_gs.py:657`) is otherwise not measuring what the eval header claims it
measures.

### 2.4 Replace the jump gate with a test, not a constant

`_jump_limit()` (`bundlesdf_gs.py:215`) already exists and already widens from observed
motion with `0.0025 m / 0.15°` floors. The principled version of that is a Mahalanobis
test on the innovation:

- compute the 6×6 information matrix from the active measurements each frame (this
  also gives the covariance the test needs, and is the observability diagnostic in P6);
- test `‖ΔT‖² / σ² ~ χ²₆` against a chi-square threshold;
- **log the innovation**, so a rejection is an observable event rather than a print.

### 2.5 Observability-aware weighting (P6)

Photometric-only LM on a sphere about the view axis is unobservable, and the current
LM (`obj_gs.optimize_wrt_image_lm`, pyramid `(4,10),(2,10),(1,20)`) converges happily
into a degenerate solution with no indication it did so. Compute the 6×6 information
matrix from the measurements actually available, then:

- down-weight the degenerate directions instead of trusting LM;
- feed the null-space directions to P2's shape prior (§3.5), which is where they get
  observability back;
- emit the condition number to the health monitor (§8.3).

This one change also converts "rejection" from a heuristic into a statement about
*what the geometry could not see*.

---

## 3. Phase 1 — Close the scale loop

The findings doc is clear that scale error dominates and its **sign is not fixed**
(`s = 0.17` compressed to `2.71` expanded). The depth buffer is the only scale oracle
in the system (`run_full_system_bundlegs_ow.py:62-70`), so nothing downstream can
detect a scale error — it can only be diagnosed after the run by `umeyama()` at
`:399-420`.

### 3.1 Two silent scale absorbers you should know about

The mapper already re-estimates scale per keyframe at `obj_gs_mapping.py:424-450`:
`s = median(render_depth / obs_depth)`, robust-filtered, then

```python
s = np.clip(s, 0.1, 10.0)
```

A ±10× per-keyframe clamp means systematic scale drift is absorbed silently rather than
reported. There is also a second, parallel depth-alignment in `align()`
(`obj_gs_mapping.py:720`), and `align_depth_tracker` defaults **False** while
`align_depth_mapper` defaults **True** (`run_full_system_bundlegs_ow.py:141-150`) —
the tracker and mapper run in different depth units by configuration, and the comment
concedes this reproduces "today's actual behavior". Unify these into one scale state.

### 3.2 Scale as a state variable, not a post-hoc correction

- **S1 — a metric anchor that isn't the depth model.** Ranked by cost:
  1. **Known object extent.** HOT3D objects have known physical sizes. Fit `s` so the
     object GS bounding extent matches the known extent. This is precisely the right
     anchor because the measured error *is* scale compression of the object trajectory,
     and the object model is the thing being compressed.
  2. **A static-environment anchor.** The scene model is fitted in the same
     non-metric units, but any metric landmark in the environment (table dimensions, an
     AprilTag, a fiducial) placed in world space anchors it. `pycolmap` 3.13 is already
     a pixi dependency and supports marker triangulation.
  3. **Robot end-effector pose**, if these are real robot runs. *Caveat:* the repo's
     `exp/exp_warp_two_view_refine_geometry_ufm.py` imports `uniflowmatch.models.ufm`,
     which is **not installed** in the pixi env (verified by filesystem search), so that
     experiment is currently un-runnable, and there is no UMI calibration data in `data/`.
     Treat that path as a design reference, not a shortcut.
- **S2 — make scale a parameter the optimizer sees.** Add `s` to the BA state with a
  slow-drift prior, estimated jointly with pose. `triangulate_tracks` already has a
  `force_scale=False` branch that hardcodes scale correction off; that machinery exists
  and is unused.
- **S3 — scale-free reprojection.** Use `p = ray_dir × range` per pixel so scale
  enters the residual exactly once and can be estimated explicitly, instead of being
  baked into the 3D points and then fought over by three consumers (tracker tracks, GS
  means, GS scales, `ray_dist`).

### 3.3 Scale error invalidates every absolute-magnitude threshold

This is the under-appreciated consequence of §3.1. A long list of constants in the
pipeline is written in meters, and the pipeline's unit is whatever the depth buffer
decided it was:

| constant | site | what it is |
|---|---|---|
| `dist < 0.05` | `geometric_tracker.py:352` | KDTree update radius |
| `occlusion_margin` 0.05 | `geometric_tracker.py:694-697` | occlusion margin (not a declared config field, so the `getattr` default always wins) |
| 5 mm baseline guard | `geometric_tracker.py:440` | triangulation minimum baseline |
| `render_depth - 0.01` | `obj_gs_mapping.py:463` | densification "closer" test |
| `scales = 0.02` | `obj_gs_mapping.py:240` | initial Gaussian radius |
| `prune_screen_size_th = 100.0` | `obj_gs_mapping.py:553-556` | pruning cutoff |
| 0.95× inlier preference | `geometric_tracker.py:761` | guess-vs-PnP tie-break |

On clip-003318, where `s ≈ 0.17`, the 50 mm `occlusion_margin` is roughly 29% of the
scene scale, and 5 mm is a large fraction of it. That is exactly the class of check
that discards an observation *before it can count toward the 20-point guard* — so the
scale error doesn't just compress the trajectory, it silently removes measurements.
The scale fix and the starvation fix are the same fix.

Make all of these relative to a running estimate of object size or mean depth, not
absolute meters.

### 3.4 A broken renderer path for `gs_type="3d"`

`render_3dgs` (`gs_dyn_obj/gs_rendering.py:195`) returns literal zeros where depth and
normal should be:

```python
# 3dgs doesn't provide direct median depth like 2dgs in info
depth = torch.zeros(1, height, width, device=device)
# Normal estimation (placeholder for 3dgs)
normal = torch.zeros(3, height, width, device=device)
```

The PGSR multi-view loss is guarded on `render_normal.any() and render_depth.any()`, so
in 3DGS mode those losses don't just underperform — they vanish silently. `use_pgsr`
is a config switch with no effect when `gs_type="3d"`. Worth fixing before any
"PGSR helps" or "PGSR doesn't help" conclusion is drawn from a 3DGS run.

### 3.5 The missing observability dimension, cheaply obtained

The repo deliberately stopped using GS-derived depth for tracker points —
`bundlesdf_gs.py:544-547` reads `curr_depth = depth` twice with the comment *"Robustly
use only monocular depth for points to avoid GS reconstruction noise."* That's
reasonable, but it throws away the one thing the model contributes that depth doesn't:
**shape**.

Use the object GS as a *shape prior* instead: constrain each track's ray range so the
point lies on the model surface, with a Gaussian kernel along the ray. The ray
*direction* becomes fully determined by (shape, camera) and only range stays free. That
is exactly the degenerate direction photometric-only pose cannot see, and it costs
nothing because the model is being rendered for the LM step anyway.

### 3.6 Consensus across depth sources (out of the box)

`DEPTH_SOURCES` at `run_full_system_bundlegs_ow.py:227-237` already enumerates 8
buffers, and the comment notes they rank differently on different clips. Instead of
picking one, treat them as redundant disagreeing sensors: estimate pose on 2–3, keep the
one whose innovation is smallest *and* most consistent with the motion model. When
sources agree you are confident; disagreement is a health signal. You get a consensus
scale anchor and a per-frame disagreement metric from plumbing that already exists.

---

## 4. Phase 2 — Perception (remove the oracle mask)

### 4.1 Define the interface first, backends second

```
Perceiver -> list[Instance(t, mask, bbox, class_id, descriptor)]
```

Write the interface, then three backends behind it: GT (current, for measurement),
SAM2 (deployment), BG-residual (§4.3). Nothing else in the system should touch mask
files.

### 4.2 The object model is the prompt — this is the load-bearing idea

You already can render the object GS at an arbitrary pose (`render_current_view`,
`bundlesdf_gs.py:299-325`) and already have a cost function that optimizes pose from a
render (`optimize_wrt_image_lm`). Close the loop:

1. predict pose cheaply (last known + velocity);
2. render the model → outline the silhouette;
3. feed that outline as a SAM2 box/point prompt → new mask;
4. refine pose on the new mask;
5. update the model.

Appearance *guides* perception, not just photometrically fits it. Each pass makes
detection easier, so tracking and perception reinforce instead of competing. Both
halves of this already exist; the loop is a few hundred lines.

### 4.3 Self-supervised mask from your own background model

The loop already renders the static scene (`run_full_system_bundlegs_ow.py:600`) and
already computes `α_bg`. Where `|img − bg_render| > τ` inside a region-growing search
you have a mask with **no oracle and no detector**, computed inside the loop. It's
noisy, which is fine: use it for *continuity* and keep the GT masks only for training
and evaluation. The `mask_loss_weight = 20.0` already assumes this signal exists.

### 4.4 Instance descriptors

Per instance, store an appearance descriptor (DINO/CLIP patch features, or a cheap
start: mean of GS SH DC terms + color histogram of the model). This is what
association (§5) and re-localization (§6) both need, and it's what turns the object GS
into a *library* entry rather than a one-shot artifact.

---

## 5. Phase 3 — Association, identity, multi-object

### 5.1 A lifecycle, not a global

Currently there is no state machine: `self.obj_gs` is one attribute, there is no path to
start a new object, and no path to recover from loss. Add

```
NEW -> TRACKING -> OCCLUDED -> LOST -> RELOCATED -> CLOSED
```

with explicit transitions and durations. `OCCLUDED` uses the motion model to hold for
`k` frames — which requires §7.2's two-velocity model.

### 5.2 Association as a chi-square, not a threshold

Fuse three independent measurements per (instance, detection) pair:

- geometry — IoU of predicted 2D bbox vs. observed;
- appearance — cosine of descriptors from §4.4;
- motion — innovation of the observed 2D displacement vs. the velocity model.

Hungarian assignment with a chi-square gate on the fused innovation. If two descriptors
get close, **do not merge** — keep both and let geometry break the tie. HOT3D has one
object per clip so this is future-proofing, but it is cheap and it's the only
association design that doesn't silently destroy identities.

### 5.3 Multi-object refactor (do this before §4 or §6 lands)

`BundleSdfGS` becomes `dict[obj_id -> ObjectGS]` and `dict[obj_id -> tracks]`, with the
camera pose as a **shared** per-frame variable and objects independent given the camera.
That separation is what makes a factor graph (§8.1) trivial later, and it's mechanical
now because the code already treats one object at a time — just make the "one" a key.

### 5.4 Dead code standing in the way

- ~~`run_ba_python` (`bundlesdf_gs.py:831`) has no caller.~~ **Resolved** — it is now
  the BA hook: called on keyframe commit when `run_ba_on_keyframe` is set, and
  rewired to `ba.py`'s bounded-window, acceptance-gated `bundle_adjust`.
- `step_informed` (`geometric_tracker.py:513`) is a parallel implementation of
  `step_informed_with_occlusion` (`:669`) with a different outlier strategy; only the
  latter is on the hot path. Pick one.
- `init_gs_model` has unreachable code after `return True` at
  `full_system_ow_working/bundlesdf_gs.py:304` (and `full_system_ow/bundlesdf_gs.py:164`
  — the same bug in both copies) — lines 305–370 including the entire point-cloud
  unprojection, normal computation and ray parameterization never run; that branch only
  survives when `pre_initialized_gs is None`, which never happens because the caller
  always builds `initial_gs` first (`run_full_system_bundlegs_ow.py:526-530`). Either
  that seed path is dead or the whole seeding story is duplicated and out of sync.

---

## 6. Phase 4 — Occlusion and re-localization

This is the part that decides whether the system survives real deployment.

### 6.1 Today, lost tracks are lost forever

`use_match_projections` defaults `False` (`geometric_tracker.py:710-711`), so there is
no track recovery path, no re-init path, and no re-localization path. When tracking
dies it stays dead.

### 6.2 The object GS is already a localizer

Re-localization needs two things: a database to retrieve from, and a cost function to
verify a candidate. **The cost function already exists** — `optimize_wrt_image_lm` is
exactly "render the model at a candidate pose and compare to the image." Only the
retrieval is missing.

- **R1 — global visual index over object keyframes.** SuperPoint descriptors (already
  in the stack) indexed by brute-force/FLANN over ~100k vectors. For tens of objects
  this is microseconds and needs no vocabulary tree. `pycolmap` 3.13 is available if
  you want a real DBoW index later.
- **R2 — retrieved candidate → verify with the LM.** No new code beyond retrieval.
- **R3 — a degraded-mode ladder**, gated on information available rather than on a
  frame counter:
  - FAST — sparse PnP + motion model (every frame, ≤5 ms);
  - MEDIUM — GS photometric LM (when tracks are thin or obs. matrix is near-singular);
  - SLOW — mapper keyframe update (when the model is meaningfully new).

  Right now all three run every frame ungated, which is why `num_steps_dyn = 150`
  coexists with a per-frame LM refinement. Gating is also what makes real-time
  possible.

### 6.3 Occlusion prediction, not occlusion reaction

`use_occlusion_check` compares `depth[iy, ix]` against the projected point depth
(`geometric_tracker.py:694-697`) — a *reaction* after the fact. Better: predict from
the static scene model *before* the frame, using the rendered `α_bg` and the object's
velocity. You can then (a) switch estimator modes ahead of time, (b) budget the
`OCCLUDED` hold duration from predicted time-to-occlusion, and (c) know in advance
whether a frame's measurements will be degenerate.

---

## 7. Phase 5 — Interaction-aware dynamics (the HOT3D-specific win)

Every frame in the dataset has a `hand_masks/` channel, and `run_full_system_bundlegs_ow.py:584-589`
already fuses it into the static mask. The hand is used only to *exclude* it from the
scene map. It is available as a hard geometric constraint and nobody is using it.

The object's dynamics change **phase**: free-floating → grasped → placed. The tracker
currently treats every frame identically, and the crudest possible guess of this shows
up at `bundlesdf_gs.py:377-383`:

```python
if v_t > 0.5 or v_R > 20.0:
    print(f"... High velocity detected ... Discarding motion model and using last T_OW.")
    self.tracker.poses[self.cnt-2] = self.tracker.poses[self.cnt-1].copy()
    # Discard motion model prediction and use last T_OW (assume stationary in world)
```

"Move fast, therefore assume the object is stationary" — this is a grab.

### 7.1 Three phases, three estimators

In `full_system_ow_working/` this branch is print-only (`:533`) — the copy in
`full_system_ow/` also freezes the pose. Either way, the intent is the same: high
velocity means "stop tracking."

Detect phase from hand-object IoU, contact persistence, and hand velocity:

- **FREE** — float `T_WO`, normal loop.
- **HELD** — rigid lock: `T_WO = T_WH @ T_HO`. `T_WH` is metric (robot), `T_HO` is
  estimated **once** at grasp onset from the current object estimate. **Stop the PnP
  and LM loops entirely.** This is a hard constraint, not a regularization, and it is
  the single largest robustness win available from data you already have: while held,
  zero drift can accumulate, by construction.
- **PLACED** — fix `T_WO` (or a one-parameter creep), and pour effort into the GS map
  instead. This phase is the *least* trackable for pose and the *most* informative for
  mapping, so the current uniform effort allocation is inverted.

### 7.2 Change-point detection for grasp and release

Detect grasp when the object-pose innovation variance collapses toward zero — the
observability matrix from §2.5 gives you this for free. Detect release when it
reopens. This replaces the velocity heuristic with something observable.

### 7.3 Two-velocity motion model

Replace the discard-and-assume-stationary branch with two coexisting models: a
recent-window velocity for short horizon, and a longer-horizon smooth model for
occlusion gaps. Select by predicted time-to-occlusion (§6.3), not by observed jump.
This handles "grab → object suddenly moves fast" without ever assuming stationarity.

### 7.4 The contact constraint as a factor, not a branch

Add a zero-relative-velocity factor between hand frame and object to the BA
 (§8.1). Then phase detection is optional — the optimizer finds the constraint itself.
This is the principled version of §7.1, and `pyceres` 2.6 is already a dependency.

### 7.5 Placed objects become landmarks (out of the box)

An object in `PLACED` is stationary, so it becomes a camera landmark: known geometry,
permanently available, never representable by the scene model. Over a long session you
accumulate a **scene-level object-landmark layer** for camera re-localization (§8.2).
This converts the phase you can't track into the asset you need most.

---

## 8. Phase 6 — Camera tracking, and the factor graph that ends the gate wars

### 8.1 One factor graph, not two pipelines

Today camera and object live in separate processes with no shared optimizer
(`data_loader_worker` / `static_scene_worker` / `dynamic_worker`
`run_full_system_bundlegs_ow.py:329-423`), and the tracker communicates with the mapper
through `multiprocessing.Queue` + `Manager().dict()`. Every "gate" in the code is a
local stand-in for what a factor graph provides.

Variables: `{camera_i, object_j_per_frame_pose, object_j_model_params, s (scale),
exposure_i, contact_phase_j}`. Factors: photometric-GS (foreground), photometric-GS
(background), depth consistency, reprojection, contact lock (§7.4), metric anchor
 (§3.2), velocity prior, shape prior (§3.5). This single change:

- replaces every hand-tuned gate with a covariance-weighted residual;
- gives you degeneracy detection and localization for free;
- enables sliding-window with pose-graph loop closure;
- makes phase switching a matter of adding or dropping factors.

`dyn_obj_ba_colmap.py` (per-frame object-pose BA, no GS coupling) and `eval_all.py`'s
`dyn_obj_ba_colmap` mode are prototypes; `test/` already contains
`test_ba_vs_e5p1_seq.py`, `test_ba_mpsfm.py`, `test_ba_dis.py`, `test_ba_hot3d.py` —
a real body of pose-solver comparison work, none of it wired into the main loop.

### 8.2 Replace the oracle camera

`T_CW` comes from GT extrinsics today. Two paths:

- **C1 — track the camera against your own static GS model.** Photometric + depth
  consistency, seeded by a short GS-SLAM bootstrap on the static regions. You already
  have the scene model, the renderer, and `gsplat`. This is the interesting option
  because it makes BG and FG share a camera estimate — today the FG pose error is
  bounded by a camera error that *nobody measures*.
- **C2 — COLMAP incremental on the static regions** (`pycolmap` 3.13 is already a dep;
  the repo already uses `poselib` for PnP). Pragmatic, less integrated.

Conveniently, **both are immediately measurable on this data**: HOT3D gives GT camera
extrinsics as input, so you can evaluate a camera tracker with `--gt_camera true`
(`eval_all.py`) before it is needed anywhere else.

### 8.3 A health monitor as a first-class output

Emit per frame: track count, inlier ratio, innovation, rejection reason, observability
condition number, map Gaussian count, per-stage latency, and a `confidence` in
`{TRACKING, DEGRADED, LOST, UNKNOWN}`.

This matters more than ATE if the system is feeding a grasp controller. The current
system has no notion of its own uncertainty, and no metric in the eval can express
"the error was concentrated in one axis" — `ate` is `‖t_est − t_gt‖` only
(`run_full_system_bundlegs_ow.py:624`), so scale compression along an axis is invisible
until the after-the-fact Umeyama.

---

## 9. Phase 7 — Evaluation, the repo's actual weak spot

The findings doc is exemplary work: it established a 4.2× run-to-run swing on identical
config and seed, and showed a "fix" that looked like a win was a 1.7× regression in a
paired test. The conclusion — *single-clip comparisons are unreadable* — is correct and
should be the operating rule. The gap is that the benchmark still can't say *why* a
number moved.

### 9.1 Decompose into five metrics, one per sub-problem

| Metric | What it isolates | Measurable today? |
|---|---|---|
| Perception: mask IoU / mAP@0.5 per frame | perception only | ✅ GT masks in `obj_masks/` |
| Association: ID switches, fragments, MOTA | identity continuity | ❌ single-object today |
| Camera: ATE/RPE in cm/deg | camera tracking | ✅ GT extrinsics, `--gt_camera` |
| Object pose: raw ATE, scale-only, Umeyama | the core | ✅ `run_full_system_bundlegs_ow.py:700-710` |
| Map quality: PSNR / SSIM / α-IoU | appearance | ✅ `eval_all.py` |

The object-pose row needs two additions: **rotation error in degrees** and **per-axis
translation error**. `umeyama()` at `:399-420` already returns `s, R, t` — the rotation
is computed and discarded.

### 9.2 A failure-injection benchmark (the highest-value addition in this document)

A script that takes a clip, applies a perturbation, and reports **degradation slope**,
not absolute ATE. This is what would have caught every current problem without
experimenting:

| Injection | Isolates |
|---|---|
| depth × s for s ∈ {0.3, 0.5, 1.5, 3.0} | scale robustness (§3) |
| drop mask every k-th frame | association + occlusion (§5, §6) |
| start at a random mid-clip frame | initialization — `--init_frame` exists |
| zero depth in the object region on some frames | the PnP input path (Fix P1) |
| jitter GT extrinsics with noise | camera sensitivity (§8.2) |
| swap in another object's mask for a few frames | association (§5.2) |

In a pipeline where one run swings 4.2×, slope across injected fault levels is the only
statistic with enough dynamic range to be readable.

### 9.3 Make the statistical rules operational

- paired seeds across configs, documented and enforced (already written down, not
  enforced);
- always report `ATE = f(rejections)` — the correlation is already found, make it a
  standard output;
- always report the depth source and scale factor next to the number, since
  `run_full_system_bundlegs_ow.py:62-70` establishes that different depth sources
  measure *different units* and their ATEs are not comparable;
- **fix the image-metric leakage.** `full_system_ow/eval_full_system.py:527-533` copies
  GT image pixels into the render wherever the GT hand mask is positive, *before*
  PSNR/SSIM/LPIPS are computed. Those numbers are measured on a composite that includes
  the ground truth, so they overstate how well the render explains the observation. This
  is harmless while perception is GT-fed — it becomes material the moment §4 makes the
  mask estimated, because a wrong mask then *helps* the score. Report the metric twice
  (masked-out and masked-in) and treat masked-out as the truth;
- a **held-out no-GT mode** where perception, camera, and pose are all estimated. That
  is the number that matters for deployment, and it is currently not measurable at all.

### 9.4 A reproducible baseline matrix

Before adding any phase above: fix `data_root` + `clip_id` list + `seed` + `depth_source`
+ `num_frames`, run every config against it, and land the results in
`eval_results_final.json`. The repo already has three of these files; the missing piece
is a single matrix, not three isolated runs.

---

## 10. Phase 8 — Engineering (prerequisite for all of the above)

### 10.1 One pipeline

`full_system_ow/`, `full_system_ow_working/`, `BundleGS/`, `BundleGS_keyframe/`,
`gs_dyn_obj/`, `gs_dyn_obj_bkup/`, `gs_dyn_obj_restored/` are not just duplicated — they
have **diverged**. md5 by file, across the top-level and the two `full_system_ow*` trees:

| file | distinct copies | verdict |
|---|---|---|
| `geometric_tracker.py` | 4 (bfc45779, fb2daf04, e18ab7a1, 2c43d345) | divergent |
| `obj_gs_mapping.py` | 4 (8eb3979a, 0c10f0d6, db86e24c, 87c252b1) | divergent |
| `bundlesdf_gs.py` | 5 (ec2b8058, a4b9eedd, 9ec3a879, 5a1e6c29, 8c080d8f) | divergent |
| `grouped_gs.py` | 1 (af24c738) | identical |
| `obj_gs.py` | 1 (673a1016) | identical |

The render stack is in sync; the tracker, the mapper and the orchestrator are not. So
fixes applied to one copy never reach the others — the most plausible explanation for
`GeoTrackerConfig` being defined **seven times** (and twice inside
`full_system_ow_working/` alone: `geometric_tracker.py:12` and `bundlesdf_gs.py:34`),
and why the
discard-on-high-velocity behavior in `full_system_ow/bundlesdf_gs.py:377-383` is
print-only in `full_system_ow_working/bundlesdf_gs.py:533`. Nothing tells you which copy
a given run actually used. Move `geometric_tracker.py` and `obj_gs_mapping.py` into a
`src/` package, keep the ~45 KB `full_system_ow_working/bundlesdf_gs.py` as
authoritative, and delete the rest. A layered architecture over five copies is
unbuildable, and over five *divergent* copies it is actively dangerous.

### 10.2 Silent failures

- **Per-phase latency has never been attributed.** `GeoTracker` measures its own stages
  and `print_timings()` is written, but its only caller is
  `BundleGS_keyframe/test_hot3d_timing.py:184` — a dead tree. The live pipeline never
  prints it. So "~1 s per frame at 221 frames" is an unattributed budget: nobody knows
  whether `num_steps_dyn=150` is expensive because of LM, because of DIS, or because of
  the mapper, and nobody can find out without adding the call. This is literally the
  cheapest item in the whole plan — one line, and it turns §8.2's latency question from
  a guess into a measurement.
- The mapper dies silently: `obj_gs_mapping.py:689-692` catches everything and `break`s
  out of the process, with no signal to the tracker. If the mapper dies at frame 40, the
  tracker keeps running happily on the stale GS and the eval reports a clean number.
- Reproducibility: `.gitignore` excludes `PoseLib/`, `gsplat/`, `third_party/` and
  `data/`, and the runner imports `gs_rendering_gsplat` from `gs_dyn_obj_bkup/`. A fresh
  clone cannot run the authoritative pipeline. The vendored `gsplat` needs to be a pin
  or a submodule, not an ignore rule.
- Unused dependencies that imply un-run code: `open3d`, `trimesh`, `plyfile`, `tensorboard`,
  `evo`, `rerun-sdk`, `scipy` are all declared in `pixi.toml` but nothing in the live
  pipeline imports them. `evo` in particular would give the trajectory analysis in §9.3
  for free.
- `kf_queue.put` has no backpressure and no timeout.
- `multiprocessing.set_start_method('spawn', force=True)` at import time
  (`full_system_ow/bundlesdf_gs.py:18`) breaks embedding and any test harness.
- `run_full_system_bundlegs_ow.py:584-589` **overwrites `dynamic_mask` in place** with
  the cumulative union of object+hand masks, so any later `disable_bg`-style use of the
  original mask is wrong.

### 10.3 Async pipeline with documented latencies

`loader → perception(GPU) → camera track → object track → mapper`, each a bounded queue
with a stated max latency. The current 3-process `spawn` + `Manager().dict()` design
works, but accidentally.

### 10.4 Structured logging

Per-frame JSONL: tracks, inliers, innovation, rejection reason, map count, per-stage
latency. Without it, §9.2 regressions are undiagnosable.

---

## 11. Things I checked that do not exist

Flagging these because they were plausible-sounding and could have shaped the plan:

1. **There is no `gsplam/` directory.** Not on disk, not in git, not a submodule, no
   branch contains it. Any plan built on "gsplam already has metric-scale GS-SLAM" is
   built on nothing.
2. **`uniflowmatch` is not installed** in the pixi env (verified by filesystem search).
   `exp/exp_warp_two_view_refine_geometry_ufm.py`, `test/test_ba_ufm.py` and
   `test/test_geometry_tracker_informed_ufm.py` all import
   `uniflowmatch.models.ufm` and will not run. It is also not declared in `pixi.toml`,
   so `pixi run` would fail. The UFM line of experiments is aspirational, not current.
3. **There is no `data/umi_calibration/`.** Nothing by that name is committed. Note
   that `data/` entries are symlinks out to `/media/shzhou/T7_2/...`, so dataset files
   genuinely do not live in this repo — the absence of an absolute scale anchor is a
   data-acquisition gap, not a code gap. §3.2's robot-anchor option needs that data
   acquired first.
4. **`eval_full_system.py` is easy to look for in the wrong place.** It does exist —
   at `full_system_ow/eval_full_system.py` (30 KB) — but not at the repo root and not
   in `full_system_ow_working/`, which is where the authoritative runner lives. The
   Umeyama/scale-aligned eval used by the runs that actually get reported is
   `run_full_system_bundlegs_ow.py:399-420` and `:700-710`, plus a `umeyama()` copy in
   each of the 10 `test/` scripts. Four implementations of the same alignment is where
   a "scale-aligned vs Umeyama-aligned" discrepancy gets born.
5. **`test_gs_to_planar_patch copy.py` and `run_full_system_ph2d copy.py` exist in
   the working tree** — uncommitted duplication; delete before the Phase 8 cleanup.

---

## 12. Roadmap

| Phase | Content | Removes | Why this order |
|---|---|---|---|
| **0** | Fix P1–P6 (§2) + §9 instrumentation | nothing | The documented blocker is here; nothing is measurable without §9 |
| **1** | Scale: S1–S3 + shape prior (§3) | depth-buffer as sole scale oracle | Dominant error, sign not fixed, no threshold can help |
| **2** | Multi-object refactor + lifecycle + association + SAM2 prompting (§4, §5) | GT masks | The interface must exist before perception can be swapped |
| **3** | Relocalization + phase detection + two-velocity model (§6, §7) | fragility to loss | This is what makes it robust in the wild |
| **4** | Camera tracking C1/C2 (§8.2) | GT camera | Measurable now with GT extrinsics as input |
| **5** | Factor graph + health monitor (§8.1, §8.3) | all hand-tuned gates | Long-term home; the gates are local stand-ins |
| — | Engineering cleanup (§10) interleaved, starts now | duplication | Prerequisite for every phase above |

**If you only do one thing:** Phase 0's Fix P1 — never let the track buffer starve,
using the object GS means as the fallback measurement source. It is the single change
that addresses the failure the findings doc identifies as upstream of everything else,
it uses a data source that already exists in object space, and it requires no new
dependency.

**If you only do one experiment:** the failure-injection benchmark (§9.2). It is the
only artifact in this document that would have predicted every other finding, and it
turns a 4.2×-noisy pipeline into one you can actually steer.

---

## Appendix: out-of-the-box ideas, ranked by how much they leverage what you have

1. **The model is the prompt** (§4.2) — closes perception with a renderer you already have.
2. **Contact lock as a hard constraint** (§7.1) — zero drift while grasped, from a mask channel you already load.
3. **Placed objects as camera landmarks** (§7.5) — turns the untrackable phase into the re-localization asset.
4. **Shape prior from the object GS** (§3.5) — buys back the degenerate pose direction for free; the renderer is already running.
5. **Consensus across the 8 depth buffers** (§3.6) — redundancy you already own, currently treated as a config knob.
6. **Learn the failure, not the estimator** — you have GT poses, GT cameras and GT masks, so you can *label* every frame "was this trustworthy?", train a small MLP on (track count, inlier ratio, depth variance in mask, motion innovation, observability condition number, previous residual) → (accept, uncertainty), and retire the gate stack. Uses the oracle as supervision for robustness, then removes it.
7. **A persistent cross-episode object library** — store each object's GS + descriptor + known extents keyed by appearance hash. New episodes match against the library and start from the stored model, so tracking becomes cumulative and re-initialization from 10k depth points becomes rare. Combined with idea 3, this is a spatial memory of the environment's objects.
8. **Uncertainty as a first-class output** (§8.3) — the value to a manipulation consumer is not ATE but *when to trust the pose*.
9. **Instrument the second opinion before trusting it.** In one sampled log window the
   photometric LM moved the PnP translation by `0.000000m`. Stage 1 is paid for every
   frame — `num_steps_dyn=150` iterations — and if it is silently contributing nothing,
   the budget is being spent on no effect. Log the pre/post-pose delta per frame and
   count how often Stage 1 actually changes the answer. Cheapest experiment in the
   document, and it decides whether Stage 1 is an asset or overhead.
9. **Degraded-mode ladder gated on observability** (§6.2) — the current system runs all three estimator tiers every frame ungated.
10. **Two-level map: cheap tracks + expensive model** — the `fast` track variant already exists and is unused; promote it to the primary and demote the GS to keyframes.
