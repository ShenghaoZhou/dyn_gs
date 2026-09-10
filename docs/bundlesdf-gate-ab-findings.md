# BundleSdfGS: multi-sequence A/B of the GS keyframe rejection gate

**Status:** decision recorded — the gate fix is **reverted**, default `gate_kf_commit=False`.
**Commit:** `548d8c5` (reversion); `51b462c` is the commit under test.
**Date:** 2026-09-09.
**Runs:** 10 (5 clips × 2 gate settings × 150 frames), seed 0, `model_infer` depth.

---

## TL;DR

A hypothesis that looked like a 31% win on one clip is **wrong** once paired. Withholding
PnP-rejected frames from the GS keyframe buffer made tracking **worse on 0 of 5 clips** on
the scale-free metric, and 1.7× worse in aggregate.

The one thing that survived the test is unrelated to the fix: the death spiral traced to
gating `is_kf`, not to committing a rejected keyframe, so `is_kf` stays ungated.

A more useful finding came out of the same 10 runs: **the dominant error source is
depth-buffer scale error that tracks PnP rejection rate.** That is where the next work
should go, not the depth source and not this gate.

---

## Why the metric is ATE/travel, not ATE

The depth buffer's units **are** the absolute scale of the whole pipeline. Tracker 3D
tracks, GS means, GS scales and `ray_dist` are all lifted from `z = depth[iy, ix]`, so no
component can observe or correct its own scale error. The Umeyama scale factor `s ≡
est/gt` measured across these 10 runs spans **0.17 → 2.71**:

- clip-003318: `s` ≈ 0.17 — the depth buffer is ~6× too small
- clip-001924: `s` ≈ 1.58–2.71 — the depth buffer is up to ~2.7× too large

Absolute ATE is therefore meaningless across clips. clip-003318's 1.83 m raw ATE is a scale
artifact, not 30× worse tracking than clip-003333's 0.09 m. Dividing by the GT trajectory
arc length (identical input data in both variants of a clip) removes the depth-buffer scale
and leaves a comparable number. That is the only honest cross-clip metric available.

Umeyama RMSE is the secondary metric. It removes scale but not rotation/translation
alignment, and its cross-clip comparability is only as good as `s` being near 1 — which it
often is not.

---

## The A/B design

```bash
CLIPS="clip-003312 clip-003318 clip-003333 clip-001910 clip-001924"
for clip in $CLIPS; do
  for flag in --gate-kf-commit --no-gate-kf-commit; do
    pixi run python full_system_ow_working/run_full_system_bundlegs_ow.py \
        --num-frames 150 --no-vis --clip-id "$clip" --seed 0 \
        $flag --traj-dump "/tmp/ab_${clip//-/}_${flag}.$ext"
  done
done
```

Same codebase, same seed, same clip, one flag flipped. Anything else constant. Runs are
strictly sequential: there is one 8 GB GPU.

Two notes on the design:

- **The flag had to be a switch, not a code change.** `gate_kf_commit` was added as a
  config field precisely so before/after could be run on one tree. tyro exposes it as a
  bare flag — `--gate-kf-commit` / `--no-gate-kf-commit`; the value form
  `--gate-kf-commit False` is rejected.
- **clip-003333 was intended as a control.** Its 30-frame smoke test showed 0 rejections,
  so the gate should have been a no-op and the two variants should match. It wasn't — it
  rejected 8 frames and skipped 3 keyframes under the gate, so it is not a true control.
  It still came out worse with the gate on, which if anything strengthens the conclusion.

Clip selection spanned the range of behavior observed in a 40-clip completeness audit:
near-metric low-rejection (003312, 003333), heavily compressed high-rejection (003318),
mid-range (001910, 001924).

---

## Results

| clip | ATE/travel **gate ON** | ATE/travel **gate OFF** | raw ATE (OFF) | Umeyama `s` (ON / OFF) | rejections (ON / OFF) |
|---|---|---|---|---|---|
| clip-003312 | 0.091 | **0.071** | 0.1039 | 1.067 / 1.111 | 17 / 13 |
| clip-003318 | 1.372 | **0.743** | 0.9898 | 0.170 / 0.179 | 89 / 116 |
| clip-003333 | 0.098 | **0.051** | 0.0452 | 0.855 / 1.011 | 8 / 17 |
| clip-001910 | 0.179 | **0.124** | 0.1882 | 0.522 / 0.983 | 72 / 85 |
| clip-001924 | 0.113 | **0.087** | 0.2361 | 2.713 / 1.577 | 54 / 51 |

Aggregate across the 5 clips:

| metric | gate ON | gate OFF | ratio |
|---|---|---|---|
| mean ATE/travel | 0.371 | **0.215** | 1.72× worse |
| median ATE/travel | 0.113 | **0.087** | 1.30× worse |
| mean raw ATE | 0.5251 | **0.3127** | 1.68× worse |
| mean Umeyama RMSE | 0.1247 | **0.1015** | 1.23× worse |

Gate ON wins **0/5** clips on ATE/travel and **0/5** on raw ATE. It wins 2/5 on Umeyama
RMSE (clip-003318 by 0.053 m, clip-003333 by 0.015 m) — both on clips where the gate's `s`
happened to land within 0.009 and 0.156 of the other variant, i.e. the win is the `s`
correction, not better tracking.

The gate-off aggregate mean raw ATE of 0.3127 is in the same neighborhood as the original
`0.3211 m` baseline from before this work began, as expected: gate off *is* the pre-fix
code path.

### Segments (where the error grows)

| clip | gate | 0–24 | 25–49 | 50–99 | 100–149 |
|---|---|---|---|---|---|
| clip-003312 | ON | 0.053 | 0.084 | 0.103 | 0.226 |
| clip-003312 | OFF | 0.024 | 0.083 | 0.088 | 0.170 |
| clip-003318 | ON | 0.125 | 0.944 | 2.285 | 2.664 |
| clip-003318 | OFF | 0.135 | 0.217 | 0.656 | 2.137 |
| clip-003333 | ON | 0.031 | 0.059 | 0.114 | 0.105 |
| clip-003333 | OFF | 0.020 | 0.038 | 0.054 | 0.053 |
| clip-001910 | ON | 0.082 | 0.192 | 0.298 | 0.382 |
| clip-001910 | OFF | 0.089 | 0.184 | 0.180 | 0.248 |
| clip-001924 | ON | 0.089 | 0.078 | 0.177 | 0.657 |
| clip-001924 | OFF | 0.074 | 0.098 | 0.142 | 0.480 |

No variant shows an unrecoverable collapse; the largest segment means are clip-003318's
late segments in both variants. Error grows roughly monotonically in every run, which is
the signature of accumulated per-frame bias rather than a single failure event.

---

## The claim that got retracted

The previous reading of these numbers was:

> The DA3METRIC depth source cut 30-frame ATE 0.1715 → 0.0290 m but regressed to 0.6591 m
> over 150 frames because the GS mapper collapsed around frame 25. That collapse was PnP
> rejection **poisoning the map**, not scale: after gating the GS keyframe commit on
> `pnp_rejected` it improved to 0.4537 m.

That comparison was **unpaired and the baseline was unseeded.** It does not survive pairing.
The mechanism it proposed — a velocity-guess pose optimizing Gaussians toward itself, then
`update_object_points()` snapping surviving tracks onto those wrong means — is plausible but
was never actually tested, and the paired test says the poisoned-view harm is smaller than
the map-sparsity harm. Withholding a view costs the mapper geometry every single time and
never reduced the error here.

The docstring in `run_full_system_bundlegs_ow.py` that recorded this as fact has been
corrected to record the measured result instead.

## What survived

`is_kf` remains deliberately **ungated**, and the A/B confirms why. The gate only withholds
from the GS keyframe buffer; it never touched `is_kf`, which is what triggers
`add_new_points_from_depth`. Gating `is_kf` is what produced the earlier unrecoverable
failure (67/150 rejections → 140/150, mean ATE 0.19 → 0.45).

The A/B shows rejection counts did **not** diverge between variants:

| clip | 003312 | 003318 | 003333 | 001910 | 001924 |
|---|---|---|---|---|---|
| gate ON | 17 | 89 | 8 | 72 | 54 |
| gate OFF | 13 | 116 | 17 | 85 | 51 |

Comparable on every clip, and the gate-OFF runs — the ones that commit rejected keyframes —
never entered the unrecoverable state. So the death spiral traced to `is_kf` starvation,
not to committing a rejected keyframe. A rejected pose is bad data, but it is still a view.

---

## New finding: scale error tracks rejection rate, not the gate

Grouping the 10 runs by rejection count:

| rejections / 150 | clips | Umeyama `s` |
|---|---|---|
| 8–17 | 003312, 003333 | 0.855 – 1.111 (near metric) |
| 51–54 | 001924 | 1.577 – 2.713 (**expanded**) |
| 72–116 | 001910, 003318 | 0.170 – 0.522 (compressed) |

The sign is not fixed — one mid-rejection clip expanded, the high-rejection ones compressed.
What is fixed is the **magnitude**: low rejection is near metric, high rejection is far
off.

Both gate variants of clip-003318 sat at `s` ≈ 0.17–0.18 (0.1701 ON, 0.1790 OFF) despite
89 vs 116 rejections. That single observation is what disproves the gate as the cause of
compression.

### The mechanism is not the jump threshold

An earlier reading of this table blamed `max_pose_jump=0.05` being calibrated on
clip-003312's median 4.6 mm/frame motion and therefore "far too tight for clips with
larger motion, firing on 36–77% of frames." That is wrong, and the logs show where.

The rejection regime split across the 10 runs:

| log | total | inliers ≥ 10 | 1 ≤ inliers < 10 | inliers = 0 | median jump | median limit |
|---|---|---|---|---|---|---|
| 003312 gate OFF | 13 | 5 | 7 | 1 | 0.299 | 0.150 |
| 003312 gate ON | 17 | 4 | 12 | 1 | 0.331 | 0.135 |
| 003318 gate OFF | 116 | 0 | 7 | 109 | 0.000 | 0.086 |
| 003318 gate ON | 89 | 0 | 0 | 89 | 0.000 | 1.597 |
| 003333 gate OFF | 17 | 3 | 14 | 0 | 0.142 | 0.092 |
| 003333 gate ON | 8 | 3 | 5 | 0 | 0.169 | 0.097 |
| 001910 gate OFF | 85 | 3 | 40 | 42 | 0.245 | 0.101 |
| 001910 gate ON | 72 | 6 | 26 | 40 | 0.143 | 0.089 |
| 001924 gate OFF | 51 | 1 | 7 | 43 | 0.258 | 0.098 |
| 001924 gate ON | 54 | 1 | 14 | 39 | 0.180 | 0.133 |

Two separate regimes, and the jump threshold only explains one of them:

- **clip-003312 (and partly 003333): the jump gate is genuinely the limiter.** 5 of 13
  rejections had ≥ 10 inliers with median jump 0.299 m against a limit of 0.150 m — valid
  estimates being vetoed. This is a threshold problem, and it is confined to the
  near-metric clips.
- **clip-003318 / 001910 / 001924: 0 of 89–116 rejections had ≥ 10 inliers, and the
  median jump is 0.000 m.** Median jump 0.000 means the majority of those frames reported
  no jump at all. Those are not threshold casualties.

Reading the clip-003318 log directly shows what that majority is:

```
[GeoTracker] Insufficient tracks at frame 11 (found 19)
[BundleSdfGS] Rejecting PnP (inliers=0, jump=0.0000m > max_jump=0.0858m); falling back to velocity guess
```

66 of the 116 rejections on that run are this path — `geometric_tracker.py:720` bailing
when `len(pts2d) < min_pnp_inliers` (default 20), returning `False, 0` as a sentinel. The
tracker had **19** observations, one short of the guard, so **PnP was never attempted.** No
threshold setting rescues that; the inlier count of 0 here is a sentinel, not a measurement.
The remaining ~50 rejections are genuine jump vetoes (observed jumps 0.25–1.22 m against a
limit of 0.0858 m).

The guard's inputs are visible in the same log and they were poor early on:

```
Mapping Frame 2: Pruned 315 -> 155     [GeoTracker] Updated 33 track points from GS model
Mapping Frame 3: Pruned 155 -> 73      [GeoTracker] Updated 29 track points from GS model
Mapping Frame 4: Pruned  73 ->  13     [GeoTracker] Updated  5 track points from GS model
```

The GS map pruned 363 → 13 Gaussians in four keyframes and the tracker had 5 updated track
points. It does recover later — tracks climb 167 → 603 → 1372 and Gaussians back to 1466 —
but the early starvation is what the 116/150 rejection count is mostly made of.

So the bottleneck is **upstream of PnP**: the map prunes to single-digit Gaussians, the
tracker lands at ~19 observations against a 20-point guard, and the frame falls back to the
velocity guess. That is a scale-bias loop as much as a starvation one — points are lifted
from a depth buffer whose scale is wrong (`s` ≈ 0.17 on this clip), so the 3D points feeding
`update_object_points()` are misplaced, and a misplaced point cloud is what makes reprojection
fail. The scale error is the root; the starvation is the symptom.

**Candidate fixes — not applied, awaiting direction:**

1. **The 20-point guard is a cliff, not a slope.** Landing at 19 observations costs the frame
   entirely. A softer floor (accept PnP below 20 with a lower inlier weight, or a larger
   `min_pnp_inliers` with denser seeding) would convert most of the 66 sentinel failures into
   usable but weak estimates.
2. **Address the scale, which is the root.** The fixed-meter constants in a pipeline whose
   scale is set by the depth buffer are the prime suspects:
   `geometric_tracker.py:440` (5 mm minimum triangulation baseline),
   `geometric_tracker.py:356` (`dist < 0.05`), `obj_gs_mapping.py:463`
   (`render_depth - 0.01`), `obj_gs_mapping.py:240` (`scales = 0.02`),
   `obj_gs_mapping.py:553-556` (`screen_size` vs `prune_screen_size_th = 100.0`).
   On a clip where `s` ≈ 0.17, 5 mm is a substantial fraction of the scene and 50 mm
   (`occlusion_margin`, `occlusion_margin` default 0.05) is ~29% of it — which is exactly
   the class of check at `geometric_tracker.py:696` that discards an observation before it
   can count toward the 20-point guard.
3. **Pruning is too aggressive early.** 363 → 13 in four keyframes is what starves the
   tracker. A frame-count- or keyframe-index-gated prune floor would keep the map dense
   while the trajectory is still establishing scale.
4. **Do not bank the velocity guess on rejection** — freeze the estimate instead. The velocity
   *history* already refuses rejected frames (`accepted_rel`, `accepted_step_t/r`) but the
   *stored pose* still takes the guess, so the bias compounds.
5. **Separately: the photometric refinement appears to do nothing to translation.** In the
   clip-003318 log, every `Photometric refinement adjusted guess:` line reads
   `0.000000m`, with only occasionally non-zero rotation (0.004–0.012 deg). If that holds
   run-wide, the second opinion the GS mapper is paying for is silent, and the `num_steps_dyn=150`
   budget is spending time to no effect. This was sampled from one log window, not counted
   run-wide, so treat it as a lead rather than a result.

### Runtime

- Steady state is **~1.0–1.2 FPS** (~0.85–1.0 s/frame) for the full 150 frames on most runs;
  a handful of runs run 6–20 FPS in isolated bands. Total wall clock ≈ 1.5–2.5 min/150 frames.
- `multiprocess_dyn` defaults to `False`, so mapping runs inline and `Mapping (Sync)` blocks
  the tracker's own step.
- The pipeline already measures what it needs — `self.timings` records
  `Pre-processing (Flow)`, `Photometric Refinement`, `Geometric Tracking`, `Mapping (Sync)`
  and `Total run()` — but **`print_timings()` at `bundlesdf_gs.py:278` is defined and never
  called anywhere.** The per-phase breakdown has never been printed. So nobody can currently
  say where the ~1 s/frame goes; the instrumentation exists and is dead code. That is the
  cheapest available next step.

---

## Caveats on the evidence

- **Run-to-run variance is large.** Three runs at *identical* config and seed 0 on
  clip-003312 gave 0.4411 / 0.1321 / 0.1039 m — a 4.2× swing. So a meaningful part of what
  was earlier attributed to seed choice was GPU contention (1× 8 GB GPU, ~3 GB already held
  by other processes). Seeding buys reproducibility of the *procedure*, not lower variance.
- **Single-clip comparisons on this pipeline are unreadable** at any ATE delta below ~2×.
  Only multi-clip aggregates carry signal. This is what made the unpaired DA3METRIC
  comparison look conclusive when it was not.
- **Each variant is a single run.** The 0/5 result is unanimous and the aggregate gap is
  1.7×, both well outside the observed 4.2× per-clip noise band at the mean-of-5 level —
  but this is still n=1 per cell, not a repeated-measures study. If a result needs to be
  load-bearing, repeat it.
- **The ATE denominator is GT travel**, chosen because it is identical across the two
  variants of a clip. If the tracker's travel diverged from GT travel, ATE/travel would
  partly measure that divergence. On the near-metric clips the two agree closely; on
  clip-003318 they do not.

---

## Reproduction

```bash
# One run (≈4 min at 150 frames). Sequential only — one 8 GB GPU.
pixi run python full_system_ow_working/run_full_system_bundlegs_ow.py \
    --num-frames 150 --no-vis --clip-id clip-003312 --seed 0 \
    --gate-kf-commit --traj-dump /tmp/probe.npz
```

- Toggle: `--gate-kf-commit` / `--no-gate-kf-commit` (tyro treats it as a bare flag; the
  value form is rejected).
- Defaults after the reversion: `gate_kf_commit=False` in both
  `run_full_system_bundlegs_ow.py` `GlobalConfig` and `bundlesdf_gs.py`
  `GeoTrackerConfig`, wired through at the `GeoTrackerConfig(...)` construction site.
- The runner dumps per-frame `T_CO_est`, `T_CO_gt` (identity-filled for invalid GT),
  `valid_gt`, and `ate` to the `--traj-dump` path. Compute travel from
  `T_CO_gt[valid_gt][:, :3, 3]`, not from the estimate — that is what makes the two
  variants comparable.
- `object_poses.txt` is 8 columns (timestamp, t×3, quat×4); frames without GT are
  skipped silently, so a low frame count in `ate` can mean missing annotations rather
  than a truncated run.

The experiment drivers (`/tmp/ab_experiment.sh`, `/tmp/analyze_ab.py`) and the 10 npz/log
pairs (`/tmp/ab_clipNNNNNN_gate{0,1}_s0.*`) are in `/tmp` and are not version-controlled.
The tables above are the durable record.

## Where this lives in the code

| file | what |
|---|---|
| `full_system_ow_working/bundlesdf_gs.py` | `GeoTrackerConfig.gate_kf_commit` (default `False`) with the measured-negative comment; the gate block in `step()`; the `is_kf` block that stays **ungated** |
| `full_system_ow_working/run_full_system_bundlegs_ow.py` | `GlobalConfig.gate_kf_commit`, the `DEPTH_SOURCES` block with the corrected DA3METRIC story, `max_pose_jump=0.05`, trajectory dump |
| `full_system_ow_working/geometric_tracker.py` | `:722` — the rejection branch that banks `T_guess`; `:440` — the 5 mm minimum triangulation baseline |

Deferred, untouched by this work: the scale-aware meter constants
(`geometric_tracker.py:356` `dist < 0.05`, `obj_gs_mapping.py:463` `render_depth - 0.01`,
`:240` `scales = 0.02`, `:553-556` `screen_size` vs `prune_screen_size_th = 100.0`).

## History

| commit | what |
|---|---|
| `9abbc21` | Break autoregressive drift — velocity history restricted to accepted frames |
| `62acbdb` | Selectable depth source, decomposed scale-aligned/Umeyama evaluation |
| `51b462c` | Gate GS keyframe commit on PnP rejection — **under test here** |
| `548d8c5` | Revert the gate default; record the measured-negative result |
