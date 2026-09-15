# Cross-sequence ATE

Dated 2026-09-10. Working tree at `d1eb6d4` **plus 6 uncommitted files (+669/−52)**.

Everything below is either quoted from a stored artifact or recomputed from one.
Nothing is estimated.

## 0. The one thing to read first

There is **no single ATE table for the whole dataset.** There are four mutually
incompatible ones, and **no two of them measure the same quantity.**

The repo has **six distinct ATE definitions**, and no two headline tables use the
same one:

| id | definition | sites |
|---|---|---|
| A | zero-filled mean of per-frame norms, **direct** tracker pose, all 150 frames | `run_full_system_bundlegs_ow.py:700` |
| B | zero-filled mean of norms over **interpolated** eval-frame poses, odd frames only | `full_system_ow/eval_full_system.py:547` |
| C | valid-masked **RMSE after full Sim3 (Umeyama) alignment** | `run_full_system_bundlegs_ow.py:758-781` |
| D | ATE / GT travel length — **not in the repo** | `/tmp/analyze_ab.py:90` |
| E | skip-guarded mean of norms (missing GT **excluded**) | `BundleGS*/test_hot3d*.py:178,265` |
| F | RMS of norms | `exp/*.py` |

A is the repo's only Umeyama implementation; B, D, E and F never align.

Presenting `eval_results_final.json`'s mean ATE 0.7460 next to the paired A/B's
0.3127 as if they measured the same thing would repeat exactly the unpaired
comparison `docs/bundlesdf-gate-ab-findings.md` already retracted.

### 0.1 Nothing here is produced by committed code

`git status` shows `?? full_system_ow_working/ba.py` and the
`full_system_ow_working/bundlesdf_gs.py` that imports it is one of the 6
uncommitted files — `HEAD:full_system_ow_working/bundlesdf_gs.py` contains no
`import ba`. So the entire BA bring-back, and every number below produced by the
working tree, is **uncommitted**. The source file mtimes (14:13–18:22 on
2026-09-10) postdate all the A/B runs (01:39–01:47), so the current tree is not
guaranteed to reproduce those dumps. (The metric/dump block at `:700-796` and
`umeyama` at `:454-475` are untouched by the diff, so the *definitions* are
stable even if the trajectories are not.)

### 0.2 The working tree has no eval harness at all

`full_system_ow_working/` contains three runner variants, `ba.py` and libs —
**no `eval_full_system.py`.** The 8-clip headline table cannot be regenerated
against the code that actually changed. The two trees are both git-tracked and
differ in tracked code (HEAD-vs-HEAD: `bundlesdf_gs.py` 442 changed lines,
`obj_gs_mapping.py` 58, `geometric_tracker.py` 38), but there is no code in the
working tree that emits a per-clip ATE across a clip list.

## 1. Record A — `eval_results_final.json`, 8 clips, 2026-06-01

Definition **B**: raw `np.linalg.norm` of the **interpolated** eval-frame pose,
averaged over only the odd frames.

The interpolation is at `full_system_ow/eval_full_system.py:468-482`:
`T_WO_eval` is built from `T_WO_prev` and `T_WO_next`, averaging translations
and Slerping rotations, then `T_CO_eval = eval_fd["extrin"] @ T_WO_eval`.
Scored at `:547` `ate = np.linalg.norm(T_CO_eval[:3, 3] - T_CO_eval_gt[:3, 3])`,
`:548-549` `else: ate = 0.0`, `:550` `ates.append(ate)` **unconditional**,
`:622` `>>> Final Mean ATE`.

- **No Umeyama, no SE3/Sim3, no scale normalisation anywhere in the file.**
- Even frames tracked, odd frames scored; of 75 odd candidates the last
  (index 149) is buffered as `pending_eval` at `:619` and **never flushed**
  → 74 scored frames, not 150.
- `eval_all.py:11` passes **only** `--clip-id`. So seed = 42 (default `:135`),
  `num_frames` = 150 (default `:71`), unseeded by any caller.
- Frozen config of this run: `max_pose_jump=1.0` (`:77`), `fix_scale=False`
  (`:120`), `use_ray_dist=False` (`:121`), `densify_error_threshold=0.8`
  (`:126`) — versus the A/B's `0.05`, `True`, `True`, `10.0`. The pose-jump
  gate is **20× looser** here.
- `eval_results_final.json` and `eval_results_temp.json` are byte-identical
  to the nanosecond (both 2026-06-01 14:28:35.836359824) and predate the
  producing script's only commit by ~20 days.
- `:527-533` copies GT hand-mask pixels into the render **before** PSNR/SSIM/
  LPIPS — the image metrics leak (already flagged in
  `docs/dynamic-tracking-proposal.md` §9.3).

| clip | ATE (m) | PSNR | SSIM | LPIPS |
|---|---|---|---|---|
| clip-001849 | 0.1008 | 21.98 | 0.7301 | 0.3668 |
| clip-001882 | 1.025 | 11.66 | 0.1702 | 0.5626 |
| clip-001924 | **3.0711** | 18.02 | 0.6402 | 0.4316 |
| clip-001959 | 0.4256 | 22.04 | 0.7247 | 0.3668 |
| clip-002002 | 0.9077 | 17.80 | 0.6062 | 0.4145 |
| clip-002793 | 0.0607 | 21.89 | 0.6885 | 0.4020 |
| clip-003312 | 0.1618 | 21.87 | 0.7409 | 0.3581 |
| clip-003363 | 0.2152 | 23.37 | 0.7721 | 0.3772 |
| **mean** | **0.7460** | 19.83 | 0.6341 | 0.4100 |

clip-001924 alone is **51.5% of the sum** (3.0711 of 5.9679). Drop it and
clip-001882 and the 6-clip mean is **0.3120** — coincidentally 0.0007 from the
A/B's 0.3127, which is why these two numbers have been compared for a week.
They are not comparable; the coincidence is what created the confusion.

## 2. Record B — the paired 5-clip A/B, 10 runs, 2026-09-09

Definition **A**: `run_full_system_bundlegs_ow.py:700`
`ate = np.linalg.norm(T_CO_est[:3, 3] - T_CO_gt[:3, 3]) if T_CO_gt is not None
else 0.0`, mean over all 150 frames, printed `:757`
`>>> Final Mean ATE (raw, unaligned)`. C at `:758-781`.

Config: `full_system_ow_working/run_full_system_bundlegs_ow.py`, seed 0
(`:61`), `--num-frames 150` (`:56`), `depth_source=model_infer` (`:71`),
`max_pose_jump=0.05`, `fix_scale=True`, `use_ray_dist=True`,
`densify_error_threshold=10.0`, one 8 GB GPU, `multiprocess_dyn=False`.
Driver `/tmp/ab_experiment.sh`; gate ON 01:39–01:42, gate OFF 01:44–01:47 —
**sequential order, so order bias is not excluded.**

Valid-GT is 150/150 in all 10 runs, and the stored `ate` matches recomputation
exactly (max diff 0.0). Every `ate` array has exactly one free zero at
frame 0 — ~0.07% deflation, because **frame 0 is seeded from GT**:
`T_CO_est[0]` and `T_CO_gt[0]` are bitwise identical. That 0.0 is not a
tracking measurement. Excluding it moves `renders/trajectory_est_gt.npz` from
0.189628 to 0.190901 (+1.3 mm).

| clip | gate | raw ATE | ATE/travel | median | max | Umeyama s | Umey RMSE | scale-aliased RMSE | PnP rej | kf skip | scale err |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 003312 | 0 | 0.103937 | 0.071429 | 0.093742 | 0.229982 | 1.1109 | 0.0715 | 0.0719 | 13/150 | 0 | +11.1% |
| 003312 | 1 | 0.132098 | 0.090783 | 0.107292 | 0.388595 | 1.0666 | 0.1048 | 0.1053 | 17/150 | 6 | +6.7% |
| 003318 | 0 | 0.989775 | 0.742872 | 0.682846 | 3.401211 | 0.1790 | 0.1292 | 0.3168 | 116/150 | 0 | −82.1% |
| 003318 | 1 | **1.827727** | 1.371792 | 2.398489 | 2.693658 | 0.1701 | 0.0758 | 0.3076 | 89/150 | 29 | −83.0% |
| 003333 | 0 | 0.045232 | 0.050656 | 0.045493 | 0.088392 | 1.0113 | 0.0465 | 0.0489 | 17/150 | 0 | +1.1% |
| 003333 | 1 | 0.087918 | 0.098461 | 0.099138 | 0.127758 | 0.8550 | 0.0318 | 0.0358 | 8/150 | 3 | −14.5% |
| 001910 | 0 | 0.188246 | 0.123977 | 0.174512 | 0.425130 | 0.9834 | 0.0952 | 0.1056 | 85/150 | 0 | −1.7% |
| 001910 | 1 | 0.272290 | 0.179329 | 0.281648 | 0.613874 | 0.5215 | 0.1593 | 0.1658 | 72/150 | 25 | −47.9% |
| 001924 | 0 | 0.236109 | 0.087220 | 0.138741 | 0.597005 | 1.5765 | 0.1652 | 0.2107 | 51/150 | 0 | +57.7% |
| 001924 | 1 | 0.305657 | 0.112912 | 0.130245 | 0.812587 | 2.7130 | 0.2519 | 0.2616 | 54/150 | 29 | +171.3% |
| **mean** | 0 | **0.312660** | 0.215231 | 0.227067 | — | — | 0.10152 | — | — | — | — |
| **mean** | 1 | **0.525138** | 0.370655 | 0.603362 | — | — | 0.12472 | — | — | — | — |
| | | **1.68×** | **1.72×** | **2.66×** | | | **1.23×** | | | | |

**This is the only genuine paired comparison in the repo.** Same clip, same
seed, same frame count, same depth source, same machine; the only variable is
the `gate_kf_commit` flag at `full_system_ow_working/bundlesdf_gs.py:982`
(`if self.tracker_cfg.gate_kf_commit and pnp_rejected:`), which withholds
`keyframes.append`/`_enqueue_keyframe`. `add_new_points_from_depth` at
`:945-953` is deliberately ungated, so depth points still enter the map. Gate 0
is the pre-fix default (`:105`, runner `:193`).

### 2.1 "Gate ON wins 0/5" is metric-specific

Recomputed per clip, gate ON vs OFF:

| metric | OFF | ON | ratio | gate ON wins |
|---|---|---|---|---|
| raw ATE mean | 0.312660 | 0.525138 | 1.68× | **0/5** |
| ATE/travel | 0.215231 | 0.370655 | 1.72× | **0/5** |
| median ATE | 0.227067 | 0.603362 | 2.66× | **1/5** (001924) |
| Umeyama RMSE | 0.10152 | 0.12472 | 1.23× | **2/5** (003318, 003333) |
| Umeyama RMSE / GT travel | 0.064389 | 0.072499 | 1.13× | **2/5** (003318, 003333) |

Gate OFF still wins on the average under **every** metric, so the revert in
`548d8c5` stands. But the margin ranges 1.13×–2.66× and the per-clip win count
ranges 0/5–2/5. The "0/5" headline and the choice of ATE/travel as *the* metric
are not defensible as such.

Those two Umeyama wins are **s-correction artifacts, not better tracking**:
003318 gate ON has s=0.1701 and 003333 gate ON has s=0.8550, so the wins come
from a larger scale correction, not from a closer trajectory.

### 2.2 clip-003318's 1.83 m is a scale artifact, not tracking failure

Raw ATE 1.827727 → Umeyama RMSE **0.0758** — a ~24× reduction from alignment.
Scale-aligned RMSE 0.3076. Depth-buffer scale error −83.0%. The tracker did
not lose the camera; the depth buffer is at one-sixth of the correct scale, and
raw ATE reports that as a 1.83 m tracking error.

This is precisely why ATE/travel is the wrong headline metric: it conflates
global scale drift with pointwise error. For that one run, ATE/travel is
1.371792 while Umeyama RMSE / travel is 0.0569.

## 3. Record C — `renders/trajectory_est_gt.npz` = clip-003312, 0.189628

The only trajectory dump inside the repo. 29581 B, mtime 2026-09-09 23:06.
Written by the runner's **default** dump path
`run_full_system_bundlegs_ow.py:95` `traj_dump: str = "renders/trajectory_est_gt.npz"`
— no `--traj-dump` override. Runner defaults: 150 frames (`:56`),
`clip-003312` (`:54`), `model_infer` (`:71`), seed 0 (`:61`).

**No clip id, seed, depth source, gate flag or commit is recorded in the
archive.** Provenance is inferred — by three independent identifications, all
converging on clip-003312: `T_CO_gt` is `np.allclose`-identical to the GT of
every clip-003312 record and not identical to any 001910/001924/003318/003333
record; GT travel 1.455093 matches; and mean ATE 0.189628 with Umeyama
s=0.943311 exactly matches the pair quoted in the runner's own docstring at
`:57-59` (*"Two 150-frame runs on the identical model_infer path measured mean
ATE 0.1896 m and 0.3998 m (Umeyama s 0.9433 vs 1.1221)"*).

The pair-mate 0.399848 / s=1.1221 is `/tmp/traj_model_infer_150.npz`. Both
runs named in that code comment are now accounted for.

Its est trajectory matches **no** existing record (`estSame=False` against all
27 `/tmp` dumps) and 0.189628 equals no other record's mean ATE → this is the
run the docstring cites but which was never saved to `/tmp`.

| | value |
|---|---|
| frames / valid GT | 150 / 150 (no identity-filled frames) |
| raw ATE mean | **0.189628** (max 0.573773 at frame 147; p50 0.15884, p90 0.32343, p95 0.44777, p99 0.56853) |
| excluding frame 0 | 0.190901 |
| Umeyama s / RMSE | 0.943311 / 0.115072 (max 0.402664) |
| scale-only RMSE | 0.124786 |
| GT travel / est travel | 1.455093 / 2.651728 (ratio 1.8224) |
| ATE/travel raw / aligned | 0.130320 / 0.079082 |
| rotation error | mean 34.43°, median 22.49°, max 91.39° |
| segments 0-24 / 25-49 / 50-99 / 100-149 | 0.1176 / 0.1686 / 0.1493 / 0.2765 |

Rotation error is large and structural, so Umeyama is compensating genuine
rotation and over-shoot, not a constant offset. Segments show monotonic
late-frame drift, no collapse.

**Gate flag is unrecorded, so this row cannot be assigned to gate ON or OFF.**

### 3.1 The per-clip noise floor is 4.24×

Six seed-0, `model_infer`, 150-frame runs on clip-003312:

| run | raw ATE |
|---|---|
| A/B gate OFF | 0.103937 |
| A/B gate ON | 0.132098 |
| `renders/trajectory_est_gt.npz` | 0.189628 |
| `traj_v3_model_infer_s1` | 0.213952 |
| `traj_model_infer_150` | 0.399848 |
| `traj_v3_model_infer_s0` | 0.441085 |

min 0.103937, max 0.441085, **spread 4.24×**, median 0.20179, mean 0.246758.
Seeding did not reduce it (0.441085 at s0 vs 0.213952 at s1).

**The A/B table's 0.1039 is the single best draw of six.** Both the gate-OFF
and gate-ON values sit at the floor and the second-from-floor of the
distribution. The gate-OFF win is real but its magnitude is far smaller than
the difference between two runs of the *same* configuration on the *same* clip.

## 4. Record D — `BundleGS_keyframe/benchmark_results.md`

Definition **E**: skip-guarded mean of norms, missing GT **excluded** —
`test_hot3d.py:178` `dist_error = np.linalg.norm(T_CO_est[:3, 3] -
T_CO_gt[:3, 3])`, `:265` `"ate": np.mean(all_dist_errors) if
all_dist_errors else None`. Different family from A/B.

| clip | ATE | NoCheat PSNR | Ideal PSNR |
|---|---|---|---|
| clip-001849 | 0.1191 | 13.16 | 15.94 |
| clip-001850 | 0.6091 | | |
| clip-001851 | 0.0028 | | |
| clip-001853 | 0.3053 | | |
| clip-001854 | **0** | | |
| clip-001855 | **0** | | |
| clip-001856 | **0** | | |
| clip-001857 | **0** | | |
| clip-001868 | **0** | | |
| clip-001870 | **0** | | |

**6 of 10 rows are dead zeros.** They are not failures recorded as failures:
the generator `benchmark_gs_performance.py` drops stderr and returncode
(`run_cmd` `:7-10`) and records `metrics.get('ATE', 0)` at `:54-56`, so a
crash is indistinguishable from a perfect 0. `extract_metrics` overwrites on
every `'ATE:'` line with a `:175` sentinel of 0.0. The script writes to
`"BundleGS/benchmark_results.md"` (`:62`), a path that does not exist — the
file we have was relocated by hand. `--n-frames-eval 10` is a NO-OP and
`--n-frames-track` is never passed, so default 30. Runs happened on a
different machine (`run_log_sync.txt` → `/home/shzhou/project/dyn_gs_exp/`).
Zero matches for `seed` in either BundleGS tree. `max_pose_jump=10.0`
("Effectively disable rejection"). Frame 0 seeded with GT.

**Different tracker codebase** — root `geometric_tracker.py` /
`obj_gs_mapping.py`, 613 and 377 diff lines against the working tree. Internal
inconsistency: `:21 from BundleGS_keyframe.bundlesdf_gs import BundleSdfGS` but
`:108 from BundleGS.bundlesdf_gs import GeoTrackerConfig`. The archived command
strings cannot be re-run (`--use_ray_dist`, `--num_steps`,
`--every-frame-mapping` are absent from the called Config).

## 5. Absolute ATE is not cross-clip meaningful

The depth buffer is the pipeline's absolute scale. Umeyama `s` across the 10
A/B runs spans **0.1701 → 2.7130**, a 16× range; raw ATE spans
0.045232 → 1.827727, a 40× range; GT travel spans only 0.893 → 2.707 m, a 3×
range. Scale error ranges −83.0% to +171.3%.

Correlation across the 10 runs: s vs raw ATE **−0.501**, s vs ATE/travel
−0.554, s vs Umeyama RMSE +0.425. Raw ATE is more strongly *inversely*
correlated with the scale factor than with the aligned error — it is partly
measuring the scale wrongness of the run.

**ATE/travel fixes the trajectory-length denominator, not the depth-buffer
scale numerator.** It is travel-length-free, not scale-free: clip-003318 gate
ON still reads 1.371792 when the aligned error is 0.0569.

## 6. Correction to the framing of "the only honest metric"

ATE/travel is **not** the only honest cross-sequence metric, and it is not the
best-behaved one. Umeyama RMSE / GT travel is:

- 1.13× ON-vs-OFF spread, versus 1.72× for ATE/travel — closer to a truth value,
  because it removes the scale artifact that dominates the absolute metrics.
- Does not let a global scale failure masquerade as local tracking failure.

Both are honest; neither is complete. The right reading of the A/B is:
**gate OFF wins on the average under all five metrics, and gate ON's two
Umeyama wins are explained by s.**

## 7. Umeyama `s` is implementation-defined

`run_full_system_bundlegs_ow.py:473`:
`s = float(np.trace(np.diag(D) @ S) / var_s)` — the sum of singular values of
the centred cross-covariance, over the estimated centred variance.

The centred-spread ratio `Sum|gt_c|² / Sum|est_c|²` gives a **different wrong
value**. On `renders/trajectory_est_gt.npz` it yields s=1.052045 / RMSE 0.119192
whereas the runner's form gives s=0.943311 / RMSE 0.115072. Across the 10 A/B
runs the logged s also disagrees with a radius-ratio recomputation (1.142 vs
1.0666 on 003312 gate ON; 3.610 vs 2.7130 on 001924 gate ON). Only the logged
s is authoritative; if you recompute, use the `:473` form.

## 8. Seven ways a number in these tables can be wrong without notice

1. Final eval frame dropped — `eval_full_system.py:619` buffers it as
   `pending_eval` and never flushes. 74 frames, not 150.
2. `eval_all.py` silently drops incomplete clips; a crash and a pass both
   disappear from the JSON.
3. `:759 if len(valid) >= 3:` silently omits the alignment lines entirely.
4. `/tmp/analyze_ab.py:87-89` computes segment means **unmasked** over
   identity-filled frames.
5. `test_best_gs_window.py:175` sentinel 0.0 for a missing metric.
6. `exp_kvtracker_pnp_seq.py:587-592` silent no-op.
7. `benchmark_gs_performance.py` overwrite-on-match regex — last `'ATE:'` line
   wins, stderr discarded.

## 9. Adversarial verification

Four claims were adversarially verified by independent agents that were told to
default to REFUTED and recompute rather than trust the extraction digests.
Verdicts are recorded in §11 — read that, not this section, for the outcomes.

- **C1** — the 8-clip table and the A/B are not comparable: **PARTLY_HOLDS**.
  The conclusion holds and all headline numbers verify exactly. Two mechanism
  clauses were overstated: "all the BA work lives in
  `full_system_ow_working/`" is **false** — `run_ba` exists in both trees
  (`full_system_ow/geometric_tracker.py:158`,
  `full_system_ow/bundlesdf_gs.py:607`;
  `full_system_ow_working/geometric_tracker.py:176` and `:918`). And
  `full_system_ow_working/ba.py` is **untracked**, so it is not evidence of a
  *tracked*-code divergence. The correct reasons are estimator + population +
  config + seed, plus the 4.24× within-config noise and the s artifacts.
- **C2** — ATE is raw, unaligned, zero-filled, unweighted: **HOLDS**. One
  refinement only: `:547`'s `T_est`/`T_gt` are `T_CO_eval`/`T_CO_eval_gt`, and
  `T_CO_eval` is the Slerp + mid-translation interpolation of the two bracketing
  tracked frames (`:468-482`) — not the tracker's direct per-frame pose. Both
  sides are multiplied by the same GT extrinsic, so the claim stands.
  **But its "biased low when GT is missing" clause never fired on any run** —
  see §11.
- **C2** — ATE is raw, unaligned, zero-filled, unweighted: **HOLDS**. §11.
- **C3** — the 13.0× gap on 001924 is definitional, not a regression:
  **PARTLY_HOLDS** — conclusion right, mechanism wrong. §11.
- **C4** — absolute ATE is meaningless across sequences: **HOLDS** with two
  corrections. §11.

## 10. Provenance, hardened

The 8-clip JSONs are **git-tracked** and were touched by exactly one commit:
`bdee8db` (2026-09-09 19:39:34, "Snapshot env + eval artifacts"), whose parent
is `a6229e4` (2026-06-20 19:16:55). `git diff HEAD -- eval_results_*.json` is
empty — disk equals HEAD.

`eval_all.py` and `full_system_ow/eval_full_system.py` each have **exactly one
commit in history: `a6229e4`, where both were ADDED** (`git show
--name-status` reports `A`). So the JSONs were written **~20 days before the
earliest commit containing the script that produced them.** The exact revision
that ran on 2026-06-01 is not in git history. Blame confirms the quoted ATE
block is the `a6229e4` version — but it **may have differed at run time**, so
even the definition is only known as of 2026-06-20, not as of the run.

Render artifacts cannot date the run either, and they actively contradict it:
`renders/eval_render_clip-003312.mp4` is 2026-06-01 14:04, **24 minutes
before** the JSON; but clip-001924/001959/002002/002793/003363 renders are
2026-06-02 12:21–13:37, **22–24 hours after**. `renders/eval_0001..eval_0297.png`
are 149 odd-indexed PNGs from 2026-06-02 13:35–14:12, implying a render session
of **~300 frames with `--save-render` on** — not the 150-frame, renders-off
run that `eval_all.py` would produce. **No mp4 exists for clip-001849 or
clip-001882.** So the JSON and the surviving renders are very likely from
**different sessions**, and the JSON's own session is unrecoverable.

The JSON records no seed, no frame count, no flags, no per-frame values, no
commit. Everything is inferred.

## 11. Verification outcomes (all four landed)

**C2 HOLDS — but its "biased low when GT is missing" clause never fired.**
The verifier confirmed the fallback did not trigger on any overlap clip:
`data/hot3d_clips_processed/clip-XXXXXX/object_poses.txt` has ≥150 lines × 8
columns for clip-003312/001924/001849/001882, so `T_WO_gt` was never None.
C2 is a true property of the code, but it produced **zero** free zeros on these
runs. The actual deflation present is only frame 0 (§2, ~0.07%).

One geometric refinement worth recording: because the same GT
`fd["extrin"]` multiplies both sides at `:482`/`:546`, the error reduces to
`T_CW_gt @ (t_est_world − t_gt_world)` — per-frame **object-pose translation
error in the camera frame**, with the tracker's own world-frame offset and
scale left unremoved, and sampled only on the interpolated non-tracked frames.

**C3 PARTLY_HOLDS — conclusion right, mechanism wrong.** The verifier
independently reconstructed the interpolated estimate from the extrinsics and
the dumped trajectory and found the definitional difference contributes
**~0** to clip-001924's 13.0× gap — it is **≥99.9% a genuine trajectory
divergence**, under a `max_pose_jump` 20× looser (`1.0` vs `0.05`) and
`fix_scale=False`. So: it is *not* a pipeline regression, and it is *not*
explained by the ATE definition. The definition was a red herring on this clip.

**C4 HOLDS, with two corrections.** The scale argument stands: `s` matched the
repo's printed values to four digits across all 10 runs, and the correlations
are s vs raw ATE −0.501, s vs ATE/travel −0.554, s vs Umeyama RMSE +0.425 —
raw ATE is more strongly *inversely* correlated with the scale factor than with
the aligned error. But: (1) "the only honest cross-sequence metric" is too
strong — Umeyama RMSE / travel is better behaved for the reasons in §6;
(2) the verifier's own Umeyama RMSE and GT-travel columns do **not** reproduce
the repo's printed values at identical `s` (0.035913 vs 0.0715 on 003312 gate
OFF; 0.048031 vs 0.1292 on 003318 gate OFF) and its travels are in a different
unit (72–128 vs 0.89–2.71), so those columns are not adopted — only its `s`
column and the log tails it quoted, which confirm §2.2.

## 12. What this table set can and cannot support

Can:
- Show that the gate does no net harm to mapping, at the cost of ~2× worse
  average raw ATE under every metric — hence the revert.
- Show the per-clip noise floor is 4.24×, so **no single-run comparison in this
  repo is statistically meaningful.**
- Show absolute ATE is dominated by depth-buffer scale, and that clip-003318's
  1.83 m is scale, not tracking.

Cannot:
- Report a per-clip ATE for the working tree across a clip list — the eval
  harness does not exist there.
- Regenerate the 8-clip table against the code that changed.
- Support any single-run claim, at any metric, without ≥3 seeds per cell.
- Compare across the four records.

The next thing that would actually move this: a single runner that writes one
JSON with (clip, seed, gate, raw ATE, median ATE, Umeyama s, Umeyama RMSE,
GT travel, aligned ATE/travel), frame 0 excluded, over ≥3 seeds, in the
**working** tree. Everything else is re-sorting the same four incompatible
tables.
