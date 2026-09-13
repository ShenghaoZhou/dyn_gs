# Proposal: Integrating Any4D into `full_system_ow` for Robust Dynamic Tracking

**Branch:** `feature/any4d-integration`  
**Status:** Proposal / Design Phase (No functional code modifications yet)

---

## 1. Motivation & Empirical Problem Statement

In `full_system_ow`, the dynamic foreground object tracking pipeline relies on a two-stage estimation process in [`bundlesdf_gs.py`](file:///home/shzhou/project/dyn_gs_exp/full_system_ow/bundlesdf_gs.py):
1. **Stage 1 (Photometric Refinement):** LM / Adam optimization of $T_{WO}$ using Gaussian Splatting rendering against the incoming RGB image.
2. **Stage 2 (Geometric Tracking):** 2D sparse feature tracking (Grid/ORB/GFTT via OpenCV) with PoseLib PnP RANSAC on 2D–3D correspondences.

### Empirical Failure Mode
Benchmarking reveals a stark contrast between texture-rich and texture-poor / challenging sequences:
- **`clip-001851` (Textured, stable):** PnP maintains ~490 track points per frame. The system achieves **ATE = 0.0031 m (0.31 cm)** and **PSNR = 21.22 dB** over 30 frames.
- **`clip-001853` (Challenging object/motion):** Suffers immediate feature track loss:
  ```
  [GeoTracker] Insufficient tracks at frame 1 (found 9)
  [GeoTracker] Insufficient tracks at frame 2 (found 1)
  [GeoTracker] Insufficient tracks at frame 3 (found 0)
  ...
  [GeoTracker] Insufficient tracks at frame 9 (found 11)
  ```
  During this 9-frame window, the system falls back entirely to a constant-velocity motion model and stationary guesses. This causes severe pose drift, resulting in **ATE = 0.0881 m (8.81 cm)** over 30 frames and deteriorating to **0.342 m** over the full sequence.

---

## 2. What Any4D Brings to the System

Any4D (`/home/shzhou/project/big_model/Any4D/`, model checkpoint `checkpoints/any4d_4v_combined.pth` [7.7 GB]) is a multi-view foundation transformer for feed-forward, metric 4D dynamic reconstruction.

When provided with a reference frame $I_0$, target frame $I_t$, and the initial binary object mask $M_0$:
1. **Metric 3D Pointmap $\mathbf{P}_0 \in \mathbb{R}^{H \times W \times 3}$:** Dense 3D surface geometry of the object in metric scale.
2. **Dense 3D Scene Flow $\Delta \mathbf{P}_t \in \mathbb{R}^{H \times W \times 3}$:** Unconstrained 3D motion field mapping every object point $\mathbf{p} \in M_0$ to its 3D location at time $t$:
   $$\mathbf{P}_t = \mathbf{P}_0 + \Delta \mathbf{P}_t$$
3. **Appearance Invariance:** Operates on DINOv2 self-supervised visual features, remaining resilient to surface texture loss, motion blur, and partial hand occlusions.

---

## 3. Proposed 3-Tier Integration Architecture

```
                       [Incoming Frame t: Color, Mask, Depth]
                                        │
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │  Stage 1: BundleSdfGS Photometric Guess  │
                   └────────────────────┬─────────────────────┘
                                        │
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │  Stage 2: Classical Geometric PnP Track  │
                   └────────────────────┬─────────────────────┘
                                        │
                         Is n_inliers >= min_inliers?
                                ├── YES ──► Accept PnP Pose
                                │
                                └── NO (Track Loss / Rapid Motion)
                                        │
                                        ▼  [ANY4D FALLBACK MODULE]
                   ┌──────────────────────────────────────────┐
                   │  Query Any4D with (I_0, I_t, Mask_0)     │
                   │  1. Compute Scene Flow: P_t = P_0 + ΔP_t │
                   │  2. Kabsch SVD on object points: T_t0    │
                   │  3. Re-seed new 2D-3D tracks for tracker │
                   └────────────────────┬─────────────────────┘
                                        │
                                        ▼
                   ┌──────────────────────────────────────────┐
                   │  Stage 3: Keyframe & GS Gaussian Mapping │
                   └──────────────────────────────────────────┘
```

### Tier 1: On-Demand Geometric Fallback (Highest Impact)
- **Trigger:** Only executed when `not success` or `n_inliers < min_pnp_inliers` (default: 20).
- **Operation:**
  1. Any4D takes $(I_0, I_t)$ and computes the displaced 3D point positions $\mathbf{P}_t$ for object mask pixels.
  2. A rigid SE(3) transformation $T_{t \leftarrow 0}$ is solved via weighted Kabsch SVD on $(\mathbf{P}_0[M_0], \mathbf{P}_t[M_0])$.
  3. This provides a clean, drift-free pose estimate $T_{CO}^{\text{any4d}} = T_{t \leftarrow 0} \cdot T_{C0O}$, preventing the catastrophic drift seen in `clip-001853`.
  4. **Track Re-seeding:** The dense 3D points $\mathbf{P}_t$ are projected back onto image $t$ via camera intrinsics $K$ to immediately repopulate the geometric tracker's point pool, preventing consecutive frames from failing.

### Tier 2: Enhanced Motion Model for Photometric LM Optimization
- Currently, when velocity jumps $> 0.5$ m or $> 20^\circ$, `bundlesdf_gs.py` discards the velocity model and assumes the object is stationary in world space.
- In rapid manipulation sequences, Any4D can supply a physically grounded coarse pose guess directly into the Levenberg-Marquardt optimizer before the geometric step.

### Tier 3: High-Fidelity Dense Metric Depth Fusion
- On Keyframe creation (`add_new_points_from_depth`), monocular depth estimators often exhibit scale drift and edge distortions around thin objects.
- Any4D's metric pointmaps can be fused into the Gaussian densification step to spawn cleaner 3D Gaussians with accurate physical boundaries.

---

## 4. Resource & Runtime Feasibility

- **VRAM Budget:**
  - RTX 4070 provides **12 GB VRAM**.
  - Current `full_system_ow` pipeline consumes **~3.5–4.5 GB VRAM** during execution.
  - Any4D in FP16 / AMP mode (`trained_with_amp=True`, input resolution $518 \times 336$) consumes **~3.2 GB VRAM**.
  - **Verdict:** Both models fit comfortably within the 12 GB limit simultaneously without OOM.
- **FPS Impact:**
  - By invoking Any4D **asynchronously or conditionally on track degradation**, normal frames run at full system speed (30–40 FPS).
  - Challenging frames incur a momentary inference latency (~80–120 ms) rather than losing the track completely.

---

## 5. Proposed File Structure Changes

1. **`full_system_ow/any4d_tracker.py`** *(New)*:
   - Encapsulates Any4D model initialization, lazy evaluation, batching of `(ref, target)` views, scene-flow extraction, and Kabsch SE(3) alignment.
2. **`full_system_ow/bundlesdf_gs.py`** *(To be updated)*:
   - Inject `any4d_tracker` as an optional fallback provider in `run()`.
3. **`full_system_ow/run_full_system_bundlegs_ow.py`** *(To be updated)*:
   - Expose `--use-any4d` (boolean, default False) and `--any4d-checkpoint` configuration flags in `GlobalConfig`.
