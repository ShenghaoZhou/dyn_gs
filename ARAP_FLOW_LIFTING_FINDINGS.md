# As-Rigid-As-Possible (ARAP) Regularization for Dynamic Gaussian Splatting & 2D-to-3D Flow Lifting

## 1. Executive Summary

This document reports findings on using the **As-Rigid-As-Possible (ARAP)** constraint in dynamic object tracking and 3D Gaussian Splatting (3DGS).

Initially, applying ARAP solely as a loss term during Gaussian Splatting mapping preserved point density and surface geometry (> 2× inliers and 14.3k Gaussians vs. 7.6k baseline), but **rigid 6-DoF Absolute Trajectory Error (ATE) did not improve**.

Investigating this revealed that the tracking failure was driven upstream by **unregularized 2D-to-3D optical flow lifting**:
1. Monocular depth boundary bleeding and hand-object interactions introduced non-rigid distortions of up to **153.4 mm** into lifted 3D point tracks.
2. Constant-velocity motion model guesses ($T_{\text{guess}}$) diverged during sudden object accelerations, causing heuristic occlusion checks to discard valid correspondences and triggering tracking collapse (baseline exploded to **1.1418 m** ATE at frame 133).

By formulating and integrating an **ARAP regularized flow lifting framework** directly into the geometric tracker:
- Non-rigid flow drift onto manipulating hands was pruned via local ARAP strain thresholding ($s_i < 0.15$).
- Depths along optical rays were regularized to enforce canonical object isometry.
- A closed-form Kabsch 6-DoF alignment was solved to provide an independent, motion-model-free visual pose prior.
- Multi-candidate pose consensus (`pnp`, `arap`, `guess`) eliminated velocity-jump tracking loss.

**Result**: Across the full 150 frames of HOT3D `clip-001851`, full sequence ATE dropped from **1.1418 m (baseline collapse) to 0.0265 m (2.65 cm)**, a **~43× error reduction**, while maintaining 13× higher tracking inliers during fast manipulation.

---

## 2. Mathematical Formulation

### 2.1 ARAP for Gaussian Mapping
For Gaussian primitives with centroids $\mathbf{P} = \{\mathbf{p}_i\}_{i=1}^N$ and quaternions $\mathbf{Q} = \{\mathbf{q}_i\}_{i=1}^N$:
- **Edge-Length Preservation (Local Isometry)**:
  $$\mathcal{L}_{\text{ARAP}} = \frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \left( \|\mathbf{p}_i - \mathbf{p}_j\|_2 - d_{ij}^{\text{ref}} \right)^2$$
- **Geodesic Rotation Consistency**:
  $$\mathcal{L}_{\text{rot}} = \frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \left( 1 - |\langle \mathbf{q}_i, \mathbf{q}_j \rangle| \right)$$
where $\mathcal{E}$ is the $k$-NN neighbor graph built in canonical object space and $d_{ij}^{\text{ref}} = \|\mathbf{p}_i^0 - \mathbf{p}_j^0\|_2$.

### 2.2 ARAP for 2D-to-3D Flow Lifting

Given 2D tracked optical flow coordinates $\mathbf{u}_i \in \mathbb{R}^2$, canonical 3D object coordinates $\mathbf{p}_{O, i} \in \mathbb{R}^3$, monocular depth map $D_t$, and camera intrinsics $K$:

#### Step 1: Camera Ray Unprojection
$$\mathbf{r}_i = K^{-1} [\mathbf{u}_i^T, 1]^T, \quad \mathbf{P}_{C, i} = D_t(\mathbf{u}_i) \mathbf{r}_i$$

#### Step 2: Local ARAP Strain Outlier Filtering
For each track $i$ and its $k$-NN canonical neighbors $\mathcal{N}(i)$:
$$s_i = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \frac{\left| \|\mathbf{P}_{C, i} - \mathbf{P}_{C, j}\| - \|\mathbf{p}_{O, i} - \mathbf{p}_{O, j}\| \right|}{\|\mathbf{p}_{O, i} - \mathbf{p}_{O, j}\|}$$
Any track with mean strain $s_i > \tau_{\text{strain}}$ (default: $0.15$, or 15% length distortion) is pruned. This rejects flow points that drifted onto the user's hand or background.

#### Step 3: Differentiable Depth Optimization Along Visual Rays
Optical flow gives accurate angular rays $\mathbf{r}_i$, but monocular depths $d_i$ are noisy. We solve a 1D per-ray depth optimization:
$$\min_{\{z_i\}} \frac{1}{M} \sum_{i=1}^M (z_i - d_i)^2 + \lambda_{\text{arap}} \frac{1}{|\mathcal{E}|} \sum_{(i,j) \in \mathcal{E}} \left( \|\mathbf{r}_i z_i - \mathbf{r}_j z_j\| - d_{ij}^{\text{ref}} \right)^2$$
Optimized in 20–25 steps using PyTorch on GPU (~3 ms). The resulting 3D points $\mathbf{P}^*_{C, i} = z_i^* \mathbf{r}_i$ satisfy canonical object isometry.

#### Step 4: Closed-Form Kabsch 6-DoF Alignment
From regularized camera coordinates $\mathbf{P}^*_C$ and canonical coordinates $\mathbf{P}_O$:
$$T_{\text{arap}} = \text{Kabsch}(\mathbf{P}_O, \mathbf{P}^*_C)$$
This produces a reliable 6-DoF pose prior derived solely from visual flow and regularized geometry, completely independent of constant-velocity extrapolation.

#### Step 5: Multi-Candidate Consensus
Poses from PnP RANSAC ($T_{\text{pnp}}$), ARAP Kabsch ($T_{\text{arap}}$), and the motion model ($T_{\text{guess}}$) are scored by reprojection inlier support and visual consistency, followed by refinement.

---

## 3. Quantitative Results & Evaluation

### 3.1 Full Sequence Tracking (150 frames) on HOT3D `clip-001851`

| Method / Configuration | Mean ATE (Full Seq) | Max ATE | Tail Inliers (Frames 146–148) | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline (Original System)** | `1.1418 m` (114.2 cm) | `12.34 m` | 0 (Lost all tracks) | **Collapsed at frame 133** |
| **Ablation: Tracker w/o ARAP Lifting** | `0.0546 m` (5.46 cm) | `0.2854 m` | 20 – 46 | Tracked, high drift in fast motion |
| **Proposed: ARAP Flow Lifting + ARAP Mapping** | **`0.0265 m` (2.65 cm)** | **`0.1190 m`** | **221 – 600 (13× higher)** | **Accurate & stable throughout** |

### 3.2 Transition at Frame 115–125 (High-Acceleration Manipulation)
At frame 115, the object accelerates at ~3.75 cm/frame while the camera is stationary:
- **Raw Lifted Edge Distortion**: Mean error 32.2 mm, max error 153.4 mm.
- **ARAP Outlier Rejection**: Identified and pruned ~40% corrupted correspondences.
- **ARAP Kabsch Pose Accuracy**:
  - Frame 115: Translation error **14.8 mm**, Rotation error **0.31°**
  - Frame 120: Translation error **9.0 mm**, Rotation error **1.39°**
  - Frame 130: Translation error **4.2 mm**, Rotation error **0.38°**

---

## 4. Code Architecture & Integration

- `gs_dyn_obj/arap.py` & `full_system_ow/gs_dyn_obj/arap.py`:
  - `build_knn_graph`: $k$-NN neighbor graph on GPU.
  - `compute_arap_loss`: Differentiable local isometry loss.
  - `compute_rotation_consistency_loss`: Geodesic quaternion alignment loss.
  - `regularize_lifted_flow_with_arap`: Flow strain computation, outlier filtering, ray-depth optimization, and Kabsch alignment.
- `geometric_tracker.py` & `full_system_ow/geometric_tracker.py`:
  - Integrated ARAP flow lifting and candidate pose consensus into `step_informed_with_occlusion`.
- `obj_gs_mapping.py` & `full_system_ow/obj_gs_mapping.py`:
  - Dynamic graph rebuilding on densify/prune events; ARAP loss injection into mapping optimization.
- `run_tracker.py` & `full_system_ow/bundlesdf_gs.py`:
  - Exposed configuration flags and integrated into tracking pipeline.
