# BundleGS: Version Comparison Summary

This document summarizes the key architectural and hyperparameter differences between the **Baseline (Old)** version (recovered in `BundleGS_old/`) and the **Optimized (New)** version (current root files).

## 1. Tracking & Geometric Robustness
The "New" version introduces several safety mechanisms to prevent catastrophic tracking failure during high-velocity motion.

| Component | Baseline (Old) | Optimized (New) | Impact |
| :--- | :--- | :--- | :--- |
| **Feature Detector** | `loftr` / `grid` (1000 pts) | **`orb` (2000 pts)** | Faster detection and better distinctiveness for PnP solvers. |
| **RANSAC Threshold** | 3.0 pixels | **1.0 pixel** | Reduces "noisy" pose estimates; requires better correspondences. |
| **PnP Safety Check** | None (Trusts PnP) | **Pose Jump Rejection** | Rejects jumps > 0.3m; falls back to CV guess to maintain continuity. |
| **Dynamic Refresh** | Keyframes only | **Auto-Redetect (< 20 tracks)** | Prevents tracking collapse by refreshing features mid-frame if count drops. |
| **Motion Model** | Pure CV | **Damped CV** | Caps predicted velocity to physical bounds (e.g., 0.1m/frame) to prevent runaway drift. |

## 2. Mapping & Reconstruction Fidelity
The mapping configuration has been tuned to maximize PSNR and ensure geometric consistency.

| Parameter | Baseline (Old) | Optimized (New) | Impact |
| :--- | :--- | :--- | :--- |
| **Optimization Steps** | 100 steps/frame | **150 steps/frame** | Improved convergence per keyframe. |
| **Mask Enforcement** | Standard | **Aggressive (20.0 weight)** | Sharper object boundaries; prevents "floaters" outside the mask. |
| **Model Constraints** | Free growth | **Fix Color/Scale** | Prevents GS kernels from drifting in appearance during tracking. |
| **Loss Function** | Standard L1 | **Ray-Distance Loss** | Better geometric alignment for 2D GS primitives. |
| **Densification** | Every 5 frames | Every 10 frames | More stable growth; prevents memory bloat in long sequences. |

## 3. Workflow & Infrastructure
- **Coordinate Anchoring**: The New version uses a more robust frame-0 object-centric anchoring system.
- **Photometric Safety**: Added a rejection loop for photometric refinement. If the LM solver suggests a jump > 10°, the result is discarded in favor of the geometric guess.
- **Evaluation Loop**: The New version uses absolute indexing for ground truth comparison, ensuring metrics are calculated against the correct timestamps even with variable strides.

## Summary of Observed Performance (clip-001849)
In recent benchmarks, the **Baseline** version exhibited a catastrophic "flip" at the end of the sequence (jumps of >2m/90°), whereas the **Optimized** version remained stable due to the pose jump protection and damped motion model.
