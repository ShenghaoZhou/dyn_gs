# BundleGS Mapping Performance Benchmark

This table summarizes the mapping performance across HOT3D sequences.

| Clip        |   No-Cheat PSNR |   No-Cheat ATE |   Ideal PSNR |
|:------------|----------------:|---------------:|-------------:|
| clip-001849 |           13.16 |         0.1191 |        15.94 |
| clip-001850 |            0    |         0.6091 |         0    |
| clip-001851 |           16    |         0.0028 |         0    |
| clip-001853 |            9.35 |         0.3053 |         0    |
| clip-001854 |            0    |         0      |         0    |
| clip-001855 |            0    |         0      |         0    |
| clip-001856 |            0    |         0      |         0    |
| clip-001857 |            0    |         0      |         0    |
| clip-001868 |            0    |         0      |         0    |
| clip-001870 |            0    |         0      |         0    |

## Analysis
- **Ideal Case**: Repesents the upper bound of GS reconstruction quality with perfect depth and pose.
- **No-Cheat Pipeline**: Evaluates the real-world performance where tracking and depth are estimated.
