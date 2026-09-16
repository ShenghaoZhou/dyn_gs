"""
Integration package for incorporating 4D Primitive-Mâché (4D_PM) into dyn_gs_exp.
Supports:
  - Option A: Multi-view Keyframe Prior (Pi3 + SAM 2)
  - Option B: Analytical Gauss-Newton BA Solver
  - Option C: Two-Pass Hybrid Reconstruction (4D_PM Geometry + Dynamic GS Appearance)
"""

from .env_bridge import is_4dpm_available, run_4dpm_script
