"""
Bridge module to interface between dyn_gs_exp and 4D_PM (4D Primitive-Mâché).
Executes 4D_PM algorithms within its dedicated isolated Python environment
(avoiding PyTorch ABI / lietorch CUDA compilation mismatches).
"""

import os
import sys
import subprocess
import json
import pickle
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List

# Standard path to 4D_PM repository and python binary
PM_ROOT = Path("/home/shzhou/project/super_primitive/4D_PM").resolve()
PM_PYTHON = PM_ROOT / ".pixi" / "envs" / "default" / "bin" / "python"
PI3_DEFAULT_PATH = PM_ROOT / "third_party" / "Pi3"


def is_4dpm_available() -> bool:
    """Check if 4D_PM directory and Python binary exist."""
    return PM_ROOT.exists() and PM_PYTHON.exists()


def run_4dpm_script(script_str: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Execute a Python script string inside the 4D_PM pixi environment."""
    if not is_4dpm_available():
        raise RuntimeError(f"4D_PM environment not found at {PM_PYTHON}")

    env = os.environ.copy()
    pythonpath = f"{str(PM_ROOT)}:{str(PI3_DEFAULT_PATH)}:{env.get('PYTHONPATH', '')}"
    env["PYTHONPATH"] = pythonpath
    env["PI3_PATH"] = str(PI3_DEFAULT_PATH)

    result = subprocess.run(
        [str(PM_PYTHON), "-c", script_str],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(PM_ROOT),
        env=env
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"4D_PM script failed (exit code {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result
