"""
Benchmark runner comparing 4D_PM integration options against the current baseline:
  - Baseline: Current state-of-the-art full_system_ow
  - Option A: Baseline + 4D_PM Keyframe Prior (Pi3 + SAM 2)
  - Option B: Baseline + 4D_PM Analytical Gauss-Newton BA
  - Option C: Two-Pass 4D_PM Geometry + Dynamic GS Appearance
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

DEFAULT_CLIPS = ["clip-001851", "clip-001853"]
DEFAULT_DATA_ROOT = "data/hot3d_clips_processed_more"


def run_experiment(
    clip_id: str,
    variant_name: str,
    extra_flags: List[str],
    data_root: str,
    num_frames: int
) -> Dict[str, Any]:
    print(f"\n{'='*70}")
    print(f"RUNNING: {variant_name} on {clip_id} ({num_frames} frames)")
    print(f"{'='*70}")

    cmd = [
        "/home/shzhou/.pixi/bin/pixi", "run", "python",
        "full_system_ow/run_full_system_bundlegs_ow.py",
        "--data-root", data_root,
        "--clip-id", clip_id,
        "--num-frames", str(num_frames),
        "--no-vis"
    ] + extra_flags

    start_t = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_t

    stdout = proc.stdout
    stderr = proc.stderr

    # Parse ATE and PSNR from stdout
    ate_match = re.search(r"Final Mean ATE:\s*([\d\.]+)m", stdout)
    psnr_match = re.search(r"Final Mean PSNR:\s*([\d\.]+)dB", stdout)

    ate = float(ate_match.group(1)) if ate_match else None
    psnr = float(psnr_match.group(1)) if psnr_match else None
    fps = (num_frames / elapsed) if elapsed > 0 else 0.0

    print(f"Finished {variant_name} in {elapsed:.2f}s | ATE: {ate}m | PSNR: {psnr}dB | FPS: {fps:.2f}")

    if proc.returncode != 0:
        print(f"Warning: Non-zero exit code {proc.returncode} for {variant_name} on {clip_id}")
        print("Stderr snippet:", stderr[-500:])

    return {
        "clip_id": clip_id,
        "variant": variant_name,
        "num_frames": num_frames,
        "ate_m": ate,
        "ate_cm": (ate * 100) if ate is not None else None,
        "psnr_db": psnr,
        "runtime_s": round(elapsed, 2),
        "fps": round(fps, 2),
        "success": (proc.returncode == 0 and ate is not None)
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark 4D_PM integration options")
    parser.add_argument("--clips", nargs="+", default=DEFAULT_CLIPS)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-csv", type=str, default="benchmark_4dpm_options_results.csv")
    args = parser.parse_args()

    configurations = [
        ("Baseline", []),
        ("Option A (4D_PM Prior)", ["--use-4dpm-prior"]),
        ("Option B (4D_PM GN BA)", ["--use-4dpm-gn"]),
        ("Option A + B (Prior + GN)", ["--use-4dpm-prior", "--use-4dpm-gn"]),
    ]

    results = []
    for clip in args.clips:
        for name, flags in configurations:
            res = run_experiment(clip, name, flags, args.data_root, args.num_frames)
            results.append(res)

    # Save to CSV
    output_path = Path(args.output_csv)
    fieldnames = ["clip_id", "variant", "num_frames", "ate_m", "ate_cm", "psnr_db", "runtime_s", "fps", "success"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"\nSuccessfully wrote benchmark results to {output_path}")

    # Print summary table
    print("\n" + "="*80)
    print(f"{'Clip':<14} | {'Variant':<26} | {'ATE (cm)':<10} | {'PSNR (dB)':<10} | {'FPS':<6} | {'Time (s)':<8}")
    print("-" * 80)
    for r in results:
        ate_str = f"{r['ate_cm']:.2f} cm" if r['ate_cm'] is not None else "FAIL"
        psnr_str = f"{r['psnr_db']:.2f} dB" if r['psnr_db'] is not None else "FAIL"
        print(f"{r['clip_id']:<14} | {r['variant']:<26} | {ate_str:<10} | {psnr_str:<10} | {r['fps']:<6.2f} | {r['runtime_s']:<8.1f}")
    print("="*80)


if __name__ == "__main__":
    main()
