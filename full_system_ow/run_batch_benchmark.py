import argparse
import subprocess
import re
import csv
import sys
from pathlib import Path
import numpy as np

CLIPS_DEFAULT = [
    "clip-001851",
    "clip-001853",
    "clip-001854",
    "clip-001855",
    "clip-001856",
    "clip-001857",
]

def parse_args():
    parser = argparse.ArgumentParser(description="Batch benchmark runner for full_system_ow")
    parser.add_argument("--data-root", type=str, default="data/hot3d_clips_processed_more")
    parser.add_argument("--clips", nargs="+", default=CLIPS_DEFAULT)
    parser.add_argument("--num-frames", type=int, default=150)
    parser.add_argument("--use-any4d", action="store_true", default=False)
    parser.add_argument("--output-csv", type=str, default="benchmark_full_system_results.csv")
    return parser.parse_args()

def run_clip(clip_id, args):
    cmd = [
        sys.executable,
        "full_system_ow/run_full_system_bundlegs_ow.py",
        "--data-root", args.data_root,
        "--clip-id", clip_id,
        "--num-frames", str(args.num_frames),
        "--no-vis"
    ]
    if args.use_any4d:
        cmd.append("--use-any4d")
    
    print(f"\n==========================================")
    print(f"Running {clip_id} (Any4D={args.use_any4d}, Frames={args.num_frames})...")
    print(f"Command: {' '.join(cmd)}")
    print(f"==========================================")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )

    ate, psnr = None, None
    ate_pattern = re.compile(r"Final Mean ATE:\s*([\d\.]+)m")
    psnr_pattern = re.compile(r"Final Mean PSNR:\s*([\d\.]+)dB")

    output_lines = []
    for line in iter(proc.stdout.readline, ""):
        output_lines.append(line)
        # Print key milestone lines to terminal
        if any(keyword in line for keyword in ["Frame ", "Final Mean", "Recovering with Any4D", "Registered new dynamic anchor", "Error", "Exception", "Traceback"]):
            print(line, end="")
        
        m_ate = ate_pattern.search(line)
        if m_ate:
            ate = float(m_ate.group(1))
        m_psnr = psnr_pattern.search(line)
        if m_psnr:
            psnr = float(m_psnr.group(1))

    proc.stdout.close()
    return_code = proc.wait()

    if return_code != 0:
        print(f"[Error] {clip_id} exited with return code {return_code}")
        # print last 10 lines of output
        for line in output_lines[-10:]:
            print(line, end="")

    return ate, psnr

def main():
    args = parse_args()
    results = []

    print(f"Evaluating {len(args.clips)} clips with Any4D={args.use_any4d}...")

    for clip in args.clips:
        ate, psnr = run_clip(clip, args)
        results.append({
            "clip_id": clip,
            "use_any4d": args.use_any4d,
            "num_frames": args.num_frames,
            "ate_m": ate,
            "psnr_db": psnr,
        })
        print(f">>> Result for {clip}: ATE={ate} m, PSNR={psnr} dB")

    # Save to CSV
    output_path = Path(args.output_csv)
    file_exists = output_path.exists()
    with open(output_path, "a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_id", "use_any4d", "num_frames", "ate_m", "psnr_db"])
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow(r)

    print("\n================ FINAL SUMMARY ================")
    print(f"{'Clip ID':<15} | {'Any4D':<8} | {'Frames':<8} | {'ATE (m)':<10} | {'PSNR (dB)':<10}")
    print("-" * 60)
    valid_ates = []
    valid_psnrs = []
    for r in results:
        ate_str = f"{r['ate_m']:.4f}" if r['ate_m'] is not None else "N/A"
        psnr_str = f"{r['psnr_db']:.2f}" if r['psnr_db'] is not None else "N/A"
        print(f"{r['clip_id']:<15} | {str(r['use_any4d']):<8} | {r['num_frames']:<8} | {ate_str:<10} | {psnr_str:<10}")
        if r['ate_m'] is not None: valid_ates.append(r['ate_m'])
        if r['psnr_db'] is not None: valid_psnrs.append(r['psnr_db'])

    print("-" * 60)
    mean_ate = np.mean(valid_ates) if valid_ates else float("nan")
    mean_psnr = np.mean(valid_psnrs) if valid_psnrs else float("nan")
    print(f"{'MEAN':<15} | {str(args.use_any4d):<8} | {'-':<8} | {mean_ate:.4f}     | {mean_psnr:.2f}")
    print(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
