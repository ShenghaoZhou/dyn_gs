import subprocess
import os
import glob
import pandas as pd
from pathlib import Path
import re
import sys
from dataclasses import dataclass
import tyro

@dataclass
class BenchmarkConfig:
    data_root: str = "data/hot3d_clips_processed"
    """Root directory for clips."""
    output_dir: str = "benchmark_results"
    """Directory for results."""
    fix_scale: bool = False
    """Flag for scale fixing."""
    fix_color: bool = False
    """Flag for color fixing."""
    use_pgsr: bool = False
    """Flag for PGSR usage."""
    feature_type: str = "orb"
    """Feature type (orb, gftt, grid)."""
    use_ray_dist: bool = False
    """Flag for ray distance usage."""
    save_rrd: bool = True
    """Flag to save Rerun recording."""

def run_benchmark(cfg: BenchmarkConfig):
    data_root = cfg.data_root
    # Find all clip directories
    clips = sorted(glob.glob(os.path.join(data_root, "clip-*")))
    
    if not clips:
        print(f"No clips found in {data_root}")
        return

    results = []
    output_dir = cfg.output_dir
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "hot3d_benchmark.csv")
    
    # Load existing results if they exist to allow resuming
    processed_clips = {}
    if os.path.exists(csv_path):
        try:
            old_df = pd.read_csv(csv_path)
            for _, row in old_df.iterrows():
                if str(row["ATE (m)"]) not in ["N/A", "ERROR"] and str(row["PSNR (dB)"]) not in ["N/A", "ERROR"]:
                    processed_clips[row["clip_id"]] = row.to_dict()
            print(f"Loaded {len(processed_clips)} existing results from {csv_path}")
        except Exception as e:
            print(f"Could not load existing CSV: {e}")

    results = []
    # If we have existing results, pre-populate the results list
    for cid, data in processed_clips.items():
        results.append(data)
    
    df = pd.DataFrame(results)

    print(f"Found {len(clips)} clips. Starting benchmark...")
    
    for clip_path in clips:
        clip_id = os.path.basename(clip_path)
        
        if clip_id in processed_clips:
            print(f"Skipping {clip_id} (already processed)")
            continue

        # Check if model_infer folder is empty
        model_infer_dir = os.path.join(clip_path, "model_infer")
        if os.path.exists(model_infer_dir) and not os.listdir(model_infer_dir):
            print(f"[Warning] model_infer folder is empty for {clip_id}. Skipping.")
            continue

        print(f"\n" + "="*50)
        print(f"Running benchmark for {clip_id}...")
        print("="*50)
        
        rrd_path = os.path.join(output_dir, f"{clip_id}.rrd")
        
        # We use the parameters from the config
        cmd = [
            "pixi", "run", "python", "full_system_ow/run_full_system_bundlegs_ow.py",
            "--clip-id", clip_id,
            "--feature-type", cfg.feature_type,
        ]
        
        if cfg.fix_scale: cmd.append("--fix-scale")
        else: cmd.append("--no-fix-scale")
        
        if cfg.fix_color: cmd.append("--fix-color")
        else: cmd.append("--no-fix-color")
        
        if cfg.use_pgsr: cmd.append("--use-pgsr")
        else: cmd.append("--no-use-pgsr")
        
        if cfg.use_ray_dist: cmd.append("--use-ray-dist")
        else: cmd.append("--no-use-ray-dist")

        if cfg.save_rrd:
            cmd.extend(["--save-rrd", rrd_path])
        
        print(f"Executing: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            ate = None
            psnr = None
            
            # Real-time output monitoring
            for line in process.stdout:
                print(line, end="")
                
                # Look for final metrics in stdout
                ate_match = re.search(r">>> Final Mean ATE: ([\d.]+)m", line)
                if ate_match:
                    ate = float(ate_match.group(1))
                
                psnr_match = re.search(r">>> Final Mean PSNR: ([\d.]+)dB", line)
                if psnr_match:
                    psnr = float(psnr_match.group(1))
            
            process.wait()
            
            if ate is not None or psnr is not None:
                results.append({
                    "clip_id": clip_id,
                    "ATE (m)": ate,
                    "PSNR (dB)": psnr,
                    "rrd_file": rrd_path
                })
                print(f"\n[Result] {clip_id}: ATE={ate}, PSNR={psnr}")
            else:
                print(f"\n[Warning] No metrics found for {clip_id}")
                results.append({
                    "clip_id": clip_id,
                    "ATE (m)": "N/A",
                    "PSNR (dB)": "N/A",
                    "rrd_file": "N/A"
                })
            
            # Save/update CSV after each run
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            
        except Exception as e:
            print(f"Error running {clip_id}: {e}")
            results.append({
                "clip_id": clip_id,
                "ATE (m)": "ERROR",
                "PSNR (dB)": "ERROR",
                "rrd_file": "ERROR"
            })
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)

    # Final summary
    print("\n" + "="*50)
    print("Benchmark complete!")
    print(f"Results saved to {csv_path}")
    print("="*50)
    
    if results:
        final_df = pd.DataFrame(results)
        print(final_df.to_string(index=False))
    else:
        print("No results to display.")

if __name__ == "__main__":
    cfg = tyro.cli(BenchmarkConfig)
    run_benchmark(cfg)
