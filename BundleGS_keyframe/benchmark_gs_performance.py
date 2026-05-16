import os
import subprocess
import json
from pathlib import Path
import pandas as pd

def run_cmd(cmd):
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout

def extract_metrics(output):
    metrics = {}
    lines = output.split('\n')
    for line in lines:
        if 'ATE:' in line:
            metrics['ATE'] = float(line.split('ATE:')[1].split('m')[0].strip())
        if 'Rot Error:' in line or 'RotErr:' in line:
            metrics['RotErr'] = float(line.split('Rot Error:')[1].split('deg')[0].strip()) if 'Rot Error:' in line else float(line.split('RotErr:')[1].split('deg')[0].strip())
        if 'Average PSNR:' in line:
            metrics['PSNR'] = float(line.split('Average PSNR:')[1].split('dB')[0].strip())
    return metrics

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    data_root = "data/hot3d_clips_processed"
    clips = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith("clip-")]
    clips.sort()
    
    if args.limit > 0:
        clips = clips[:args.limit]
    
    results = []
    
    for clip in clips:
        print(f"\n>>> Benchmarking {clip}")
        
        # 1. No-Cheat Pipeline
        cmd_no_cheat = f"pixi run python BundleGS/test_hot3d.py --clip_id {clip} --gs-type 3d --fix-color --fix-scale --num_steps 30 --use_ray_dist --n-frames-eval 10 --no-vis"
        out_no_cheat = run_cmd(cmd_no_cheat)
        metrics_no_cheat = extract_metrics(out_no_cheat)
        
        # 2. Ideal Case (Upper Bound)
        cmd_ideal = f"pixi run python BundleGS/test_hot3d_with_gt.py --clip_id {clip} --use-gt-pose --use-gt-depth --gs-type 2d --num_steps 100 --use_ray_dist --every-frame-mapping --n-frames-eval 10 --no-vis"
        out_ideal = run_cmd(cmd_ideal)
        metrics_ideal = extract_metrics(out_ideal)
        
        results.append({
            'Clip': clip,
            'No-Cheat PSNR': metrics_no_cheat.get('PSNR', 0),
            'No-Cheat ATE': metrics_no_cheat.get('ATE', 0),
            'Ideal PSNR': metrics_ideal.get('PSNR', 0),
        })
    
    df = pd.DataFrame(results)
    print("\n" + df.to_markdown())
    
    with open("BundleGS/benchmark_results.md", "w") as f:
        f.write("# BundleGS Mapping Performance Benchmark\n\n")
        f.write("This table summarizes the mapping performance across HOT3D sequences.\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Analysis\n")
        f.write("- **Ideal Case**: Repesents the upper bound of GS reconstruction quality with perfect depth and pose.\n")
        f.write("- **No-Cheat Pipeline**: Evaluates the real-world performance where tracking and depth are estimated.\n")

if __name__ == "__main__":
    main()
