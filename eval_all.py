import subprocess
import re
import sys
import json
from pathlib import Path

CLIPS = ["clip-001849", "clip-001882", "clip-001924", "clip-001959", "clip-002002", "clip-002793", "clip-003312", "clip-003363"]

def run_clip(clip_id):
    print(f"========================================\nRunning {clip_id}...\n========================================")
    cmd = ["pixi", "run", "python", "full_system_ow/eval_full_system.py", "--clip-id", clip_id]
    
    # Run the evaluation process and print output in real-time
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    ate, psnr, ssim, lpips = None, None, None, None
    
    for line in process.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        
        # Parse final metrics
        if "Final Mean ATE" in line:
            m = re.search(r"Final Mean ATE:\s*([\d\.]+)", line)
            if m: ate = float(m.group(1))
        elif "Final Mean PSNR" in line:
            m = re.search(r"Final Mean PSNR:\s*([\d\.]+)", line)
            if m: psnr = float(m.group(1))
        elif "Final Mean SSIM" in line:
            m = re.search(r"Final Mean SSIM:\s*([\d\.]+)", line)
            if m: ssim = float(m.group(1))
        elif "Final Mean LPIPS" in line:
            m = re.search(r"Final Mean LPIPS:\s*([\d\.]+)", line)
            if m: lpips = float(m.group(1))
            
    process.wait()
    return {"clip_id": clip_id, "ATE": ate, "PSNR": psnr, "SSIM": ssim, "LPIPS": lpips}

def main():
    results = []
    for clip in CLIPS:
        res = run_clip(clip)
        results.append(res)
        
        # Intermediate save
        with open("eval_results_temp.json", "w") as f:
            json.dump(results, f, indent=4)
            
    print("\n\n========================================")
    print("EVALUATION COMPLETED. SUMMARY RESULTS:")
    print("========================================")
    
    # Format and calculate averages
    valid_ates = [r["ATE"] for r in results if r["ATE"] is not None]
    valid_psnrs = [r["PSNR"] for r in results if r["PSNR"] is not None]
    valid_ssims = [r["SSIM"] for r in results if r["SSIM"] is not None]
    valid_lpipss = [r["LPIPS"] for r in results if r["LPIPS"] is not None]
    
    avg_ate = sum(valid_ates) / len(valid_ates) if valid_ates else 0.0
    avg_psnr = sum(valid_psnrs) / len(valid_psnrs) if valid_psnrs else 0.0
    avg_ssim = sum(valid_ssims) / len(valid_ssims) if valid_ssims else 0.0
    avg_lpips = sum(valid_lpipss) / len(valid_lpipss) if valid_lpipss else 0.0
    
    headers = ["Sequence", "ATE (m)", "PSNR (dB)", "SSIM", "LPIPS"]
    row_fmt = "{:<15} | {:<8} | {:<9} | {:<8} | {:<8}"
    
    print(row_fmt.format(*headers))
    print("-" * 60)
    for r in results:
        print(row_fmt.format(
            r["clip_id"],
            f"{r['ATE']:.4f}" if r["ATE"] is not None else "N/A",
            f"{r['PSNR']:.2f}" if r["PSNR"] is not None else "N/A",
            f"{r['SSIM']:.4f}" if r["SSIM"] is not None else "N/A",
            f"{r['LPIPS']:.4f}" if r["LPIPS"] is not None else "N/A"
        ))
    print("-" * 60)
    print(row_fmt.format(
        "AVERAGE",
        f"{avg_ate:.4f}",
        f"{avg_psnr:.2f}",
        f"{avg_ssim:.4f}",
        f"{avg_lpips:.4f}"
    ))
    
    # Save final results to json
    with open("eval_results_final.json", "w") as f:
        json.dump({
            "results": results,
            "average": {
                "ATE": avg_ate,
                "PSNR": avg_psnr,
                "SSIM": avg_ssim,
                "LPIPS": avg_lpips
            }
        }, f, indent=4)

if __name__ == "__main__":
    main()
