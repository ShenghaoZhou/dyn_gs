import torch
from gsplat import fully_fused_projection
import math

def test_projection():
    print("Testing gsplat projection...")
    N = 100
    means3d = torch.randn((N, 3), device="cuda")
    quats = torch.randn((N, 4), device="cuda")
    scales = torch.exp(torch.randn((N, 3), device="cuda"))
    viewmat = torch.eye(4, device="cuda")
    K = torch.tensor([[500, 0, 256], [0, 500, 256], [0, 0, 1]], device="cuda", dtype=torch.float32)
    W, H = 512, 512
    
    try:
        # Just a dummy call to see if it triggers any issues
        full_proj = fully_fused_projection(
            means3d,
            quats,
            scales,
            None, # opacities
            viewmat,
            K,
            W, H,
            render_mode="RGB"
        )
        print("Projection successful!")
        return True
    except Exception as e:
        print(f"Projection failed: {e}")
        return False

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("CUDA is NOT available!")
    else:
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        test_projection()
