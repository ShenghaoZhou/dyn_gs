#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

@torch.no_grad()
def adamUpdateBasic(param, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps):
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    denom = exp_avg_sq.sqrt().add_(eps)
    param.addcdiv_(exp_avg, denom, value=-lr)

@torch.no_grad()
def adamUpdate(param, grad, exp_avg, exp_avg_sq, visibility, lr, beta1, beta2, eps, N, M):
    if visibility.any():
        v_grad = grad[visibility]
        v_exp_avg = exp_avg[visibility]
        v_exp_avg_sq = exp_avg_sq[visibility]
        v_param = param[visibility]
        
        v_exp_avg.mul_(beta1).add_(v_grad, alpha=1 - beta1)
        v_exp_avg_sq.mul_(beta2).addcmul_(v_grad, v_grad, value=1 - beta2)
        
        denom = v_exp_avg_sq.sqrt().add_(eps)
        
        if lr.dim() > 0:
            v_lr = lr[visibility].view(-1, *([1] * (v_param.ndim - 1)))
            v_param.sub_(v_lr * v_exp_avg / denom)
        else:
            v_param.addcdiv_(v_exp_avg, denom, value=-lr)
        
        param[visibility] = v_param
        exp_avg[visibility] = v_exp_avg
        exp_avg_sq[visibility] = v_exp_avg_sq
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 /
                         float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def fused_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device).type_as(img1)
    
    mu1 = torch.nn.functional.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = torch.nn.functional.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = torch.nn.functional.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class BaseAdam:
    """Adam optimizer for regular parameters. This is simpler than torch.optim.Adam and initializes faster."""
    @torch.no_grad()
    def __init__(self, params, betas=(0.9, 0.999), eps=1e-15):
        self.params = params
        self.betas = betas
        self.eps = eps
        # Initialize moments if not already done
        for param in self.params.values():
            if "exp_avg" not in param:
                param["exp_avg"] = torch.zeros_like(
                    param["val"], memory_format=torch.preserve_format
                )
                param["exp_avg_sq"] = torch.zeros_like(
                    param["val"], memory_format=torch.preserve_format
                )

    def zero_grad(self):
        for param in self.params.values():
            param["val"].grad = None

    @torch.no_grad()
    def step(self):
        for param_dict in self.params.values():
            lr = param_dict["lr"]
            param = param_dict["val"]
            if param.grad is None:
                continue

            exp_avg = param_dict["exp_avg"]
            exp_avg_sq = param_dict["exp_avg_sq"]
            adamUpdateBasic(
                param,
                param.grad,
                exp_avg,
                exp_avg_sq,
                lr,
                self.betas[0],
                self.betas[1],
                self.eps,
            )


class SparseGaussianAdam(BaseAdam):
    """Adam optimizer for primitive parameters that can be optimized with sparse updates."""

    def __init__(self, params, betas=(0.9, 0.999), eps=1e-15, lr_dict={}):
        super().__init__(params=params, betas=betas, eps=eps)

        self.lr_dict = lr_dict
        # Convert learning rates to tensors
        for key, param in self.params.items():
            if "lr" not in param or type(param["lr"]) is not torch.Tensor:
                if key not in self.lr_dict:
                    param["lr"] = torch.tensor(
                        param["lr"], dtype=torch.float, device="cuda"
                    )
                else:
                    param["lr"] = torch.empty(0, dtype=torch.float, device="cuda")

    @torch.no_grad()
    def step(self, visibility, N):
        for key, param_dict in self.params.items():
            # Run the update
            lr = param_dict["lr"]
            param = param_dict["val"]
            if param.grad is None:
                continue

            exp_avg = param_dict["exp_avg"]
            exp_avg_sq = param_dict["exp_avg_sq"]
            M = param.numel() // N
            adamUpdate(
                param,
                param.grad,
                exp_avg,
                exp_avg_sq,
                visibility,
                lr,
                self.betas[0],
                self.betas[1],
                self.eps,
                N,
                M,
            )

            # Update the learning rate
            if key in self.lr_dict:
                param_dict["lr"][visibility] *= self.lr_dict[key]["lr_decay"]
                param_dict["lr"].clamp_min_(self.lr_dict[key]["lr_init"] * 0.1)

    def add_and_prune(self, extension_tensors, valid_mask):
        for key, param in self.params.items():
            extension_tensor = extension_tensors[key]
            param["val"] = torch.cat(
                [param["val"].detach()[valid_mask], extension_tensor], dim=0
            ).contiguous()
            param["val"].requires_grad = True
            param["exp_avg"] = torch.cat(
                [param["exp_avg"][valid_mask], torch.zeros_like(extension_tensor)],
                dim=0,
            ).contiguous()
            param["exp_avg_sq"] = torch.cat(
                [param["exp_avg_sq"][valid_mask], torch.zeros_like(extension_tensor)],
                dim=0,
            ).contiguous()

            if key in self.lr_dict:
                param["lr"] = torch.cat(
                    [
                        param["lr"][valid_mask],
                        torch.ones(extension_tensor.shape[0], device=extension_tensor.device, dtype=torch.float) * self.lr_dict[key]["lr_init"],
                    ],
                    dim=0,
                ).contiguous()
