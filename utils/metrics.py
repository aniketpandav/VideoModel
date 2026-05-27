"""Training and evaluation metrics."""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio.
    
    Args:
        pred: Predicted tensor.
        target: Ground truth tensor.
        max_val: Maximum possible pixel value.
    
    Returns:
        PSNR value in dB.
    """
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(max_val ** 2 / mse)


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    max_val: float = 1.0,
) -> float:
    """Compute Structural Similarity Index (simplified).
    
    Args:
        pred: Predicted tensor [B, C, H, W].
        target: Ground truth tensor [B, C, H, W].
        window_size: Gaussian window size.
        max_val: Maximum possible pixel value.
    
    Returns:
        SSIM value.
    """
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    
    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)
    
    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_cross = mu_pred * mu_target
    
    sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size // 2) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size // 2) - mu_target_sq
    sigma_cross = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_cross
    
    ssim_map = ((2 * mu_cross + C1) * (2 * sigma_cross + C2)) / (
        (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )
    return ssim_map.mean().item()


def compute_video_metrics(
    pred_video: torch.Tensor,
    target_video: torch.Tensor,
) -> dict[str, float]:
    """Compute per-frame PSNR and SSIM for video tensors.
    
    Args:
        pred_video: Predicted video [B, C, T, H, W] in [0, 1].
        target_video: Ground truth video [B, C, T, H, W] in [0, 1].
    
    Returns:
        Dictionary with 'psnr' and 'ssim' values averaged over frames.
    """
    B, C, T, H, W = pred_video.shape
    psnr_vals = []
    ssim_vals = []
    
    for t in range(T):
        pred_frame = pred_video[:, :, t]
        target_frame = target_video[:, :, t]
        psnr_vals.append(compute_psnr(pred_frame, target_frame))
        ssim_vals.append(compute_ssim(pred_frame, target_frame))
    
    return {
        'psnr': np.mean(psnr_vals),
        'ssim': np.mean(ssim_vals),
    }
