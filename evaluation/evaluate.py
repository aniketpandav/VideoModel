"""Video generation evaluation metrics.

Metrics:
  - FID (Fréchet Inception Distance): per-frame image quality
  - SSIM (Structural Similarity): pixel-level similarity
  - PSNR (Peak Signal-to-Noise Ratio): reconstruction quality
  - Temporal Consistency Score: smoothness between frames
"""

from __future__ import annotations
import numpy as np, torch
from typing import Optional


def compute_psnr(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute PSNR between predicted and target frames. Higher = better."""
    mse = np.mean((pred.astype(float) - target.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255.0 ** 2 / mse)


def compute_ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute SSIM between two images. Range: [-1, 1], higher = better."""
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    pred_f, target_f = pred.astype(np.float64), target.astype(np.float64)

    mu1, mu2 = pred_f.mean(), target_f.mean()
    sigma1_sq = ((pred_f - mu1) ** 2).mean()
    sigma2_sq = ((target_f - mu2) ** 2).mean()
    sigma12 = ((pred_f - mu1) * (target_f - mu2)).mean()

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    return float(num / den)


def compute_temporal_consistency(frames: np.ndarray) -> float:
    """Measure temporal consistency as average frame-to-frame SSIM. Higher = smoother."""
    if len(frames) < 2:
        return 1.0
    scores = [compute_ssim(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    return float(np.mean(scores))


def evaluate_video(pred_frames: np.ndarray, target_frames: Optional[np.ndarray] = None) -> dict[str, float]:
    """Run all evaluation metrics on a generated video.

    Args:
        pred_frames: Generated frames [T, H, W, C] uint8.
        target_frames: Optional ground truth frames for comparison.

    Returns:
        Dictionary of metric name -> value.
    """
    results = {"temporal_consistency": compute_temporal_consistency(pred_frames),
               "num_frames": len(pred_frames)}

    if target_frames is not None:
        T = min(len(pred_frames), len(target_frames))
        psnr_scores = [compute_psnr(pred_frames[i], target_frames[i]) for i in range(T)]
        ssim_scores = [compute_ssim(pred_frames[i], target_frames[i]) for i in range(T)]
        results["psnr"] = float(np.mean(psnr_scores))
        results["ssim"] = float(np.mean(ssim_scores))

    return results
