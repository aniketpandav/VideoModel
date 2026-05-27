"""Training losses for the diffusion model.

Diffusion Loss:
  L_diffusion = E_{t,x_0,ε} [ ||ε - ε_θ(x_t, t, c)||² ]
  - The model predicts noise ε added at random timestep t
  - MSE between predicted and actual noise

Min-SNR-γ Weighting:
  L_weighted = min(SNR(t), γ) / SNR(t) * L_diffusion
  - Reweights loss by signal-to-noise ratio per timestep
  - Prevents high-SNR timesteps from dominating training
  - γ = 5.0 is the recommended default (Hang et al., 2023)

Temporal Consistency Loss:
  L_temporal = Σ_t ||f(frame_t) - f(frame_{t+1})||²
  - Applied to noise predictions in latent space
  - Encourages temporally smooth denoising predictions
  - Reduces flickering artifacts in generated video

Perceptual Loss:
  L_perceptual = Σ_l ||VGG_l(x) - VGG_l(x̂)||²
  - Feature-matching in VGG feature space
  - Captures structural and semantic similarity beyond pixel-level
"""

from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F


class DiffusionLoss(nn.Module):
    """MSE loss for noise prediction in diffusion training.

    Supports optional Min-SNR-γ weighting for improved convergence.

    Args:
        loss_type: Loss function type ('mse', 'l1', 'huber').
        snr_gamma: Min-SNR-γ parameter. Set to 0 to disable SNR weighting.
    """

    def __init__(self, loss_type: str = "mse", snr_gamma: float = 0.0):
        super().__init__()
        self.loss_type = loss_type
        self.snr_gamma = snr_gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                timesteps: torch.Tensor | None = None,
                alphas_cumprod: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """Compute diffusion loss with optional SNR weighting.

        Args:
            pred: Model prediction [B, C, T, H, W].
            target: Training target [B, C, T, H, W].
            timesteps: Diffusion timesteps [B] (needed for SNR weighting).
            alphas_cumprod: Cumulative alpha schedule (needed for SNR weighting).
            mask: Optional loss mask.

        Returns:
            Scalar loss value.
        """
        if self.loss_type == "mse":
            loss = F.mse_loss(pred, target, reduction="none")
        elif self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="none")
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(pred, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        if mask is not None:
            loss = loss * mask

        # Per-sample mean (reduce all dims except batch)
        loss = loss.mean(dim=list(range(1, loss.ndim)))  # [B]

        # Apply Min-SNR-γ weighting
        if self.snr_gamma > 0 and timesteps is not None and alphas_cumprod is not None:
            snr = self._compute_snr(timesteps, alphas_cumprod)
            snr_weight = torch.clamp(snr, max=self.snr_gamma) / snr
            loss = loss * snr_weight

        return loss.mean()

    @staticmethod
    def _compute_snr(timesteps: torch.Tensor, alphas_cumprod: torch.Tensor) -> torch.Tensor:
        """Compute signal-to-noise ratio: SNR(t) = ᾱ_t / (1 - ᾱ_t).

        Args:
            timesteps: Timestep indices [B].
            alphas_cumprod: Cumulative product of alphas [T].

        Returns:
            SNR values [B].
        """
        alpha_cumprod_t = alphas_cumprod.to(timesteps.device)[timesteps]
        snr = alpha_cumprod_t / (1.0 - alpha_cumprod_t)
        return snr


class TemporalConsistencyLoss(nn.Module):
    """Penalize inconsistency between consecutive frames in latent space.

    Applied to noise predictions [B, C, T, H, W] to encourage temporally
    smooth denoising, which reduces flickering in generated videos.

    Args:
        weight: Loss weight multiplier.
    """

    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """Args: video [B, C, T, H, W]. Returns scalar loss."""
        if video.shape[2] < 2:
            return torch.tensor(0.0, device=video.device)
        diff = video[:, :, 1:] - video[:, :, :-1]
        return self.weight * diff.pow(2).mean()
