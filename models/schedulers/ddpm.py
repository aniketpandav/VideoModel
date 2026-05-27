"""DDPM Noise Scheduler — Denoising Diffusion Probabilistic Models.

Mathematical Foundation:
  Forward process (adding noise):
    q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1 - ᾱ_t) * I)
    
    Where:
      β_t = noise schedule (small values ~0.0001 to ~0.02)
      α_t = 1 - β_t
      ᾱ_t = Π_{s=1}^{t} α_s  (cumulative product)

  Training objective:
    L = E_{t, x_0, ε} [ ||ε - ε_θ(x_t, t)||² ]
    
    The model learns to predict the noise ε added at timestep t.

  Reverse process (sampling):
    p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)
    
    μ_θ = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
from typing import Optional


class DDPMScheduler:
    """DDPM noise scheduler with linear or cosine beta schedule.

    Handles both the forward (noising) and reverse (denoising) processes.

    Args:
        num_timesteps: Total number of diffusion timesteps.
        beta_schedule: Type of beta schedule ('linear' or 'cosine').
        beta_start: Starting beta value (for linear schedule).
        beta_end: Ending beta value (for linear schedule).
        prediction_type: What the model predicts ('epsilon' or 'v_prediction').
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: str = "cosine",
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
    ):
        self.num_timesteps = num_timesteps
        self.prediction_type = prediction_type

        # Compute beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif beta_schedule == "cosine":
            betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        # Pre-compute useful quantities
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Quantities for q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Quantities for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _cosine_beta_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule as proposed in 'Improved DDPM'."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _extract(self, arr: torch.Tensor, t: torch.Tensor, shape: tuple) -> torch.Tensor:
        """Extract values from arr at indices t, broadcast to shape."""
        batch_size = t.shape[0]
        out = arr.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(shape) - 1)))

    def add_noise(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward process: add noise to clean data.

        q(x_t | x_0) = √ᾱ_t * x_0 + √(1-ᾱ_t) * ε

        Args:
            x_0: Clean data [B, ...].
            noise: Gaussian noise [B, ...] (same shape as x_0).
            timesteps: Timestep indices [B].

        Returns:
            Noisy data x_t [B, ...].
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_0.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_0.shape
        )
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def get_training_target(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Get the training target based on prediction type.

        For epsilon prediction: target = noise
        For v-prediction: target = √ᾱ_t * ε - √(1-ᾱ_t) * x_0

        Args:
            x_0: Clean data.
            noise: Added noise.
            timesteps: Timestep indices.

        Returns:
            Training target tensor.
        """
        if self.prediction_type == "epsilon":
            return noise
        elif self.prediction_type == "v_prediction":
            sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_0.shape)
            sqrt_one_minus_alpha = self._extract(
                self.sqrt_one_minus_alphas_cumprod, timesteps, x_0.shape
            )
            return sqrt_alpha * noise - sqrt_one_minus_alpha * x_0
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

    def predict_x0(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Predict x_0 from model output.

        From ε prediction:  x_0 = (x_t - √(1-ᾱ_t) * ε_pred) / √ᾱ_t
        """
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_t.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape
        )

        if self.prediction_type == "epsilon":
            return (x_t - sqrt_one_minus_alpha * noise_pred) / sqrt_alpha
        elif self.prediction_type == "v_prediction":
            return sqrt_alpha * x_t - sqrt_one_minus_alpha * noise_pred
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: int,
        x_t: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Single reverse diffusion step: x_t -> x_{t-1}.

        Args:
            noise_pred: Predicted noise from the model.
            timestep: Current timestep.
            x_t: Current noisy sample.
            generator: Random generator for reproducibility.

        Returns:
            Denoised sample x_{t-1}.
        """
        t = torch.tensor([timestep], device=x_t.device)

        # Predict x_0
        x_0_pred = self.predict_x0(x_t, noise_pred, t)
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Compute posterior mean
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        posterior_mean = coef1 * x_0_pred + coef2 * x_t

        # Add noise (except at t=0)
        if timestep > 0:
            noise = torch.randn_like(x_t, generator=generator)
            posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
            x_prev = posterior_mean + torch.sqrt(posterior_var) * noise
        else:
            x_prev = posterior_mean

        return x_prev
