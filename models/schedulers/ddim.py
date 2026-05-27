"""DDIM Sampler — Denoising Diffusion Implicit Models.

DDIM enables deterministic, faster sampling by using a non-Markovian process.
Instead of the 1000-step DDPM reverse process, DDIM can generate quality
samples in as few as 20-50 steps.

Mathematical Foundation:
  DDIM sampling formula:
    x_{t-1} = √ᾱ_{t-1} * x̂_0  +  √(1-ᾱ_{t-1}-σ²_t) * ε_θ(x_t, t)  +  σ_t * z
    
  Where:
    x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t     (predicted clean sample)
    σ_t = η * √((1-ᾱ_{t-1})/(1-ᾱ_t)) * √(1-ᾱ_t/ᾱ_{t-1})
    z ~ N(0, I) if σ_t > 0, else z = 0

  When η=0: fully deterministic (same seed → same output)
  When η=1: equivalent to DDPM
"""

from __future__ import annotations

import torch
import numpy as np
from typing import Optional

from models.schedulers.ddpm import DDPMScheduler


class DDIMScheduler(DDPMScheduler):
    """DDIM sampler for accelerated deterministic sampling.

    Extends DDPMScheduler with DDIM-specific sampling that supports
    variable step counts and deterministic generation.

    Args:
        num_timesteps: Training timesteps (typically 1000).
        num_inference_steps: Sampling timesteps (e.g., 50).
        eta: Stochasticity parameter (0=deterministic, 1=DDPM-like).
        **kwargs: Arguments passed to DDPMScheduler.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        **kwargs,
    ):
        super().__init__(num_timesteps=num_timesteps, **kwargs)
        self.num_inference_steps = num_inference_steps
        self.eta = eta

        # Compute the timestep subsequence
        self.timesteps = self._compute_timesteps(num_inference_steps)

    def _compute_timesteps(self, num_steps: int) -> torch.Tensor:
        """Compute evenly spaced timestep subsequence.

        Args:
            num_steps: Number of inference steps.

        Returns:
            Tensor of timesteps in descending order.
        """
        step_ratio = self.num_timesteps // num_steps
        timesteps = (np.arange(0, num_steps) * step_ratio).round()[::-1].copy()
        return torch.from_numpy(timesteps).long()

    def set_timesteps(self, num_inference_steps: int) -> None:
        """Update the number of inference steps.

        Args:
            num_inference_steps: New step count.
        """
        self.num_inference_steps = num_inference_steps
        self.timesteps = self._compute_timesteps(num_inference_steps)

    def step(
        self,
        noise_pred: torch.Tensor,
        timestep: int,
        x_t: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        prev_timestep: Optional[int] = None,
    ) -> torch.Tensor:
        """DDIM reverse step: x_t -> x_{t-1}.

        Uses the DDIM update rule for potentially faster, deterministic sampling.

        Args:
            noise_pred: Model's noise prediction.
            timestep: Current timestep.
            x_t: Current noisy sample.
            generator: Random generator for reproducibility.
            prev_timestep: Previous timestep (auto-computed if None).

        Returns:
            Denoised sample x_{t-1}.
        """
        t = timestep

        # Determine previous timestep
        if prev_timestep is None:
            t_idx = (self.timesteps == t).nonzero(as_tuple=True)[0]
            if len(t_idx) > 0 and t_idx[0] < len(self.timesteps) - 1:
                prev_t = self.timesteps[t_idx[0] + 1].item()
            else:
                prev_t = 0
        else:
            prev_t = prev_timestep

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t > 0 else torch.tensor(1.0)

        # Move to device
        alpha_prod_t = alpha_prod_t.to(x_t.device)
        alpha_prod_t_prev = alpha_prod_t_prev.to(x_t.device)

        # Predict x_0
        if self.prediction_type == "epsilon":
            x_0_pred = (x_t - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
        elif self.prediction_type == "v_prediction":
            x_0_pred = torch.sqrt(alpha_prod_t) * x_t - torch.sqrt(1 - alpha_prod_t) * noise_pred
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Clamp predicted x_0
        x_0_pred = torch.clamp(x_0_pred, -1.0, 1.0)

        # Compute sigma for stochastic component
        sigma = self.eta * torch.sqrt(
            (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
            * (1 - alpha_prod_t / alpha_prod_t_prev)
        )

        # DDIM formula
        # Direction pointing to x_t
        pred_direction = torch.sqrt(1 - alpha_prod_t_prev - sigma**2) * noise_pred

        # x_{t-1}
        x_prev = torch.sqrt(alpha_prod_t_prev) * x_0_pred + pred_direction

        # Add stochastic component
        if sigma > 0 and t > 0:
            noise = torch.randn_like(x_t, generator=generator)
            x_prev = x_prev + sigma * noise

        return x_prev

    def sample_loop(
        self,
        model_fn,
        shape: tuple[int, ...],
        device: torch.device,
        conditioning: Optional[dict] = None,
        guidance_scale: float = 7.5,
        generator: Optional[torch.Generator] = None,
        callback: Optional[callable] = None,
    ) -> torch.Tensor:
        """Full DDIM sampling loop.

        Implements classifier-free guidance (CFG):
          ε̃ = ε_uncond + s * (ε_cond - ε_uncond)

        Args:
            model_fn: Function(x_t, t, context) -> noise_pred.
            shape: Output tensor shape.
            device: Target device.
            conditioning: Dict with 'context', 'context_mask', and optionally
                         'unconditional_context', 'unconditional_mask'.
            guidance_scale: CFG scale (1.0 = no guidance, 7.5 = typical).
            generator: Random generator.
            callback: Optional callback(step, timestep, x_t) called each step.

        Returns:
            Generated sample tensor.
        """
        # Start from pure noise
        x_t = torch.randn(shape, device=device, generator=generator)

        for i, t in enumerate(self.timesteps):
            t_tensor = torch.full((shape[0],), t.item(), device=device, dtype=torch.long)

            if guidance_scale > 1.0 and conditioning is not None:
                # Classifier-free guidance: run model twice
                # Conditional prediction
                noise_cond = model_fn(
                    x_t, t_tensor,
                    context=conditioning.get("context"),
                    context_mask=conditioning.get("context_mask"),
                )
                # Unconditional prediction
                noise_uncond = model_fn(
                    x_t, t_tensor,
                    context=conditioning.get("unconditional_context"),
                    context_mask=conditioning.get("unconditional_mask"),
                )
                # CFG formula
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = model_fn(
                    x_t, t_tensor,
                    context=conditioning.get("context") if conditioning else None,
                    context_mask=conditioning.get("context_mask") if conditioning else None,
                )

            # Compute previous timestep
            prev_t = self.timesteps[i + 1].item() if i < len(self.timesteps) - 1 else 0

            x_t = self.step(noise_pred, t.item(), x_t, generator=generator, prev_timestep=prev_t)

            if callback is not None:
                callback(i, t.item(), x_t)

        return x_t
