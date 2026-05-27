"""Normalization layers for the DiT model.

Includes:
  - AdaLayerNorm: LayerNorm modulated by timestep, used in DiT blocks
  - RMSNorm: Efficient normalization (no mean centering)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaLayerNorm(nn.Module):
    """Adaptive Layer Normalization modulated by timestep embedding.

    Applies LayerNorm then scales and shifts using learned projections
    of the timestep embedding:
      AdaLN(x, t) = γ(t) * LayerNorm(x) + β(t)

    This allows the normalization to be timestep-aware, which is critical
    for diffusion models where behavior changes across noise levels.

    Args:
        hidden_size: Feature dimension.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )

    def forward(self, x: torch.Tensor, timestep_emb: torch.Tensor) -> torch.Tensor:
        """Apply adaptive layer norm.

        Args:
            x: Input tensor [B, N, D].
            timestep_emb: Timestep embedding [B, D].

        Returns:
            Normalized and modulated tensor [B, N, D].
        """
        # Project timestep to scale and shift
        params = self.projection(timestep_emb)  # [B, 2*D]
        scale, shift = params.unsqueeze(1).chunk(2, dim=-1)  # [B, 1, D] each

        return self.norm(x) * (1 + scale) + shift


class AdaLayerNormZero(nn.Module):
    """Adaptive LayerNorm with zero-initialized gating.

    Used in DiT blocks. Projects timestep embedding to produce:
      - scale_msa, shift_msa: for self-attention LayerNorm
      - gate_msa: gating for self-attention output
      - scale_mlp, shift_mlp: for MLP LayerNorm
      - gate_mlp: gating for MLP output

    Zero-initialized gates ensure the block starts as identity at init.

    Args:
        hidden_size: Feature dimension.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size * 6),
        )
        # Zero-initialize the projection output
        nn.init.zeros_(self.projection[-1].weight)
        nn.init.zeros_(self.projection[-1].bias)

    def forward(
        self, x: torch.Tensor, timestep_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute adaptive normalization parameters.

        Args:
            x: Input tensor [B, N, D].
            timestep_emb: Timestep embedding [B, D].

        Returns:
            Tuple of (normed_x, gate_msa, scale_msa, shift_msa, scale_mlp, shift_mlp, gate_mlp).
        """
        params = self.projection(timestep_emb).unsqueeze(1)  # [B, 1, 6*D]
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = params.chunk(6, dim=-1)

        normed = self.norm(x) * (1 + scale_msa) + shift_msa
        return normed, gate_msa, shift_msa, scale_msa, scale_mlp, shift_mlp, gate_mlp


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm as it skips mean centering:
      RMSNorm(x) = x / RMS(x) * γ
      RMS(x) = √(mean(x²))

    Args:
        dim: Feature dimension.
        eps: Epsilon for numerical stability.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight
