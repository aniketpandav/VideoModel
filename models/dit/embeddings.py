"""Embeddings for the Diffusion Transformer.

Includes:
  - Sinusoidal timestep embeddings
  - 3D Rotary Positional Embeddings (RoPE)
  - Patch embedding: converts video latents to token sequences

Attention mechanism:
  Standard self-attention: softmax(QK^T / √d) * V
  
  With RoPE, positional information is injected by rotating Q and K:
    Q' = R(θ_pos) * Q
    K' = R(θ_pos) * K
  
  This allows the model to learn relative positional relationships
  across spatial (H, W) and temporal (T) dimensions simultaneously.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from einops import rearrange


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Maps integer timestep t to a high-dimensional vector using
    sine and cosine functions at different frequencies.

    Args:
        dim: Embedding dimension.
        max_period: Maximum period for the sinusoidal functions.
    """

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

        # MLP to project sinusoidal features
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute timestep embeddings.

        Args:
            timesteps: Integer timesteps [B].

        Returns:
            Timestep embeddings [B, dim].
        """
        half_dim = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        if self.dim % 2 == 1:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)

        return self.mlp(embedding)


class RotaryPositionalEmbedding3D(nn.Module):
    """3D Rotary Positional Embedding (RoPE) for spatial-temporal tokens.

    Injects positional information by rotating query and key vectors,
    allowing the model to learn relative positions across T, H, W dimensions.

    Args:
        dim: Per-head dimension (must be divisible by 6 for 3D: T, H, W each get dim//3).
        max_temporal: Maximum temporal length.
        max_spatial: Maximum spatial length (H or W).
    """

    def __init__(
        self,
        dim: int,
        max_temporal: int = 64,
        max_spatial: int = 64,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        # Split dim equally among T, H, W
        self.dim_per_axis = dim // 3
        if self.dim_per_axis * 3 != dim:
            self.dim_per_axis = dim // 3
            # Pad remaining to first axis
            self._remainder = dim - self.dim_per_axis * 3
        else:
            self._remainder = 0

        # Precompute frequency bands
        inv_freq = 1.0 / (theta ** (torch.arange(0, self.dim_per_axis, 2).float() / self.dim_per_axis))
        self.register_buffer("inv_freq", inv_freq)

    def _compute_rope(self, positions: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute rotary embeddings for a single axis.

        Args:
            positions: Position indices [N].
            dim: Feature dimension for this axis.

        Returns:
            cos, sin tensors [N, dim].
        """
        inv_freq = self.inv_freq[:dim // 2].to(positions.device)
        freqs = torch.einsum("i,j->ij", positions.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        temporal_ids: torch.Tensor,
        height_ids: torch.Tensor,
        width_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply 3D RoPE to query and key tensors.

        Args:
            q: Query tensor [B, heads, N, head_dim].
            k: Key tensor [B, heads, N, head_dim].
            temporal_ids: Temporal position IDs [N].
            height_ids: Height position IDs [N].
            width_ids: Width position IDs [N].

        Returns:
            Rotated (q, k) tensors.
        """
        d = self.dim_per_axis

        # Compute RoPE for each axis
        cos_t, sin_t = self._compute_rope(temporal_ids, d)
        cos_h, sin_h = self._compute_rope(height_ids, d)
        cos_w, sin_w = self._compute_rope(width_ids, d)

        # Concatenate cos/sin for all axes
        cos = torch.cat([cos_t, cos_h, cos_w], dim=-1)  # [N, 3*d]
        sin = torch.cat([sin_t, sin_h, sin_w], dim=-1)  # [N, 3*d]

        # Handle remainder dimensions (no rotation)
        total = cos.shape[-1]
        head_dim = q.shape[-1]
        if total < head_dim:
            pad = head_dim - total
            cos = torch.cat([cos, torch.ones(cos.shape[0], pad, device=cos.device)], dim=-1)
            sin = torch.cat([sin, torch.zeros(sin.shape[0], pad, device=sin.device)], dim=-1)

        cos = cos[:total].unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
        sin = sin[:total].unsqueeze(0).unsqueeze(0)

        # Apply rotation
        q_rot = self._rotate(q[..., :total], cos, sin)
        k_rot = self._rotate(k[..., :total], cos, sin)

        # Concatenate unrotated remainder
        if total < head_dim:
            q_rot = torch.cat([q_rot, q[..., total:]], dim=-1)
            k_rot = torch.cat([k_rot, k[..., total:]], dim=-1)

        return q_rot, k_rot

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Apply rotation using complex number multiplication trick."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        cos = cos[..., : x.shape[-1]]
        sin = sin[..., : x.shape[-1]]
        c1 = cos[..., : x1.shape[-1]]
        c2 = cos[..., x1.shape[-1] :]
        s1 = sin[..., : x1.shape[-1]]
        s2 = sin[..., x1.shape[-1] :]
        return torch.cat([x1 * c1 - x2 * s1, x2 * c2 + x1 * s2], dim=-1)


class PatchEmbed3D(nn.Module):
    """Convert video latent tensor to a sequence of patch tokens.

    Converts [B, C, T, H, W] -> [B, N, hidden_size] where N = (T/pt)*(H/ph)*(W/pw).

    Args:
        in_channels: Input latent channels.
        hidden_size: Output embedding dimension.
        patch_size: (patch_t, patch_h, patch_w) — patch dimensions.
    """

    def __init__(
        self,
        in_channels: int = 4,
        hidden_size: int = 512,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.proj = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int]]:
        """Patchify video latent.

        Args:
            x: Video latent [B, C, T, H, W].

        Returns:
            Tuple of (tokens [B, N, D], grid_size (T', H', W')).
        """
        B, C, T, H, W = x.shape
        x = self.proj(x)  # [B, D, T', H', W']
        T_out, H_out, W_out = x.shape[2], x.shape[3], x.shape[4]

        # Flatten to sequence: [B, D, T', H', W'] -> [B, N, D]
        x = rearrange(x, "b d t h w -> b (t h w) d")
        x = self.norm(x)

        return x, (T_out, H_out, W_out)

    def unpatchify(
        self, x: torch.Tensor, grid_size: tuple[int, int, int]
    ) -> torch.Tensor:
        """Convert token sequence back to spatial tensor.

        Args:
            x: Token sequence [B, N, D].
            grid_size: (T', H', W') grid dimensions.

        Returns:
            Spatial tensor [B, D, T', H', W'].
        """
        T, H, W = grid_size
        return rearrange(x, "b (t h w) d -> b d t h w", t=T, h=H, w=W)
