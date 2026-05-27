"""DiT Transformer Blocks — core building blocks of the diffusion transformer."""

from __future__ import annotations
import torch, torch.nn as nn
from typing import Optional
from models.dit.attention import Attention, SpatialTemporalAttention
from models.dit.normalization import AdaLayerNorm, RMSNorm


class FeedForward(nn.Module):
    """MLP feed-forward with GELU activation."""
    def __init__(self, hidden_size: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        mlp_dim = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_size),
            nn.Dropout(dropout),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiTBlock(nn.Module):
    """Single DiT transformer block with adaptive layer norm.

    Architecture:
      x -> AdaLN -> Self-Attn -> + -> AdaLN -> Cross-Attn -> + -> AdaLN -> FFN -> +
      
    Uses AdaLayerNorm modulated by timestep embedding for diffusion-aware processing.
    """
    def __init__(self, hidden_size: int = 512, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.0, attention_mode: str = "joint",
                 cross_attention_dim: Optional[int] = None, use_flash: bool = True):
        super().__init__()
        # Self-attention
        self.norm1 = AdaLayerNorm(hidden_size)
        self.self_attn = SpatialTemporalAttention(
            hidden_size, num_heads, mode=attention_mode, dropout=dropout, use_flash=use_flash)

        # Cross-attention (for text/image conditioning)
        self.norm2 = AdaLayerNorm(hidden_size)
        self.cross_attn = Attention(
            hidden_size, num_heads, dropout=dropout,
            is_cross_attention=True,
            cross_attention_dim=cross_attention_dim or hidden_size,
            use_flash=use_flash)

        # Feed-forward
        self.norm3 = AdaLayerNorm(hidden_size)
        self.ffn = FeedForward(hidden_size, mlp_ratio, dropout)

    def forward(self, x: torch.Tensor, timestep_emb: torch.Tensor,
                context: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None,
                grid_size: Optional[tuple[int, int, int]] = None) -> torch.Tensor:
        # Self-attention with residual
        x = x + self.self_attn(self.norm1(x, timestep_emb), grid_size=grid_size)
        # Cross-attention with residual
        if context is not None:
            x = x + self.cross_attn(self.norm2(x, timestep_emb), context=context, context_mask=context_mask)
        # FFN with residual
        x = x + self.ffn(self.norm3(x, timestep_emb))
        return x
