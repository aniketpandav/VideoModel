"""Attention mechanisms for the DiT model.

Supports self-attention, cross-attention, and spatial-temporal modes
with optional xFormers / Flash Attention backends.
"""

from __future__ import annotations
import math, torch, torch.nn as nn, torch.nn.functional as F
from typing import Optional
from einops import rearrange

_ATTENTION_BACKEND = "pytorch"
try:
    import xformers.ops as xops
    _ATTENTION_BACKEND = "xformers"
except ImportError:
    if hasattr(F, "scaled_dot_product_attention"):
        _ATTENTION_BACKEND = "sdpa"


class Attention(nn.Module):
    """Multi-head attention with optional efficient backends."""

    def __init__(self, hidden_size: int = 512, num_heads: int = 8, dropout: float = 0.0,
                 is_cross_attention: bool = False, cross_attention_dim: Optional[int] = None,
                 use_flash: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.is_cross_attention = is_cross_attention
        self.use_flash = use_flash and _ATTENTION_BACKEND != "pytorch"
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=False)
        kv_dim = cross_attention_dim if (is_cross_attention and cross_attention_dim) else hidden_size
        self.to_k = nn.Linear(kv_dim, hidden_size, bias=False)
        self.to_v = nn.Linear(kv_dim, hidden_size, bias=False)
        self.to_out = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, D = x.shape
        kv_input = context if self.is_cross_attention and context is not None else x
        q = rearrange(self.to_q(x), "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(self.to_k(kv_input), "b m (h d) -> b h m d", h=self.num_heads)
        v = rearrange(self.to_v(kv_input), "b m (h d) -> b h m d", h=self.num_heads)

        if self.use_flash and _ATTENTION_BACKEND == "sdpa":
            mask = context_mask[:, None, None, :].bool() if context_mask is not None else None
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            if context_mask is not None:
                attn = attn.masked_fill(~context_mask[:, None, None, :].bool(), float("-inf"))
            out = torch.matmul(F.softmax(attn, dim=-1), v)

        return self.to_out(rearrange(out, "b h n d -> b n (h d)"))


class SpatialTemporalAttention(nn.Module):
    """Joint or decomposed spatial-temporal self-attention."""

    def __init__(self, hidden_size: int = 512, num_heads: int = 8, mode: str = "joint",
                 dropout: float = 0.0, use_flash: bool = True):
        super().__init__()
        self.mode = mode
        if mode == "joint":
            self.attn = Attention(hidden_size, num_heads, dropout, use_flash=use_flash)
        else:
            self.spatial_attn = Attention(hidden_size, num_heads, dropout, use_flash=use_flash)
            self.temporal_attn = Attention(hidden_size, num_heads, dropout, use_flash=use_flash)

    def forward(self, x: torch.Tensor, grid_size: Optional[tuple[int, int, int]] = None) -> torch.Tensor:
        if self.mode == "joint":
            return self.attn(x)
        T, H, W = grid_size
        B, N, D = x.shape
        x = rearrange(x, "b (t h w) d -> (b t) (h w) d", t=T, h=H, w=W)
        x = self.spatial_attn(x)
        x = rearrange(x, "(b t) (h w) d -> (b h w) t d", b=B, t=T, h=H, w=W)
        x = self.temporal_attn(x)
        return rearrange(x, "(b h w) t d -> b (t h w) d", b=B, h=H, w=W)
