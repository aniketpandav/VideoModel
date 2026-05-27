"""3D Causal VAE Encoder for video compression.

Compresses video [B, C, T, H, W] into latent space [B, latent_ch, T', H', W']
using 3D causal convolutions with spatial 8x and temporal 4x downsampling.

Architecture:
  Input (3, T, H, W) -> Conv3D -> [ResBlock + Downsample] x N -> Attention -> Conv3D -> (2*Z, T/4, H/8, W/8)

The encoder outputs mean and logvar for reparameterization.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CausalConv3d(nn.Module):
    """3D causal convolution - pads only the past in temporal dimension.

    Ensures that each output frame only depends on current and past input frames,
    which is critical for autoregressive generation and variable-length support.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.kernel_size = kernel_size
        self.stride = stride

        # Temporal: pad only the past (causal)
        self.temporal_pad = kernel_size[0] - 1
        # Spatial: symmetric padding
        self.spatial_pad_h = kernel_size[1] // 2
        self.spatial_pad_w = kernel_size[2] // 2

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(0, self.spatial_pad_h, self.spatial_pad_w),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        # Causal padding: pad temporal dim on left side only
        if self.temporal_pad > 0:
            x = F.pad(x, (0, 0, 0, 0, self.temporal_pad, 0))
        return self.conv(x)


class ResBlock3D(nn.Module):
    """3D Residual block with GroupNorm and SiLU activation.

    ResBlock(x) = x + Conv3D(SiLU(GN(Conv3D(SiLU(GN(x))))))
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_groups: int = 32,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        self.norm1 = nn.GroupNorm(min(num_groups, in_channels), in_channels)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)

        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class SpatialDownsample(nn.Module):
    """Downsample spatial dimensions by 2x using strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, channels,
            kernel_size=(1, 3, 3),
            stride=(1, 2, 2),
            padding=(0, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TemporalDownsample(nn.Module):
    """Downsample temporal dimension by 2x using strided causal convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv3d(
            channels, channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionBlock3D(nn.Module):
    """Self-attention block operating on the spatial dimensions.

    For efficiency, attention is applied per-frame (flattened H*W tokens).
    """

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        residual = x

        x = self.norm(x)
        # Reshape to [B*T, C, H*W] for spatial attention
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H * W)

        qkv = self.qkv(x).reshape(B * T, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        # Reshape to [B*T, heads, HW, d] for attention
        q = q.permute(0, 1, 3, 2)  # [B*T, heads, HW, d]
        k = k.permute(0, 1, 3, 2)  # [B*T, heads, HW, d]
        v = v.permute(0, 1, 3, 2)  # [B*T, heads, HW, d]

        # Use efficient attention when available
        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            scale = (C // self.num_heads) ** -0.5
            attn = torch.matmul(q, k.transpose(-1, -2)) * scale
            attn = F.softmax(attn, dim=-1)
            out = torch.matmul(attn, v)  # [B*T, heads, HW, d]

        out = out.permute(0, 1, 3, 2).reshape(B * T, C, H * W)
        out = self.proj(out)
        out = out.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

        return out + residual


class Encoder3D(nn.Module):
    """3D Video Encoder with causal convolutions.

    Compresses video tensor from pixel space to latent space:
      [B, 3, T, H, W] -> [B, 2*latent_channels, T//4, H//8, W//8]

    Output contains concatenated mean and logvar for reparameterization.

    Args:
        in_channels: Input channels (3 for RGB).
        latent_channels: Latent space channels (typically 4).
        base_channels: Base channel count for the first layer.
        channel_multipliers: Channel multiplier at each encoder stage.
        num_res_blocks: Number of residual blocks per stage.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.latent_channels = latent_channels

        # Initial projection
        self.conv_in = CausalConv3d(in_channels, base_channels, kernel_size=3)

        # Encoder stages
        self.stages = nn.ModuleList()
        channels = [base_channels * m for m in channel_multipliers]
        in_ch = base_channels

        for i, out_ch in enumerate(channels):
            stage_blocks = nn.ModuleList()

            # Residual blocks
            for j in range(num_res_blocks):
                stage_blocks.append(
                    ResBlock3D(in_ch if j == 0 else out_ch, out_ch)
                )

            # Downsampling (spatial for all, temporal for first two stages)
            downsample = nn.ModuleList()
            if i < len(channels) - 1:  # No downsample at last stage
                downsample.append(SpatialDownsample(out_ch))
                if i < 2:  # Temporal downsample in first 2 stages (4x total)
                    downsample.append(TemporalDownsample(out_ch))

            self.stages.append(nn.ModuleDict({
                "blocks": stage_blocks,
                "downsample": downsample,
            }))
            in_ch = out_ch

        # Middle block with attention
        self.mid_block1 = ResBlock3D(channels[-1])
        self.mid_attn = AttentionBlock3D(channels[-1])
        self.mid_block2 = ResBlock3D(channels[-1])

        # Output projection (mean + logvar)
        self.norm_out = nn.GroupNorm(32, channels[-1])
        self.conv_out = CausalConv3d(channels[-1], 2 * latent_channels, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode video to latent distribution parameters.

        Args:
            x: Input video [B, C, T, H, W] in [-1, 1].

        Returns:
            Concatenated mean and logvar [B, 2*latent_channels, T', H', W'].
        """
        h = self.conv_in(x)

        for stage in self.stages:
            for block in stage["blocks"]:
                h = block(h)
            for ds in stage["downsample"]:
                h = ds(h)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)

        return h  # [B, 2*Z, T', H', W']
