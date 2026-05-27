"""3D Causal VAE Decoder for video reconstruction.

Reconstructs video [B, C, T, H, W] from latent [B, latent_ch, T', H', W']
using 3D causal transposed convolutions with spatial 8x and temporal 4x upsampling.

Architecture:
  Input (Z, T', H', W') -> Conv3D -> Attention -> [ResBlock + Upsample] x N -> Conv3D -> (3, T, H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from models.vae.encoder import CausalConv3d, ResBlock3D, AttentionBlock3D


class SpatialUpsample(nn.Module):
    """Upsample spatial dimensions by 2x using interpolation + convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, channels,
            kernel_size=(1, 3, 3),
            padding=(0, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upsample spatial dims only
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        _, _, H2, W2 = x.shape
        x = x.reshape(B, T, C, H2, W2).permute(0, 2, 1, 3, 4)
        return self.conv(x)


class TemporalUpsample(nn.Module):
    """Upsample temporal dimension by 2x using interpolation + causal conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = CausalConv3d(channels, channels, kernel_size=(3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T, H, W = x.shape
        # Reshape for temporal interpolation
        x = x.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        T2 = x.shape[2]
        x = x.reshape(B, H, W, C, T2).permute(0, 3, 4, 1, 2)
        return self.conv(x)


class Decoder3D(nn.Module):
    """3D Video Decoder with causal convolutions.

    Reconstructs video from latent space:
      [B, latent_channels, T', H', W'] -> [B, 3, T, H, W]

    Mirror architecture of the Encoder3D with upsampling.

    Args:
        out_channels: Output channels (3 for RGB).
        latent_channels: Latent space channels.
        base_channels: Base channel count.
        channel_multipliers: Channel multiplier at each decoder stage (reversed from encoder).
        num_res_blocks: Number of residual blocks per stage.
    """

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
    ):
        super().__init__()

        self.out_channels = out_channels

        # Reverse multipliers for decoder (go from deepest to shallowest)
        channels = [base_channels * m for m in channel_multipliers]
        reversed_channels = list(reversed(channels))

        # Input projection from latent space
        self.conv_in = CausalConv3d(latent_channels, reversed_channels[0], kernel_size=3)

        # Middle block with attention
        self.mid_block1 = ResBlock3D(reversed_channels[0])
        self.mid_attn = AttentionBlock3D(reversed_channels[0])
        self.mid_block2 = ResBlock3D(reversed_channels[0])

        # Decoder stages (reversed from encoder)
        self.stages = nn.ModuleList()
        num_stages = len(reversed_channels)

        for i in range(num_stages):
            in_ch = reversed_channels[i]
            out_ch = reversed_channels[i + 1] if i < num_stages - 1 else reversed_channels[-1]

            stage_blocks = nn.ModuleList()

            # Residual blocks
            for j in range(num_res_blocks + 1):  # +1 for decoder
                block_in = in_ch if j == 0 else out_ch
                stage_blocks.append(ResBlock3D(block_in, out_ch))

            # Upsampling
            upsample = nn.ModuleList()
            if i < num_stages - 1:  # No upsample at last stage
                upsample.append(SpatialUpsample(out_ch))
                # Temporal upsample in last stages of decoder (mirrors first stages of encoder)
                stage_idx_from_end = num_stages - 2 - i
                if stage_idx_from_end < 2:
                    upsample.append(TemporalUpsample(out_ch))

            self.stages.append(nn.ModuleDict({
                "blocks": stage_blocks,
                "upsample": upsample,
            }))

        # Output projection
        final_ch = reversed_channels[-1]
        self.norm_out = nn.GroupNorm(32, final_ch)
        self.conv_out = CausalConv3d(final_ch, out_channels, kernel_size=3)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to video.

        Args:
            z: Latent tensor [B, latent_channels, T', H', W'].

        Returns:
            Reconstructed video [B, 3, T, H, W] in [-1, 1].
        """
        h = self.conv_in(z)

        h = self.mid_block1(h)
        h = self.mid_attn(h)
        h = self.mid_block2(h)

        for stage in self.stages:
            for block in stage["blocks"]:
                h = block(h)
            for us in stage["upsample"]:
                h = us(h)

        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)

        return torch.tanh(h)  # [-1, 1]
