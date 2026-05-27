"""Full 3D Video VAE combining Encoder + Decoder with reparameterization.

The VAE compresses video from pixel space to a lower-dimensional latent space,
which dramatically reduces the computational cost of the diffusion model.

Mathematical foundation:
  ELBO = E_q[log p(x|z)] - KL(q(z|x) || p(z))
  
  Where:
    - q(z|x) = N(μ_encoder(x), σ_encoder(x)) is the approximate posterior
    - p(z) = N(0, I) is the prior
    - p(x|z) is the decoder reconstruction
    
  The KL divergence has a closed-form solution:
    KL = -0.5 * Σ(1 + log(σ²) - μ² - σ²)

  The reconstruction term is approximated by L1/L2 loss + perceptual loss.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, NamedTuple

from models.vae.encoder import Encoder3D
from models.vae.decoder import Decoder3D

logger = logging.getLogger(__name__)


class VAEOutput(NamedTuple):
    """Output of the VAE forward pass."""
    reconstruction: torch.Tensor  # [B, C, T, H, W]
    mean: torch.Tensor            # [B, Z, T', H', W']
    logvar: torch.Tensor          # [B, Z, T', H', W']
    latent: torch.Tensor          # [B, Z, T', H', W'] (sampled)


class VideoVAE(nn.Module):
    """3D Video Variational Autoencoder.

    Compresses video from pixel space [B, 3, T, H, W] to latent [B, Z, T/4, H/8, W/8]
    and back, using 3D causal convolutions for temporal consistency.

    Args:
        in_channels: Input image channels (3 for RGB).
        latent_channels: Latent space dimension.
        base_channels: Base feature channels.
        channel_multipliers: Multipliers per encoder stage.
        num_res_blocks: Residual blocks per stage.
        scaling_factor: Scaling factor for latent values (stabilizes diffusion training).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_multipliers: list[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        scaling_factor: float = 0.18215,
    ):
        super().__init__()

        self.latent_channels = latent_channels
        self.scaling_factor = scaling_factor
        self._gradient_checkpointing = False
        self._temporal_chunk_size = 0  # 0 = no chunking

        self.encoder = Encoder3D(
            in_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )

        self.quant_conv = nn.Identity()  # placeholder for future quantization

        self.decoder = Decoder3D(
            out_channels=in_channels,
            latent_channels=latent_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers,
            num_res_blocks=num_res_blocks,
        )

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing to trade compute for memory."""
        self._gradient_checkpointing = True
        logger.info("VAE gradient checkpointing enabled")

    def set_temporal_chunk_size(self, chunk_size: int) -> None:
        """Set temporal chunk size for chunked encode/decode (0 = disabled)."""
        self._temporal_chunk_size = chunk_size
        if chunk_size > 0:
            logger.info(f"VAE temporal chunking enabled: chunk_size={chunk_size}")

    def encode(self, x: torch.Tensor, sample: bool = True) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode video to latent space.

        Args:
            x: Input video [B, C, T, H, W] in [-1, 1].
            sample: If True, sample from the distribution. Otherwise, return mean.

        Returns:
            Tuple of (latent, mean, logvar).
        """
        if self._gradient_checkpointing and self.training:
            h = grad_checkpoint(self.encoder, x, use_reentrant=False)
        else:
            h = self.encoder(x)  # [B, 2*Z, T', H', W']
        mean, logvar = torch.chunk(h, 2, dim=1)

        # Clamp logvar for stability
        logvar = torch.clamp(logvar, -30.0, 20.0)

        if sample:
            # Reparameterization trick: z = μ + σ * ε
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + std * eps
        else:
            z = mean

        # Scale latent for diffusion model
        z = z * self.scaling_factor

        return z, mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent back to video.

        Args:
            z: Latent tensor [B, Z, T', H', W'].

        Returns:
            Reconstructed video [B, C, T, H, W] in [-1, 1].
        """
        # Unscale
        z = z / self.scaling_factor

        if self._gradient_checkpointing and self.training:
            return grad_checkpoint(self.decoder, z, use_reentrant=False)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> VAEOutput:
        """Full VAE forward: encode -> sample -> decode.

        Args:
            x: Input video [B, C, T, H, W] in [-1, 1].

        Returns:
            VAEOutput with reconstruction, mean, logvar, and sampled latent.
        """
        z, mean, logvar = self.encode(x, sample=True)
        recon = self.decode(z)

        return VAEOutput(
            reconstruction=recon,
            mean=mean,
            logvar=logvar,
            latent=z,
        )

    def get_latent_shape(self, video_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Calculate the latent tensor shape for a given video input shape.

        Args:
            video_shape: (T, H, W) or (B, C, T, H, W).

        Returns:
            Tuple of latent dimensions.
        """
        if len(video_shape) == 3:
            T, H, W = video_shape
        elif len(video_shape) == 5:
            _, _, T, H, W = video_shape
        else:
            raise ValueError(f"Expected 3 or 5 dims, got {len(video_shape)}")

        return (self.latent_channels, T // 4, H // 8, W // 8)
