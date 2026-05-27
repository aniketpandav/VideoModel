"""Main Diffusion Transformer (DiT) model for video generation.

Full architecture:
  Input latent [B, Z, T, H, W]
  -> PatchEmbed3D -> [B, N, D] tokens
  -> + timestep embedding
  -> N x DiTBlock (self-attn + cross-attn + FFN)
  -> Final LayerNorm + Linear projection
  -> Unpatchify -> [B, Z, T, H, W] noise prediction
"""

from __future__ import annotations
import torch, torch.nn as nn
from typing import Optional
from models.dit.embeddings import SinusoidalTimestepEmbedding, PatchEmbed3D
from models.dit.blocks import DiTBlock
from models.dit.normalization import AdaLayerNorm


class VideoDiT(nn.Module):
    """Diffusion Transformer for latent video generation.

    Predicts noise (or v-prediction) from noisy video latents, conditioned
    on timestep, text embeddings, and optional image embeddings.

    Args:
        in_channels: Latent input channels (from VAE).
        hidden_size: Transformer hidden dimension.
        num_layers: Number of DiT blocks.
        num_heads: Number of attention heads.
        mlp_ratio: MLP expansion ratio.
        patch_size: 3D patch size (T, H, W).
        dropout: Dropout probability.
        attention_mode: 'joint' or 'decomposed' spatial-temporal attention.
        cross_attention_dim: Dimension of conditioning context.
        use_flash: Use efficient attention backends.
        gradient_checkpointing: Enable gradient checkpointing to save memory.
    """

    def __init__(
        self, in_channels: int = 4, hidden_size: int = 512, num_layers: int = 12,
        num_heads: int = 8, mlp_ratio: float = 4.0, patch_size: tuple[int, int, int] = (1, 2, 2),
        dropout: float = 0.0, attention_mode: str = "joint", cross_attention_dim: int = 512,
        use_flash: bool = True, gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.in_channels = in_channels

        # Patch embedding
        self.patch_embed = PatchEmbed3D(in_channels, hidden_size, patch_size)

        # Timestep embedding
        self.time_embed = SinusoidalTimestepEmbedding(hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(
                hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
                dropout=dropout, attention_mode=attention_mode,
                cross_attention_dim=cross_attention_dim, use_flash=use_flash,
            )
            for _ in range(num_layers)
        ])

        # Final output projection
        self.final_norm = nn.LayerNorm(hidden_size)
        self.final_proj = nn.Linear(hidden_size, in_channels * patch_size[0] * patch_size[1] * patch_size[2])

        self.patch_size = patch_size
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        # Zero-initialize output projection for residual-friendly start
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(
        self, x: torch.Tensor, timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass: predict noise from noisy latent.

        Args:
            x: Noisy video latent [B, C, T, H, W].
            timesteps: Diffusion timestep [B].
            context: Conditioning context [B, seq_len, D] from text/image encoder.
            context_mask: Attention mask [B, seq_len].

        Returns:
            Predicted noise [B, C, T, H, W] (same shape as input).
        """
        B, C, T, H, W = x.shape

        # Patchify: [B, C, T, H, W] -> [B, N, D]
        tokens, grid_size = self.patch_embed(x)

        # Add timestep embedding
        t_emb = self.time_embed(timesteps)  # [B, D]

        # Process through transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                tokens = torch.utils.checkpoint.checkpoint(
                    block, tokens, t_emb, context, context_mask, grid_size,
                    use_reentrant=False,
                )
            else:
                tokens = block(tokens, t_emb, context, context_mask, grid_size)

        # Final projection
        tokens = self.final_norm(tokens)  # [B, N, D]
        tokens = self.final_proj(tokens)  # [B, N, C*pt*ph*pw]

        # Unpatchify: [B, N, C*p] -> [B, C, T, H, W]
        T_g, H_g, W_g = grid_size
        pt, ph, pw = self.patch_size
        tokens = tokens.reshape(B, T_g, H_g, W_g, C, pt, ph, pw)
        output = tokens.permute(0, 4, 1, 5, 2, 6, 3, 7)  # [B, C, T_g, pt, H_g, ph, W_g, pw]
        output = output.reshape(B, C, T_g * pt, H_g * ph, W_g * pw)

        return output

    def get_param_count(self) -> dict[str, int]:
        """Get parameter counts by component."""
        counts = {
            "patch_embed": sum(p.numel() for p in self.patch_embed.parameters()),
            "time_embed": sum(p.numel() for p in self.time_embed.parameters()),
            "blocks": sum(p.numel() for p in self.blocks.parameters()),
            "final": sum(p.numel() for p in self.final_norm.parameters())
                    + sum(p.numel() for p in self.final_proj.parameters()),
        }
        counts["total"] = sum(counts.values())
        return counts
