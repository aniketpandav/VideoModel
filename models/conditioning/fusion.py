"""Cross-attention conditioning fusion for text and image signals.

Fuses text embeddings, image embeddings, and optional structural guidance
into a unified conditioning signal for the diffusion model.

Supports:
  - Text-only conditioning (text-to-video)
  - Image-only conditioning (image-to-video)
  - Combined text + image conditioning
  - Classifier-free guidance with random dropout
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


class ConditioningFusion(nn.Module):
    """Fuses multiple conditioning signals into cross-attention context.

    During training with classifier-free guidance, randomly drops conditioning
    signals with probability `cfg_dropout_prob` to enable unconditional generation.

    Args:
        hidden_size: Model hidden dimension.
        text_dim: Text embedding dimension.
        image_dim: Image embedding dimension (0 to disable).
        cfg_dropout_prob: Probability of dropping conditions during training.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        text_dim: int = 512,
        image_dim: int = 512,
        cfg_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.cfg_dropout_prob = cfg_dropout_prob

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Image projection (if used)
        self.use_image = image_dim > 0
        if self.use_image:
            self.image_proj = nn.Sequential(
                nn.Linear(image_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )

        # Null embeddings for unconditional generation (CFG)
        self.null_text_embed = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.01)
        if self.use_image:
            self.null_image_embed = nn.Parameter(torch.randn(1, 1, hidden_size) * 0.01)

    def forward(
        self,
        text_embeds: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        force_unconditional: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Fuse conditioning signals.

        Args:
            text_embeds: Text embeddings [B, seq_len, text_dim].
            text_mask: Text attention mask [B, seq_len].
            image_embeds: Image embeddings [B, num_tokens, image_dim].
            force_unconditional: Force unconditional (for CFG inference).

        Returns:
            Dictionary with:
              - 'context': Fused context [B, total_tokens, hidden_size]
              - 'context_mask': Attention mask [B, total_tokens]
        """
        batch_size = (text_embeds.shape[0] if text_embeds is not None
                      else image_embeds.shape[0])
        device = (text_embeds.device if text_embeds is not None
                  else image_embeds.device)

        context_parts = []
        mask_parts = []

        # CFG dropout during training
        drop_text = False
        drop_image = False
        if self.training and not force_unconditional:
            if torch.rand(1).item() < self.cfg_dropout_prob:
                drop_text = True
                drop_image = True
        if force_unconditional:
            drop_text = True
            drop_image = True

        # Process text
        if text_embeds is not None:
            if drop_text:
                # Use null embeddings
                null = self.null_text_embed.expand(batch_size, text_embeds.shape[1], -1)
                context_parts.append(null)
                mask_parts.append(torch.ones(batch_size, text_embeds.shape[1], device=device))
            else:
                projected = self.text_proj(text_embeds)
                context_parts.append(projected)
                if text_mask is not None:
                    mask_parts.append(text_mask.float())
                else:
                    mask_parts.append(torch.ones(batch_size, text_embeds.shape[1], device=device))

        # Process image
        if image_embeds is not None and self.use_image:
            if drop_image:
                null = self.null_image_embed.expand(batch_size, image_embeds.shape[1], -1)
                context_parts.append(null)
            else:
                projected = self.image_proj(image_embeds)
                context_parts.append(projected)
            mask_parts.append(torch.ones(batch_size, image_embeds.shape[1], device=device))

        # Concatenate all conditioning tokens
        context = torch.cat(context_parts, dim=1)  # [B, total_tokens, D]
        context_mask = torch.cat(mask_parts, dim=1)  # [B, total_tokens]

        return {
            "context": context,
            "context_mask": context_mask,
        }
