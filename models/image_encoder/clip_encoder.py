"""CLIP-style Image Encoder for image-to-video conditioning.

Uses a local CLIP ViT model to extract visual features from reference images.
These features are fused with text embeddings via cross-attention to guide
video generation based on visual reference.

Supports:
  - Single reference image conditioning
  - Multiple reference images (averaged or concatenated)
  - Style transfer via global CLIP features
  - Structural guidance via patch-level features
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


class CLIPImageEncoder(nn.Module):
    """Local CLIP ViT image encoder for visual conditioning.

    Extracts both global (CLS token) and local (patch tokens) features
    from reference images for different conditioning strategies.

    Args:
        model_name: HuggingFace CLIP model ID.
        output_hidden_size: Target hidden size for projection.
        device: Device for the model.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch16",
        output_hidden_size: int = 512,
        device: str = "cuda",
    ):
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._output_hidden_size = output_hidden_size

        self._model = None
        self._processor = None
        self._loaded = False

        # Projection from CLIP hidden size to target
        self.projection = None

    def _load_model(self) -> None:
        """Load CLIP model locally."""
        if self._loaded:
            return

        from transformers import CLIPVisionModel, CLIPImageProcessor

        logger.info(f"Loading image encoder: {self.model_name}")
        self._processor = CLIPImageProcessor.from_pretrained(self.model_name)
        self._model = CLIPVisionModel.from_pretrained(self.model_name)
        self._model = self._model.to(self.device)
        self._model.eval()

        for param in self._model.parameters():
            param.requires_grad = False

        clip_hidden = self._model.config.hidden_size
        if clip_hidden != self._output_hidden_size:
            self.projection = nn.Linear(clip_hidden, self._output_hidden_size).to(self.device)

        self._loaded = True
        logger.info(f"Image encoder loaded: {self.model_name} (hidden={clip_hidden})")

    def encode_images(
        self,
        images: list[Image.Image] | torch.Tensor,
        return_patch_features: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Encode reference images into conditioning features.

        Args:
            images: List of PIL images or tensor [B, C, H, W].
            return_patch_features: If True, return per-patch features for cross-attention.

        Returns:
            Dictionary with:
              - 'global_embeds': [B, hidden_size] — CLS token features
              - 'patch_embeds': [B, num_patches, hidden_size] — patch-level features
        """
        self._load_model()

        # Process images
        if isinstance(images, list):
            inputs = self._processor(images=images, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
        else:
            pixel_values = images.to(self.device)

        with torch.no_grad():
            outputs = self._model(pixel_values=pixel_values, output_hidden_states=True)

        # Global features from CLS token
        global_embeds = outputs.pooler_output  # [B, hidden]

        result = {}

        if self.projection is not None:
            global_embeds = self.projection(global_embeds)

        result["global_embeds"] = global_embeds

        if return_patch_features:
            # Patch features from last hidden state (excluding CLS)
            patch_embeds = outputs.last_hidden_state[:, 1:, :]  # [B, num_patches, hidden]
            if self.projection is not None:
                patch_embeds = self.projection(patch_embeds)
            result["patch_embeds"] = patch_embeds

        return result

    def encode_for_conditioning(
        self,
        images: list[Image.Image],
        num_tokens: int = 16,
    ) -> torch.Tensor:
        """Encode images into a fixed number of conditioning tokens.

        Uses adaptive average pooling to reduce patch tokens to a fixed count,
        making it compatible with cross-attention regardless of image resolution.

        Args:
            images: List of PIL images.
            num_tokens: Number of output conditioning tokens.

        Returns:
            Conditioning tokens [B, num_tokens, hidden_size].
        """
        features = self.encode_images(images, return_patch_features=True)
        patch_embeds = features["patch_embeds"]  # [B, N, D]

        # Reduce to fixed token count
        # [B, N, D] -> [B, D, N] -> pool -> [B, D, num_tokens] -> [B, num_tokens, D]
        tokens = patch_embeds.permute(0, 2, 1)
        tokens = F.adaptive_avg_pool1d(tokens, num_tokens)
        tokens = tokens.permute(0, 2, 1)

        return tokens

    def forward(self, images: list[Image.Image]) -> torch.Tensor:
        """Simple forward: encode and return patch features.

        Args:
            images: List of PIL images.

        Returns:
            Patch embeddings [B, num_patches, hidden_size].
        """
        features = self.encode_images(images)
        return features["patch_embeds"]
