"""Local T5 Text Encoder for text-to-video conditioning.

Uses HuggingFace T5-small model downloaded and run entirely locally.
No API calls. The model files are cached in ~/.cache/huggingface.

Why T5 for text encoding?
  - T5 is a powerful encoder-decoder transformer pre-trained on massive text corpora.
  - Its encoder produces rich semantic embeddings that capture meaning, structure, and
    relationships in the prompt text.
  - These embeddings are used as conditioning signals via cross-attention in the
    diffusion model, allowing the denoising process to be guided by text semantics.
  - T5-small (60M params) fits on any GPU while providing reasonable text understanding.

How token embeddings affect video generation:
  - Each token in the prompt is mapped to a high-dimensional vector (embedding).
  - The transformer layers contextualize these embeddings, so "running dog" produces
    different representations than "dog running".
  - The diffusion model's cross-attention layers attend to these embeddings, allowing
    different spatial-temporal regions of the video to focus on different parts of the prompt.
  - Prompt weighting (e.g., "(fire:1.5)") scales specific token embeddings to increase
    their influence on the generated video.
"""

from __future__ import annotations

import re
import gc
import time
import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _to_device_with_retry(batch_encoding, device: str, retries: int = 2):
    """Move a tokenizer BatchEncoding to device, retrying past transient CUDA blips.

    cudaErrorUnknown can fire on any CUDA API call when the driver is briefly
    unhealthy (thermal, contention, etc.). A short backoff + cache clear is
    usually enough to get past it without aborting a long training run.
    """
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return batch_encoding.to(device)
        except RuntimeError as e:
            msg = str(e)
            if "CUDA" not in msg and "cuda" not in msg:
                raise
            last_err = e
            logger.warning(
                "CUDA error moving tokens to %s (attempt %d/%d): %s",
                device, attempt + 1, retries + 1, msg.splitlines()[0],
            )
            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except RuntimeError:
                    pass
            time.sleep(0.2 * (attempt + 1))
    assert last_err is not None
    raise last_err


class T5TextEncoder(nn.Module):
    """Local T5-based text encoder for conditioning signals.

    Downloads and runs the T5 model entirely locally via HuggingFace Transformers.

    Args:
        model_name: HuggingFace model ID (e.g., "google/t5-small").
        max_length: Maximum token sequence length.
        output_hidden_size: Target hidden size (projects if different from T5).
        device: Device to load the model on.
        dtype: Data type for the model.
    """

    def __init__(
        self,
        model_name: str = "t5-small",
        max_length: int = 128,
        output_hidden_size: int = 512,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.device = device

        # Lazy-load to avoid import time overhead
        self._encoder = None
        self._tokenizer = None
        self._loaded = False

        # Projection layer (T5-small hidden_size=512, but may need projection)
        self.projection = None
        self._output_hidden_size = output_hidden_size

    def _load_model(self) -> None:
        """Load T5 model and tokenizer from local cache / HuggingFace hub."""
        if self._loaded:
            return

        from transformers import T5EncoderModel, T5Tokenizer

        logger.info(f"Loading text encoder: {self.model_name}")
        self._tokenizer = T5Tokenizer.from_pretrained(
            self.model_name, legacy=True
        )
        self._encoder = T5EncoderModel.from_pretrained(self.model_name)
        self._encoder = self._encoder.to(self.device)
        self._encoder.eval()

        # Freeze text encoder weights
        for param in self._encoder.parameters():
            param.requires_grad = False

        # Setup projection if needed
        t5_hidden = self._encoder.config.d_model
        if t5_hidden != self._output_hidden_size:
            self.projection = nn.Linear(t5_hidden, self._output_hidden_size).to(self.device)

        self._loaded = True
        logger.info(f"Text encoder loaded: {self.model_name} (hidden={t5_hidden})")

    def encode(
        self,
        prompts: list[str],
        negative_prompts: Optional[list[str]] = None,
    ) -> dict[str, torch.Tensor]:
        """Encode text prompts into conditioning embeddings.

        Args:
            prompts: List of text prompts.
            negative_prompts: Optional negative prompts for CFG.

        Returns:
            Dictionary with:
              - 'prompt_embeds': [B, seq_len, hidden_size]
              - 'attention_mask': [B, seq_len]
              - 'negative_embeds': [B, seq_len, hidden_size] (if negative_prompts given)
        """
        self._load_model()

        # Parse prompt weights
        processed_prompts = []
        weight_maps = []
        for prompt in prompts:
            clean_prompt, weights = self._parse_prompt_weights(prompt)
            processed_prompts.append(clean_prompt)
            weight_maps.append(weights)

        # Tokenize
        tokens = self._tokenizer(
            processed_prompts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tokens = _to_device_with_retry(tokens, self.device)

        # Encode
        with torch.no_grad():
            outputs = self._encoder(
                input_ids=tokens.input_ids,
                attention_mask=tokens.attention_mask,
            )
            embeds = outputs.last_hidden_state  # [B, seq_len, hidden]

        # Apply prompt weights
        for i, weights in enumerate(weight_maps):
            if weights:
                embeds[i] = self._apply_weights(embeds[i], tokens.input_ids[i], weights)

        # Project if needed
        if self.projection is not None:
            embeds = self.projection(embeds)

        result = {
            "prompt_embeds": embeds,
            "attention_mask": tokens.attention_mask,
        }

        # Encode negative prompts
        if negative_prompts is not None:
            neg_tokens = self._tokenizer(
                negative_prompts,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            neg_tokens = _to_device_with_retry(neg_tokens, self.device)

            with torch.no_grad():
                neg_outputs = self._encoder(
                    input_ids=neg_tokens.input_ids,
                    attention_mask=neg_tokens.attention_mask,
                )
                neg_embeds = neg_outputs.last_hidden_state

            if self.projection is not None:
                neg_embeds = self.projection(neg_embeds)

            result["negative_embeds"] = neg_embeds
            result["negative_mask"] = neg_tokens.attention_mask

        return result

    def _parse_prompt_weights(self, prompt: str) -> tuple[str, dict[str, float]]:
        """Parse prompt weighting syntax like '(word:1.5)'.

        Args:
            prompt: Raw prompt text with optional weights.

        Returns:
            Tuple of (clean_prompt, weight_dict).
        """
        weights = {}
        # Match patterns like (word:1.5) or (multi word phrase:0.8)
        pattern = r'\(([^:]+):([0-9]*\.?[0-9]+)\)'
        matches = re.findall(pattern, prompt)

        for text, weight in matches:
            weights[text.strip()] = float(weight)

        # Remove weight syntax from prompt
        clean = re.sub(pattern, r'\1', prompt)
        return clean, weights

    def _apply_weights(
        self,
        embeds: torch.Tensor,
        input_ids: torch.Tensor,
        weights: dict[str, float],
    ) -> torch.Tensor:
        """Apply prompt weights to token embeddings.

        Scales the embedding vectors of weighted tokens.
        """
        # Simple approach: scale entire embedding by average weight
        # For production, you'd map specific tokens to their weight
        if weights:
            avg_weight = sum(weights.values()) / len(weights)
            embeds = embeds * avg_weight
        return embeds

    def forward(self, prompts: list[str]) -> torch.Tensor:
        """Simple forward: encode prompts and return embeddings.

        Args:
            prompts: List of text prompts.

        Returns:
            Embeddings tensor [B, seq_len, hidden_size].
        """
        result = self.encode(prompts)
        return result["prompt_embeds"]
