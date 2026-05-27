"""Exponential Moving Average (EMA) for model weights.

EMA smooths training by maintaining a running average of model parameters:
  θ_ema = decay * θ_ema + (1 - decay) * θ_model

This reduces noise in the final weights and typically produces better samples.
"""

from __future__ import annotations
import copy, torch, torch.nn as nn


class EMA:
    """Exponential Moving Average wrapper for any nn.Module.

    Args:
        model: The model to track.
        decay: EMA decay rate (e.g., 0.9999).
        update_every: Update EMA every N steps.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, update_every: int = 10):
        self.decay = decay
        self.update_every = update_every
        self.step_count = 0
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        self.ema_model.requires_grad_(False)
        self._device_synced = False

    def _sync_device(self, model: nn.Module) -> None:
        """Move EMA model to the same device as the training model."""
        if not self._device_synced:
            device = next(model.parameters()).device
            self.ema_model.to(device)
            self._device_synced = True

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Update EMA weights from the training model."""
        self.step_count += 1
        if self.step_count % self.update_every != 0:
            return
        # Ensure EMA model is on the same device as the training model
        self._sync_device(model)
        for ema_p, model_p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1 - self.decay)
        for ema_b, model_b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.data.copy_(model_b.data)

    def to(self, device: torch.device | str) -> 'EMA':
        """Move the EMA model to the specified device."""
        self.ema_model.to(device)
        self._device_synced = True
        return self

    def state_dict(self):
        return self.ema_model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        self.ema_model.load_state_dict(state_dict, strict=strict)
