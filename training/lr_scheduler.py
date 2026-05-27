"""Learning rate scheduler factory for the video diffusion training pipeline.

Provides warmup + decay schedules compatible with ``torch.optim.lr_scheduler.LambdaLR``.
All schedules start with a linear warmup phase and then transition to the
chosen decay strategy (cosine annealing, constant, or linear decay).

Example::

    from training.lr_scheduler import create_lr_scheduler

    scheduler = create_lr_scheduler(
        optimizer,
        scheduler_type="cosine",
        num_warmup_steps=1000,
        num_training_steps=100_000,
    )
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch.optim.lr_scheduler import LambdaLR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_warmup_factor(current_step: int, warmup_steps: int) -> float:
    """Return the linear warmup multiplier for *current_step*.

    During warmup the learning rate scales linearly from 0 to 1:

        factor = current_step / warmup_steps

    After warmup the factor is clamped to 1.0 so that the caller's decay
    schedule takes over without interference.

    Args:
        current_step: The global training step (0-indexed).
        warmup_steps: Total number of warmup steps.  When set to 0 no
            warmup is applied and the factor is always 1.0.

    Returns:
        A float in ``[0.0, 1.0]``.
    """
    if warmup_steps <= 0:
        return 1.0
    if current_step >= warmup_steps:
        return 1.0
    return float(current_step) / float(warmup_steps)


# ---------------------------------------------------------------------------
# Schedule builders (each returns a ``lr_lambda`` callable)
# ---------------------------------------------------------------------------

def _cosine_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float,
) -> Callable[[int], float]:
    """Linear warmup followed by cosine annealing to *min_lr_ratio*."""

    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return _get_warmup_factor(current_step, num_warmup_steps)

        # Cosine decay phase
        decay_steps = max(num_training_steps - num_warmup_steps, 1)
        progress = float(current_step - num_warmup_steps) / float(decay_steps)
        progress = min(progress, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return lr_lambda


def _constant_schedule(
    num_warmup_steps: int,
) -> Callable[[int], float]:
    """Linear warmup followed by a constant learning rate."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return _get_warmup_factor(current_step, num_warmup_steps)
        return 1.0

    return lr_lambda


def _linear_schedule(
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float,
) -> Callable[[int], float]:
    """Linear warmup followed by linear decay to *min_lr_ratio*."""

    def lr_lambda(current_step: int) -> float:
        # Warmup phase
        if current_step < num_warmup_steps:
            return _get_warmup_factor(current_step, num_warmup_steps)

        # Linear decay phase
        decay_steps = max(num_training_steps - num_warmup_steps, 1)
        progress = float(current_step - num_warmup_steps) / float(decay_steps)
        progress = min(progress, 1.0)
        return max(min_lr_ratio, 1.0 - progress * (1.0 - min_lr_ratio))

    return lr_lambda


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

_SCHEDULE_BUILDERS: dict[str, Callable[..., Callable[[int], float]]] = {
    "cosine": _cosine_schedule,
    "constant": _constant_schedule,
    "linear": _linear_schedule,
}


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    """Create a learning rate scheduler with linear warmup.

    Args:
        optimizer: The optimizer whose learning rate will be adjusted.
        scheduler_type: One of ``'cosine'``, ``'constant'``, or ``'linear'``.
        num_warmup_steps: Number of steps for the linear warmup phase.
        num_training_steps: Total number of training steps (used by cosine
            and linear schedules to compute decay).
        min_lr_ratio: Minimum learning rate expressed as a fraction of the
            initial learning rate.  Only used by cosine and linear schedules.
            Defaults to ``0.0`` (decay all the way to zero).

    Returns:
        A ``LambdaLR`` scheduler instance.

    Raises:
        ValueError: If *scheduler_type* is not one of the supported types.
    """
    scheduler_type = scheduler_type.lower()
    if scheduler_type not in _SCHEDULE_BUILDERS:
        supported = ", ".join(sorted(_SCHEDULE_BUILDERS))
        raise ValueError(
            f"Unknown scheduler type '{scheduler_type}'. "
            f"Supported types: {supported}"
        )

    if scheduler_type == "constant":
        lr_lambda = _constant_schedule(num_warmup_steps)
    else:
        lr_lambda = _SCHEDULE_BUILDERS[scheduler_type](
            num_warmup_steps, num_training_steps, min_lr_ratio,
        )

    return LambdaLR(optimizer, lr_lambda)
