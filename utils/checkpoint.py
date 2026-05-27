"""Checkpoint save/load utilities with EMA support."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    epoch: int,
    config: Any,
    save_dir: str | Path,
    ema_model: Optional[nn.Module] = None,
    scheduler: Optional[Any] = None,
    max_checkpoints: int = 5,
    prefix: str = "checkpoint",
    extra_modules: Optional[dict[str, nn.Module]] = None,
) -> Path:
    """Save a training checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer state.
        step: Current global step.
        epoch: Current epoch.
        config: Training config (saved for reproducibility).
        save_dir: Directory to save checkpoints.
        ema_model: Optional EMA model weights.
        scheduler: Optional LR scheduler state.
        max_checkpoints: Maximum number of checkpoints to keep.
        prefix: Filename prefix.

    Returns:
        Path to the saved checkpoint.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
    }

    if ema_model is not None:
        checkpoint["ema_state_dict"] = ema_model.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if extra_modules:
        checkpoint["extra_state_dicts"] = {
            name: module.state_dict()
            for name, module in extra_modules.items()
        }

    # Save checkpoint
    ckpt_path = save_dir / f"{prefix}_step{step:08d}.pt"
    torch.save(checkpoint, ckpt_path)
    logger.info(f"Saved checkpoint to {ckpt_path}")

    # Save a 'latest' symlink/copy
    latest_path = save_dir / f"{prefix}_latest.pt"
    if latest_path.exists():
        latest_path.unlink()
    # Prefer symlink (no extra disk space), fall back to copy on Windows
    try:
        latest_path.symlink_to(ckpt_path.name)
    except (OSError, NotImplementedError):
        shutil.copy2(ckpt_path, latest_path)

    # Cleanup old checkpoints
    _cleanup_checkpoints(save_dir, prefix, max_checkpoints)

    return ckpt_path


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[nn.Module] = None,
    scheduler: Optional[Any] = None,
    map_location: str = "cpu",
    strict: bool = True,
    extra_modules: Optional[dict[str, nn.Module]] = None,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load weights into.
        optimizer: Optional optimizer to restore state.
        ema_model: Optional EMA model to restore state.
        scheduler: Optional LR scheduler to restore state.
        map_location: Device to map tensors to.
        strict: Whether to strictly enforce state_dict key matching.

    Returns:
        Dictionary with 'step' and 'epoch' from the checkpoint.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Load model weights
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Optionally restore optimizer
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Optionally restore EMA
    if ema_model is not None and "ema_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_state_dict"], strict=strict)

    # Optionally restore scheduler
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    if extra_modules:
        extra_state = checkpoint.get("extra_state_dicts", {})
        for name, module in extra_modules.items():
            if name in extra_state:
                module.load_state_dict(extra_state[name], strict=strict)
            else:
                logger.warning("Checkpoint has no extra module state for %s", name)

    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
    }


def find_latest_checkpoint(save_dir: str | Path, prefix: str = "checkpoint") -> Optional[Path]:
    """Find the latest checkpoint in a directory.

    Searches first for the specified prefix, then falls back to any
    checkpoint prefix found in the directory.

    Args:
        save_dir: Directory containing checkpoints.
        prefix: Checkpoint filename prefix.

    Returns:
        Path to the latest checkpoint, or None.
    """
    save_dir = Path(save_dir)
    if not save_dir.exists():
        return None

    # Try exact prefix first
    latest = save_dir / f"{prefix}_latest.pt"
    if latest.exists():
        return latest

    # Fallback: find by step number with exact prefix
    checkpoints = sorted(save_dir.glob(f"{prefix}_step*.pt"))
    if checkpoints:
        return checkpoints[-1]

    # Fallback: search for any *_latest.pt or *_step*.pt
    any_latest = sorted(save_dir.glob("*_latest.pt"))
    if any_latest:
        return any_latest[-1]

    any_step = sorted(save_dir.glob("*_step*.pt"))
    if any_step:
        return any_step[-1]

    return None


def _cleanup_checkpoints(save_dir: Path, prefix: str, max_keep: int) -> None:
    """Remove old checkpoints, keeping only the most recent ones."""
    checkpoints = sorted(save_dir.glob(f"{prefix}_step*.pt"))
    if len(checkpoints) > max_keep:
        for ckpt in checkpoints[:-max_keep]:
            ckpt.unlink()
            logger.info(f"Removed old checkpoint: {ckpt}")
