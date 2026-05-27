"""Memory optimization utilities for GPU training and inference."""

from __future__ import annotations

import gc
import logging
from typing import Optional
from contextlib import contextmanager

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_gpu_memory_info() -> dict[str, float]:
    """Get current GPU memory usage in GB.

    Returns:
        Dictionary with allocated, reserved, and free memory.
    """
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0, "total": 0}

    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    total = torch.cuda.get_device_properties(0).total_mem / 1024**3
    free = total - allocated

    return {
        "allocated": round(allocated, 2),
        "reserved": round(reserved, 2),
        "free": round(free, 2),
        "total": round(total, 2),
    }


def log_gpu_memory(prefix: str = "") -> None:
    """Log current GPU memory usage."""
    info = get_gpu_memory_info()
    logger.info(
        f"{prefix}GPU Memory: {info['allocated']:.2f}GB allocated, "
        f"{info['free']:.2f}GB free / {info['total']:.2f}GB total"
    )


def estimate_model_memory(model: nn.Module, precision: str = "fp32") -> float:
    """Estimate model memory usage in GB.

    Args:
        model: PyTorch model.
        precision: One of 'fp32', 'fp16', 'bf16'.

    Returns:
        Estimated memory in GB.
    """
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total = param_bytes + buffer_bytes

    if precision in ("fp16", "bf16"):
        total = total // 2

    return total / 1024**3


def enable_gradient_checkpointing(model: nn.Module) -> None:
    """Enable gradient checkpointing on supported model modules.

    This trades compute for memory by recomputing intermediate
    activations during backward pass instead of storing them.
    """
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        logger.info("Enabled gradient checkpointing via model method")
    elif hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Enabled gradient checkpointing via HF method")
    else:
        # Manual: wrap forward methods with checkpoint
        logger.warning("Model does not support gradient checkpointing natively")


def clear_gpu_cache() -> None:
    """Aggressively clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@contextmanager
def gpu_memory_tracker(label: str = ""):
    """Context manager to track GPU memory usage of a code block."""
    if not torch.cuda.is_available():
        yield
        return

    torch.cuda.synchronize()
    start_mem = torch.cuda.memory_allocated()
    yield
    torch.cuda.synchronize()
    end_mem = torch.cuda.memory_allocated()
    delta = (end_mem - start_mem) / 1024**3
    logger.info(f"[{label}] GPU memory delta: {delta:+.3f} GB")


def setup_memory_efficient_attention() -> str:
    """Detect and configure the best attention backend.

    Returns:
        Name of the backend: 'xformers', 'flash_attention', or 'pytorch'.
    """
    # Try xFormers first
    try:
        import xformers.ops  # noqa: F401
        logger.info("Using xFormers memory-efficient attention")
        return "xformers"
    except ImportError:
        pass

    # Try PyTorch native Flash Attention (2.0+)
    if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        logger.info("Using PyTorch native scaled_dot_product_attention")
        return "flash_attention"

    logger.info("Using standard PyTorch attention (no optimization)")
    return "pytorch"


class MemoryEfficientSequential(nn.Sequential):
    """Sequential module with optional gradient checkpointing."""

    def __init__(self, *args, use_checkpointing: bool = False):
        super().__init__(*args)
        self.use_checkpointing = use_checkpointing

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for module in self:
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    module, x, use_reentrant=False, **kwargs
                )
            else:
                x = module(x, **kwargs)
        return x
