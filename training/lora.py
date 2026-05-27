"""LoRA (Low-Rank Adaptation) for efficient fine-tuning.

LoRA injects trainable low-rank matrices into attention layers:
  W' = W + α/r * BA
  
Where B ∈ R^{d×r}, A ∈ R^{r×k}, r << min(d,k).
Only A, B are trained (typically <1% of total parameters).
"""

from __future__ import annotations
import torch, torch.nn as nn, logging
from typing import Optional

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Args:
        original: Original nn.Linear layer (frozen).
        rank: LoRA rank (lower = fewer params, higher = more expressive).
        alpha: LoRA scaling factor.
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.original(x)
        lora = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base + lora


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 1.0,
                target_modules: list[str] = ["to_q", "to_v"]) -> dict[str, LoRALinear]:
    """Inject LoRA into target attention layers of a model.

    Freezes all original parameters and only makes LoRA params trainable.

    Args:
        model: The model to inject LoRA into.
        rank: LoRA rank.
        alpha: LoRA alpha scaling.
        target_modules: Names of linear layers to target.

    Returns:
        Dictionary of injected LoRA modules.
    """
    model.requires_grad_(False)
    lora_modules = {}
    
    for name, module in model.named_modules():
        for target in target_modules:
            if hasattr(module, target):
                original = getattr(module, target)
                if isinstance(original, nn.Linear):
                    lora_layer = LoRALinear(original, rank=rank, alpha=alpha)
                    setattr(module, target, lora_layer)
                    lora_modules[f"{name}.{target}"] = lora_layer

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"LoRA injected: {len(lora_modules)} layers, {trainable:,}/{total:,} params trainable "
                f"({100*trainable/total:.2f}%)")
    return lora_modules


def save_lora_weights(model: nn.Module, path: str) -> None:
    """Save only LoRA weights."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lora_state = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state, path)
    logger.info(f"Saved {len(lora_state)} LoRA parameters to {path}")


def load_lora_weights(model: nn.Module, path: str) -> None:
    """Load LoRA weights into model."""
    lora_state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(lora_state, strict=False)
    logger.info(f"Loaded LoRA weights from {path}")
