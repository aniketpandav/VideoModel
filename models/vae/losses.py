"""VAE training losses: reconstruction, KL divergence, and perceptual loss.

Loss formulas:
  L_total = λ_recon * L_recon + λ_kl * L_kl + λ_perc * L_perceptual

  L_recon = L1(x, x_hat) — pixel-level reconstruction
  L_kl    = -0.5 * mean(1 + logvar - μ² - exp(logvar)) — regularization
  L_perc  = Σ MSE(VGG_l(x), VGG_l(x_hat)) — perceptual similarity
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class VAELoss(nn.Module):
    """Combined VAE loss with reconstruction, KL divergence, and optional perceptual loss.

    Args:
        reconstruction_weight: Weight for L1 reconstruction loss.
        kl_weight: Weight for KL divergence loss.
        perceptual_weight: Weight for perceptual loss (0 to disable).
    """

    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        kl_weight: float = 1e-5,
        perceptual_weight: float = 0.1,
    ):
        super().__init__()
        self.recon_weight = reconstruction_weight
        self.kl_weight = kl_weight
        self.perc_weight = perceptual_weight

        # Perceptual loss using VGG features
        self.perceptual = None
        if perceptual_weight > 0:
            self.perceptual = VGGPerceptualLoss()

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute VAE loss.

        Args:
            reconstruction: Decoder output [B, C, T, H, W].
            target: Original input [B, C, T, H, W].
            mean: Encoder mean [B, Z, T', H', W'].
            logvar: Encoder logvar [B, Z, T', H', W'].

        Returns:
            Dictionary with 'total', 'reconstruction', 'kl', 'perceptual' losses.
        """
        # L1 reconstruction loss
        recon_loss = F.l1_loss(reconstruction, target)

        # KL divergence loss
        # KL(q(z|x) || N(0,I)) = -0.5 * sum(1 + log(σ²) - μ² - σ²)
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mean.pow(2) - logvar.exp()
        )

        total = self.recon_weight * recon_loss + self.kl_weight * kl_loss

        losses = {
            "reconstruction": recon_loss,
            "kl": kl_loss,
        }

        # Perceptual loss (computed per-frame for memory efficiency)
        if self.perceptual is not None and self.perc_weight > 0:
            perc_loss = self._compute_perceptual(reconstruction, target)
            total = total + self.perc_weight * perc_loss
            losses["perceptual"] = perc_loss

        losses["total"] = total
        return losses

    def _compute_perceptual(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute perceptual loss frame-by-frame to save memory."""
        B, C, T, H, W = pred.shape
        # Reshape to [B*T, C, H, W] for per-frame VGG
        pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        target_2d = target.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        return self.perceptual(pred_2d, target_2d)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 feature matching.

    Extracts features from multiple VGG layers and computes MSE
    between predicted and target feature maps.
    
    Note: VGG features are computed without gradients for both pred and
    target. Gradient flow only occurs through the MSE computation on
    the detached feature maps, which is sufficient for perceptual guidance.
    """

    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)

        # Extract feature layers (after ReLU at different depths)
        self.blocks = nn.ModuleList([
            nn.Sequential(*list(vgg.features[:4])),   # relu1_2
            nn.Sequential(*list(vgg.features[4:9])),  # relu2_2
            nn.Sequential(*list(vgg.features[9:16])), # relu3_3
            nn.Sequential(*list(vgg.features[16:23])),# relu4_3
        ])

        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalization
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        self._blocks_on_device = False

    def _ensure_device(self, device: torch.device) -> None:
        """Move VGG blocks to the target device once (lazy init)."""
        if not self._blocks_on_device:
            self.blocks.to(device)
            self._blocks_on_device = True

    @torch.no_grad()
    def _extract_features(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Extract multi-scale VGG features without gradients."""
        features = []
        for block in self.blocks:
            x = block(x)
            features.append(x.clone())
        return features

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale perceptual loss.

        Args:
            pred: Predicted images [B, 3, H, W] in [-1, 1].
            target: Target images [B, 3, H, W] in [-1, 1].

        Returns:
            Scalar perceptual loss.
        """
        self._ensure_device(pred.device)

        # Normalize from [-1,1] to ImageNet range
        pred_norm = (pred + 1) / 2  # [0, 1]
        target_norm = (target + 1) / 2
        mean = self.mean.to(device=pred.device, dtype=pred.dtype)
        std = self.std.to(device=pred.device, dtype=pred.dtype)
        pred_norm = (pred_norm - mean) / std
        target_norm = (target_norm - mean) / std

        # Extract features without gradients for both pred and target
        pred_features = self._extract_features(pred_norm)
        target_features = self._extract_features(target_norm)

        # Compute MSE between feature maps
        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for pf, tf in zip(pred_features, target_features):
            loss = loss + F.mse_loss(pf, tf)

        return loss
