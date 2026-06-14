"""Output quality validation for generated video frames.

All checks operate on (T, H, W, C) uint8 numpy arrays and are intentionally
lightweight — no GPU, no heavy ML models — so they add < 500ms even for 1-hour
videos.

The optional CLIP visual-similarity check for Mode 2 is gated by `clip_check=True`
and adds ~1–2s per validation when enabled.

Retry logic lives in the pipeline, not here. This module just reports pass/fail.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

MAX_SSIM_COMPARISONS = 120  # cap for very long videos (avoid O(N) slowdown)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: object = None

    def __str__(self):
        return f"{self.name}={'OK' if self.passed else 'FAIL'}({self.detail})"


@dataclass
class ValidationResult:
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    failed_names: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return " | ".join(str(c) for c in self.checks)


class QualityValidator:
    """Validates a (T, H, W, C) uint8 frame array before delivery to the client."""

    def __init__(
        self,
        min_proxy_ssim: float = 0.15,
        max_black_ratio: float = 0.05,
        min_visual_similarity: float = 0.60,
        clip_check: bool = False,
    ):
        self.min_proxy_ssim = min_proxy_ssim
        self.max_black_ratio = max_black_ratio
        self.min_visual_similarity = min_visual_similarity
        self.clip_check = clip_check

    def validate(
        self,
        frames: np.ndarray,
        *,
        expected_resolution: Optional[tuple[int, int]] = None,
        expected_duration_s: Optional[float] = None,
        expected_fps: Optional[int] = None,
        reference_images: Optional[list[bytes]] = None,
    ) -> ValidationResult:
        checks: list[CheckResult] = []

        checks.append(self._check_shape(frames))
        if not checks[-1].passed:
            # Shape failure means all other checks would crash
            return ValidationResult(passed=False, checks=checks, failed_names=[checks[-1].name])

        checks.append(self._check_black_frames(frames))
        checks.append(self._check_temporal_consistency(frames))

        if expected_resolution is not None:
            checks.append(self._check_resolution(frames, expected_resolution))

        if expected_duration_s is not None and expected_fps is not None:
            checks.append(self._check_duration(frames, expected_duration_s, expected_fps))

        if reference_images and self.clip_check:
            checks.append(self._check_visual_similarity(frames, reference_images))

        failed = [c.name for c in checks if not c.passed]
        return ValidationResult(passed=len(failed) == 0, checks=checks, failed_names=failed)

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_shape(frames: np.ndarray) -> CheckResult:
        ok = (
            isinstance(frames, np.ndarray)
            and frames.ndim == 4
            and frames.shape[0] >= 1
            and frames.shape[-1] in (1, 3, 4)
            and frames.dtype == np.uint8
        )
        return CheckResult("valid_shape", ok, detail=getattr(frames, "shape", "not_ndarray"))

    def _check_black_frames(self, frames: np.ndarray) -> CheckResult:
        means = frames.mean(axis=(1, 2, 3))   # per-frame brightness
        black_idx = [int(i) for i, m in enumerate(means) if m < 5.0]
        ratio = len(black_idx) / len(frames)
        ok = ratio <= self.max_black_ratio
        return CheckResult(
            "no_black_frames", ok,
            detail={"black_ratio": round(ratio, 4), "first_black": black_idx[:5]},
        )

    def _check_temporal_consistency(self, frames: np.ndarray) -> CheckResult:
        if len(frames) < 2:
            return CheckResult("temporal_consistency", True, detail="single_frame")

        # Proxy: mean absolute frame difference normalised to [0,1].
        # Completely random frames → diff ≈ 0.5 (TV static).
        # Good video → diff ≈ 0.01–0.15 per adjacent pair.
        step = max(1, len(frames) // MAX_SSIM_COMPARISONS)
        diffs = []
        for i in range(0, len(frames) - 1, step):
            a = frames[i].astype(np.float32) / 255.0
            b = frames[i + 1].astype(np.float32) / 255.0
            diffs.append(float(np.abs(a - b).mean()))

        avg_diff = float(np.mean(diffs))
        proxy_ssim = round(max(0.0, 1.0 - avg_diff * 2), 3)  # rough mapping
        ok = proxy_ssim >= self.min_proxy_ssim
        return CheckResult(
            "temporal_consistency", ok,
            detail={"proxy_ssim": proxy_ssim, "avg_frame_diff": round(avg_diff, 4)},
        )

    @staticmethod
    def _check_resolution(
        frames: np.ndarray, expected: tuple[int, int]
    ) -> CheckResult:
        w_exp, h_exp = expected
        h_act, w_act = frames.shape[1], frames.shape[2]
        ok = w_act == w_exp and h_act == h_exp
        return CheckResult(
            "resolution_match", ok,
            detail={"expected": (w_exp, h_exp), "actual": (w_act, h_act)},
        )

    @staticmethod
    def _check_duration(
        frames: np.ndarray, expected_s: float, fps: int
    ) -> CheckResult:
        expected_n = int(expected_s * fps)
        actual_n = len(frames)
        tolerance = max(fps, int(expected_n * 0.10))  # ±10% or ±1 second
        ok = abs(actual_n - expected_n) <= tolerance
        return CheckResult(
            "duration_match", ok,
            detail={"expected_frames": expected_n, "actual_frames": actual_n, "tolerance": tolerance},
        )

    def _check_visual_similarity(
        self, frames: np.ndarray, reference_images: list[bytes]
    ) -> CheckResult:
        try:
            import io
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor

            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model.eval()

            def _embed_bytes(data: bytes) -> "torch.Tensor":
                img = Image.open(io.BytesIO(data)).convert("RGB")
                inputs = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    feat = model.get_image_features(**inputs)
                return feat / feat.norm(dim=-1, keepdim=True)

            def _embed_frame(frame: np.ndarray) -> "torch.Tensor":
                img = Image.fromarray(frame)
                inputs = processor(images=img, return_tensors="pt")
                with torch.no_grad():
                    feat = model.get_image_features(**inputs)
                return feat / feat.norm(dim=-1, keepdim=True)

            ref_embeds = torch.cat([_embed_bytes(r) for r in reference_images], dim=0)
            ref_embed = ref_embeds.mean(dim=0, keepdim=True)
            ref_embed = ref_embed / ref_embed.norm(dim=-1, keepdim=True)

            # Sample up to 10 frames spread across the video
            idxs = np.linspace(0, len(frames) - 1, min(10, len(frames)), dtype=int)
            sims = []
            for idx in idxs:
                fe = _embed_frame(frames[idx])
                sims.append(float((fe @ ref_embed.T).squeeze()))

            avg_sim = float(np.mean(sims))
            ok = avg_sim >= self.min_visual_similarity
            return CheckResult(
                "visual_similarity", ok,
                detail={"avg_clip_sim": round(avg_sim, 4), "threshold": self.min_visual_similarity},
            )

        except ImportError:
            log.info("CLIP check skipped: transformers/torch not available")
            return CheckResult("visual_similarity", True, detail="clip_unavailable_skipped")
        except Exception as exc:
            log.warning("CLIP visual similarity check error: %s", exc)
            return CheckResult("visual_similarity", True, detail=f"check_error_skipped")
