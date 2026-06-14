"""Real-ESRGAN 4K upscaling module.

Converts any backbone output (even 32×32 toy frames) to near-4K quality:
  - 4× scale:  704×480  → 2816×1920  (~3K,  fits GTX 1650 in tiles)
  - 4× scale:   32×32   →  128×128   (demo quality, not true 4K)

The upscaler runs locally on the GTX 1650 (inference only, ~2 GB VRAM).
Training the upscaler would require an entirely different dataset and is not needed —
Real-ESRGAN weights are pretrained and general-purpose.

Install:
    pip install realesrgan basicsr

Usage:
    Set VDM_UPSCALE=4k  (or "none" to disable, default)
    The finish_clip() hook in render.py calls this automatically.

4K pipeline:
    1. Generate with LTX-Video on Kaggle T4 (704×480, 24fps)
    2. Download the checkpoint / generate locally
    3. Apply 4× Real-ESRGAN → 2816×1920
    4. Apply RIFE 2× frame interpolation → 48fps (optional)
    5. Encode with H.265 for bandwidth-efficient 4K delivery
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Available upscale modes
# ---------------------------------------------------------------------------
#  none   → identity (default, no extra packages needed)
#  2x     → Real-ESRGAN × 2  (fast, moderate quality boost)
#  4k     → Real-ESRGAN × 4  (production; 704×480 → 2816×1920)
#  8k     → Real-ESRGAN × 4, then lanczos ×2 to 5632×3840 (crops to true 4K)

UPSCALE_MODE = os.environ.get("VDM_UPSCALE", "none").lower()


def upscale_frames(
    frames: np.ndarray,
    scale: int = 4,
    tile: int = 256,
    half_precision: bool = True,
) -> np.ndarray:
    """Upscale a (T, H, W, C) uint8 array using Real-ESRGAN.

    Args:
        frames:          (T, H, W, C) uint8 ndarray.
        scale:           Upscale factor (2 or 4).
        tile:            Tile size for tiled inference (lower = less VRAM).
                         256 fits on 4 GB; use 512 on 8 GB+.
        half_precision:  Use fp16 on CUDA (saves ~40% VRAM, negligible quality loss).

    Returns:
        (T, H*scale, W*scale, C) uint8 ndarray.
    """
    upsampler = _get_upsampler(scale=scale, tile=tile, half=half_precision)
    if upsampler is None:
        log.info("Real-ESRGAN unavailable; returning frames unchanged")
        return frames

    T = len(frames)
    results = []
    for i, frame in enumerate(frames):
        if i % 10 == 0:
            log.debug("Upscaling frame %d/%d …", i + 1, T)
        out, _ = upsampler.enhance(frame, outscale=scale)
        results.append(out)

    return np.stack(results, axis=0).astype(np.uint8)


def auto_upscale(frames: np.ndarray) -> np.ndarray:
    """Apply upscaling based on VDM_UPSCALE env var. Called from finish_clip()."""
    mode = UPSCALE_MODE
    if mode == "none" or not mode:
        return frames

    h, w = frames.shape[1], frames.shape[2]
    log.info("Real-ESRGAN upscale mode=%s on %dx%d×%d frames", mode, w, h, len(frames))

    if mode in ("4k", "4x"):
        return upscale_frames(frames, scale=4, tile=256)
    if mode in ("2x", "2k"):
        return upscale_frames(frames, scale=2, tile=512)
    if mode == "8k":
        # 4× Real-ESRGAN, then Lanczos ×2 to reach 8K-class resolution
        frames = upscale_frames(frames, scale=4, tile=256)
        return _lanczos_resize(frames, scale=2)

    log.warning("Unknown VDM_UPSCALE=%r; skipping", mode)
    return frames


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _patch_basicsr_torchvision() -> None:
    """Shim torchvision.transforms.functional_tensor for basicsr compat.

    basicsr imports rgb_to_grayscale from functional_tensor, which was removed
    in torchvision 0.17+. We inject functional as the shim so basicsr finds all
    the functions it needs without requiring a torchvision downgrade.
    """
    import sys
    if "torchvision.transforms.functional_tensor" not in sys.modules:
        try:
            import torchvision.transforms.functional as _tvf
            sys.modules["torchvision.transforms.functional_tensor"] = _tvf  # type: ignore[assignment]
        except ImportError:
            pass


@lru_cache(maxsize=1)
def _get_upsampler(scale: int = 4, tile: int = 256, half: bool = True):
    """Load and cache the Real-ESRGAN model. Returns None if not installed."""
    _patch_basicsr_torchvision()
    try:
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer
    except ImportError:
        log.warning(
            "Real-ESRGAN not installed. Run:\n"
            "  pip install realesrgan basicsr\n"
            "to enable 4K upscaling."
        )
        return None

    # Select model weights based on scale
    model_map = {
        2: ("RealESRGAN_x2plus.pth",
            RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2)),
        4: ("RealESRGAN_x4plus.pth",
            RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4)),
    }
    if scale not in model_map:
        raise ValueError(f"scale must be 2 or 4, got {scale}")

    model_name, model = model_map[scale]

    # Download path: weights/ dir in project root
    weights_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights"
    )
    os.makedirs(weights_dir, exist_ok=True)
    model_path = os.path.join(weights_dir, model_name)

    # Auto-download if not present
    if not os.path.exists(model_path):
        _download_weights(model_name, model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_half = half and device == "cuda"

    log.info(
        "Loading Real-ESRGAN %s on %s (half=%s, tile=%d) …",
        model_name, device, use_half, tile,
    )

    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=tile,
        tile_pad=10,
        pre_pad=0,
        half=use_half,
        device=device,
    )
    log.info("Real-ESRGAN loaded successfully")
    return upsampler


def _download_weights(model_name: str, dest_path: str):
    """Download Real-ESRGAN weights from GitHub releases."""
    import urllib.request

    base_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0"
    url = f"{base_url}/{model_name}"
    log.info("Downloading Real-ESRGAN weights from %s …", url)
    try:
        urllib.request.urlretrieve(url, dest_path)
        log.info("Weights saved to %s", dest_path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {model_name}: {exc}\n"
            f"Download manually from {url} and place in weights/"
        ) from exc


def _lanczos_resize(frames: np.ndarray, scale: int) -> np.ndarray:
    """Simple Lanczos resize for the 8K step (4× ESRGAN + 2× Lanczos = 8×)."""
    try:
        from PIL import Image
    except ImportError:
        return frames

    h, w = frames.shape[1], frames.shape[2]
    target_h, target_w = h * scale, w * scale
    result = np.empty((len(frames), target_h, target_w, frames.shape[3]), dtype=np.uint8)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        result[i] = np.asarray(img.resize((target_w, target_h), Image.LANCZOS))
    return result
