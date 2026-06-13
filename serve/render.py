"""Render helpers: turn model output into a saved video file.

Kept deliberately small. The production render *tail* (RIFE frame-interpolation +
Real-ESRGAN tiled upscale + FFmpeg color-grade/encode) plugs in here later via
`finish_clip()` — for now it is an identity pass so the slice runs end-to-end.
"""
import os

import numpy as np


def to_uint8_clip(samples) -> np.ndarray:
    """(B,C,T,H,W) float in [-1,1]  ->  (T,H,W,C) uint8 for the first clip in the batch."""
    import torch
    if isinstance(samples, torch.Tensor):
        x = ((samples.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
        x = x.permute(0, 2, 3, 4, 1).cpu().numpy()  # (B,T,H,W,C)
    else:
        x = samples
    clip = x[0]
    if clip.shape[-1] == 1:  # grayscale -> RGB so every encoder is happy
        clip = np.repeat(clip, 3, axis=-1)
    return clip


def save_clip(clip: np.ndarray, path: str, fps: int = 8) -> str:
    """Write a (T,H,W,C) uint8 clip to .mp4 (default) or .gif."""
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    frames = list(clip)
    if path.lower().endswith(".gif"):
        imageio.mimsave(path, frames, fps=fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps, codec="libx264",
                        pixelformat="yuv420p", macro_block_size=1)
    return path


def finish_clip(clip: np.ndarray) -> np.ndarray:
    """Production render tail hook (RIFE -> ESRGAN -> grade). Identity for now.

    Phase 2: interpolate fps with RIFE, upscale with tiled Real-ESRGAN, grade via
    FFmpeg lut3d. See BLUEPRINT.md sec 6.
    """
    return clip
