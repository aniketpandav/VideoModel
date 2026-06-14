"""Render helpers: turn model output into a saved video file.

Format support (via FFmpeg subprocess — must be installed on the host):
  mp4  → H.264 + yuv420p   (universal playback, streaming-friendly)
  mov  → H.264 + yuv420p   (Apple ecosystem)
  avi  → MPEG-4 Part 2     (legacy compatibility)
  mkv  → H.265/HEVC + yuv420p (high efficiency, smaller files)
  webm → VP9                (web streaming, royalty-free)
  gif  → imageio (no FFmpeg needed, low quality — for previews only)

Quality presets → CRF values:
  draft=35, standard=23, high=18, cinematic=14

Resolution scaling: frames are always rescaled to the target WxH using Lanczos
before encoding to ensure the output file exactly matches the requested spec.
"""
from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# CRF (Constant Rate Factor): lower = better quality, larger file
_QUALITY_CRF = {"draft": 35, "standard": 23, "high": 18, "cinematic": 14}

# Codec + pixel format per container
_FORMAT_CODEC: dict[str, tuple[str, str]] = {
    "mp4":  ("libx264", "yuv420p"),
    "mov":  ("libx264", "yuv420p"),
    "avi":  ("mpeg4",   "yuv420p"),
    "mkv":  ("libx265", "yuv420p"),
    "webm": ("libvpx-vp9", "yuv420p"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def to_uint8_clip(samples) -> np.ndarray:
    """(B, C, T, H, W) float in [-1, 1]  →  (T, H, W, C) uint8 for the first clip."""
    try:
        import torch
        if isinstance(samples, torch.Tensor):
            x = ((samples.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
            x = x.permute(0, 2, 3, 4, 1).cpu().numpy()  # (B, T, H, W, C)
        else:
            x = np.asarray(samples)
    except ImportError:
        x = np.asarray(samples)

    clip = x[0] if x.ndim == 5 else x
    if clip.shape[-1] == 1:  # grayscale → RGB so every encoder is happy
        clip = np.repeat(clip, 3, axis=-1)
    return clip.astype(np.uint8)


def finish_clip(clip: np.ndarray) -> np.ndarray:
    """Production render tail hook: Real-ESRGAN upscaling → colour grade.

    Activated by environment variable:
        VDM_UPSCALE=none    → identity pass (default, no extra packages)
        VDM_UPSCALE=2x      → Real-ESRGAN ×2 upscale
        VDM_UPSCALE=4k      → Real-ESRGAN ×4 (704×480 → 2816×1920 near-4K)
        VDM_UPSCALE=8k      → Real-ESRGAN ×4 + Lanczos ×2 (~5632×3840)

    Real-ESRGAN runs on GTX 1650 in tiled mode (~2 GB VRAM).
    Requires:  pip install realesrgan basicsr
    """
    import os
    if os.environ.get("VDM_UPSCALE", "none").lower() == "none":
        return clip
    try:
        from .upscale import auto_upscale
        return auto_upscale(clip)
    except Exception as exc:
        log.warning("finish_clip: upscaling failed (%s); returning original frames", exc)
        return clip


def save_clip(clip: np.ndarray, path: str, fps: int = 8) -> str:
    """Write a (T, H, W, C) uint8 clip to .mp4 (or any supported format).

    Backward-compatible entry point used by the legacy /v1/videos endpoint.
    """
    ext = Path(path).suffix.lstrip(".").lower() or "mp4"
    return encode_video(clip, path, fps=fps, quality="standard", fmt=ext)


def encode_video(
    frames: np.ndarray,
    output_path: str,
    *,
    fps: int = 24,
    quality: str = "standard",
    fmt: str | None = None,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """Encode (T, H, W, C) uint8 frames to a video file via FFmpeg.

    Falls back to imageio for GIF or when FFmpeg is not available.

    Args:
        frames:      (T, H, W, C) uint8 ndarray.
        output_path: Destination file path (extension determines format if fmt is None).
        fps:         Output frame rate.
        quality:     draft | standard | high | cinematic.
        fmt:         Container format override (mp4/mov/avi/mkv/webm/gif).
        width/height: Resize frames before encoding (None = use frame dimensions).

    Returns:
        output_path (for chaining).
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if fmt is None:
        fmt = Path(output_path).suffix.lstrip(".").lower() or "mp4"

    if fmt == "gif":
        return _encode_gif(frames, output_path, fps)

    if width or height:
        frames = _resize_frames(frames, width, height)

    if _ffmpeg_available():
        return _encode_ffmpeg(frames, output_path, fps=fps, quality=quality, fmt=fmt)

    log.info("FFmpeg not found; using imageio for %s (MP4 only — install FFmpeg for all formats)", fmt)
    return _encode_imageio(frames, output_path, fps)


def make_preview_gif(frames: np.ndarray, path: str, fps: int = 8, max_frames: int = 72) -> str:
    """Write a short low-res preview GIF (≤ 3 seconds). Used for job thumbnails."""
    # Subsample to at most max_frames
    if len(frames) > max_frames:
        idxs = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = frames[idxs]
    small = _resize_frames(frames, 320, None)  # cap width at 320px for size
    return _encode_gif(small, path, fps=min(fps, 10))


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _encode_ffmpeg(
    frames: np.ndarray,
    output_path: str,
    *,
    fps: int,
    quality: str,
    fmt: str,
) -> str:
    codec, pix_fmt = _FORMAT_CODEC.get(fmt, ("libx264", "yuv420p"))
    crf = _QUALITY_CRF.get(quality, 23)
    h, w = frames.shape[1], frames.shape[2]

    # H.264/H.265 require even dimensions
    w_enc = w if w % 2 == 0 else w - 1
    h_enc = h if h % 2 == 0 else h - 1

    vf = ""
    if w_enc != w or h_enc != h:
        vf = f"-vf scale={w_enc}:{h_enc}"

    # Build codec-specific quality flag
    if fmt == "webm":
        quality_flags = ["-crf", str(crf), "-b:v", "0"]
    elif fmt in ("mp4", "mov", "mkv"):
        quality_flags = ["-crf", str(crf)]
    else:  # avi / mpeg4
        quality_flags = ["-qscale:v", str(max(1, crf // 4))]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-vcodec", codec,
        "-pix_fmt", pix_fmt,
        *quality_flags,
        "-movflags", "+faststart",  # web-optimised MP4/MOV
    ]
    if vf:
        cmd += vf.split()
    cmd.append(output_path)

    raw_bytes = frames.tobytes()
    log.debug("FFmpeg encode: %s %dx%d %dfps %d frames → %s",
              fmt, w, h, fps, len(frames), output_path)
    try:
        result = subprocess.run(
            cmd, input=raw_bytes,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=600,
        )
        if result.returncode != 0:
            log.error("FFmpeg stderr: %s", result.stderr.decode(errors="replace"))
            raise RuntimeError(f"FFmpeg exited with code {result.returncode}")
    except FileNotFoundError:
        log.info("FFmpeg not on PATH; using imageio fallback (install FFmpeg for MOV/MKV/WEBM support)")
        return _encode_imageio(frames, output_path, fps)

    return output_path


def _encode_imageio(frames: np.ndarray, output_path: str, fps: int) -> str:
    """Fallback encoder using imageio (supports mp4 via imageio-ffmpeg plugin)."""
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    frame_list = list(frames)
    ext = Path(output_path).suffix.lower()
    if ext == ".gif":
        imageio.mimsave(output_path, frame_list, fps=fps, loop=0)
    else:
        imageio.mimsave(output_path, frame_list, fps=fps,
                        codec="libx264", pixelformat="yuv420p", macro_block_size=1)
    return output_path


def _encode_gif(frames: np.ndarray, output_path: str, fps: int) -> str:
    """Write a GIF using imageio (no FFmpeg dependency)."""
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    duration = 1.0 / max(fps, 1)
    imageio.mimsave(output_path, list(frames), duration=duration, loop=0)
    return output_path


def _resize_frames(
    frames: np.ndarray, width: Optional[int], height: Optional[int]
) -> np.ndarray:
    """Resize all frames to (height, width) using Lanczos interpolation."""
    try:
        from PIL import Image
    except ImportError:
        log.warning("Pillow not installed; skipping frame resize")
        return frames

    T, h_src, w_src, C = frames.shape
    if height is None:
        height = int(h_src * width / w_src)  # type: ignore[operator]
    if width is None:
        width = int(w_src * height / h_src)

    # Enforce even dimensions (H.264/H.265 requirement)
    width = width if width % 2 == 0 else width - 1
    height = height if height % 2 == 0 else height - 1

    if (width, height) == (w_src, h_src):
        return frames

    out = np.empty((T, height, width, C), dtype=np.uint8)
    for i, frame in enumerate(frames):
        img = Image.fromarray(frame)
        out[i] = np.asarray(img.resize((width, height), Image.LANCZOS))
    return out


def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
