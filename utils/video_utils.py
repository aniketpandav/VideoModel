"""Video I/O utilities using OpenCV and FFmpeg."""

from __future__ import annotations

import random
import subprocess
import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def _as_uint8_frames(frames: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert frame arrays/tensors to [T, H, W, C] uint8."""
    if isinstance(frames, torch.Tensor):
        frames = frames.detach().cpu().numpy()
    if frames.dtype == np.float32 or frames.dtype == np.float64:
        frames = (np.clip(frames, 0, 1) * 255).astype(np.uint8)
    if frames.dtype != np.uint8:
        frames = frames.astype(np.uint8)
    return frames


class VideoFrameWriter:
    """Incremental video writer for long, chunked generation."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float = 8.0,
        codec: str = "libx264",
        quality: int = 23,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self._process: subprocess.Popen | None = None
        self._opencv_writer = None
        self._size: tuple[int, int] | None = None

    def __enter__(self) -> "VideoFrameWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _start(self, width: int, height: int) -> None:
        self._size = (width, height)
        cmd = [
            "ffmpeg", "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
            "-c:v", self.codec,
            "-crf", str(self.quality),
            "-pix_fmt", "yuv420p",
            str(self.output_path),
        ]
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.warning("FFmpeg not found, falling back to OpenCV VideoWriter")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._opencv_writer = cv2.VideoWriter(
                str(self.output_path), fourcc, self.fps, (width, height)
            )

    def write(self, frames: Union[np.ndarray, torch.Tensor]) -> None:
        frames = _as_uint8_frames(frames)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape [T, H, W, 3]")

        _, height, width, _ = frames.shape
        if self._size is None:
            self._start(width, height)
        elif self._size != (width, height):
            raise ValueError(
                f"all chunks must share size {self._size}, got {(width, height)}"
            )

        if self._process is not None:
            assert self._process.stdin is not None
            self._process.stdin.write(frames.tobytes())
        elif self._opencv_writer is not None:
            for frame in frames:
                self._opencv_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        if self._process is not None:
            assert self._process.stdin is not None
            self._process.stdin.close()
            self._process.wait()
            if self._process.returncode != 0:
                stderr = self._process.stderr.read().decode(errors="replace")
                logger.warning(f"FFmpeg warning: {stderr[:200]}")
            self._process = None
        if self._opencv_writer is not None:
            self._opencv_writer.release()
            self._opencv_writer = None


def read_video_frames(
    path: str | Path,
    num_frames: Optional[int] = None,
    target_fps: Optional[float] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
) -> tuple[np.ndarray, float]:
    """Read video frames using OpenCV.

    Args:
        path: Path to video file.
        num_frames: Number of frames to sample (uniformly). None = all frames.
        target_fps: Target FPS for resampling. None = original FPS.
        height: Target frame height. None = original.
        width: Target frame width. None = original.

    Returns:
        Tuple of (frames array [T, H, W, C] in uint8, original fps).
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {path}")

    # Determine which frames to read
    if target_fps is not None and target_fps < original_fps:
        # Subsample by FPS
        frame_interval = original_fps / target_fps
        indices = [int(i * frame_interval) for i in range(int(total_frames / frame_interval))]
    else:
        indices = list(range(total_frames))

    if num_frames is not None and len(indices) > num_frames:
        # Uniformly sample num_frames
        step = len(indices) / num_frames
        indices = [indices[int(i * step)] for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if height is not None and width is not None:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)

    cap.release()

    if not frames:
        raise ValueError(f"No frames read from video: {path}")

    return np.stack(frames, axis=0), original_fps


def read_video_clip(
    path: str | Path,
    num_frames: int,
    target_fps: Optional[float] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    random_start: bool = True,
) -> tuple[np.ndarray, float]:
    """Read a contiguous fixed-length training clip from a video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video has no frames: {path}")

    if target_fps is not None and target_fps > 0 and target_fps < original_fps:
        frame_interval = max(1, int(round(original_fps / target_fps)))
    else:
        frame_interval = 1

    span = (num_frames - 1) * frame_interval + 1
    max_start = max(0, total_frames - span)
    start = random.randint(0, max_start) if random_start and max_start > 0 else 0
    indices = [min(start + i * frame_interval, total_frames - 1) for i in range(num_frames)]

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if height is not None and width is not None:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        frames.append(frame)

    cap.release()
    if not frames:
        raise ValueError(f"No frames read from video: {path}")
    while len(frames) < num_frames:
        frames.append(frames[-1].copy())

    return np.stack(frames, axis=0), original_fps


def save_video_frames(
    frames: Union[np.ndarray, torch.Tensor],
    output_path: str | Path,
    fps: float = 8.0,
    codec: str = "libx264",
    quality: int = 23,
) -> Path:
    """Save frames as a video file using FFmpeg.

    Args:
        frames: Video frames [T, H, W, C] in uint8 or float [0, 1].
        output_path: Output video file path.
        fps: Frames per second.
        codec: Video codec (libx264 for mp4).
        quality: CRF quality (lower = better, 0-51).

    Returns:
        Path to the saved video.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frames = _as_uint8_frames(frames)

    T, H, W, C = frames.shape
    if C != 3:
        raise ValueError("frames must have shape [T, H, W, 3]")

    with VideoFrameWriter(output_path, fps=fps, codec=codec, quality=quality) as writer:
        writer.write(frames)

    logger.info(f"Saved video to {output_path} ({T} frames, {fps} FPS)")
    return output_path


def _save_with_opencv(
    frames: np.ndarray, output_path: Path, fps: float
) -> None:
    """Fallback: save video using OpenCV VideoWriter."""
    T, H, W, C = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))
    for i in range(T):
        bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        writer.write(bgr)
    writer.release()


def frames_to_tensor(
    frames: np.ndarray,
    normalize: bool = True,
) -> torch.Tensor:
    """Convert numpy frames to PyTorch tensor.

    Args:
        frames: [T, H, W, C] uint8 array.
        normalize: If True, normalize to [-1, 1]. Otherwise [0, 1].

    Returns:
        Tensor of shape [C, T, H, W] in float32.
    """
    # [T, H, W, C] -> [T, C, H, W]
    tensor = torch.from_numpy(frames).float().permute(0, 3, 1, 2)
    tensor = tensor / 255.0  # [0, 1]
    if normalize:
        tensor = tensor * 2.0 - 1.0  # [-1, 1]
    # [T, C, H, W] -> [C, T, H, W]
    tensor = tensor.permute(1, 0, 2, 3)
    return tensor


def tensor_to_frames(
    tensor: torch.Tensor,
    denormalize: bool = True,
) -> np.ndarray:
    """Convert PyTorch tensor back to numpy frames.

    Args:
        tensor: [C, T, H, W] or [B, C, T, H, W] tensor.
        denormalize: If True, maps [-1,1] to [0,255].

    Returns:
        Numpy array [T, H, W, C] in uint8.
    """
    if tensor.dim() == 5:
        tensor = tensor[0]  # Take first batch

    # [C, T, H, W] -> [T, H, W, C]
    tensor = tensor.permute(1, 2, 3, 0).detach().cpu().float()

    if denormalize:
        tensor = (tensor + 1.0) / 2.0  # [-1,1] -> [0,1]

    frames = (tensor.clamp(0, 1) * 255).numpy().astype(np.uint8)
    return frames


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path,
    fps: Optional[float] = None,
    format: str = "png",
) -> list[Path]:
    """Extract individual frames from a video file.

    Args:
        video_path: Input video path.
        output_dir: Directory to save frames.
        fps: Target FPS. None = original.
        format: Image format (png, jpg).

    Returns:
        List of paths to extracted frame images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames, original_fps = read_video_frames(
        video_path, target_fps=fps
    )

    paths = []
    for i, frame in enumerate(frames):
        frame_path = output_dir / f"frame_{i:06d}.{format}"
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(frame_path), frame_bgr)
        paths.append(frame_path)

    logger.info(f"Extracted {len(paths)} frames to {output_dir}")
    return paths
