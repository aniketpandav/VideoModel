"""Video preprocessing utilities for dataset preparation.

Includes frame extraction, resizing, FPS normalization, scene detection, and motion filtering.
"""

from __future__ import annotations
import logging, subprocess
from pathlib import Path
from typing import Optional
import cv2, numpy as np

logger = logging.getLogger(__name__)


def extract_clips(video_path: str | Path, output_dir: str | Path, clip_duration: float = 2.0,
                  target_fps: int = 8, target_height: int = 256, target_width: int = 256,
                  min_motion: float = 0.01) -> list[Path]:
    """Extract clips from a video with preprocessing.

    Args:
        video_path: Source video path.
        output_dir: Directory for output clips.
        clip_duration: Duration per clip in seconds.
        target_fps: Target FPS.
        target_height: Output height.
        target_width: Output width.
        min_motion: Minimum average optical flow magnitude to keep clip.

    Returns:
        List of paths to extracted clips.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = int(clip_duration * target_fps)
    frame_interval = max(1, int(original_fps / target_fps))

    clips = []
    clip_idx = 0
    frame_buffer = []

    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num % frame_interval != 0:
            continue

        frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        frame_buffer.append(frame)

        if len(frame_buffer) >= frames_per_clip:
            # Check motion
            if _has_sufficient_motion(frame_buffer, min_motion):
                clip_path = output_dir / f"clip_{clip_idx:06d}.mp4"
                _save_clip(frame_buffer, clip_path, target_fps)
                clips.append(clip_path)
                clip_idx += 1
            frame_buffer = []

    cap.release()
    logger.info(f"Extracted {len(clips)} clips from {video_path}")
    return clips


def _has_sufficient_motion(frames: list[np.ndarray], threshold: float) -> bool:
    """Check if frames contain enough motion using optical flow."""
    if len(frames) < 2:
        return True
    total_flow = 0.0
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for frame in frames[1:]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        total_flow += np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
        prev_gray = gray
    avg_flow = total_flow / (len(frames) - 1)
    return avg_flow > threshold


def _save_clip(frames: list[np.ndarray], path: Path, fps: int) -> None:
    """Save frames as a video clip."""
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def center_crop(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Center crop a frame to target dimensions."""
    h, w = frame.shape[:2]
    start_h = max(0, (h - target_h) // 2)
    start_w = max(0, (w - target_w) // 2)
    return frame[start_h:start_h + target_h, start_w:start_w + target_w]
