"""Frame-count and duration helpers for video training and inference."""

from __future__ import annotations

import math

MIN_DURATION_SECONDS = 4.0
MAX_DURATION_SECONDS = 60.0 * 60.0
TEMPORAL_MULTIPLE = 4


def round_up_to_multiple(value: int, multiple: int = TEMPORAL_MULTIPLE) -> int:
    """Round an integer up to the next positive temporal multiple."""
    if multiple <= 0:
        raise ValueError("multiple must be positive")
    if value <= 0:
        raise ValueError("frame count must be positive")
    return int(math.ceil(value / multiple) * multiple)


def normalize_num_frames(
    num_frames: int,
    temporal_multiple: int = TEMPORAL_MULTIPLE,
) -> int:
    """Return a VAE-safe frame count.

    The project VAE downsamples time by 4x, so frame counts must be divisible
    by 4 or the decoded video can be shorter than the requested input.
    """
    return round_up_to_multiple(int(num_frames), temporal_multiple)


def frames_from_duration(
    duration_seconds: float,
    fps: float,
    temporal_multiple: int = TEMPORAL_MULTIPLE,
) -> int:
    """Convert a requested duration to a VAE-safe frame count."""
    duration_seconds = float(duration_seconds)
    fps = float(fps)

    if fps <= 0:
        raise ValueError("fps must be greater than zero")
    if duration_seconds < MIN_DURATION_SECONDS:
        raise ValueError(
            f"duration_seconds must be at least {MIN_DURATION_SECONDS:g} seconds"
        )
    if duration_seconds > MAX_DURATION_SECONDS:
        raise ValueError(
            f"duration_seconds cannot exceed {MAX_DURATION_SECONDS:g} seconds"
        )

    return normalize_num_frames(
        int(math.ceil(duration_seconds * fps)),
        temporal_multiple=temporal_multiple,
    )


def resolve_frame_count(
    num_frames: int,
    duration_seconds: float | None,
    fps: float,
    temporal_multiple: int = TEMPORAL_MULTIPLE,
) -> int:
    """Resolve either explicit frames or duration into a safe frame count."""
    if duration_seconds is not None:
        return frames_from_duration(
            duration_seconds, fps, temporal_multiple=temporal_multiple
        )
    return normalize_num_frames(num_frames, temporal_multiple=temporal_multiple)


def split_frame_count(total_frames: int, chunk_frames: int) -> list[int]:
    """Split a total frame count into VAE-safe chunk sizes."""
    total_frames = normalize_num_frames(total_frames)
    chunk_frames = normalize_num_frames(chunk_frames)
    chunks: list[int] = []
    remaining = total_frames
    while remaining > 0:
        current = min(chunk_frames, remaining)
        chunks.append(normalize_num_frames(current))
        remaining -= current
    return chunks
