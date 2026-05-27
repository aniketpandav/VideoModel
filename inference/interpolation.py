"""Frame interpolation for smoother video output.

Implements simple optical-flow-based frame interpolation to
increase frame rate (e.g., 8fps -> 24fps) for smoother playback.
"""

from __future__ import annotations
import numpy as np, cv2, torch


def interpolate_frames(frames: np.ndarray, factor: int = 2) -> np.ndarray:
    """Interpolate between frames to increase frame count.

    Uses optical-flow-based warping for motion-aware interpolation.

    Args:
        frames: Video frames [T, H, W, C] in uint8.
        factor: Interpolation factor (2 = double frame count).

    Returns:
        Interpolated frames [T*factor-(factor-1), H, W, C].
    """
    if factor <= 1:
        return frames
    T, H, W, C = frames.shape
    result = []

    for i in range(T - 1):
        result.append(frames[i])
        frame1 = frames[i]
        frame2 = frames[i + 1]

        for j in range(1, factor):
            alpha = j / factor
            interp = _flow_interpolate(frame1, frame2, alpha)
            result.append(interp)

    result.append(frames[-1])
    return np.stack(result, axis=0)


def _flow_interpolate(frame1: np.ndarray, frame2: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate between two frames using optical flow.

    Args:
        frame1: First frame [H, W, C] uint8.
        frame2: Second frame [H, W, C] uint8.
        alpha: Interpolation weight (0=frame1, 1=frame2).

    Returns:
        Interpolated frame [H, W, C] uint8.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)

    # Compute forward optical flow
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    H, W = gray1.shape
    coords = np.stack(np.meshgrid(np.arange(W), np.arange(H)), axis=-1).astype(np.float32)

    # Warp frame1 forward by alpha * flow
    map1 = coords + flow * alpha
    warped1 = cv2.remap(frame1, map1[..., 0], map1[..., 1], cv2.INTER_LINEAR)

    # Warp frame2 backward by (1-alpha) * flow
    map2 = coords - flow * (1 - alpha)
    warped2 = cv2.remap(frame2, map2[..., 0], map2[..., 1], cv2.INTER_LINEAR)

    # Blend
    result = ((1 - alpha) * warped1.astype(np.float32) + alpha * warped2.astype(np.float32))
    return np.clip(result, 0, 255).astype(np.uint8)
