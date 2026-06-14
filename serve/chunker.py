"""Long-video chunked generation and cross-fade blending.

Strategy
--------
1. Expand the master prompt into N sequential scene descriptions (via PromptEngine).
2. Generate each chunk sequentially using the backbone. The last frame of chunk i
   is passed as `first_frame` to chunk i+1 for temporal continuity.
3. Blend at boundaries with a linear cross-fade over `overlap_frames`.
4. Trim the final concatenated array to the exact requested duration.

Chunk size is fixed at CHUNK_DURATION_S seconds. For a 1-hour video at 24 fps:
  720 chunks × 5s × 24fps = 86,400 frames  (≈ 3 GB RAM for 512×512 RGB uint8)

For production GPU clusters, each chunk can be a separate Celery subtask with
the last-frame tensor serialised via Redis. The blend step runs as a final reducer.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from .backbones import Backbone
    from .prompt_engine import EnhancedRequest, PromptEngine

log = logging.getLogger(__name__)

CHUNK_DURATION_S: int = 5   # seconds per generation chunk
OVERLAP_S: float = 1.0      # seconds of cross-fade overlap between chunks


class LongVideoChunker:
    """Generates videos of arbitrary duration by stitching short backbone clips."""

    def __init__(self, backbone: "Backbone", prompt_engine: "PromptEngine"):
        self.backbone = backbone
        self.engine = prompt_engine

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        request: "EnhancedRequest",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """Return (T, H, W, C) uint8 array for the full video duration.

        progress_callback receives a float in [0, 100] as chunks complete.
        """
        fps = request.fps
        duration = request.duration_seconds
        overlap_frames = max(0, int(OVERLAP_S * fps))
        n_chunks = max(1, int(np.ceil(duration / CHUNK_DURATION_S)))

        log.info(
            "LongVideoChunker: %.1fs → %d chunks @ %dfps, overlap=%d frames",
            duration, n_chunks, fps, overlap_frames,
        )

        scene_prompts = self.engine.expand_narrative(request.enhanced_prompt, n_chunks)
        chunks: list[np.ndarray] = []
        prev_last_frame: Optional[np.ndarray] = None

        for i, scene_prompt in enumerate(scene_prompts):
            if progress_callback:
                progress_callback(i / n_chunks * 95.0)

            remaining_s = duration - i * CHUNK_DURATION_S
            this_s = min(float(CHUNK_DURATION_S) + OVERLAP_S, remaining_s + OVERLAP_S)
            num_frames = max(8, int(this_s * fps))

            chunk = self._generate_chunk(
                scene_prompt,
                num_frames=num_frames,
                steps=request.inference_steps,
                seed=(request.seed or 0) + i,
                width=request.resolution[0],
                height=request.resolution[1],
                first_frame=prev_last_frame,
            )
            chunks.append(chunk)
            prev_last_frame = chunk[-1].copy()
            log.debug("Chunk %d/%d done: %d frames", i + 1, n_chunks, len(chunk))

        if progress_callback:
            progress_callback(96.0)

        blended = self._blend_chunks(chunks, overlap_frames)

        # Trim to exact requested frame count
        target_frames = int(duration * fps)
        result = blended[:target_frames] if len(blended) > target_frames else blended

        if progress_callback:
            progress_callback(100.0)

        log.info("LongVideoChunker done: %d frames (%.1fs)", len(result), len(result) / fps)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_chunk(
        self,
        prompt: str,
        num_frames: int,
        steps: int,
        seed: int,
        width: int,
        height: int,
        first_frame: Optional[np.ndarray],
    ) -> np.ndarray:
        """Call backbone with progressive kwarg fallback for compatibility."""
        first_frame_bytes = _frame_to_bytes(first_frame) if first_frame is not None else None

        # Try most-capable signature first, fall back gracefully
        for kwargs in [
            dict(num_frames=num_frames, steps=steps, seed=seed,
                 width=width, height=height, first_frame=first_frame_bytes),
            dict(num_frames=num_frames, steps=steps, seed=seed,
                 width=width, height=height),
            dict(num_frames=num_frames, steps=steps, seed=seed),
            dict(steps=steps, seed=seed),
        ]:
            try:
                clip, _ = self.backbone.generate(prompt, **kwargs)
                return clip
            except TypeError:
                continue

        raise RuntimeError("Backbone.generate() rejected all signature variants")

    @staticmethod
    def _blend_chunks(chunks: list[np.ndarray], overlap_frames: int) -> np.ndarray:
        """Linear cross-fade between consecutive chunks at shared boundary frames."""
        if len(chunks) == 0:
            raise ValueError("No chunks to blend")
        if len(chunks) == 1:
            return chunks[0]
        if overlap_frames <= 1:
            return np.concatenate(chunks, axis=0)

        result = chunks[0].astype(np.float32)
        for chunk in chunks[1:]:
            chunk_f = chunk.astype(np.float32)
            ov = min(overlap_frames, len(result), len(chunk_f))
            if ov <= 1:
                result = np.concatenate([result, chunk_f], axis=0)
                continue
            # Linear alpha: 0→1 over overlap region
            alpha = np.linspace(0.0, 1.0, ov, dtype=np.float32)
            alpha = alpha[:, np.newaxis, np.newaxis, np.newaxis]  # (ov,1,1,1)
            tail = result[-ov:]
            head = chunk_f[:ov]
            blended = (1.0 - alpha) * tail + alpha * head
            result = np.concatenate([result[:-ov], blended, chunk_f[ov:]], axis=0)

        return np.clip(result, 0.0, 255.0).astype(np.uint8)


def _frame_to_bytes(frame: np.ndarray) -> bytes:
    """Convert (H, W, C) uint8 frame to PNG bytes for backbone consumption."""
    import io
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(frame.astype(np.uint8)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        import struct, zlib
        # Minimal PNG writer for RGB frames (no Pillow dependency)
        h, w, c = frame.shape
        raw = b''.join(b'\x00' + frame[y].tobytes() for y in range(h))
        def png_chunk(tag: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(tag + data) & 0xFFFFFFFF
            return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', crc)
        colour_type = 2 if c == 3 else (6 if c == 4 else 0)
        ihdr = struct.pack('>IIBBBBB', w, h, 8, colour_type, 0, 0, 0)
        idat = zlib.compress(raw)
        return (b'\x89PNG\r\n\x1a\n'
                + png_chunk(b'IHDR', ihdr)
                + png_chunk(b'IDAT', idat)
                + png_chunk(b'IEND', b''))
