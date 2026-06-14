"""Generation orchestrator: routes EnhancedRequests to the correct backbone/chunker.

Mode 1 — Text → Video
  - duration <= LONG_VIDEO_THRESHOLD_S : direct backbone.generate()
  - duration >  LONG_VIDEO_THRESHOLD_S : LongVideoChunker.generate()

Mode 2 — Reference Image(s) + Text → Video
  - Short : backbone.generate_i2v() with CLIP-averaged multi-image style
  - Long  : LongVideoChunker with carry-forward first_frame anchored to original reference

Backbone kwargs fall back gracefully so ToyBackbone (which lacks width/height/
first_frame args) still works end-to-end without modification.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np

if TYPE_CHECKING:
    from .backbones import Backbone
    from .chunker import LongVideoChunker
    from .prompt_engine import EnhancedRequest

log = logging.getLogger(__name__)

LONG_VIDEO_THRESHOLD_S: float = 30.0


class VideoPipeline:
    """Stateless generation orchestrator. One instance per server process."""

    def __init__(self, backbone: "Backbone", chunker: "LongVideoChunker"):
        self.backbone = backbone
        self.chunker = chunker

    # ------------------------------------------------------------------
    # Mode 1: Text → Video
    # ------------------------------------------------------------------

    def text_to_video(
        self,
        request: "EnhancedRequest",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        if request.duration_seconds > LONG_VIDEO_THRESHOLD_S:
            log.info("Mode 1 long video (%.1fs) → chunker", request.duration_seconds)
            return self.chunker.generate(request, progress_callback=progress_callback)

        log.info("Mode 1 short video (%.1fs) → direct backbone", request.duration_seconds)
        num_frames = max(8, int(request.duration_seconds * request.fps))
        w, h = request.resolution

        if progress_callback:
            progress_callback(10.0)

        clip = self._call_t2v(
            request.enhanced_prompt,
            num_frames=num_frames,
            steps=request.inference_steps,
            seed=request.seed or 0,
            width=w,
            height=h,
        )

        if progress_callback:
            progress_callback(90.0)
        return clip

    # ------------------------------------------------------------------
    # Mode 2: Reference Image(s) + Text → Video
    # ------------------------------------------------------------------

    def image_to_video(
        self,
        request: "EnhancedRequest",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        if not request.reference_images:
            raise ValueError("Mode 2 requires at least one reference image.")

        if request.duration_seconds > LONG_VIDEO_THRESHOLD_S:
            log.info("Mode 2 long video (%.1fs) → chunker with I2V anchoring", request.duration_seconds)
            return self._long_i2v(request, progress_callback)

        log.info("Mode 2 short video (%.1fs) → backbone I2V", request.duration_seconds)
        num_frames = max(8, int(request.duration_seconds * request.fps))
        w, h = request.resolution
        primary_image = request.reference_images[0]

        if progress_callback:
            progress_callback(10.0)

        clip = self._call_i2v(
            image=primary_image,
            prompt=request.enhanced_prompt,
            num_frames=num_frames,
            steps=request.inference_steps,
            seed=request.seed or 0,
            width=w,
            height=h,
            reference_images=request.reference_images,
        )

        if progress_callback:
            progress_callback(90.0)
        return clip

    # ------------------------------------------------------------------
    # Long I2V: chunked with reference image as anchor
    # ------------------------------------------------------------------

    def _long_i2v(
        self,
        request: "EnhancedRequest",
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        from .chunker import CHUNK_DURATION_S, OVERLAP_S

        fps = request.fps
        duration = request.duration_seconds
        overlap_frames = max(0, int(OVERLAP_S * fps))
        n_chunks = max(1, int(np.ceil(duration / CHUNK_DURATION_S)))
        scene_prompts = self.chunker.engine.expand_narrative(
            request.enhanced_prompt, n_chunks
        )

        chunks: list[np.ndarray] = []
        prev_last_frame: Optional[np.ndarray] = None
        primary_image = request.reference_images[0]
        w, h = request.resolution

        for i, scene_prompt in enumerate(scene_prompts):
            if progress_callback:
                progress_callback(i / n_chunks * 95.0)

            remaining_s = duration - i * CHUNK_DURATION_S
            this_s = min(float(CHUNK_DURATION_S) + OVERLAP_S, remaining_s + OVERLAP_S)
            num_frames = max(8, int(this_s * fps))

            # Anchor = last frame of previous chunk (for continuity),
            # but keep the original reference images for style guidance.
            anchor_bytes = (
                _frame_to_bytes(prev_last_frame)
                if prev_last_frame is not None
                else primary_image
            )

            chunk = self._call_i2v(
                image=anchor_bytes,
                prompt=scene_prompt,
                num_frames=num_frames,
                steps=request.inference_steps,
                seed=(request.seed or 0) + i,
                width=w,
                height=h,
                reference_images=request.reference_images,
            )
            chunks.append(chunk)
            prev_last_frame = chunk[-1].copy()

        if progress_callback:
            progress_callback(96.0)

        blended = self.chunker._blend_chunks(chunks, overlap_frames)
        target = int(duration * fps)
        result = blended[:target] if len(blended) > target else blended

        if progress_callback:
            progress_callback(100.0)
        return result

    # ------------------------------------------------------------------
    # Backbone call helpers (graceful kwarg fallback)
    # ------------------------------------------------------------------

    def _call_t2v(self, prompt: str, **kwargs) -> np.ndarray:
        """Call backbone.generate() with progressive kwarg fallback."""
        # Full kwargs → drop width/height → drop num_frames
        for strip_keys in [[], ["width", "height"], ["width", "height", "num_frames"]]:
            kw = {k: v for k, v in kwargs.items() if k not in strip_keys}
            try:
                clip, _ = self.backbone.generate(prompt, **kw)
                return clip
            except TypeError:
                continue
        raise RuntimeError("backbone.generate() rejected all kwarg combinations")

    def _call_i2v(self, image: bytes, prompt: str, **kwargs) -> np.ndarray:
        """Call backbone.generate_i2v() with fallback to generate() if unsupported."""
        i2v_fn = getattr(self.backbone, "generate_i2v", None)
        if i2v_fn is not None:
            for strip_keys in [[], ["width", "height", "reference_images"],
                                ["width", "height", "reference_images", "num_frames"]]:
                kw = {k: v for k, v in kwargs.items() if k not in strip_keys}
                try:
                    clip, _ = i2v_fn(image=image, prompt=prompt, **kw)
                    return clip
                except TypeError:
                    continue

        log.warning("Backbone has no generate_i2v(); falling back to text-to-video")
        return self._call_t2v(prompt, **{
            k: v for k, v in kwargs.items()
            if k not in ("image", "reference_images")
        })


def _frame_to_bytes(frame: np.ndarray) -> bytes:
    """(H, W, C) uint8 → PNG bytes for backbone image conditioning."""
    import io
    try:
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(frame.astype(np.uint8)).save(buf, format="PNG")
        return buf.getvalue()
    except ImportError:
        import struct
        import zlib
        h, w, c = frame.shape
        raw = b''.join(b'\x00' + frame[y].tobytes() for y in range(h))
        def chunk(tag: bytes, data: bytes) -> bytes:
            crc = zlib.crc32(tag + data) & 0xFFFFFFFF
            return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', crc)
        colour_type = 2 if c == 3 else 6
        ihdr = struct.pack('>IIBBBBB', w, h, 8, colour_type, 0, 0, 0)
        return (b'\x89PNG\r\n\x1a\n'
                + chunk(b'IHDR', ihdr)
                + chunk(b'IDAT', zlib.compress(raw))
                + chunk(b'IEND', b''))
