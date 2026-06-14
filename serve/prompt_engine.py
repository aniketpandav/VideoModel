"""Prompt intelligence layer.

Responsibilities:
  1. Input validation (prompt length, encoding, image type/size)
  2. Intent detection: Mode 1 (text→video) vs Mode 2 (image+text→video)
  3. Safety filtering: keyword blocklist + optional Claude moderation
  4. Prompt enhancement: rewrite for cinematic quality via Claude API
  5. Parameter derivation: extract aspect ratio, quality level, motion hints
  6. Narrative expansion: split master prompt into N sequential scenes for long video

Claude API is optional. All operations degrade gracefully when ANTHROPIC_API_KEY
is absent — prompts pass through unmodified, moderation is skipped.
"""
from __future__ import annotations

import base64
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety: hard keyword blocklist (checked before any LLM call)
# ---------------------------------------------------------------------------
_BLOCKLIST = frozenset([
    "csam", "child pornography", "child sexual", "minor sexual",
    "snuff", "real murder", "real death video", "beheading",
    "bomb making", "drug synthesis", "meth recipe",
])

# ---------------------------------------------------------------------------
# Keyword → parameter derivation tables
# ---------------------------------------------------------------------------
_ASPECT_KEYWORDS: dict[str, str] = {
    "portrait": "9:16", "vertical": "9:16", "phone": "9:16",
    "landscape": "16:9", "widescreen": "16:9", "wide shot": "16:9",
    "square": "1:1", "instagram": "1:1",
    "4:3": "4:3",
}

_QUALITY_KEYWORDS: dict[str, str] = {
    "cinematic": "cinematic", "film": "high", "4k": "high", "8k": "high",
    "photorealistic": "high", "hd": "high", "high quality": "high",
    "high-quality": "high",
    "draft": "draft", "preview": "draft", "quick": "draft", "fast": "draft",
}

# Soft text hints that suggest Mode 2 (image-to-video) even without an image attached
_I2V_HINTS = frozenset([
    "animate this", "from this image", "based on this photo",
    "bring to life", "make this move", "animate the image",
    "animate my photo", "from the picture", "from the photo",
    "from the image", "make it move",
])

# ---------------------------------------------------------------------------
# Claude API system prompts
# ---------------------------------------------------------------------------
_ENHANCE_SYSTEM = (
    "You are a professional video generation prompt engineer. "
    "Rewrite the user's prompt to be cinematically detailed. "
    "Include: shot type (close-up/wide/aerial/POV), lighting quality (golden hour/"
    "neon/overcast/studio), camera motion (pan/tilt/dolly/handheld/static), "
    "color palette, temporal progression (what changes over time), mood, and visual style. "
    "Keep the rewritten prompt under 200 words. "
    "Return ONLY the rewritten prompt, no preamble or explanation."
)

_SAFETY_SYSTEM = (
    "You are a content safety classifier for a video generation service. "
    "Reply ONLY with the single word 'SAFE' or 'UNSAFE'. "
    "UNSAFE means the prompt explicitly requests CSAM, real acts of violence, "
    "detailed instructions for illegal weapons or drug synthesis, or terrorism. "
    "Creative fiction, action movies, horror themes, and mature themes are SAFE. "
    "When in doubt, reply SAFE."
)

_SCENE_EXPAND_SYSTEM = (
    "You are a video storyboard writer. Given a master prompt and a number N, "
    "write N brief scene descriptions (one per line) that together tell a "
    "coherent, visually consistent story. Each scene should flow naturally into "
    "the next, maintaining consistent visual style and subject. "
    "Return exactly N lines. No numbering, bullets, or explanations."
)

MAX_PROMPT_LEN = 2_000
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EnhancedRequest:
    mode: int                           # 1 = text→video, 2 = image+text→video
    original_prompt: str
    enhanced_prompt: str
    duration_seconds: float
    output_format: str
    resolution: tuple[int, int]         # (width, height)
    fps: int
    quality: str                        # draft | standard | high | cinematic
    seed: Optional[int]
    reference_images: list[bytes] = field(default_factory=list)
    inference_steps: int = 50
    aspect_ratio: str = "16:9"


class PromptValidationError(ValueError):
    """Raised for malformed or oversized inputs."""


class SafetyError(ValueError):
    """Raised when content fails safety checks."""


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PromptEngine:
    """Stateless prompt intelligence layer.

    Instantiate once at startup and call process_text_to_video() /
    process_image_to_video() per request.
    """

    def __init__(self, anthropic_api_key: str | None = None):
        self._api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None
        if self._api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
                log.info("PromptEngine: Claude API client initialised (enhancement enabled)")
            except ImportError:
                log.warning("PromptEngine: 'anthropic' package not installed; enhancement disabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_text_to_video(
        self,
        prompt: str,
        *,
        duration_seconds: float = 5.0,
        output_format: str = "mp4",
        resolution: str = "512x512",
        aspect_ratio: str = "16:9",
        fps: int = 24,
        quality: str = "standard",
        seed: Optional[int] = None,
        enhance_prompt: bool = True,
    ) -> EnhancedRequest:
        self._validate_prompt(prompt)
        self._safety_check_text(prompt)
        mode = self._detect_mode_from_text(prompt)
        effective_aspect = self._derive_aspect(prompt, aspect_ratio)
        w, h = self._parse_resolution(resolution, effective_aspect)
        effective_quality = self._derive_quality(prompt, quality)
        enhanced = (
            self._enhance(prompt, mode=1, duration=duration_seconds)
            if enhance_prompt else prompt
        )
        return EnhancedRequest(
            mode=mode,
            original_prompt=prompt,
            enhanced_prompt=enhanced,
            duration_seconds=duration_seconds,
            output_format=output_format,
            resolution=(w, h),
            fps=fps,
            quality=effective_quality,
            seed=seed,
            reference_images=[],
            inference_steps=self._quality_to_steps(effective_quality),
            aspect_ratio=effective_aspect,
        )

    def process_image_to_video(
        self,
        prompt: str,
        reference_images_b64: list[str],
        *,
        duration_seconds: float = 5.0,
        output_format: str = "mp4",
        resolution: str = "512x512",
        fps: int = 24,
        quality: str = "standard",
        seed: Optional[int] = None,
        enhance_prompt: bool = True,
    ) -> EnhancedRequest:
        self._validate_prompt(prompt)
        self._safety_check_text(prompt)
        if not reference_images_b64:
            raise PromptValidationError("At least one reference image is required for Mode 2.")
        images = [self._decode_image(b64) for b64 in reference_images_b64]
        for img in images:
            self._validate_image(img)
        w, h = self._parse_resolution(resolution, "16:9")
        effective_quality = self._derive_quality(prompt, quality)
        enhanced = (
            self._enhance(prompt, mode=2, duration=duration_seconds)
            if enhance_prompt else prompt
        )
        return EnhancedRequest(
            mode=2,
            original_prompt=prompt,
            enhanced_prompt=enhanced,
            duration_seconds=duration_seconds,
            output_format=output_format,
            resolution=(w, h),
            fps=fps,
            quality=effective_quality,
            seed=seed,
            reference_images=images,
            inference_steps=self._quality_to_steps(effective_quality),
            aspect_ratio="16:9",
        )

    def expand_narrative(self, master_prompt: str, n_scenes: int) -> list[str]:
        """Split master_prompt into n_scenes sequential scene descriptions."""
        if n_scenes <= 1:
            return [master_prompt]
        if self._client is None:
            # Fallback: append scene number to maintain basic temporal tagging
            return [f"{master_prompt}, scene {i + 1} of {n_scenes}" for i in range(n_scenes)]
        try:
            msg = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                system=_SCENE_EXPAND_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": f"Master prompt: {master_prompt}\nNumber of scenes: {n_scenes}",
                }],
            )
            lines = [l.strip() for l in msg.content[0].text.strip().split("\n") if l.strip()]
            # Pad or trim to exactly n_scenes
            while len(lines) < n_scenes:
                lines.append(master_prompt)
            return lines[:n_scenes]
        except Exception as exc:
            log.warning("expand_narrative API call failed (%s); using fallback", exc)
            return [f"{master_prompt}, scene {i + 1} of {n_scenes}" for i in range(n_scenes)]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_prompt(prompt: str):
        if not prompt or not prompt.strip():
            raise PromptValidationError("Prompt must not be empty.")
        if len(prompt) > MAX_PROMPT_LEN:
            raise PromptValidationError(
                f"Prompt length {len(prompt)} exceeds maximum {MAX_PROMPT_LEN} characters."
            )
        try:
            prompt.encode("utf-8")
        except UnicodeEncodeError:
            raise PromptValidationError("Prompt contains invalid UTF-8 characters.")

    def _safety_check_text(self, prompt: str):
        lower = prompt.lower()
        for term in _BLOCKLIST:
            if term in lower:
                raise SafetyError("Prompt contains blocked content and cannot be processed.")
        if self._client:
            try:
                resp = self._client.messages.create(
                    model="claude-haiku-4-5-20251001",
                    max_tokens=10,
                    system=_SAFETY_SYSTEM,
                    messages=[{"role": "user", "content": prompt}],
                )
                verdict = resp.content[0].text.strip().upper()
                if verdict == "UNSAFE":
                    raise SafetyError("Prompt failed content safety review.")
            except SafetyError:
                raise
            except Exception as exc:
                log.warning("Safety check API call failed (%s); allowing request", exc)

    @staticmethod
    def _detect_mode_from_text(prompt: str) -> int:
        lower = prompt.lower()
        for hint in _I2V_HINTS:
            if hint in lower:
                return 2
        return 1

    @staticmethod
    def _decode_image(b64: str) -> bytes:
        # Strip data URI prefix if present (e.g. "data:image/png;base64,...")
        if "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        try:
            return base64.b64decode(b64)
        except Exception:
            raise PromptValidationError("Reference image is not valid base64.")

    @staticmethod
    def _validate_image(data: bytes):
        if len(data) > MAX_IMAGE_BYTES:
            raise PromptValidationError(
                f"Reference image exceeds {MAX_IMAGE_BYTES // 1024 // 1024} MB limit."
            )
        valid_magic = (
            data[:8] == b'\x89PNG\r\n\x1a\n'      # PNG
            or data[:3] == b'\xff\xd8\xff'          # JPEG
            or data[:4] == b'RIFF'                  # WebP / BMP (RIFF container)
            or data[:6] in (b'GIF87a', b'GIF89a')   # GIF
        )
        if not valid_magic:
            raise PromptValidationError(
                "Reference image must be PNG, JPEG, GIF, or WebP."
            )

    def _enhance(self, prompt: str, mode: int, duration: float) -> str:
        if self._client is None:
            return prompt
        mode_hint = "text-to-video generation" if mode == 1 else "image-to-video animation"
        try:
            msg = self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=350,
                system=_ENHANCE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Original prompt: {prompt}\n"
                        f"Target duration: {duration:.1f}s\n"
                        f"Generation mode: {mode_hint}"
                    ),
                }],
            )
            enhanced = msg.content[0].text.strip()
            log.debug("Prompt enhanced: %d → %d chars", len(prompt), len(enhanced))
            return enhanced
        except Exception as exc:
            log.warning("Prompt enhancement failed (%s); using original", exc)
            return prompt

    @staticmethod
    def _parse_resolution(resolution: str, aspect_ratio: str) -> tuple[int, int]:
        try:
            parts = resolution.lower().replace("×", "x").replace(" ", "").split("x")
            if len(parts) == 2:
                w, h = int(parts[0]), int(parts[1])
                if w > 0 and h > 0:
                    return w, h
        except (ValueError, IndexError):
            pass
        # Derive sensible defaults from aspect ratio
        _defaults: dict[str, tuple[int, int]] = {
            "16:9": (704, 480), "9:16": (480, 704),
            "1:1": (512, 512), "4:3": (640, 480),
        }
        return _defaults.get(aspect_ratio, (512, 512))

    @staticmethod
    def _derive_quality(prompt: str, requested: str) -> str:
        if requested == "draft":
            return "draft"  # never upgrade from explicit draft
        lower = prompt.lower()
        for kw, q in _QUALITY_KEYWORDS.items():
            if kw in lower:
                # Map to the stronger of detected vs requested
                order = ["draft", "standard", "high", "cinematic"]
                detected_idx = order.index(q) if q in order else 1
                requested_idx = order.index(requested) if requested in order else 1
                return order[max(detected_idx, requested_idx)]
        return requested

    @staticmethod
    def _derive_aspect(prompt: str, requested: str) -> str:
        lower = prompt.lower()
        for kw, ratio in _ASPECT_KEYWORDS.items():
            if kw in lower:
                return ratio
        return requested

    @staticmethod
    def _quality_to_steps(quality: str) -> int:
        return {"draft": 15, "standard": 50, "high": 75, "cinematic": 100}.get(quality, 50)
