"""Unified video-generation backbone interface.

The API and pipeline depend ONLY on `Backbone.generate()` / `Backbone.generate_i2v()`.
Swapping the backbone is a one-line env-var change — no caller changes needed.

Track A backbones (production, cloud GPU 8 GB+):
  ltx        → Lightricks/LTX-Video  (Apache-2.0, Mode 1 + Mode 2)
  cogvideox  → THUDM/CogVideoX-5b-I2V (Mode 2 specialist)

Track B backbone (local GTX 1650 demo):
  toy        → From-scratch 3D U-Net DDPM checkpoint

Factory: get_backbone(kind) or BackboneFactory.create(kind, device)
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
from abc import ABC, abstractmethod

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class Backbone(ABC):
    name: str = "backbone"

    # Subclasses override to declare what they support.
    # Used by the job executor to skip checks that are always wrong for a given backbone.
    capabilities: dict = {
        "dynamic_resolution": True,   # can produce arbitrary output resolution
        "dynamic_duration": True,     # can produce arbitrary frame counts
        "image_conditioning": False,  # supports image-to-video (generate_i2v)
    }

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        num_frames: int | None = None,
        steps: int = 50,
        seed: int = 0,
        width: int | None = None,
        height: int | None = None,
        first_frame: bytes | None = None,
    ) -> tuple[np.ndarray, int]:
        """Return (clip, fps) where clip is (T, H, W, C) uint8."""

    def generate_i2v(
        self,
        *,
        image: bytes,
        prompt: str,
        num_frames: int | None = None,
        steps: int = 50,
        seed: int = 0,
        width: int | None = None,
        height: int | None = None,
        reference_images: list[bytes] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Image-to-video generation. Defaults to text-to-video fallback."""
        _ = image, reference_images  # unused in base fallback; subclasses consume them
        log.warning("%s: generate_i2v() not implemented; falling back to t2v", self.name)
        return self.generate(prompt, num_frames=num_frames, steps=steps, seed=seed,
                             width=width, height=height)


# ---------------------------------------------------------------------------
# Track B: toy 3D U-Net DDPM checkpoint (runs on GTX 1650)
# ---------------------------------------------------------------------------

class ToyBackbone(Backbone):
    """Serves the trained 3D U-Net DDPM checkpoint (EMA weights).

    Class-conditional on motion direction (left/right/up/down), not text.
    Prompt keywords are mapped to direction labels as a demo bridge.

    Output is always (cfg.frames, cfg.image_size, cfg.image_size) — fixed by the
    training config. Resolution and duration requests are ignored.
    """
    DIRECTIONS = {"left": 0, "right": 1, "up": 2, "down": 3}
    capabilities: dict = {
        "dynamic_resolution": False,  # fixed at cfg.image_size (e.g. 32px)
        "dynamic_duration": False,    # fixed at cfg.frames (e.g. 8 or 16)
        "image_conditioning": False,
    }

    def __init__(self, ckpt_path: str, device: str | None = None):
        import torch
        from vdm import GaussianDiffusion, UNet3D, seed_everything

        self.name = "toy"
        self._torch = torch
        self._seed_everything = seed_everything

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path!r}. Train one first:\n"
                "  python scripts/train.py --config configs/train_local.yaml"
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        cfg = ckpt["cfg"]
        m = cfg["model"]

        net = UNet3D(
            in_ch=cfg["channels"], base=m["base"], ch_mult=tuple(m["ch_mult"]),
            num_res_blocks=m["num_res_blocks"], attn_resolutions=tuple(m["attn_resolutions"]),
            heads=m["heads"], dropout=0.0, num_classes=cfg["num_classes"],
            image_size=cfg["image_size"], use_checkpoint=False,
        ).to(self.device)

        # Load EMA weights for best sample quality
        msd = net.state_dict()
        for k, v in ckpt["ema"].items():
            if k in msd:
                msd[k].copy_(v.to(self.device))
        net.eval()

        self.net = net
        self.diffusion = GaussianDiffusion(
            net, timesteps=cfg["diffusion"]["timesteps"],
            schedule=cfg["diffusion"]["schedule"],
            predict=cfg["diffusion"].get("predict", "v"),
        ).to(self.device)
        self.cfg = cfg
        self.step = int(ckpt.get("step", -1)) + 1

    def _prompt_to_label(self, prompt: str) -> int | None:
        if not self.cfg["num_classes"]:
            return None
        p = (prompt or "").lower()
        for word, lbl in self.DIRECTIONS.items():
            if word in p:
                return lbl
        h = int(hashlib.sha1(p.encode()).hexdigest(), 16)
        return h % self.cfg["num_classes"]

    def generate(self, prompt, *, num_frames=None, steps=50, seed=0,
                 width=None, height=None, first_frame=None):
        _ = num_frames, width, height, first_frame  # ToyBackbone uses fixed size from config
        torch = self._torch
        self._seed_everything(seed)
        cfg = self.cfg
        label = self._prompt_to_label(prompt)
        y = torch.tensor([label], device=self.device) if label is not None else None
        shape = (1, cfg["channels"], cfg["frames"], cfg["image_size"], cfg["image_size"])
        with torch.no_grad():
            samples = self.diffusion.ddim_sample(shape, y=y, steps=steps, device=self.device)
        from .render import finish_clip, to_uint8_clip
        clip = finish_clip(to_uint8_clip(samples))
        fps = max(4, cfg["frames"] // 2)
        return clip, fps


# ---------------------------------------------------------------------------
# Track A: LTX-Video (Lightricks, Apache-2.0) — production text-to-video
# ---------------------------------------------------------------------------

class LTXVideoBackbone(Backbone):
    """Wraps Lightricks/LTX-Video for both text-to-video and image-to-video.

    Requires a cloud GPU with ≥8 GB VRAM and:
      pip install 'diffusers>=0.32' transformers accelerate sentencepiece
    """
    capabilities: dict = {
        "dynamic_resolution": True,
        "dynamic_duration": True,
        "image_conditioning": True,
    }

    def __init__(
        self,
        model_id: str = "Lightricks/LTX-Video",
        device: str | None = None,
    ):
        self.name = "ltx"
        self.model_id = model_id

        try:
            import torch
            from diffusers import LTXPipeline, LTXImageToVideoPipeline
        except ImportError as e:
            raise RuntimeError(
                "LTXVideoBackbone requires: pip install 'diffusers>=0.32' "
                "transformers accelerate sentencepiece"
            ) from e

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        log.info("Loading LTX-Video t2v pipeline from %s …", model_id)
        self._t2v = LTXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self._t2v.enable_model_cpu_offload()
        self._t2v.vae.enable_tiling()

        # I2V variant — same weights, different pipeline
        try:
            log.info("Loading LTX-Video i2v pipeline …")
            self._i2v = LTXImageToVideoPipeline.from_pretrained(
                model_id, torch_dtype=torch.bfloat16
            )
            self._i2v.enable_model_cpu_offload()
            self._i2v.vae.enable_tiling()
        except Exception as exc:
            log.warning("LTX I2V pipeline unavailable (%s); i2v will fall back to t2v", exc)
            self._i2v = None

    def generate(self, prompt, *, num_frames=None, steps=50, seed=0,
                 width=None, height=None, first_frame=None):
        torch = self._torch
        nf = _ltx_frame_count(num_frames)
        w = width or 704
        h = height or 480
        g = torch.Generator(device="cpu").manual_seed(seed)

        kwargs: dict = dict(
            prompt=prompt, num_frames=nf, num_inference_steps=steps,
            width=w, height=h, generator=g,
        )
        if first_frame is not None:
            kwargs["image"] = _bytes_to_pil(first_frame)

        out = self._t2v(**kwargs)
        clip = _frames_to_clip(out.frames[0])
        return finish_clip(clip), 24

    def generate_i2v(self, *, image: bytes, prompt: str, num_frames=None,
                     steps=50, seed=0, width=None, height=None,
                     reference_images=None):
        _ = reference_images  # multi-image style blending is a Phase 2 enhancement
        if self._i2v is None:
            return self.generate(prompt, num_frames=num_frames, steps=steps,
                                 seed=seed, width=width, height=height,
                                 first_frame=image)

        torch = self._torch
        nf = _ltx_frame_count(num_frames)
        w = width or 704
        h = height or 480
        g = torch.Generator(device="cpu").manual_seed(seed)
        pil_image = _bytes_to_pil(image)

        out = self._i2v(
            image=pil_image, prompt=prompt, num_frames=nf,
            num_inference_steps=steps, width=w, height=h, generator=g,
        )
        clip = _frames_to_clip(out.frames[0])
        return finish_clip(clip), 24


# ---------------------------------------------------------------------------
# Track A: CogVideoX-5b-I2V (THUDM) — image-to-video specialist
# ---------------------------------------------------------------------------

class CogVideoXBackbone(Backbone):
    """Wraps THUDM/CogVideoX-5b-I2V for high-quality image-to-video generation.

    Specialised for Mode 2. Falls back to text-to-video via CogVideoX-5b for Mode 1.
    Requires: pip install 'diffusers>=0.32' transformers accelerate
    """
    capabilities: dict = {
        "dynamic_resolution": True,
        "dynamic_duration": True,
        "image_conditioning": True,
    }

    def __init__(
        self,
        i2v_model_id: str = "THUDM/CogVideoX-5b-I2V",
        t2v_model_id: str = "THUDM/CogVideoX-5b",
        device: str | None = None,
    ):
        self.name = "cogvideox"

        try:
            import torch
            from diffusers import CogVideoXImageToVideoPipeline
        except ImportError as e:
            raise RuntimeError(
                "CogVideoXBackbone requires: pip install 'diffusers>=0.32' transformers accelerate"
            ) from e

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        log.info("Loading CogVideoX I2V pipeline from %s …", i2v_model_id)
        self._i2v_pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            i2v_model_id, torch_dtype=torch.bfloat16
        )
        self._i2v_pipe.enable_model_cpu_offload()
        self._i2v_pipe.vae.enable_tiling()

        # T2V pipeline loaded lazily (Mode 1 fallback)
        self._t2v_model_id = t2v_model_id
        self._t2v_pipe = None

    def _ensure_t2v(self):
        if self._t2v_pipe is None:
            from diffusers import CogVideoXPipeline
            log.info("Loading CogVideoX T2V pipeline from %s …", self._t2v_model_id)
            self._t2v_pipe = CogVideoXPipeline.from_pretrained(
                self._t2v_model_id, torch_dtype=self._torch.bfloat16
            )
            self._t2v_pipe.enable_model_cpu_offload()

    def generate(self, prompt, *, num_frames=None, steps=50, seed=0,
                 width=None, height=None, first_frame=None):
        _ = first_frame  # CogVideoX t2v does not support first-frame conditioning
        self._ensure_t2v()
        torch = self._torch
        nf = num_frames or 49   # CogVideoX default
        w = width or 720
        h = height or 480
        g = torch.Generator(device="cpu").manual_seed(seed)
        out = self._t2v_pipe(
            prompt=prompt, num_frames=nf, num_inference_steps=steps,
            width=w, height=h, generator=g,
        )
        clip = _frames_to_clip(out.frames[0])
        return finish_clip(clip), 8

    def generate_i2v(self, *, image: bytes, prompt: str, num_frames=None,
                     steps=50, seed=0, width=None, height=None,
                     reference_images=None):
        _ = reference_images  # multi-image style blending is a Phase 2 enhancement
        torch = self._torch
        nf = num_frames or 49
        w = width or 720
        h = height or 480
        g = torch.Generator(device="cpu").manual_seed(seed)
        pil_image = _bytes_to_pil(image)
        out = self._i2v_pipe(
            image=pil_image, prompt=prompt, num_frames=nf,
            num_inference_steps=steps, width=w, height=h, generator=g,
        )
        clip = _frames_to_clip(out.frames[0])
        return finish_clip(clip), 8


# ---------------------------------------------------------------------------
# Shared render helpers (re-exported for backward compat)
# ---------------------------------------------------------------------------

def finish_clip(clip: np.ndarray) -> np.ndarray:
    """Delegate to render.finish_clip() which handles Real-ESRGAN upscaling."""
    try:
        from .render import finish_clip as _finish
        return _finish(clip)
    except Exception:
        return clip


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ltx_frame_count(n: int | None) -> int:
    """LTX-Video requires frame counts of the form 8k+1. Round up if needed."""
    n = n or 97
    remainder = (n - 1) % 8
    return n if remainder == 0 else n + (8 - remainder)


def _bytes_to_pil(data: bytes):
    """Decode raw PNG/JPEG bytes into a PIL Image."""
    from PIL import Image
    return Image.open(io.BytesIO(data)).convert("RGB")


def _frames_to_clip(frames) -> np.ndarray:
    """Convert a list of PIL Images to (T, H, W, C) uint8 ndarray."""
    import numpy as np
    return np.stack([np.asarray(f.convert("RGB")) for f in frames]).astype(np.uint8)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class BackboneFactory:
    @staticmethod
    def create(kind: str, device: str | None = None, **kwargs) -> Backbone:
        kind = kind.lower().strip()
        if kind == "toy":
            ckpt = kwargs.get("ckpt_path") or os.environ.get("VDM_CKPT", "runs/local/last.pt")
            return ToyBackbone(ckpt_path=ckpt, device=device)
        if kind == "ltx":
            model_id = kwargs.get("model_id", "Lightricks/LTX-Video")
            return LTXVideoBackbone(model_id=model_id, device=device)
        if kind in ("cogvideox", "cogvx"):
            return CogVideoXBackbone(device=device)
        raise ValueError(f"Unknown backbone {kind!r}. Valid: toy | ltx | cogvideox")


def get_backbone(kind: str | None = None, **kw) -> Backbone:
    """Backward-compatible factory. Reads VDM_BACKBONE env if kind is None."""
    kind = (kind or os.environ.get("VDM_BACKBONE", "toy")).lower()
    device = kw.get("device") or os.environ.get("VDM_DEVICE")
    return BackboneFactory.create(kind, device=device, **kw)
