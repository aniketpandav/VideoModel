"""Unified video-generation backbone interface.

The API and website depend ONLY on `Backbone.generate(...)`. Swapping the toy model
for a pretrained LTX-Video model on a cloud GPU is a one-line config change
(`VDM_BACKBONE=ltx`) — no caller changes.
"""
from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod

import numpy as np


class Backbone(ABC):
    name: str = "backbone"

    @abstractmethod
    def generate(self, prompt: str, *, num_frames: int | None = None,
                 steps: int = 50, seed: int = 0) -> tuple[np.ndarray, int]:
        """Return (clip, fps) where clip is a (T,H,W,C) uint8 array."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Track B: the from-scratch toy model (runs on the local GTX 1650).            #
# --------------------------------------------------------------------------- #
class ToyBackbone(Backbone):
    """Serves the trained 3D U-Net DDPM checkpoint (EMA weights).

    The toy is class-conditional on motion *direction* (left/right/up/down), not text.
    As a bridge until LTX is wired, we map prompt keywords -> a direction label so the
    same `prompt` API works end-to-end. This is a demo, not production T2V.
    """
    DIRECTIONS = {"left": 0, "right": 1, "up": 2, "down": 3}

    def __init__(self, ckpt_path: str, device: str | None = None):
        import torch
        from vdm import GaussianDiffusion, UNet3D, seed_everything

        self.name = "toy"
        self._torch = torch
        self._seed_everything = seed_everything
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint at {ckpt_path}. Train one first:\n"
                f"  python scripts/train.py --config configs/train_local.yaml")
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
        msd = net.state_dict()  # load EMA weights for best quality
        for k, v in ckpt["ema"].items():
            if k in msd:
                msd[k].copy_(v.to(self.device))
        net.eval()
        self.net = net
        self.diffusion = GaussianDiffusion(
            net, timesteps=cfg["diffusion"]["timesteps"],
            schedule=cfg["diffusion"]["schedule"],
            predict=cfg["diffusion"].get("predict", "v")).to(self.device)
        self.cfg = cfg
        self.step = int(ckpt.get("step", -1)) + 1

    def _prompt_to_label(self, prompt: str) -> int | None:
        if not self.cfg["num_classes"]:
            return None
        p = (prompt or "").lower()
        for word, lbl in self.DIRECTIONS.items():
            if word in p:
                return lbl
        # deterministic fallback so the same prompt always yields the same direction
        h = int(hashlib.sha1(p.encode()).hexdigest(), 16)
        return h % self.cfg["num_classes"]

    def generate(self, prompt, *, num_frames=None, steps=50, seed=0):
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


# --------------------------------------------------------------------------- #
# Track A: pretrained LTX-Video (production quality; needs a cloud GPU 8 GB+). #
# --------------------------------------------------------------------------- #
class LTXBackbone(Backbone):
    """Wraps Lightricks/LTX-Video (Apache-2.0). Lazy-imports diffusers so the API
    starts without it. This is the engine the website ships in production."""

    def __init__(self, model_id: str = "Lightricks/LTX-Video", device: str | None = None):
        self.name = "ltx"
        self.model_id = model_id
        try:
            import torch
            from diffusers import LTXPipeline
        except ImportError as e:
            raise RuntimeError(
                "LTXBackbone needs a cloud GPU with: pip install "
                "'diffusers>=0.32' transformers accelerate sentencepiece. "
                "The local 4 GB GTX 1650 cannot serve this model.") from e
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = LTXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload()   # keep only the active module on GPU
        self.pipe.vae.enable_tiling()          # decode big frames in tiles

    def generate(self, prompt, *, num_frames=None, steps=50, seed=0):
        import numpy as np
        torch = self._torch
        g = torch.Generator(device=self.device).manual_seed(seed)
        nf = num_frames or 97  # LTX likes 8k+1
        out = self.pipe(prompt=prompt, num_frames=nf, num_inference_steps=steps,
                        width=704, height=480, generator=g)
        frames = out.frames[0]  # list[PIL]
        clip = np.stack([np.asarray(f.convert("RGB")) for f in frames])
        from .render import finish_clip
        return finish_clip(clip), 24


# --------------------------------------------------------------------------- #
def get_backbone(kind: str | None = None, **kw) -> Backbone:
    """Factory. `kind` defaults to env VDM_BACKBONE (toy|ltx), else 'toy'."""
    kind = (kind or os.environ.get("VDM_BACKBONE", "toy")).lower()
    device = kw.get("device") or os.environ.get("VDM_DEVICE")  # e.g. "cpu" while GPU trains
    if kind == "toy":
        ckpt = kw.get("ckpt_path") or os.environ.get("VDM_CKPT", "runs/local/last.pt")
        return ToyBackbone(ckpt_path=ckpt, device=device)
    if kind == "ltx":
        return LTXBackbone(model_id=kw.get("model_id", "Lightricks/LTX-Video"),
                           device=device)
    raise ValueError(f"Unknown backbone: {kind!r} (use 'toy' or 'ltx')")
