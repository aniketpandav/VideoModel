"""Streaming online video dataset — no local downloads required.

Streams directly from HuggingFace Hub (streaming=True) so clips are fetched
frame-by-frame during training. Includes a lightweight Haar-cascade face filter
so every yielded clip is faceless.

Supported sources (all public, no login required):
  kabr         → legacy metadata source, skipped in streaming mode
  mpala        → imageomics/mmla_mpala    — Drone footage of zebras/giraffes
  wilds_drones → imageomics/thewilds_drones — Safari park drone monitoring
  deepsea      → MBARI-org/DeepSea-MOT    — Deep-sea ROV tracking footage
  pexels       → zengxianyu/open-sora-pexels-subset — curated nature/landscape
  internvid    → OpenGVLab/InternVid-10M-FLT (nature-keyword filtered)
  synthetic    → procedural colour gradients (always works, zero bandwidth)

Quick usage:
    from datasets.online_dataset import OnlineVideoDataset, online_collate_fn
    from torch.utils.data import DataLoader

    ds     = OnlineVideoDataset(sources=["kabr","pexels","synthetic"],
                                 num_frames=16, height=256, width=256)
    loader = DataLoader(ds, batch_size=2, collate_fn=online_collate_fn)
    batch  = next(iter(loader))
    # batch["video"]   → Tensor [B, 3, T, H, W]  float32 in [-1,1]
    # batch["caption"] → list[str]
    # batch["source"]  → list[str]
"""

from __future__ import annotations

import logging
import os
import random
import tempfile
from typing import Iterator

import cv2
import numpy as np
import torch
from torch.utils.data import IterableDataset

# Use our loader that bypasses the local datasets/ namespace conflict
from datasets._hf_loader import load_hf_dataset
from datasets.captions import clean_caption, truncate_caption

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Face detection (OpenCV Haar cascade — ships with opencv-python, no download)
# ─────────────────────────────────────────────────────────────────────────────

_FACE_CASCADE: cv2.CascadeClassifier | None = None


def _get_face_cascade() -> cv2.CascadeClassifier:
    global _FACE_CASCADE
    if _FACE_CASCADE is None:
        xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        _FACE_CASCADE = cv2.CascadeClassifier(xml)
    return _FACE_CASCADE


def _has_face(frame_bgr: np.ndarray, scale: float = 0.25) -> bool:
    """Cheap face check on a downscaled frame. Returns True → skip this clip."""
    h, w = frame_bgr.shape[:2]
    small = cv2.resize(frame_bgr, (max(1, int(w * scale)), max(1, int(h * scale))))
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    faces = _get_face_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
    return len(faces) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Video byte → frame array decoder
# ─────────────────────────────────────────────────────────────────────────────

def _decode_video_bytes(
    video_bytes: bytes | None,
    num_frames: int,
    height: int,
    width: int,
    face_filter: bool,
) -> np.ndarray | None:
    """Decode raw video bytes → [T, H, W, 3] uint8, or None to skip clip."""
    if not video_bytes or len(video_bytes) < 512:
        return None

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_bytes)
            tmp_path = f.name

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return None

        total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        step   = max(1, total // (num_frames * 2))
        frames: list[np.ndarray] = []
        idx    = 0

        while cap.isOpened() and len(frames) < num_frames * 2:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            frames.append(frame)
            idx += step

        cap.release()

        if len(frames) < 2:
            return None

        # Face filter: check middle frame only (fast)
        if face_filter and _has_face(frames[len(frames) // 2]):
            return None

        # Sample exactly num_frames
        if len(frames) >= num_frames:
            idxs   = np.linspace(0, len(frames) - 1, num_frames, dtype=int)
            frames = [frames[i] for i in idxs]
        else:
            while len(frames) < num_frames:
                frames.append(frames[-1])

        # BGR → RGB
        frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        return np.stack(frames, axis=0)          # [T, H, W, 3]

    except Exception as e:
        logger.debug(f"Decode failed: {e}")
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


def _extract_bytes(ex: dict, key: str = "video") -> bytes | None:
    """Pull raw bytes from a HuggingFace dataset example."""
    v = ex.get(key)
    if v is None:
        return None
    if isinstance(v, dict):
        raw = v.get("bytes")
        if isinstance(raw, bytes):
            return raw
        path = v.get("path")
        if path:
            try:
                with open(path, "rb") as f:
                    return f.read()
            except Exception:
                pass
    if isinstance(v, bytes):
        return v
    if hasattr(v, "read"):        # file-like object
        return v.read()
    if hasattr(v, "path") and v.path:   # VideoFile proxy
        try:
            with open(v.path, "rb") as f:
                return f.read()
        except Exception:
            pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-source streaming generators
# ─────────────────────────────────────────────────────────────────────────────

def _iter_kabr(num_frames, height, width, face_filter, max_samples) -> Iterator[dict]:
    """Skip the legacy KABR metadata dataset in online streaming mode."""
    logger.warning(
        "[KABR] imageomics/KABR is not a streamable video-byte dataset with "
        "current Hugging Face Datasets. Use pexels/synthetic online training "
        "or prepare KABR raw videos locally with scripts/prepare_dataset.py."
    )
    return
    yield  # pragma: no cover - keeps this function a generator

def _iter_mpala(num_frames, height, width, face_filter, max_samples) -> Iterator[dict]:
    """Mpala Research Center: drone footage of zebras and giraffes."""
    try:
        ds = load_hf_dataset("imageomics/mmla_mpala", split="train",
                             streaming=True)
        count = 0
        for ex in ds:
            if count >= max_samples:
                break
            vb = _extract_bytes(ex, "video")
            frames = _decode_video_bytes(vb, num_frames, height, width, face_filter)
            if frames is None:
                continue
            caption = "drone footage of wildlife at mpala research center"
            count  += 1
            yield {"frames": frames, "caption": caption, "source": "mpala"}
    except Exception as e:
        logger.warning(f"[Mpala] source failed: {e}")

def _iter_wilds_drones(num_frames, height, width, face_filter, max_samples) -> Iterator[dict]:
    """The Wilds: ecological monitoring drone footage."""
    try:
        ds = load_hf_dataset("imageomics/thewilds_drones", split="train",
                             streaming=True)
        count = 0
        for ex in ds:
            if count >= max_samples:
                break
            vb = _extract_bytes(ex, "video")
            frames = _decode_video_bytes(vb, num_frames, height, width, face_filter)
            if frames is None:
                continue
            caption = "aerial ecological monitoring drone footage of safari park"
            count  += 1
            yield {"frames": frames, "caption": caption, "source": "wilds_drones"}
    except Exception as e:
        logger.warning(f"[Wilds Drones] source failed: {e}")

def _iter_deepsea(num_frames, height, width, face_filter, max_samples) -> Iterator[dict]:
    """DeepSea-MOT: ROV deep sea underwater footage."""
    try:
        ds = load_hf_dataset("MBARI-org/DeepSea-MOT", split="train",
                             streaming=True)
        count = 0
        for ex in ds:
            if count >= max_samples:
                break
            vb = _extract_bytes(ex, "video")
            frames = _decode_video_bytes(vb, num_frames, height, width, face_filter)
            if frames is None:
                continue
            caption = "underwater deep sea exploration footage captured by ROV"
            count  += 1
            yield {"frames": frames, "caption": caption, "source": "deepsea"}
    except Exception as e:
        logger.warning(f"[DeepSea] source failed: {e}")


def _iter_pexels(num_frames, height, width, face_filter, max_samples) -> Iterator[dict]:
    """Open-Sora curated Pexels nature/landscape subset."""
    try:
        ds = load_hf_dataset("zengxianyu/open-sora-pexels-subset", split="train",
                             streaming=True)
        count = 0
        for ex in ds:
            if count >= max_samples:
                break
            # Try multiple ways to get video bytes
            vb = _extract_bytes(ex, "video")
            if vb is None:
                # HF Datasets may return a dict with 'path' pointing to a
                # cached/temporary file that requires torchcodec to decode.
                # Try reading raw bytes from the path directly via cv2.
                v = ex.get("video")
                if v is not None and hasattr(v, "path") and v.path:
                    try:
                        with open(v.path, "rb") as f:
                            vb = f.read()
                    except Exception:
                        pass
                if vb is None:
                    # Try mp4 key (webdataset tar format)
                    vb = _extract_bytes(ex, "mp4")
                if vb is None:
                    continue
            frames = _decode_video_bytes(vb, num_frames, height, width, face_filter)
            if frames is None:
                continue
            caption = str(ex.get("caption", ex.get("text",
                                  "a beautiful nature landscape scene")))
            count += 1
            yield {"frames": frames, "caption": caption, "source": "pexels"}
    except Exception as e:
        logger.warning(f"[Pexels] source failed: {e}")


def _iter_internvid(num_frames, height, width, face_filter, max_samples) -> Iterator[dict]:
    """InternVid-10M filtered to nature/scenery keywords — streams clip URLs on-the-fly."""
    NATURE_KW = {
        "forest", "ocean", "mountain", "sunset", "sunrise", "nature",
        "landscape", "waterfall", "river", "beach", "sky", "cloud",
        "field", "valley", "canyon", "desert", "snow", "glacier",
        "wildlife", "animal", "bird", "tree", "flower", "rain",
        "jungle", "lake", "meadow", "cave", "reef", "wave",
    }
    try:
        import urllib.request
        ds = load_hf_dataset("OpenGVLab/InternVid-10M-FLT", split="train",
                             streaming=True)
        count = 0
        for ex in ds:
            if count >= max_samples:
                break
            caption = str(ex.get("caption", ex.get("text", ""))).lower()
            if not any(kw in caption for kw in NATURE_KW):
                continue
            url = ex.get("video_url") or ex.get("url")
            if not url:
                continue
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0 VideoGen/1.0"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    vb = resp.read()
            except Exception:
                continue
            frames = _decode_video_bytes(vb, num_frames, height, width, face_filter)
            if frames is None:
                continue
            count += 1
            yield {"frames": frames,
                   "caption": ex.get("caption", "nature scenery"),
                   "source": "internvid"}
    except Exception as e:
        logger.warning(f"[InternVid] source failed: {e}")


def _iter_synthetic(num_frames, height, width, max_samples) -> Iterator[dict]:
    """Procedural colour-gradient frames — zero bandwidth, always works."""
    PALETTES = [
        ("forest",  (34, 85),   (80, 180),  (60, 130),
         "sunlight filtering through a lush green forest"),
        ("sky",     (90, 130),  (100, 220), (150, 230),
         "blue sky with drifting white clouds"),
        ("sunset",  (5,  30),   (180, 255), (180, 255),
         "warm golden sunset over the horizon"),
        ("ocean",   (95, 125),  (120, 200), (100, 180),
         "calm ocean waves reflecting blue sky"),
        ("desert",  (15,  35),  (60,  150), (160, 220),
         "sandy desert dunes in soft warm light"),
        ("snow",    (100, 140), (10,  50),  (210, 255),
         "pristine snow-covered mountain peaks"),
        ("jungle",  (40,  75),  (100, 200), (40,  110),
         "dense tropical jungle with dappled sunlight"),
        ("valley",  (25,  65),  (70,  160), (80,  150),
         "a green valley between rolling hills at golden hour"),
    ]
    rng = random.Random(42)
    for i in range(max_samples):
        name, h_range, s_range, v_range, caption = rng.choice(PALETTES)
        base_h = rng.randint(*h_range)
        frames = []
        for t in range(num_frames):
            h_val = (base_h + t * 2) % 180
            s_val = rng.randint(*s_range)
            v_val = rng.randint(*v_range)
            img_hsv = np.zeros((height, width, 3), dtype=np.uint8)
            for row in range(height):
                row_h = (h_val + int(row * 4 / height)) % 180
                row_v = int(np.clip(v_val - row * 25 / height, 40, 255))
                img_hsv[row, :] = [row_h, s_val, row_v]
            img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
            noise   = np.random.randint(-10, 10,
                                        (height, width, 3), dtype=np.int16)
            img_rgb = np.clip(img_rgb.astype(np.int16) + noise,
                              0, 255).astype(np.uint8)
            frames.append(img_rgb)
        yield {"frames": np.stack(frames), "caption": caption, "source": "synthetic"}


# ─────────────────────────────────────────────────────────────────────────────
# Main dataset class
# ─────────────────────────────────────────────────────────────────────────────

class OnlineVideoDataset(IterableDataset):
    """PyTorch IterableDataset that streams faceless video clips from online sources.

    No files are persisted to disk. Data is fetched progressively while the
    training loop iterates — identical interface to VideoTextDataset.

    Args:
        sources:        Ordered list of source names to round-robin.
                        Choose from pexels, internvid, synthetic, and legacy sources.
        num_frames:     Frames per clip (must match model config).
        height:         Frame height in pixels.
        width:          Frame width in pixels.
        face_filter:    Drop clips that contain a detected face (Haar cascade).
        max_per_source: How many clips to pull from each source before cycling.
        limit:          Optional total number of clips exposed by this dataset.
        shuffle_buffer: In-memory shuffle buffer size.
        seed:           RNG seed for reproducibility.
    """

    AVAILABLE_SOURCES = ("kabr", "mpala", "wilds_drones", "deepsea", "pexels", "internvid", "synthetic")

    def __init__(
        self,
        sources: list[str] | tuple[str, ...] = ("pexels", "synthetic"),
        num_frames: int = 16,
        height: int = 256,
        width: int = 256,
        face_filter: bool = True,
        max_per_source: int = 5_000,
        limit: int | None = None,
        shuffle_buffer: int = 256,
        seed: int = 42,
    ):
        super().__init__()
        # Allow env-var override for CLI --sources flag
        env_src = os.environ.get("VIDEOGEN_ONLINE_SOURCES")
        if env_src:
            import json
            sources = json.loads(env_src)

        env_limit = os.environ.get("VIDEOGEN_ONLINE_LIMIT")
        if env_limit and limit is None:
            limit = int(env_limit)
        if limit is not None and limit <= 0:
            raise ValueError("Online dataset limit must be a positive integer")

        self.sources       = [s for s in sources if s in self.AVAILABLE_SOURCES]
        self.num_frames    = num_frames
        self.height        = height
        self.width         = width
        self.face_filter   = face_filter
        self.max_per_source = max_per_source
        self.limit         = limit
        clip_bytes = max(1, num_frames * height * width * 3)
        max_buffer_by_memory = max(1, (64 * 1024 * 1024) // clip_bytes)
        self.shuffle_buffer = min(
            shuffle_buffer,
            max_buffer_by_memory,
            limit if limit is not None else shuffle_buffer,
        )
        self.seed          = seed

        if not self.sources:
            logger.warning("No valid sources — falling back to synthetic")
            self.sources = ["synthetic"]

        logger.info(
            f"OnlineVideoDataset | sources={self.sources} | "
            f"{num_frames}f @ {height}×{width} | "
            f"face_filter={face_filter} | max_per_source={max_per_source} | "
            f"limit={limit} | shuffle_buffer={self.shuffle_buffer}"
        )

    # ── source routing ──────────────────────────────────────────────────────

    def _source_iter(self, source: str) -> Iterator[dict]:
        kw = dict(num_frames=self.num_frames, height=self.height,
                  width=self.width, max_samples=self.max_per_source)
        if source == "kabr":
            yield from _iter_kabr(**kw, face_filter=self.face_filter)
        elif source == "mpala":
            yield from _iter_mpala(**kw, face_filter=self.face_filter)
        elif source == "wilds_drones":
            yield from _iter_wilds_drones(**kw, face_filter=self.face_filter)
        elif source == "deepsea":
            yield from _iter_deepsea(**kw, face_filter=self.face_filter)
        elif source == "pexels":
            yield from _iter_pexels(**kw, face_filter=self.face_filter)
        elif source == "internvid":
            yield from _iter_internvid(**kw, face_filter=self.face_filter)
        elif source == "synthetic":
            yield from _iter_synthetic(**kw)

    # ── tensor conversion ───────────────────────────────────────────────────

    @staticmethod
    def _to_tensor(frames: np.ndarray) -> torch.Tensor:
        """[T,H,W,3] uint8 → [3,T,H,W] float32 in [-1, 1]."""
        t = torch.from_numpy(frames.copy()).float() / 127.5 - 1.0
        return t.permute(3, 0, 1, 2).contiguous()

    # ── main iterator ────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[dict]:
        iters     = [self._source_iter(s) for s in self.sources]
        exhausted = [False] * len(iters)
        rng       = random.Random(self.seed)
        buf: list[dict] = []
        fetched = 0
        yielded = 0

        def _try_fill(src_idx: int) -> bool:
            nonlocal fetched
            if self.limit is not None and fetched >= self.limit:
                return False
            if exhausted[src_idx]:
                return False
            try:
                buf.append(next(iters[src_idx]))
                fetched += 1
                return True
            except StopIteration:
                exhausted[src_idx] = True
                logger.info(f"Source '{self.sources[src_idx]}' exhausted")
                return False

        # Prime the buffer
        targets = min(self.shuffle_buffer, self.max_per_source * len(iters))
        if self.limit is not None:
            targets = min(targets, self.limit)
        for _ in range(targets):
            active = [i for i, ex in enumerate(exhausted) if not ex]
            if not active:
                break
            _try_fill(rng.choice(active))

        while buf and (self.limit is None or yielded < self.limit):
            pick = rng.randrange(len(buf))
            item = buf.pop(pick)

            yielded += 1
            yield {
                "video":   self._to_tensor(item["frames"]),
                "caption": clean_caption(truncate_caption(item["caption"])),
                "source":  item.get("source", "unknown"),
            }

            # Refill one slot
            active = [i for i, ex in enumerate(exhausted) if not ex]
            if active:
                _try_fill(rng.choice(active))

        if yielded == 0 and "synthetic" not in self.sources:
            fallback_count = self.limit if self.limit is not None else self.max_per_source
            logger.warning(
                "No configured online source yielded any clips; falling back "
                "to synthetic clips so the training epoch is not empty."
            )
            for item in _iter_synthetic(
                self.num_frames, self.height, self.width, fallback_count
            ):
                yield {
                    "video": self._to_tensor(item["frames"]),
                    "caption": item["caption"],
                    "source": "synthetic_fallback",
                }


# ─────────────────────────────────────────────────────────────────────────────
# DataLoader collate
# ─────────────────────────────────────────────────────────────────────────────

def online_collate_fn(batch: list[dict]) -> dict:
    """Stack a list of OnlineVideoDataset items into a training batch."""
    return {
        "video":   torch.stack([b["video"]   for b in batch]),
        "caption": [b["caption"] for b in batch],
        "source":  [b.get("source", "?") for b in batch],
    }
