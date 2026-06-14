"""Training data preprocessing pipeline.

Steps
-----
  1. Scan the raw data directory for video files
  2. Filter: duration 2–30s, resolution ≥ target, no pure-black clips
  3. Resize to target resolution (Lanczos)
  4. Extract frames at target FPS
  5. Auto-caption using BLIP-2 (or manual captions if provided)
  6. Tag: motion speed, dominant colour, scene type
  7. Save as {id}.mp4 + {id}.json (caption, tags, metadata)
  8. Split into train / val sets (95 / 5)

Usage
-----
    # Preprocess UCF-101 raw data:
    python scripts/prepare_data.py --input data/raw/ucf101 --output data/processed \
        --resolution 256 --fps 16 --min-duration 2 --max-duration 30

    # With BLIP-2 auto-captioning (requires GPU + transformers):
    python scripts/prepare_data.py --input data/raw --output data/processed \
        --caption blip2

    # Preview only (no writes):
    python scripts/prepare_data.py --input data/raw --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import time
from pathlib import Path
from typing import Optional

import numpy as np


def _get_ffmpeg() -> str:
    """Return path to ffmpeg binary: bundled via imageio_ffmpeg, or fall back to PATH."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}
DEFAULT_RESOLUTION = 256
DEFAULT_FPS = 16
DEFAULT_MIN_DURATION = 2.0
DEFAULT_MAX_DURATION = 30.0
TRAIN_SPLIT = 0.95

BLIP2_PROMPT = "Question: What is happening in this video frame? Answer:"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess raw video datasets for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, help="Raw data root directory")
    p.add_argument("--output", default="data/processed",
                   help="Processed output directory")
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                   help="Target square resolution in pixels (default: 256)")
    p.add_argument("--fps", type=int, default=DEFAULT_FPS,
                   help="Target frame rate (default: 16)")
    p.add_argument("--min-duration", type=float, default=DEFAULT_MIN_DURATION,
                   help="Minimum clip duration in seconds (default: 2)")
    p.add_argument("--max-duration", type=float, default=DEFAULT_MAX_DURATION,
                   help="Maximum clip duration in seconds (default: 30)")
    p.add_argument("--min-short-side", type=int, default=0,
                   help="Minimum short-side resolution to accept (default: 0 = no filter). "
                        "FFmpeg upscales smaller videos to --resolution automatically.")
    p.add_argument("--caption", choices=["none", "blip2", "filename"],
                   default="filename",
                   help="Caption strategy: none | blip2 (GPU required) | filename")
    p.add_argument("--max-clips", type=int, default=None,
                   help="Process at most N clips (for quick testing)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split")
    p.add_argument("--dry-run", action="store_true",
                   help="Scan and report stats without writing files")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel workers (default: 1; >1 requires ffmpeg)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        log.error("Input directory does not exist: %s", input_dir)
        return

    # Discover all video files
    all_videos = sorted(
        p for p in input_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS
    )
    log.info("Found %d video files under %s", len(all_videos), input_dir)

    if args.max_clips:
        random.shuffle(all_videos)
        all_videos = all_videos[: args.max_clips]
        log.info("Capped to %d clips (--max-clips)", len(all_videos))

    # Load optional BLIP-2 captioner
    captioner = _load_captioner(args.caption)

    # Check tools are available before processing anything
    try:
        import cv2 as _cv2  # noqa: F401
    except ImportError:
        log.error("OpenCV not found. Install it: pip install opencv-python-headless")
        return
    ffmpeg_bin = _get_ffmpeg()
    log.info("Using ffmpeg: %s", ffmpeg_bin)

    # Process
    accepted = []
    reject_reasons: dict[str, int] = {}
    probe_warn_count = 0

    for i, video_path in enumerate(all_videos):
        if i % 100 == 0:
            log.info("Processing %d/%d …", i, len(all_videos))

        try:
            meta = _probe_video(video_path)
        except Exception as exc:
            probe_warn_count += 1
            if probe_warn_count <= 5:
                log.warning("Probe failed for %s: %s", video_path.name, exc)
            elif probe_warn_count == 6:
                log.warning("(suppressing further probe-failure warnings…)")
            reject_reasons["probe_failed"] = reject_reasons.get("probe_failed", 0) + 1
            continue

        duration = meta.get("duration", 0)
        w, h = meta.get("width", 0), meta.get("height", 0)
        short_side = min(w, h)

        # Filter
        if not (args.min_duration <= duration <= args.max_duration):
            key = f"duration_out_of_range"
            reject_reasons[key] = reject_reasons.get(key, 0) + 1
            continue
        if args.min_short_side > 0 and short_side < args.min_short_side:
            key = "short_side_too_small"
            reject_reasons[key] = reject_reasons.get(key, 0) + 1
            continue

        if args.dry_run:
            accepted.append(video_path)
            continue

        # Process this clip
        try:
            result = _process_clip(
                video_path=video_path,
                output_dir=output_dir,
                resolution=args.resolution,
                fps=args.fps,
                captioner=captioner,
                meta=meta,
            )
            if result:
                accepted.append(result)
        except Exception as exc:
            log.warning("Failed to process %s: %s", video_path.name, exc)
            reject_reasons["process_failed"] = reject_reasons.get("process_failed", 0) + 1

    total_rejected = sum(reject_reasons.values())
    log.info("Accepted: %d | Rejected: %d | Total: %d",
             len(accepted), total_rejected, len(all_videos))
    if reject_reasons:
        for reason, count in sorted(reject_reasons.items(), key=lambda x: -x[1]):
            log.info("  rejected[%s]: %d", reason, count)

    if args.dry_run:
        log.info("Dry run complete. No files written.")
        return

    # Train / val split
    _write_split(accepted, output_dir, args.seed)
    log.info("Processed data saved to: %s", output_dir.resolve())
    log.info("Next: train with configs/train_small.yaml pointing to %s", output_dir)


# ---------------------------------------------------------------------------
# Per-clip processing
# ---------------------------------------------------------------------------

def _process_clip(
    video_path: Path,
    output_dir: Path,
    resolution: int,
    fps: int,
    captioner,
    meta: dict,
) -> Optional[Path]:
    clip_id = f"{video_path.stem}_{hash(str(video_path)) % 10**8:08d}"
    out_video = output_dir / "videos" / f"{clip_id}.mp4"
    out_meta = output_dir / "metadata" / f"{clip_id}.json"
    out_video.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    # Re-encode via FFmpeg: resize + set fps
    cmd = [
        _get_ffmpeg(), "-y", "-i", str(video_path),
        "-vf", f"scale={resolution}:{resolution}:force_original_aspect_ratio=decrease,"
               f"pad={resolution}:{resolution}:(ow-iw)/2:(oh-ih)/2,"
               f"fps={fps}",
        "-c:v", "libx264", "-crf", "23", "-an",
        str(out_video),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        log.debug("FFmpeg failed for %s: %s", video_path.name,
                  result.stderr.decode(errors="replace")[:200])
        return None

    # Extract frames for quality check and captioning
    frames = _extract_frames_cv2(out_video, max_frames=8)
    if frames is None or len(frames) == 0:
        out_video.unlink(missing_ok=True)
        return None

    # Black frame check
    means = np.array([f.mean() for f in frames])
    if (means < 10).mean() > 0.5:
        out_video.unlink(missing_ok=True)
        return None

    # Caption
    caption = ""
    if captioner is not None:
        caption = captioner(frames[len(frames) // 2])  # middle frame
    if not caption:
        caption = _filename_caption(video_path)

    # Tags
    tags = _extract_tags(frames, meta)

    record = {
        "id": clip_id,
        "source": str(video_path.name),
        "caption": caption,
        "tags": tags,
        "duration": meta.get("duration"),
        "original_resolution": [meta.get("width"), meta.get("height")],
        "processed_resolution": [resolution, resolution],
        "fps": fps,
        "frames": len(frames),
    }
    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return out_video


# ---------------------------------------------------------------------------
# Video probe (cv2-based, no ffprobe binary required)
# ---------------------------------------------------------------------------

def _probe_video(path: Path) -> dict:
    """Probe video metadata using OpenCV — no ffprobe binary required."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"cv2 could not open {path.name}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0.0
    finally:
        cap.release()

    if width == 0 or height == 0:
        raise RuntimeError(f"Could not read dimensions from {path.name}")

    return {
        "duration": duration,
        "width": width,
        "height": height,
        "codec": path.suffix.lstrip("."),
        "fps": fps,
    }


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _extract_frames_cv2(path: Path, max_frames: int = 8) -> Optional[list]:
    try:
        import cv2
    except ImportError:
        return _extract_frames_ffmpeg(path, max_frames)

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None
    idxs = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames if frames else None


def _extract_frames_ffmpeg(path: Path, max_frames: int = 8) -> Optional[list]:
    """Fallback frame extractor using FFmpeg rawvideo output (no ffprobe needed)."""
    try:
        meta = _probe_video(path)
        w, h = meta["width"], meta["height"]
        fps_val = meta["fps"] or 25.0
        duration = meta["duration"]
        total = int(duration * fps_val)

        if total <= 0 or w <= 0 or h <= 0:
            return None

        idxs = np.linspace(0, total - 1, min(max_frames, total), dtype=int)
        frames = []
        for idx in idxs:
            cmd = [
                _get_ffmpeg(), "-y", "-ss", str(idx / fps_val),
                "-i", str(path), "-vframes", "1",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
            ]
            raw = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, timeout=10)
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            frames.append(frame)
        return frames if frames else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Auto-captioning
# ---------------------------------------------------------------------------

def _load_captioner(strategy: str):
    if strategy == "none":
        return None
    if strategy == "filename":
        return None  # handled in _filename_caption
    if strategy == "blip2":
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            log.info("Loading BLIP-2 captioner on %s …", device)
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            model.eval()

            def caption_fn(frame: np.ndarray) -> str:
                from PIL import Image
                img = Image.fromarray(frame)
                inputs = processor(img, BLIP2_PROMPT, return_tensors="pt").to(device)
                with torch.no_grad():
                    ids = model.generate(**inputs, max_new_tokens=50)
                return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()

            return caption_fn
        except ImportError:
            log.warning("BLIP-2 requires: pip install transformers accelerate Pillow. "
                        "Falling back to filename captions.")
            return None
    return None


def _filename_caption(path: Path) -> str:
    """Generate a basic caption from the file path components."""
    parts = list(path.parts)
    name = path.stem.replace("_", " ").replace("-", " ")
    # For UCF-101: parent folder = action class
    if len(parts) >= 2:
        action = parts[-2].replace("_", " ").replace("-", " ")
        return f"a video of {action}: {name}"
    return f"a video: {name}"


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------

def _extract_tags(frames: list, meta: dict) -> dict:
    """Lightweight rule-based tag extraction from frames and metadata."""
    tags: dict = {}

    # Motion speed: mean of consecutive frame differences
    if len(frames) >= 2:
        diffs = []
        for a, b in zip(frames[:-1], frames[1:]):
            diff = np.abs(a.astype(float) - b.astype(float)).mean() / 255.0
            diffs.append(diff)
        avg_diff = float(np.mean(diffs))
        if avg_diff < 0.02:
            tags["motion"] = "static"
        elif avg_diff < 0.08:
            tags["motion"] = "slow"
        elif avg_diff < 0.20:
            tags["motion"] = "medium"
        else:
            tags["motion"] = "fast"

    # Dominant colour (median RGB of middle frame)
    if frames:
        mid = frames[len(frames) // 2]
        r = int(np.median(mid[:, :, 0]))
        g = int(np.median(mid[:, :, 1]))
        b = int(np.median(mid[:, :, 2]))
        tags["dominant_rgb"] = [r, g, b]
        # Simple brightness
        brightness = (r + g + b) / 3
        tags["brightness"] = "bright" if brightness > 128 else "dark"

    # Duration bucket
    dur = meta.get("duration", 0)
    if dur < 5:
        tags["duration_bucket"] = "short"
    elif dur < 15:
        tags["duration_bucket"] = "medium"
    else:
        tags["duration_bucket"] = "long"

    return tags


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def _write_split(video_paths: list, output_dir: Path, seed: int):
    random.seed(seed)
    paths = [str(p) for p in video_paths]
    random.shuffle(paths)
    n_train = int(len(paths) * TRAIN_SPLIT)
    train_paths = paths[:n_train]
    val_paths = paths[n_train:]

    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, subset in [("train.txt", train_paths), ("val.txt", val_paths)]:
        with open(splits_dir / name, "w", encoding="utf-8") as f:
            f.write("\n".join(subset))

    log.info("Split: %d train / %d val", len(train_paths), len(val_paths))
    log.info("Split files: %s/train.txt, %s/val.txt", splits_dir, splits_dir)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
