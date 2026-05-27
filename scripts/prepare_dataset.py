"""Dataset preparation CLI script.

Processes raw videos into a training-ready dataset:
  1. Extracts fixed-length clips
  2. Resizes and normalizes frames
  3. Filters out static scenes
  4. Generates manifest CSV
"""

from __future__ import annotations
import argparse, csv, logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from datasets.preprocessing import extract_clips

logger = logging.getLogger(__name__)


def prepare_dataset(input_dir: str, output_dir: str, clip_duration: float = 4.0,
                    fps: int = 8, height: int = 256, width: int = 256,
                    min_motion: float = 0.01):
    """Prepare a video dataset from raw videos."""
    logging.basicConfig(level=logging.INFO)
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    videos = [f for f in input_path.rglob("*") if f.suffix.lower() in extensions]
    logger.info(f"Found {len(videos)} videos in {input_dir}")

    manifest = []
    for video in videos:
        try:
            clip_dir = output_path / "clips" / video.stem
            clips = extract_clips(video, clip_dir, clip_duration=clip_duration,
                                 target_fps=fps, target_height=height, target_width=width,
                                 min_motion=min_motion)
            for clip in clips:
                caption = video.parent.name  # Use parent folder as caption
                manifest.append({"path": str(clip), "caption": caption})
        except Exception as e:
            logger.warning(f"Error processing {video}: {e}")

    # Write manifest
    manifest_path = output_path / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "caption"])
        writer.writeheader()
        writer.writerows(manifest)

    logger.info(f"Dataset prepared: {len(manifest)} clips -> {manifest_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare video dataset")
    p.add_argument("--input", type=str, required=True, help="Input directory with raw videos")
    p.add_argument("--output", type=str, default="data", help="Output directory")
    p.add_argument("--clip_duration", type=float, default=4.0)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    args = p.parse_args()
    prepare_dataset(args.input, args.output, args.clip_duration, args.fps, args.height, args.width)
