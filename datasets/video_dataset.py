"""Video dataset for training video generation models.

Supports:
  - Folder-based datasets (videos in directories)
  - CSV manifest files (path, caption pairs)
  - Fixed-length temporal sampling from variable-length videos
  - On-the-fly augmentation
"""

from __future__ import annotations
import csv, logging, random
from pathlib import Path
from typing import Optional
import numpy as np, torch
from torch.utils.data import Dataset
from utils.video_utils import read_video_clip, frames_to_tensor
from datasets.captions import clean_caption, truncate_caption

logger = logging.getLogger(__name__)


class VideoTextDataset(Dataset):
    """Dataset for video-caption pairs.

    Loads videos from a manifest CSV or folder, extracts fixed-length clips,
    and returns (video_tensor, caption) pairs for training.

    CSV format: path,caption
    Folder format: root_dir/class_name/video.mp4 (caption = class_name)

    Args:
        root_dir: Root directory containing videos.
        manifest_path: Path to CSV manifest (overrides folder mode).
        num_frames: Number of frames per clip.
        height: Target frame height.
        width: Target frame width.
        fps: Target FPS for frame sampling.
        random_flip: Enable random horizontal flip augmentation.
        random_crop: Enable random crop augmentation.
    """

    def __init__(self, root_dir: str = "data/videos", manifest_path: Optional[str] = None,
                 num_frames: int = 16, height: int = 256, width: int = 256, fps: int = 8,
                 random_flip: bool = True, random_crop: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.fps = fps
        self.random_flip = random_flip
        self.random_crop = random_crop

        # Load entries: list of (video_path, caption)
        self.entries: list[tuple[str, str]] = []

        if manifest_path and Path(manifest_path).exists():
            self._load_manifest(manifest_path)
        elif Path(root_dir).exists():
            self._load_from_folder(root_dir)
        else:
            logger.warning(f"No data found at {root_dir} or {manifest_path}")

        logger.info(f"Loaded {len(self.entries)} video-caption pairs")

    def _load_manifest(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_path = row.get("path", row.get("video_path", ""))
                caption = row.get("caption", row.get("text", ""))
                if video_path and Path(video_path).exists():
                    self.entries.append((video_path, caption))

    def _load_from_folder(self, root_dir: str) -> None:
        root = Path(root_dir)
        extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        for video_path in sorted(root.rglob("*")):
            if video_path.suffix.lower() in extensions:
                caption = video_path.parent.name  # Use folder name as caption
                self.entries.append((str(video_path), caption))

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        """Load a video-caption pair, with retry logic on failure."""
        # Retry up to 3 times with different samples on failure
        max_retries = 3
        for attempt in range(max_retries + 1):
            try_idx = idx if attempt == 0 else random.randint(0, len(self.entries) - 1)
            video_path, caption = self.entries[try_idx]

            try:
                frames, _ = read_video_clip(
                    video_path, num_frames=self.num_frames,
                    target_fps=self.fps, height=self.height, width=self.width,
                    random_start=True,
                )
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Error loading {video_path}: {e}, retrying with another sample")
                    continue
                else:
                    logger.error(f"Failed to load after {max_retries} retries, using blank frames")
                    frames = np.zeros((self.num_frames, self.height, self.width, 3), dtype=np.uint8)

            # Pad if too few frames
            if len(frames) < self.num_frames:
                pad = np.repeat(frames[-1:], self.num_frames - len(frames), axis=0)
                frames = np.concatenate([frames, pad], axis=0)

            # Augmentation
            if self.random_flip and random.random() > 0.5:
                frames = frames[:, :, ::-1, :].copy()

            # Convert to tensor [C, T, H, W] normalized to [-1, 1]
            video_tensor = frames_to_tensor(frames, normalize=True)

            # Clean caption
            caption = clean_caption(caption)
            caption = truncate_caption(caption)

            return {"video": video_tensor, "caption": caption, "path": video_path}
