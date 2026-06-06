"""Datasets. Fully self-contained: a procedural synthetic video generator (no downloads,
no third-party data) and an optional loader for your own local video files."""
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticShapes(Dataset):
    """Procedurally generated clips of a colored shape translating in one of 4 directions.

    The motion direction is returned as a class label (0=left, 1=right, 2=up, 3=down),
    giving a self-contained conditional video task to train and verify the model on.
    """
    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (dx, dy): left, right, up, down

    def __init__(self, size=32, frames=16, length=4000, channels=3, seed=0):
        self.size, self.frames, self.length = size, frames, length
        self.channels, self.seed = channels, seed

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 100003 + idx)
        S, T, C = self.size, self.frames, self.channels
        d = int(rng.integers(0, 4))
        dx, dy = self.DIRS[d]
        r = max(2, S // 6)
        speed = (S - 1 - 2 * r) / max(1, T - 1)

        if dx != 0:
            x0 = r if dx > 0 else S - 1 - r
            y0 = int(rng.integers(r, S - r))
        else:
            y0 = r if dy > 0 else S - 1 - r
            x0 = int(rng.integers(r, S - r))

        color = rng.uniform(0.5, 1.0, size=3).astype(np.float32) if C == 3 else np.ones(1, np.float32)
        is_square = bool(rng.integers(0, 2))

        yy, xx = np.mgrid[0:S, 0:S]
        vid = np.zeros((T, S, S, C), dtype=np.float32)
        for t in range(T):
            cx, cy = x0 + dx * speed * t, y0 + dy * speed * t
            if is_square:
                mask = (np.abs(xx - cx) <= r) & (np.abs(yy - cy) <= r)
            else:
                mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            for c in range(C):
                vid[t, ..., c][mask] = color[c % color.shape[0]]

        vid = vid * 2.0 - 1.0  # [0,1] -> [-1,1]
        video = torch.from_numpy(vid).permute(3, 0, 1, 2).contiguous()  # (C,T,H,W)
        return video, d


class VideoFolder(Dataset):
    """Loads .mp4/.gif/.avi/... from a directory. Samples `frames` consecutive frames,
    resizes to `size`, returns (C,T,H,W) in [-1,1] with label 0 (single-class)."""
    EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif")

    def __init__(self, root, size=32, frames=16, channels=3):
        self.size, self.frames, self.channels = size, frames, channels
        self.files = [p for p in glob.glob(os.path.join(root, "**", "*"), recursive=True)
                      if p.lower().endswith(self.EXTS)]
        if not self.files:
            raise ValueError(f"No video files found under {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        import cv2
        path = self.files[idx]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.frames
        start = max(0, int(np.random.randint(0, max(1, total - self.frames + 1))))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frames = []
        for _ in range(self.frames):
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            f = cv2.resize(f, (self.size, self.size))
            frames.append(f)
        cap.release()

        if not frames:
            frames = [np.zeros((self.size, self.size, 3), np.uint8)]
        while len(frames) < self.frames:
            frames.append(frames[-1])

        arr = np.stack(frames).astype(np.float32) / 127.5 - 1.0  # (T,H,W,3)
        if self.channels == 1:
            arr = arr.mean(axis=-1, keepdims=True)
        video = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        return video, 0


def build_dataset(cfg):
    t = cfg["train"]
    name = t.get("dataset", "synthetic")
    if name == "synthetic":
        return SyntheticShapes(size=cfg["image_size"], frames=cfg["frames"],
                               channels=cfg["channels"], length=t.get("length", 4000))
    if name == "folder":
        return VideoFolder(root=t["data_dir"], size=cfg["image_size"],
                           frames=cfg["frames"], channels=cfg["channels"])
    raise ValueError(f"Unknown dataset: {name}")
