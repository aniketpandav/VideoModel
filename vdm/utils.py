"""Training utilities: seeding, EMA, and video saving."""
import math
import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cycle(loader):
    while True:
        for batch in loader:
            yield batch


class EMA:
    """Exponential moving average of (float) model parameters."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items() if v.dtype.is_floating_point}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def store(self, model):
        self.backup = {k: v.detach().clone()
                       for k, v in model.state_dict().items() if k in self.shadow}

    @torch.no_grad()
    def copy_to(self, model):
        msd = model.state_dict()
        for k in self.shadow:
            msd[k].copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model):
        msd = model.state_dict()
        for k in self.backup:
            msd[k].copy_(self.backup[k])
        self.backup = {}


def _to_uint8(samples: torch.Tensor) -> np.ndarray:
    # samples: (B,C,T,H,W) in [-1,1] -> (B,T,H,W,C) uint8
    x = ((samples.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)
    return x.permute(0, 2, 3, 4, 1).cpu().numpy()


def save_video(samples: torch.Tensor, path: str, fps: int = 8, nrow: int | None = None):
    """Save a batch (B,C,T,H,W) as an animated grid (.gif or .mp4 via imageio)."""
    import imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    s = _to_uint8(samples)                       # (B,T,H,W,C)
    B, T, H, W, C = s.shape
    nrow = nrow or int(math.ceil(math.sqrt(B)))
    ncol = int(math.ceil(B / nrow))

    frames = []
    for t in range(T):
        grid = np.zeros((nrow * H, ncol * W, C), dtype=np.uint8)
        for b in range(B):
            r, c = b // ncol, b % ncol
            grid[r * H:(r + 1) * H, c * W:(c + 1) * W] = s[b, t]
        frames.append(grid[..., 0] if C == 1 else grid)

    if path.lower().endswith(".gif"):
        imageio.mimsave(path, frames, fps=fps, loop=0)
    else:
        imageio.mimsave(path, frames, fps=fps)   # mp4 requires imageio-ffmpeg
