"""3D U-Net denoiser for video diffusion. Pure PyTorch, no pretrained weights.

Spatiotemporal U-Net: 3D convolutions with spatial-only down/up-sampling (temporal
length stays fixed), per-block spatial + temporal self-attention, sinusoidal timestep
embeddings, and optional class conditioning. Designed to fit a 4 GB GPU at 32x32x16.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


def _gn(channels: int, groups: int = 8) -> nn.GroupNorm:
    return nn.GroupNorm(math.gcd(groups, channels), channels)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, emb_dim, dropout=0.0, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = _gn(in_ch)
        self.conv1 = nn.Conv3d(in_ch, out_ch, 3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.norm2 = _gn(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv3d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def _forward(self, x, emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb_proj(F.silu(emb))[:, :, None, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)

    def forward(self, x, emb):
        if self.use_checkpoint and self.training and x.requires_grad:
            return checkpoint(self._forward, x, emb, use_reentrant=False)
        return self._forward(x, emb)


class AttentionBlock(nn.Module):
    """Self-attention over the spatial (H*W) or temporal (T) axis."""

    def __init__(self, ch, heads=4, mode="spatial"):
        super().__init__()
        assert mode in ("spatial", "temporal")
        assert ch % heads == 0, f"channels {ch} not divisible by heads {heads}"
        self.mode = mode
        self.heads = heads
        self.norm = _gn(ch)
        self.qkv = nn.Conv3d(ch, ch * 3, 1)
        self.proj = nn.Conv3d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, T, H, W = x.shape
        nh, hd = self.heads, C // self.heads
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)

        if self.mode == "spatial":
            def split(t):  # -> (B*T, nh, H*W, hd)
                t = t.permute(0, 2, 1, 3, 4).reshape(B * T, C, H * W)
                return t.reshape(B * T, nh, hd, H * W).permute(0, 1, 3, 2).contiguous()
            q, k, v = split(q), split(k), split(v)
            o = F.scaled_dot_product_attention(q, k, v)            # (B*T, nh, HW, hd)
            o = o.permute(0, 1, 3, 2).reshape(B * T, C, H * W)
            o = o.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4).contiguous()
        else:
            def split(t):  # -> (B*H*W, nh, T, hd)
                t = t.permute(0, 3, 4, 1, 2).reshape(B * H * W, C, T)
                return t.reshape(B * H * W, nh, hd, T).permute(0, 1, 3, 2).contiguous()
            q, k, v = split(q), split(k), split(v)
            o = F.scaled_dot_product_attention(q, k, v)            # (B*HW, nh, T, hd)
            o = o.permute(0, 1, 3, 2).reshape(B * H * W, C, T)
            o = o.reshape(B, H, W, C, T).permute(0, 3, 4, 1, 2).contiguous()

        return x + self.proj(o)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv3d(ch, ch, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv3d(ch, ch, kernel_size=(1, 3, 3), padding=(0, 1, 1))

    def forward(self, x):
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="nearest")
        return self.conv(x)


class EmbSequential(nn.Sequential):
    """nn.Sequential that forwards the embedding only to timestep-aware blocks."""

    def forward(self, x, emb):
        for layer in self:
            x = layer(x, emb) if isinstance(layer, ResBlock3D) else layer(x)
        return x


class UNet3D(nn.Module):
    def __init__(self, in_ch=3, base=64, ch_mult=(1, 2, 4), num_res_blocks=2,
                 attn_resolutions=(16, 8), heads=4, dropout=0.0, num_classes=0,
                 image_size=32, use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.sin_dim = base
        emb_dim = base * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        if num_classes and num_classes > 0:
            self.class_emb = nn.Embedding(num_classes, emb_dim)

        attn_set = set(attn_resolutions)
        rb_kw = dict(dropout=dropout, use_checkpoint=use_checkpoint)

        # ---- down ----
        self.input_blocks = nn.ModuleList([EmbSequential(nn.Conv3d(in_ch, base, 3, padding=1))])
        chs = [base]
        cur = base
        res = image_size
        for i, mult in enumerate(ch_mult):
            out = base * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock3D(cur, out, emb_dim, **rb_kw)]
                cur = out
                if res in attn_set:
                    layers += [AttentionBlock(cur, heads, "spatial"),
                               AttentionBlock(cur, heads, "temporal")]
                self.input_blocks.append(EmbSequential(*layers))
                chs.append(cur)
            if i != len(ch_mult) - 1:
                self.input_blocks.append(EmbSequential(Downsample(cur)))
                chs.append(cur)
                res //= 2

        # ---- middle ----
        self.middle = EmbSequential(
            ResBlock3D(cur, cur, emb_dim, **rb_kw),
            AttentionBlock(cur, heads, "spatial"),
            AttentionBlock(cur, heads, "temporal"),
            ResBlock3D(cur, cur, emb_dim, **rb_kw),
        )

        # ---- up ----
        self.output_blocks = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            out = base * mult
            for j in range(num_res_blocks + 1):
                skip = chs.pop()
                layers = [ResBlock3D(cur + skip, out, emb_dim, **rb_kw)]
                cur = out
                if res in attn_set:
                    layers += [AttentionBlock(cur, heads, "spatial"),
                               AttentionBlock(cur, heads, "temporal")]
                if i != 0 and j == num_res_blocks:
                    layers.append(Upsample(cur))
                    res *= 2
                self.output_blocks.append(EmbSequential(*layers))

        self.out = nn.Sequential(_gn(cur), nn.SiLU(), nn.Conv3d(cur, in_ch, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self, x, t, y=None):
        emb = self.time_mlp(sinusoidal_embedding(t, self.sin_dim))
        if self.num_classes and y is not None:
            emb = emb + self.class_emb(y)
        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)
