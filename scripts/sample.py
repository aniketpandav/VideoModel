"""Generate videos from a trained checkpoint (uses the EMA weights).

Usage:
    python scripts/sample.py --ckpt runs/small/last.pt --n 4 --out out.gif
    python scripts/sample.py --ckpt runs/small/last.pt --label 1 --steps 100 --out right.mp4
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdm import GaussianDiffusion, UNet3D, save_video


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default="sample.gif")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--label", type=int, default=None, help="fixed class label; default cycles classes")
    p.add_argument("--steps", type=int, default=100, help="DDIM steps")
    p.add_argument("--ddpm", action="store_true", help="use full DDPM sampling instead of DDIM")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["cfg"]
    m = cfg["model"]

    net = UNet3D(
        in_ch=cfg["channels"], base=m["base"], ch_mult=tuple(m["ch_mult"]),
        num_res_blocks=m["num_res_blocks"], attn_resolutions=tuple(m["attn_resolutions"]),
        heads=m["heads"], dropout=0.0, num_classes=cfg["num_classes"],
        image_size=cfg["image_size"], use_checkpoint=False,
    ).to(device)

    # load EMA weights into the net for best quality
    msd = net.state_dict()
    for k, v in ckpt["ema"].items():
        msd[k].copy_(v.to(device))
    net.eval()

    diffusion = GaussianDiffusion(net, timesteps=cfg["diffusion"]["timesteps"],
                                  schedule=cfg["diffusion"]["schedule"],
                                  predict=cfg["diffusion"].get("predict", "v")).to(device)

    nc = cfg["num_classes"]
    if nc:
        if args.label is not None:
            y = torch.full((args.n,), args.label, device=device, dtype=torch.long)
        else:
            y = torch.arange(args.n, device=device) % nc
    else:
        y = None

    shape = (args.n, cfg["channels"], cfg["frames"], cfg["image_size"], cfg["image_size"])
    with torch.no_grad():
        if args.ddpm:
            samples = diffusion.sample(shape, y=y, device=device)
        else:
            samples = diffusion.ddim_sample(shape, y=y, steps=args.steps, device=device)

    save_video(samples, args.out, fps=max(4, cfg["frames"] // 2))
    print(f"[sample] wrote {args.out}")


if __name__ == "__main__":
    main()
