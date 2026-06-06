"""Train the video diffusion model from scratch.

Usage:
    python scripts/train.py --config configs/train_small.yaml
    python scripts/train.py --config configs/train_small.yaml --steps 50   # quick smoke test
    python scripts/train.py --config configs/train_small.yaml --resume runs/small/last.pt
"""
import argparse
import os
import sys

import torch
import yaml
from torch import amp
from torch.utils.data import DataLoader
from tqdm import trange

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from vdm import EMA, GaussianDiffusion, UNet3D, build_dataset, cycle, save_video, seed_everything


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None)
    p.add_argument("--device", default=None)
    # quick overrides
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--dataset", default=None)
    p.add_argument("--data_dir", default=None)
    return p.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    t = cfg["train"]
    if args.steps is not None:
        t["steps"] = args.steps
    if args.batch_size is not None:
        t["batch_size"] = args.batch_size
    if args.dataset is not None:
        t["dataset"] = args.dataset
    if args.data_dir is not None:
        t["data_dir"] = args.data_dir
    return cfg


def build_model(cfg, device):
    m = cfg["model"]
    net = UNet3D(
        in_ch=cfg["channels"], base=m["base"], ch_mult=tuple(m["ch_mult"]),
        num_res_blocks=m["num_res_blocks"], attn_resolutions=tuple(m["attn_resolutions"]),
        heads=m["heads"], dropout=m.get("dropout", 0.0), num_classes=cfg["num_classes"],
        image_size=cfg["image_size"], use_checkpoint=m.get("use_checkpoint", False),
    )
    diffusion = GaussianDiffusion(net, timesteps=cfg["diffusion"]["timesteps"],
                                  schedule=cfg["diffusion"]["schedule"]).to(device)
    return net.to(device), diffusion


@torch.no_grad()
def preview(diffusion, ema, net, cfg, device, step, out_dir):
    t = cfg["train"]
    n = t["sample_count"]
    nc = cfg["num_classes"]
    y = (torch.arange(n, device=device) % nc) if nc else None
    shape = (n, cfg["channels"], cfg["frames"], cfg["image_size"], cfg["image_size"])

    ema.store(net); ema.copy_to(net); net.eval()
    samples = diffusion.ddim_sample(shape, y=y, steps=t.get("sample_steps", 50), device=device)
    net.train(); ema.restore(net)

    save_video(samples, os.path.join(out_dir, f"sample_{step:06d}.gif"),
               fps=max(4, cfg["frames"] // 2))


def main():
    args = parse_args()
    cfg = load_config(args)
    t = cfg["train"]
    seed_everything(0)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(t.get("amp", False)) and device.startswith("cuda")
    out_dir = t["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    print(f"[train] device={device} amp={use_amp} out={out_dir}")

    ds = build_dataset(cfg)
    dl = DataLoader(ds, batch_size=t["batch_size"], shuffle=True, drop_last=True,
                    num_workers=t["num_workers"], pin_memory=device.startswith("cuda"))
    data = cycle(dl)

    net, diffusion = build_model(cfg, device)
    n_params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"[train] UNet3D params: {n_params:.1f}M")

    opt = torch.optim.AdamW(net.parameters(), lr=t["lr"])
    scaler = amp.GradScaler("cuda", enabled=use_amp)
    ema = EMA(net, decay=t["ema_decay"])

    start = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        net.load_state_dict(ckpt["model"])
        ema.shadow = {k: v.to(device) for k, v in ckpt["ema"].items()}
        opt.load_state_dict(ckpt["opt"])
        start = ckpt["step"] + 1
        print(f"[train] resumed from {args.resume} @ step {start}")

    nc = cfg["num_classes"]
    timesteps = cfg["diffusion"]["timesteps"]
    accum = t["grad_accum"]
    pbar = trange(start, t["steps"], initial=start, total=t["steps"], dynamic_ncols=True)
    for step in pbar:
        opt.zero_grad(set_to_none=True)
        total = 0.0
        for _ in range(accum):
            x, y = next(data)
            x = x.to(device)
            y = y.to(device) if nc else None
            tt = torch.randint(0, timesteps, (x.size(0),), device=device)
            with amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = diffusion.p_losses(x, tt, y) / accum
            scaler.scale(loss).backward()
            total += loss.item()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        ema.update(net)
        pbar.set_postfix(loss=f"{total:.4f}")

        if (step + 1) % t["sample_every"] == 0 or step == t["steps"] - 1:
            preview(diffusion, ema, net, cfg, device, step + 1, out_dir)
        if (step + 1) % t["ckpt_every"] == 0 or step == t["steps"] - 1:
            torch.save({"step": step, "model": net.state_dict(), "ema": ema.shadow,
                        "opt": opt.state_dict(), "cfg": cfg},
                       os.path.join(out_dir, "last.pt"))


if __name__ == "__main__":
    main()
