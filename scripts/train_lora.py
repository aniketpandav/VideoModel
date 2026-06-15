"""LTX-Video LoRA fine-tuning script.

Fine-tunes Lightricks/LTX-Video (Apache-2.0) using LoRA adapters via PEFT.
Only the attention layers get trainable rank-16 adapters — the rest of the
model stays frozen. This reduces VRAM from ~40 GB (full fine-tune) to ~16 GB
(LoRA), making it feasible on Kaggle T4 (16 GB) or a single A100 (40 GB).

Hardware requirements
---------------------
  Kaggle T4   (16 GB) — batch=1, grad_accum=4, fp16=true   → ~14 GB VRAM
  A100 40 GB  (cloud) — batch=2, grad_accum=2, fp16=true   → ~30 GB VRAM
  GTX 1650    (4 GB)  → NOT SUPPORTED (model alone needs 8 GB)

Run on Kaggle T4
----------------
  1. Upload your processed dataset to a Kaggle Dataset (data/processed/).
  2. Create a new Kaggle Notebook, GPU=T4, Internet=On.
  3. Add this repo as a dataset or clone via GitHub token.
  4. Run:
       !pip install 'diffusers>=0.32' transformers accelerate peft sentencepiece
       !python scripts/train_lora.py --config configs/train_lora.yaml

  5. Save the output adapter weights and download to your local machine.
  6. Load them in serve/backbones.py via `pipe.load_lora_weights(adapter_path)`.

What LoRA training gives you
-----------------------------
  - Preserves LTX-Video's general video generation quality
  - Adapts to your specific domain (style, subject, motion)
  - Training time: ~2 h on Kaggle T4 for 1 000 steps on 500 clips
  - Result: noticeably more on-brand output without full retraining cost

Usage
-----
    python scripts/train_lora.py --config configs/train_lora.yaml
    python scripts/train_lora.py --config configs/train_lora.yaml --resume runs/lora/last.pt
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import time
from pathlib import Path

import yaml

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune LTX-Video with LoRA")
    p.add_argument("--config", default="configs/train_lora.yaml")
    p.add_argument("--resume", default=None, help="Path to a previous LoRA checkpoint")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    _check_prerequisites()

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from diffusers import LTXPipeline
    from diffusers.optimization import get_cosine_schedule_with_warmup
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Training on: %s", device)

    if device == "cuda":
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log.info("VRAM: %.1f GB", vram)
        if vram < 14:
            log.error(
                "Insufficient VRAM (%.1f GB). LTX-Video LoRA requires ≥16 GB.\n"
                "  → Use Kaggle T4 (free 16 GB) or a cloud A100.\n"
                "  → GTX 1650 (4 GB) cannot run this script.", vram
            )
            return

    set_seed(cfg.get("seed", 42))
    accelerator = Accelerator(
        gradient_accumulation_steps=cfg["train"]["grad_accum"],
        mixed_precision="fp16" if cfg["train"].get("amp", True) else "no",
        log_with=None,
    )

    # ------------------------------------------------------------------
    # Load pretrained model
    # ------------------------------------------------------------------
    model_id = cfg.get("model_id", "Lightricks/LTX-Video")
    log.info("Loading LTX-Video from %s …", model_id)

    pipe = LTXPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if cfg["train"].get("amp", True) else torch.float32,
    )

    # Freeze VAE and text encoder — only train the transformer
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    transformer = pipe.transformer

    # ------------------------------------------------------------------
    # Inject LoRA adapters into attention layers
    # ------------------------------------------------------------------
    lora_cfg = cfg.get("lora", {})
    lora_config = LoraConfig(
        r=lora_cfg.get("rank", 16),
        lora_alpha=lora_cfg.get("alpha", 16),
        target_modules=lora_cfg.get("target_modules",
                                    ["to_q", "to_k", "to_v", "to_out.0"]),
        lora_dropout=lora_cfg.get("dropout", 0.0),
        bias="none",
    )
    transformer = get_peft_model(transformer, lora_config)
    transformer.print_trainable_parameters()

    pipe.transformer = transformer

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    data_dir = Path(cfg["dataset"]["path"])
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{data_dir.resolve()}'. "
            "Run Cell 5 (download) and Cell 6 (prepare) before training."
        )

    from vdm.data import CaptionedVideoFolder
    metadata_dir = data_dir / "metadata"
    dataset = CaptionedVideoFolder(
        videos_root=str(data_dir / "videos"),
        metadata_root=str(metadata_dir) if metadata_dir.exists() else None,
        size=cfg["dataset"].get("resolution", 256),
        frames=cfg["dataset"].get("frames", 25),
    )
    log.info("Dataset: %d clips from %s", len(dataset), data_dir)

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=0,    # always 0 on Windows; safe on Linux too
        pin_memory=(device == "cuda"),
        drop_last=True,
    )

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    trainable = [p for p in transformer.parameters() if p.requires_grad]
    log.info("Trainable LoRA params: %s",
             f"{sum(p.numel() for p in trainable):,}")

    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"].get("weight_decay", 1e-2),
    )

    total_steps = cfg["train"]["steps"]
    warmup_steps = cfg["train"].get("warmup_steps", 100)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # Prepare with accelerator
    # ------------------------------------------------------------------
    transformer, optimizer, loader, scheduler = accelerator.prepare(
        transformer, optimizer, loader, scheduler
    )

    # Resume from checkpoint
    start_step = 0
    out_dir = Path(cfg["train"].get("out_dir", "runs/lora"))
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and Path(args.resume).exists():
        log.info("Resuming from %s", args.resume)
        ckpt = torch.load(args.resume, map_location="cpu")
        accelerator.unwrap_model(transformer).load_state_dict(ckpt["lora_state"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_step = ckpt["step"] + 1
        log.info("Resumed at step %d", start_step)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    def _cycle(dl):
        while True:
            yield from dl

    data_iter = _cycle(loader)
    pipe.vae = pipe.vae.to(device)
    pipe.text_encoder = pipe.text_encoder.to(device)

    log.info("Starting LoRA training: %d steps from step %d", total_steps, start_step)
    t0 = time.time()

    for step in range(start_step, total_steps):
        transformer.train()
        optimizer.zero_grad()
        accum = cfg["train"]["grad_accum"]
        loss_accum = 0.0

        for _ in range(accum):
            batch, captions = next(data_iter)
            batch = batch.to(device)
            # DataLoader collates str items as a tuple; ensure list of str
            captions = [str(c) if c else "" for c in captions]

            with accelerator.accumulate(transformer):
                # Encode video frames to latent space via VAE
                with torch.no_grad():
                    # batch: (B, C, T, H, W) in [-1, 1]
                    B, C, T, H, W = batch.shape
                    frames_flat = batch.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
                    latents = pipe.vae.encode(frames_flat).latent_dist.sample()
                    latents = latents * pipe.vae.config.scaling_factor
                    latents = latents.reshape(B, T, *latents.shape[1:])
                    latents = latents.permute(0, 2, 1, 3, 4)  # (B, C, T, h, w)

                # Sample noise and timestep
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, pipe.scheduler.config.num_train_timesteps,
                    (B,), device=device,
                ).long()

                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

                # Encode prompts using YouTube titles loaded from metadata JSONs
                prompt_embeds, pooled_embeds = _encode_prompt(pipe, captions, device)

                # Forward pass through LoRA-patched transformer
                pred = transformer(
                    hidden_states=noisy_latents,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]

                # v-prediction loss (matches LTX-Video's training objective)
                target = pipe.scheduler.get_velocity(latents, noise, timesteps)
                loss = torch.nn.functional.mse_loss(pred, target)
                loss_accum += loss.item() / accum

                accelerator.backward(loss / accum)

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()

        # Logging
        if step % 10 == 0:
            elapsed = time.time() - t0
            sps = (step - start_step + 1) / max(elapsed, 1)
            eta = (total_steps - step - 1) / max(sps, 1e-6)
            log.info(
                "step %5d/%d  loss=%.4f  lr=%.2e  %.1f s/step  ETA %.0f min",
                step + 1, total_steps, loss_accum,
                optimizer.param_groups[0]["lr"],
                1.0 / max(sps, 1e-6),
                eta / 60,
            )

        # Checkpoint
        ckpt_every = cfg["train"].get("ckpt_every", 50)
        if (step + 1) % ckpt_every == 0 or step == total_steps - 1:
            _save_checkpoint(
                out_dir, step, accelerator.unwrap_model(transformer),
                optimizer, scheduler, cfg,
            )

    log.info("Training complete in %.1f min", (time.time() - t0) / 60)
    log.info("LoRA weights saved to: %s", out_dir)
    log.info(
        "\nTo use in production:\n"
        "  VDM_BACKBONE=ltx  (start the API)\n"
        "  Then patch LTXVideoBackbone.__init__() to add:\n"
        "    self.pipe.load_lora_weights('%s/last_lora/')", out_dir
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_prompt(pipe, prompts: list[str], device: str):
    import torch
    inputs = pipe.tokenizer(
        prompts,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        enc_out = pipe.text_encoder(**inputs)
    return enc_out.last_hidden_state, enc_out.pooler_output


def _save_checkpoint(out_dir: Path, step: int, model, optimizer, scheduler, cfg: dict):
    import torch
    from peft import get_peft_model_state_dict

    ckpt_dir = out_dir / f"step_{step+1:06d}"
    ckpt_dir.mkdir(exist_ok=True)

    # Save LoRA adapter weights only (much smaller than full model)
    lora_state = get_peft_model_state_dict(model)
    torch.save({
        "step": step,
        "lora_state": lora_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": cfg,
    }, ckpt_dir / "lora.pt")

    # Also save in diffusers format for direct pipe.load_lora_weights() use
    model.save_pretrained(str(ckpt_dir / "adapter"))

    # Keep a "last" symlink
    last = out_dir / "last.pt"
    if last.exists() or last.is_symlink():
        last.unlink()
    try:
        last.symlink_to(ckpt_dir / "lora.pt")
    except OSError:
        import shutil
        shutil.copy2(ckpt_dir / "lora.pt", last)

    log.info("Checkpoint saved: %s", ckpt_dir)


def _check_prerequisites():
    missing = []
    for pkg in ["diffusers", "peft", "accelerate", "transformers"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        log.error(
            "Missing packages: %s\n"
            "Install with:\n"
            "  pip install 'diffusers>=0.32' peft accelerate transformers sentencepiece",
            ", ".join(missing),
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
