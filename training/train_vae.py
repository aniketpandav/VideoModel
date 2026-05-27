"""VAE training script — supports both local and online (streaming) datasets."""

from __future__ import annotations
import logging, sys, torch
from pathlib import Path
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import load_configs
from utils.duration import normalize_num_frames
from utils.memory import clear_gpu_cache
from models.vae.vae import VideoVAE
from models.vae.losses import VAELoss
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from training.lr_scheduler import create_lr_scheduler
from training.trainer import _flatten_config

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parent.parent


def _load_training_config(model_config_path: str, config_path: str, use_online: bool):
    paths = [model_config_path]
    online_config = PROJECT_ROOT / "configs" / "dataset" / "online.yaml"
    if use_online and online_config.exists():
        paths.append(online_config)
    paths.append(config_path)
    return load_configs(*paths)


def _build_dataloader(
    config,
    use_online: bool,
    batch_size: int,
    num_workers: int,
    online_limit: int | None = None,
):
    """Build dataloader from online streaming sources or local files."""

    if use_online:
        from datasets.online_dataset import OnlineVideoDataset, online_collate_fn

        # Prefer online_dataset config if present, else sensible defaults
        od_cfg = config.get("online_dataset") if hasattr(config, "get") else None

        sources = ["pexels", "synthetic"]
        face_filter = True
        max_per_source = 5000
        shuffle_buffer = 256
        seed = 42

        if od_cfg is not None:
            sources = list(od_cfg.get("sources", sources))
            face_filter = od_cfg.get("face_filter", face_filter)
            max_per_source = od_cfg.get("max_per_source", max_per_source)
            shuffle_buffer = od_cfg.get("shuffle_buffer", shuffle_buffer)
            seed = od_cfg.get("seed", seed)

        ds_cfg = config.dataset
        dataset = OnlineVideoDataset(
            sources=sources,
            num_frames=ds_cfg.num_frames,
            height=ds_cfg.frame_height,
            width=ds_cfg.frame_width,
            face_filter=face_filter,
            max_per_source=max_per_source,
            limit=online_limit,
            shuffle_buffer=shuffle_buffer,
            seed=seed,
        )
        # IterableDataset — no shuffle in DataLoader, collate handles batching
        return DataLoader(dataset, batch_size=batch_size,
                          num_workers=0,
                          collate_fn=online_collate_fn,
                          pin_memory=True)
    else:
        from datasets.video_dataset import VideoTextDataset
        ds_cfg = config.dataset
        dataset = VideoTextDataset(
            root_dir=ds_cfg.root_dir,
            manifest_path=ds_cfg.get("manifest_path"),
            num_frames=ds_cfg.num_frames,
            height=ds_cfg.frame_height,
            width=ds_cfg.frame_width,
            fps=ds_cfg.fps,
        )

        if len(dataset) == 0:
            return None   # caller handles the empty case

        return DataLoader(dataset, batch_size=batch_size,
                          shuffle=True, num_workers=num_workers,
                          pin_memory=True, drop_last=True)


def train_vae(config_path: str = "configs/training/train_vae.yaml",
              model_config_path: str = "configs/model/dit_small.yaml",
              use_online: bool = False,
              online_limit: int | None = None,
              overrides: list[str] | None = None):
    """Train the 3D Video VAE.

    Args:
        use_online: If True, stream video data from HuggingFace Hub
                    (no local videos required).
        online_limit: Optional number of online clips to expose per epoch.
    """
    config = _load_training_config(model_config_path, config_path, use_online)
    if overrides:
        config.merge_overrides(overrides)
    logging.basicConfig(level=logging.INFO)
    safe_frames = normalize_num_frames(config.dataset.num_frames)
    if safe_frames != config.dataset.num_frames:
        logger.warning(
            "Rounding VAE dataset.num_frames from %s to %s for 4x temporal VAE downsampling",
            config.dataset.num_frames, safe_frames,
        )
        config.dataset.num_frames = safe_frames

    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=config.training.log_dir,
    )

    # CUDA memory optimization for low-VRAM GPUs
    if torch.cuda.is_available():
        import os
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Build VAE
    vae_cfg = config.model.vae
    vae = VideoVAE(
        in_channels=vae_cfg.in_channels, latent_channels=vae_cfg.latent_channels,
        base_channels=vae_cfg.base_channels, channel_multipliers=vae_cfg.channel_multipliers,
        num_res_blocks=vae_cfg.num_res_blocks,
    )

    # Enable gradient checkpointing to reduce VRAM usage
    gc = config.training.get("gradient_checkpointing", True) if hasattr(config.training, "get") else True
    if gc:
        vae.enable_gradient_checkpointing()

    accelerator.print(f"VAE params: {sum(p.numel() for p in vae.parameters()):,}")

    # Loss
    loss_cfg = config.loss
    criterion = VAELoss(
        reconstruction_weight=loss_cfg.reconstruction_weight,
        kl_weight=loss_cfg.kl_weight,
        perceptual_weight=loss_cfg.perceptual_weight,
    )

    # Dataset / DataLoader
    accelerator.print(f"Dataset mode: {'[Online streaming]' if use_online else '[Local files]'}")
    dataloader = _build_dataloader(
        config, use_online,
        batch_size=config.training.batch_size,
        num_workers=config.training.dataloader_num_workers,
        online_limit=online_limit,
    )

    if dataloader is None:
        accelerator.print("\n" + "=" * 60)
        accelerator.print("ERROR: No local training videos found!")
        accelerator.print("=" * 60)
        accelerator.print("\nRun with --online to stream training data from the internet:")
        accelerator.print("  python scripts/train.py --mode vae --online\n")
        accelerator.print("Or to use local videos:")
        accelerator.print("  1. Place .mp4/.avi/.mov videos in: data/videos/")
        accelerator.print("  2. Or run: python scripts/prepare_dataset.py --input /path/to/videos --output data/")
        accelerator.print("  3. Or create a CSV manifest at: data/manifest.csv\n")
        return

    # Optimizer
    optimizer = torch.optim.AdamW(
        vae.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.weight_decay,
    )

    vae, criterion, optimizer, dataloader = accelerator.prepare(
        vae, criterion, optimizer, dataloader
    )

    # LR Scheduler
    scheduler_type = config.training.get("lr_scheduler", "constant") if hasattr(config.training, "get") else "constant"
    warmup_steps = config.training.get("lr_warmup_steps", 500) if hasattr(config.training, "get") else 500
    try:
        steps_per_epoch = len(dataloader)
    except TypeError:
        steps_per_epoch = 10000
    total_steps = steps_per_epoch * config.training.num_epochs
    if config.training.max_steps > 0:
        total_steps = min(total_steps, config.training.max_steps)
    total_steps = total_steps // config.training.gradient_accumulation_steps

    lr_scheduler = create_lr_scheduler(
        optimizer,
        scheduler_type=scheduler_type,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    lr_scheduler = accelerator.prepare(lr_scheduler)

    # Resume from checkpoint
    global_step = 0
    start_epoch = 0
    tc = config.training
    if tc.get("resume_from") if hasattr(tc, "get") else None:
        info = load_checkpoint(tc.resume_from, accelerator.unwrap_model(vae),
                              optimizer, scheduler=lr_scheduler)
        global_step = info["step"]
        start_epoch = info["epoch"]
        accelerator.print(f"Resumed from step {global_step}, epoch {start_epoch}")
    else:
        latest = find_latest_checkpoint(tc.checkpoint_dir, prefix="vae")
        if latest:
            info = load_checkpoint(str(latest), accelerator.unwrap_model(vae),
                                  optimizer, scheduler=lr_scheduler)
            global_step = info["step"]
            start_epoch = info["epoch"]
            accelerator.print(f"Auto-resumed from step {global_step}, epoch {start_epoch}")

    # Initialize TensorBoard tracker
    if accelerator.is_main_process:
        accelerator.init_trackers("vae_training", config=_flatten_config(config.to_dict()))

    # Training loop
    should_stop = False
    for epoch in range(start_epoch, config.training.num_epochs):
        vae.train()
        epoch_steps = 0
        pbar = tqdm(dataloader, desc=f"VAE Epoch {epoch + 1}",
                    disable=not accelerator.is_local_main_process)
        for batch in pbar:
            with accelerator.accumulate(vae):
                video = batch["video"]
                output = vae(video)
                losses = criterion(output.reconstruction, video, output.mean, output.logvar)
                accelerator.backward(losses["total"])

                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(vae.parameters(), config.training.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Only count optimizer steps, not micro-batches
            if accelerator.sync_gradients:
                global_step += 1
                epoch_steps += 1

                if global_step % config.training.log_every == 0 and accelerator.is_main_process:
                    src = batch.get("source", ["?"])[0] if isinstance(
                        batch.get("source"), list) else "?"
                    lr = optimizer.param_groups[0]["lr"]
                    pbar.set_postfix(
                        loss=f"{losses['total'].item():.4f}",
                        recon=f"{losses['reconstruction'].item():.4f}",
                        kl=f"{losses['kl'].item():.5f}",
                        lr=f"{lr:.2e}",
                        src=src,
                    )
                    logger.info(
                        f"Step {global_step} | src={src} "
                        f"recon={losses['reconstruction'].item():.4f} "
                        f"kl={losses['kl'].item():.5f} "
                        f"total={losses['total'].item():.4f}"
                    )

                    # TensorBoard logging
                    log_dict = {
                        "train/loss_total": losses["total"].item(),
                        "train/loss_reconstruction": losses["reconstruction"].item(),
                        "train/loss_kl": losses["kl"].item(),
                        "train/learning_rate": lr,
                        "train/epoch": epoch,
                    }
                    if "perceptual" in losses:
                        log_dict["train/loss_perceptual"] = losses["perceptual"].item()
                    accelerator.log(log_dict, step=global_step)

                if global_step % config.training.checkpoint_every == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_checkpoint(
                            accelerator.unwrap_model(vae), optimizer, global_step, epoch,
                            config, config.training.checkpoint_dir, prefix="vae",
                            scheduler=lr_scheduler,
                        )

                if config.training.max_steps > 0 and global_step >= config.training.max_steps:
                    should_stop = True
                    break

        if epoch_steps == 0:
            accelerator.print(
                "WARNING: VAE dataloader produced 0 batches. Check online sources or local data paths."
            )

        # Clear GPU cache between epochs
        clear_gpu_cache()

        if should_stop:
            break

    accelerator.end_training()
    accelerator.print("VAE training complete!")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--online", action="store_true", help="Use online streaming dataset")
    p.add_argument("--limit", type=int, default=None,
                   help="With --online, limit online video clips per epoch")
    args = p.parse_args()
    train_vae(use_online=args.online, online_limit=args.limit)
