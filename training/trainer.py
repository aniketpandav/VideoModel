"""Main training loop for the DiT video diffusion model.

Uses HuggingFace Accelerate for distributed training, mixed precision,
gradient accumulation, and gradient checkpointing.
"""

from __future__ import annotations
import logging, time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from PIL import Image
from tqdm import tqdm

from utils.config import Config
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint
from utils.memory import clear_gpu_cache
from training.ema import EMA
from training.losses import DiffusionLoss, TemporalConsistencyLoss
from training.lr_scheduler import create_lr_scheduler

logger = logging.getLogger(__name__)


def _flatten_config(cfg_dict: dict, prefix: str = "") -> dict[str, int | float | str | bool]:
    """Flatten a nested config dict to scalar values for TensorBoard hparams."""
    flat = {}
    for k, v in cfg_dict.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            flat.update(_flatten_config(v, key))
        elif isinstance(v, (list, tuple)):
            flat[key] = str(v)
        elif isinstance(v, (int, float, str, bool)) or v is None:
            flat[key] = v if v is not None else "null"
        else:
            flat[key] = str(v)
    return flat


class Trainer:
    """Training orchestrator for the video DiT model.

    Manages the full training loop including:
    - Mixed precision (fp16/bf16) via Accelerate
    - Gradient accumulation
    - EMA weight averaging
    - Learning rate scheduling (cosine/constant/linear with warmup)
    - Min-SNR-γ loss weighting
    - Automatic checkpointing
    - TensorBoard logging
    - Sample generation during training

    Args:
        model: DiT model.
        vae: Video VAE (frozen during DiT training).
        text_encoder: Text encoder (frozen).
        noise_scheduler: DDPM noise scheduler.
        config: Training configuration.
        optimizer: Optimizer (created if None).
        scheduler: LR scheduler (created if None).
        image_encoder: Optional image encoder for image-to-video conditioning.
        conditioning: Optional conditioning fusion module.
    """

    def __init__(self, model: nn.Module, vae: nn.Module, text_encoder: nn.Module,
                 noise_scheduler, config: Config, optimizer=None, scheduler=None,
                 image_encoder: nn.Module | None = None,
                 conditioning: nn.Module | None = None):
        self.config = config
        self.noise_scheduler = noise_scheduler

        # CUDA memory optimization for low-VRAM GPUs
        if torch.cuda.is_available():
            import os
            os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # Setup accelerator
        self.accelerator = Accelerator(
            mixed_precision=config.training.mixed_precision,
            gradient_accumulation_steps=config.training.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir=config.training.log_dir,
        )

        # Create optimizer if not provided
        if optimizer is None:
            trainable_params = list(model.parameters())
            if conditioning is not None:
                trainable_params.extend(conditioning.parameters())
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=config.training.learning_rate,
                betas=(config.training.adam_beta1, config.training.adam_beta2),
                eps=config.training.adam_epsilon,
                weight_decay=config.training.weight_decay,
            )

        self.model = model
        self.vae = vae
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.conditioning = conditioning
        self.optimizer = optimizer
        self.lr_scheduler = scheduler

        # Losses
        snr_gamma = config.training.get("snr_gamma", 5.0) if hasattr(config.training, "get") else 5.0
        self.diffusion_loss = DiffusionLoss(
            loss_type=config.diffusion.get("loss_type", "mse"),
            snr_gamma=snr_gamma,
        )
        temporal_weight = config.training.get("temporal_loss_weight", 0.01) if hasattr(config.training, "get") else 0.01
        self.temporal_loss = TemporalConsistencyLoss(weight=temporal_weight)

        # EMA — will be created AFTER accelerator.prepare() to ensure correct device
        self._use_ema = config.training.use_ema
        self._ema_decay = config.training.ema_decay if hasattr(config.training, "ema_decay") else 0.9999
        self._ema_update_every = config.training.ema_update_every if hasattr(config.training, "ema_update_every") else 10
        self.ema = None

        # Freeze VAE, text encoder, and image encoder
        self.vae.eval()
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if self.image_encoder is not None:
            self.image_encoder.eval()
            self.image_encoder.requires_grad_(False)

        # State
        self.global_step = 0
        self.epoch = 0

    def train(self, train_dataloader: DataLoader) -> None:
        """Run the full training loop."""
        tc = self.config.training

        # Prepare with accelerator
        if self.conditioning is not None:
            self.model, self.conditioning, self.optimizer, train_dataloader = self.accelerator.prepare(
                self.model, self.conditioning, self.optimizer, train_dataloader
            )
        else:
            self.model, self.optimizer, train_dataloader = self.accelerator.prepare(
                self.model, self.optimizer, train_dataloader
            )
        self.vae = self.vae.to(self.accelerator.device)
        if hasattr(self.text_encoder, "device"):
            self.text_encoder.device = str(self.accelerator.device)
        self.text_encoder = self.text_encoder.to(self.accelerator.device)
        if self.image_encoder is not None and hasattr(self.image_encoder, "device"):
            self.image_encoder.device = str(self.accelerator.device)
            self.image_encoder = self.image_encoder.to(self.accelerator.device)
        self.vae.eval()
        self.text_encoder.eval()
        if self.image_encoder is not None:
            self.image_encoder.eval()

        # Create EMA AFTER model is on the correct device
        if self._use_ema:
            model_for_ema = self.accelerator.unwrap_model(self.model)
            self.ema = EMA(model_for_ema, decay=self._ema_decay,
                          update_every=self._ema_update_every)

        # Create LR scheduler if not provided
        if self.lr_scheduler is None:
            scheduler_type = tc.get("lr_scheduler", "cosine") if hasattr(tc, "get") else "cosine"
            warmup_steps = tc.get("lr_warmup_steps", 1000) if hasattr(tc, "get") else 1000
            # Estimate total training steps
            try:
                steps_per_epoch = len(train_dataloader)
            except TypeError:
                steps_per_epoch = 10000  # Fallback for IterableDataset
            total_steps = steps_per_epoch * tc.num_epochs
            if tc.max_steps > 0:
                total_steps = min(total_steps, tc.max_steps)
            total_steps = total_steps // tc.gradient_accumulation_steps

            self.lr_scheduler = create_lr_scheduler(
                self.optimizer,
                scheduler_type=scheduler_type,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            )

        if self.lr_scheduler:
            self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

        # Gradient checkpointing
        if tc.gradient_checkpointing:
            self.model.module.enable_gradient_checkpointing() if hasattr(self.model, 'module') \
                else self.model.enable_gradient_checkpointing()

        # Resume from checkpoint
        if tc.resume_from:
            self._resume(tc.resume_from)
        else:
            latest = find_latest_checkpoint(tc.checkpoint_dir)
            if latest:
                self._resume(str(latest))

        # Initialize TensorBoard tracker
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dit_training", config=_flatten_config(self.config.to_dict()))

        # Training loop
        num_epochs = tc.num_epochs
        self.accelerator.print(f"Starting training for {num_epochs} epochs")
        self.accelerator.print(f"  Batch size: {tc.batch_size} x {tc.gradient_accumulation_steps} accum")
        self.accelerator.print(f"  Mixed precision: {tc.mixed_precision}")
        self.accelerator.print(f"  LR scheduler: {tc.get('lr_scheduler', 'cosine') if hasattr(tc, 'get') else 'cosine'}")
        self.accelerator.print(f"  EMA: {self._use_ema}")

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            should_stop = self._train_epoch(train_dataloader, epoch)
            # Clear GPU cache between epochs
            clear_gpu_cache()
            if should_stop:
                break

        self.accelerator.end_training()

    def _train_epoch(self, dataloader: DataLoader, epoch: int) -> bool:
        """Train for one epoch."""
        self.model.train()
        if self.conditioning is not None:
            self.conditioning.train()
        tc = self.config.training
        progress = tqdm(dataloader, desc=f"Epoch {epoch}", disable=not self.accelerator.is_local_main_process)
        epoch_steps = 0
        # Transient CUDA error tolerance: skip the batch, clear cache, keep going.
        # Bail only if many consecutive failures suggest a real (non-transient) problem.
        max_consecutive_cuda_errors = 5
        consecutive_cuda_errors = 0

        for batch in progress:
            try:
                with self.accelerator.accumulate(self.model):
                    loss_dict = self._train_step(batch)
                    total_loss = loss_dict["total"]

                    self.accelerator.backward(total_loss)

                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), tc.max_grad_norm)

                    self.optimizer.step()
                    if self.lr_scheduler:
                        self.lr_scheduler.step()
                    self.optimizer.zero_grad()
                consecutive_cuda_errors = 0
            except RuntimeError as e:
                # Catches torch.AcceleratorError (subclass of RuntimeError) and other
                # CUDA-flavored RuntimeErrors. OOM is a different recovery story so
                # handle it explicitly, but transient cudaErrorUnknown / illegal-access
                # blips get a retry by skipping the batch.
                msg = str(e)
                is_cuda_err = "CUDA" in msg or "cuda" in msg or "cuDNN" in msg
                if not is_cuda_err:
                    raise
                consecutive_cuda_errors += 1
                logger.warning(
                    "Transient CUDA error at step %d (consecutive=%d/%d): %s",
                    self.global_step, consecutive_cuda_errors,
                    max_consecutive_cuda_errors, msg.splitlines()[0],
                )
                # Drop any half-built grads from the failed step, then reset CUDA state.
                try:
                    self.optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
                clear_gpu_cache()
                if consecutive_cuda_errors >= max_consecutive_cuda_errors:
                    logger.error(
                        "Aborting: %d consecutive CUDA errors — not transient.",
                        consecutive_cuda_errors,
                    )
                    raise
                # Skip this batch and continue with the next one.
                continue

            if self.accelerator.sync_gradients:
                self.global_step += 1
                epoch_steps += 1
                if self.ema:
                    self.ema.update(self.accelerator.unwrap_model(self.model))

                # Logging
                if self.global_step % tc.log_every == 0:
                    lr = self.optimizer.param_groups[0]["lr"]
                    progress.set_postfix(loss=f"{total_loss.item():.4f}", lr=f"{lr:.2e}",
                                        step=self.global_step)

                    # TensorBoard logging
                    if self.accelerator.is_main_process:
                        log_dict = {
                            "train/loss_total": total_loss.item(),
                            "train/loss_diffusion": loss_dict["diffusion"].item(),
                            "train/loss_temporal": loss_dict["temporal"].item(),
                            "train/learning_rate": lr,
                            "train/epoch": epoch,
                        }
                        self.accelerator.log(log_dict, step=self.global_step)

                # Checkpointing
                if self.global_step % tc.checkpoint_every == 0:
                    self.accelerator.wait_for_everyone()
                    self._save_checkpoint()

                # Sample generation
                sample_every = tc.get("sample_every", 0) if hasattr(tc, "get") else 0
                if sample_every > 0 and self.global_step % sample_every == 0:
                    self._generate_samples()

                # Max steps check
                if tc.max_steps > 0 and self.global_step >= tc.max_steps:
                    return True

        if epoch_steps == 0:
            self.accelerator.print(
                "WARNING: DiT dataloader produced 0 batches. Check online sources or local data paths."
            )
        return False

    def _train_step(self, batch: dict) -> dict[str, torch.Tensor]:
        """Single training step: encode -> add noise -> predict -> loss."""
        video = batch["video"]  # [B, C, T, H, W]
        captions = batch["caption"]

        # Encode video to latent space
        with torch.no_grad():
            latent, _, _ = self.vae.encode(video)  # [B, Z, T', H', W']

        # Encode text
        with torch.no_grad():
            text_out = self.text_encoder.encode(list(captions))
        text_embeds = text_out["prompt_embeds"]
        text_mask = text_out["attention_mask"]

        if self.conditioning is not None and self.image_encoder is not None:
            with torch.no_grad():
                ref_images = self._reference_images_from_video(video)
                image_embeds = self.image_encoder.encode_for_conditioning(ref_images)
            cond = self.conditioning(
                text_embeds=text_embeds,
                text_mask=text_mask,
                image_embeds=image_embeds,
            )
            text_embeds = cond["context"]
            text_mask = cond["context_mask"]
            # NOTE: ConditioningFusion handles CFG dropout internally.
            # Do NOT apply CFG dropout here to avoid double-dropping.
        else:
            # CFG: randomly drop conditioning (only when NOT using ConditioningFusion)
            if self.training and torch.rand(1).item() < self.config.training.cfg_dropout_prob:
                text_embeds = torch.zeros_like(text_embeds)

        # Sample noise and timestep
        noise = torch.randn_like(latent)
        timesteps = torch.randint(0, self.noise_scheduler.num_timesteps,
                                  (latent.shape[0],), device=latent.device)

        # Add noise
        noisy_latent = self.noise_scheduler.add_noise(latent, noise, timesteps)

        # Get target
        target = self.noise_scheduler.get_training_target(latent, noise, timesteps)

        # Predict noise
        noise_pred = self.model(noisy_latent, timesteps, context=text_embeds, context_mask=text_mask)

        # Compute losses
        diff_loss = self.diffusion_loss(
            noise_pred, target,
            timesteps=timesteps,
            alphas_cumprod=self.noise_scheduler.alphas_cumprod,
        )
        temp_loss = self.temporal_loss(noise_pred)

        total = diff_loss + temp_loss
        return {"total": total, "diffusion": diff_loss, "temporal": temp_loss}

    def _generate_samples(self) -> None:
        """Generate and save sample videos during training."""
        if not self.accelerator.is_main_process:
            return

        try:
            from training.validation import generate_training_samples, save_sample_videos, DEFAULT_SAMPLE_PROMPTS

            model = self.ema.ema_model if self.ema else self.accelerator.unwrap_model(self.model)
            model.eval()

            ds_cfg = self.config.dataset
            videos = generate_training_samples(
                model=model,
                vae=self.vae,
                text_encoder=self.text_encoder,
                noise_scheduler=self.noise_scheduler,
                prompts=DEFAULT_SAMPLE_PROMPTS[:2],  # Generate 2 samples for speed
                num_frames=min(ds_cfg.num_frames, 16),  # Limit frames for speed
                height=ds_cfg.frame_height,
                width=ds_cfg.frame_width,
                num_inference_steps=20,
                guidance_scale=4.0,
                device=self.accelerator.device,
                image_encoder=self.image_encoder,
                conditioning=self.conditioning,
            )

            save_dir = Path(self.config.training.log_dir) / "samples"
            save_sample_videos(
                videos=videos,
                prompts=DEFAULT_SAMPLE_PROMPTS[:2],
                save_dir=save_dir,
                step=self.global_step,
            )
            logger.info(f"Saved {len(videos)} sample videos at step {self.global_step}")

            model.train()
        except Exception as e:
            logger.warning(f"Sample generation failed: {e}")

    def _save_checkpoint(self) -> None:
        """Save training checkpoint."""
        if self.accelerator.is_main_process:
            model = self.accelerator.unwrap_model(self.model)
            extra_modules = None
            if self.conditioning is not None:
                extra_modules = {
                    "conditioning": self.accelerator.unwrap_model(self.conditioning)
                }

            # Get underlying scheduler (AcceleratedScheduler wraps it in .scheduler)
            lr_scheduler_to_save = None
            if self.lr_scheduler is not None:
                lr_scheduler_to_save = getattr(self.lr_scheduler, 'scheduler', self.lr_scheduler)

            save_checkpoint(
                model=model, optimizer=self.optimizer,
                step=self.global_step, epoch=self.epoch,
                config=self.config,
                save_dir=self.config.training.checkpoint_dir,
                ema_model=self.ema.ema_model if self.ema else None,
                scheduler=lr_scheduler_to_save,
                max_checkpoints=self.config.training.max_checkpoints,
                extra_modules=extra_modules,
            )

    def _resume(self, path: str) -> None:
        """Resume training from checkpoint."""
        model = self.accelerator.unwrap_model(self.model)
        extra_modules = None
        if self.conditioning is not None:
            extra_modules = {
                "conditioning": self.accelerator.unwrap_model(self.conditioning)
            }

        # Get underlying scheduler for loading
        lr_scheduler_to_load = None
        if self.lr_scheduler is not None:
            lr_scheduler_to_load = getattr(self.lr_scheduler, 'scheduler', self.lr_scheduler)

        info = load_checkpoint(path, model, self.optimizer,
                              ema_model=self.ema.ema_model if self.ema else None,
                              scheduler=lr_scheduler_to_load,
                              extra_modules=extra_modules)
        self.global_step = info["step"]
        self.epoch = info["epoch"]
        logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")

    @property
    def training(self) -> bool:
        """Whether the trainer is in training mode."""
        return self.model.training

    @staticmethod
    def _reference_images_from_video(video: torch.Tensor) -> list[Image.Image]:
        """Convert the first frame of each training clip to PIL images."""
        first = video[:, :, 0].detach().clamp(-1, 1)
        first = ((first + 1.0) * 127.5).byte()
        first = first.permute(0, 2, 3, 1).cpu().numpy()
        return [Image.fromarray(frame) for frame in first]
