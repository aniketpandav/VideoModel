"""DiT diffusion model training script — supports online streaming datasets and LoRA."""

from __future__ import annotations
import logging, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import load_configs
from utils.duration import normalize_num_frames
from models.vae.vae import VideoVAE
from models.dit.dit import VideoDiT
from models.text_encoder.t5_encoder import T5TextEncoder
from models.image_encoder.clip_encoder import CLIPImageEncoder
from models.conditioning.fusion import ConditioningFusion
from models.schedulers.ddpm import DDPMScheduler
from training.trainer import Trainer
from training.lora import inject_lora, save_lora_weights

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).parent.parent


def _load_training_config(model_config: str, train_config: str, use_online: bool):
    paths = [model_config]
    online_config = PROJECT_ROOT / "configs" / "dataset" / "online.yaml"
    if use_online and online_config.exists():
        paths.append(online_config)
    paths.append(train_config)
    return load_configs(*paths)


def _build_dataloader(config, use_online: bool, online_limit: int | None = None):
    """Return a DataLoader from online streaming or local files."""
    tc = config.training
    if use_online:
        from datasets.online_dataset import OnlineVideoDataset, online_collate_fn
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
        return DataLoader(dataset, batch_size=tc.batch_size,
                          num_workers=0,
                          collate_fn=online_collate_fn, pin_memory=True)
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
            random_flip=ds_cfg.random_flip,
        )
        if len(dataset) == 0:
            logger.error(
                "No local videos found. Run with --online to stream from the internet:\n"
                "  python scripts/train.py --mode dit --online"
            )
            return None
        return DataLoader(dataset, batch_size=tc.batch_size,
                          shuffle=True,
                          num_workers=tc.dataloader_num_workers,
                          pin_memory=True, drop_last=True,
                          collate_fn=_local_collate_fn)


def _local_collate_fn(batch):
    """Collate local dataset items."""
    import torch
    videos = torch.stack([b["video"] for b in batch])
    captions = [b["caption"] for b in batch]
    return {"video": videos, "caption": captions}


def train_dit(model_config: str = "configs/model/dit_small.yaml",
              train_config: str = "configs/training/train_dit.yaml",
              use_online: bool = False,
              online_limit: int | None = None,
              use_lora: bool = False,
              lora_rank: int = 8,
              lora_alpha: float = 1.0,
              lora_target_modules: list[str] | None = None,
              overrides: list[str] | None = None):
    """Train the DiT video diffusion model.

    Args:
        use_online: Stream training data from HuggingFace Hub (no local files needed).
        online_limit: Optional number of online clips to expose per epoch.
        use_lora: If True, inject LoRA adapters and train only those parameters.
        lora_rank: LoRA rank (lower = fewer params, higher = more expressive).
        lora_alpha: LoRA scaling factor.
        lora_target_modules: Which attention layers to target (default: ["to_q", "to_v"]).
    """
    config = _load_training_config(model_config, train_config, use_online)
    if overrides:
        config.merge_overrides(overrides)
    logging.basicConfig(level=logging.INFO)
    safe_frames = normalize_num_frames(config.dataset.num_frames)
    if safe_frames != config.dataset.num_frames:
        logger.warning(
            "Rounding DiT dataset.num_frames from %s to %s for 4x temporal VAE downsampling",
            config.dataset.num_frames, safe_frames,
        )
        config.dataset.num_frames = safe_frames

    # Build VAE (frozen encoder)
    vae_cfg = config.model.vae
    vae = VideoVAE(
        in_channels=vae_cfg.in_channels, latent_channels=vae_cfg.latent_channels,
        base_channels=vae_cfg.base_channels, channel_multipliers=vae_cfg.channel_multipliers,
        num_res_blocks=vae_cfg.num_res_blocks,
    )

    # Build DiT
    dit_cfg = config.model.dit
    dit = VideoDiT(
        in_channels=vae_cfg.latent_channels, hidden_size=dit_cfg.hidden_size,
        num_layers=dit_cfg.num_layers, num_heads=dit_cfg.num_heads,
        mlp_ratio=dit_cfg.mlp_ratio, patch_size=tuple(dit_cfg.patch_size),
        dropout=dit_cfg.dropout,
        cross_attention_dim=config.model.text_encoder.hidden_size,
        use_flash=dit_cfg.use_flash_attention,
        gradient_checkpointing=dit_cfg.gradient_checkpointing,
    )
    params = dit.get_param_count()
    logger.info(f"DiT params: {params['total']:,}")

    # LoRA injection
    lora_modules = None
    if use_lora:
        target_modules = lora_target_modules or ["to_q", "to_v"]
        lora_modules = inject_lora(
            dit, rank=lora_rank, alpha=lora_alpha,
            target_modules=target_modules,
        )
        logger.info(f"LoRA mode: rank={lora_rank}, alpha={lora_alpha}, targets={target_modules}")

    # Text encoder
    text_cfg = config.model.text_encoder
    text_encoder = T5TextEncoder(
        model_name=text_cfg.name, max_length=text_cfg.max_length,
        output_hidden_size=text_cfg.hidden_size,
    )

    image_encoder = None
    conditioning = None
    image_conditioning_cfg = config.training.get("image_conditioning", {})
    if image_conditioning_cfg.get("enabled", False):
        img_cfg = config.model.image_encoder
        image_encoder = CLIPImageEncoder(
            model_name=img_cfg.name,
            output_hidden_size=img_cfg.hidden_size,
        )
        conditioning = ConditioningFusion(
            hidden_size=text_cfg.hidden_size,
            text_dim=text_cfg.hidden_size,
            image_dim=img_cfg.hidden_size,
            cfg_dropout_prob=config.training.cfg_dropout_prob,
        )
        logger.info("Image conditioning enabled: first video frame -> CLIP tokens")

    # Noise scheduler
    diff_cfg = config.diffusion
    scheduler = DDPMScheduler(
        num_timesteps=diff_cfg.num_timesteps, beta_schedule=diff_cfg.beta_schedule,
        prediction_type=diff_cfg.prediction_type,
    )

    # DataLoader
    logger.info(f"Dataset mode: {'[Online streaming]' if use_online else '[Local files]'}")
    dataloader = _build_dataloader(config, use_online, online_limit=online_limit)
    if dataloader is None:
        return

    # Create optimizer — only LoRA params if using LoRA
    optimizer = None
    if use_lora:
        lora_params = [p for p in dit.parameters() if p.requires_grad]
        logger.info(f"LoRA trainable params: {sum(p.numel() for p in lora_params):,}")
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=config.training.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            eps=config.training.adam_epsilon,
            weight_decay=config.training.weight_decay,
        )

    # Train
    trainer = Trainer(model=dit, vae=vae, text_encoder=text_encoder,
                      noise_scheduler=scheduler, config=config,
                      optimizer=optimizer,
                      image_encoder=image_encoder, conditioning=conditioning)
    trainer.train(dataloader)

    # Save LoRA weights separately after training
    if use_lora:
        lora_path = Path(config.training.checkpoint_dir) / "lora_weights.pt"
        save_lora_weights(dit, str(lora_path))
        logger.info(f"LoRA weights saved to {lora_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--online", action="store_true")
    p.add_argument("--limit", type=int, default=None,
                   help="With --online, limit online video clips per epoch")
    p.add_argument("--lora", action="store_true", help="Use LoRA fine-tuning")
    p.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    p.add_argument("--lora_alpha", type=float, default=1.0, help="LoRA alpha")
    args = p.parse_args()
    train_dit(use_online=args.online, online_limit=args.limit,
              use_lora=args.lora, lora_rank=args.lora_rank,
              lora_alpha=args.lora_alpha)
