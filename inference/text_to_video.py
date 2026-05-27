"""Text-to-video generation entry point."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.config import load_configs
from inference.pipeline import VideoPipeline
from models.vae.vae import VideoVAE
from models.dit.dit import VideoDiT
from models.text_encoder.t5_encoder import T5TextEncoder
from models.schedulers.ddim import DDIMScheduler
from utils.checkpoint import load_checkpoint


def text_to_video(
    prompt: str, model_config: str = "configs/model/dit_small.yaml",
    checkpoint: str = "checkpoints/checkpoint_latest.pt",
    vae_checkpoint: str = "checkpoints/vae/vae_latest.pt",
    output_path: str = "output/generated.mp4",
    negative_prompt: str = "", num_frames: int = 16,
    height: int = 256, width: int = 256,
    guidance_scale: float = 7.5, steps: int = 50,
    seed: int = 42, fps: float = 8.0,
    duration_seconds: float | None = None,
    chunk_frames: int | None = None,
):
    """Generate a video from a text prompt.

    Example:
        python inference/text_to_video.py --prompt "a cat walking in a garden"
    """
    config = load_configs(model_config)

    # Build models
    vae_cfg = config.model.vae
    vae = VideoVAE(in_channels=vae_cfg.in_channels, latent_channels=vae_cfg.latent_channels,
                   base_channels=vae_cfg.base_channels, channel_multipliers=vae_cfg.channel_multipliers)
    if Path(vae_checkpoint).exists():
        load_checkpoint(vae_checkpoint, vae)

    dit_cfg = config.model.dit
    dit = VideoDiT(in_channels=vae_cfg.latent_channels, hidden_size=dit_cfg.hidden_size,
                   num_layers=dit_cfg.num_layers, num_heads=dit_cfg.num_heads,
                   mlp_ratio=dit_cfg.mlp_ratio, patch_size=tuple(dit_cfg.patch_size),
                   cross_attention_dim=config.model.text_encoder.hidden_size)
    if Path(checkpoint).exists():
        load_checkpoint(checkpoint, dit)

    text_encoder = T5TextEncoder(model_name=config.model.text_encoder.name,
                                  max_length=config.model.text_encoder.max_length,
                                  output_hidden_size=config.model.text_encoder.hidden_size)

    scheduler = DDIMScheduler(num_timesteps=config.diffusion.num_timesteps,
                              num_inference_steps=steps,
                              beta_schedule=config.diffusion.beta_schedule,
                              prediction_type=config.diffusion.prediction_type)

    pipeline = VideoPipeline(dit=dit, vae=vae, text_encoder=text_encoder, scheduler=scheduler)

    result = pipeline.generate(
        prompt=prompt, negative_prompt=negative_prompt,
        num_frames=num_frames, height=height, width=width,
        num_inference_steps=steps, guidance_scale=guidance_scale,
        seed=seed, fps=fps, duration_seconds=duration_seconds,
        chunk_frames=chunk_frames, output_path=output_path,
    )

    print(f"Video saved to: {result.get('video_path', 'N/A')}")
    return result


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Text-to-Video Generation")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--negative_prompt", type=str, default="")
    p.add_argument("--output", type=str, default="output/generated.mp4")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--height", type=int, default=256)
    p.add_argument("--width", type=int, default=256)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--cfg", type=float, default=7.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--duration", type=float, default=None,
                   help="Optional duration in seconds (4 to 3600)")
    p.add_argument("--chunk_frames", type=int, default=None,
                   help="Frames per generation chunk for long videos")
    p.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_latest.pt")
    args = p.parse_args()
    text_to_video(args.prompt, output_path=args.output, negative_prompt=args.negative_prompt,
                  num_frames=args.frames, height=args.height, width=args.width,
                  guidance_scale=args.cfg, steps=args.steps, seed=args.seed,
                  fps=args.fps, duration_seconds=args.duration,
                  chunk_frames=args.chunk_frames, checkpoint=args.checkpoint)
