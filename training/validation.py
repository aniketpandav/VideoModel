"""Validation and sample generation during training.

Periodically generates sample videos using DDIM sampling to monitor training
progress.  This provides qualitative feedback on model quality without
interrupting the training loop for long.

Usage from the trainer::

    from training.validation import generate_training_samples, save_sample_videos

    videos = generate_training_samples(model, vae, text_encoder, noise_scheduler, prompts)
    save_sample_videos(videos, prompts, save_dir="samples", step=global_step)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from models.schedulers.ddim import DDIMScheduler
from utils.video_utils import save_video_frames, tensor_to_frames

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default prompts used when no explicit prompt list is provided.
# ---------------------------------------------------------------------------
DEFAULT_SAMPLE_PROMPTS: list[str] = [
    "a serene mountain landscape at sunset with golden clouds",
    "ocean waves gently crashing on a sandy beach",
    "a dense forest with sunlight filtering through the canopy",
]


# ---------------------------------------------------------------------------
# Sample generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_training_samples(
    model: nn.Module,
    vae: nn.Module,
    text_encoder: nn.Module,
    noise_scheduler,  # DDPMScheduler — used only to read schedule config
    prompts: list[str],
    num_frames: int = 32,
    height: int = 256,
    width: int = 256,
    num_inference_steps: int = 20,
    guidance_scale: float = 4.0,
    device: torch.device | str = "cuda",
    image_encoder: nn.Module | None = None,
    conditioning: nn.Module | None = None,
) -> list[torch.Tensor]:
    """Generate sample videos using DDIM for quality monitoring.

    Creates one video per prompt by running the full inference pipeline:
    encode text → sample latent noise → DDIM denoise → VAE decode.

    Args:
        model: DiT denoising model (will be set to eval mode temporarily).
        vae: Video VAE used to decode latents back to pixel space.
        text_encoder: T5 (or compatible) text encoder.
        noise_scheduler: The training DDPMScheduler.  Its ``num_timesteps``,
            ``prediction_type``, and beta schedule parameters are forwarded to
            a temporary :class:`DDIMScheduler` for inference.
        prompts: Text prompts to generate videos for.
        num_frames: Number of video frames to generate.
        height: Output frame height in pixels.
        width: Output frame width in pixels.
        num_inference_steps: DDIM denoising steps (fewer = faster).
        guidance_scale: Classifier-free guidance weight.
        device: Target device.
        image_encoder: Optional image encoder for image-conditioned models.
        conditioning: Optional conditioning module that fuses text + image.

    Returns:
        List of video tensors, each of shape ``[C, T, H, W]`` in ``[0, 1]``.
    """
    device = torch.device(device) if isinstance(device, str) else device

    # Put the model in eval mode; restore afterwards.
    was_training = model.training
    model.eval()

    # ------------------------------------------------------------------
    # Build a DDIM scheduler that mirrors the training noise schedule.
    # ------------------------------------------------------------------
    ddim = DDIMScheduler(
        num_timesteps=noise_scheduler.num_timesteps,
        num_inference_steps=num_inference_steps,
        eta=0.0,  # deterministic sampling
        beta_schedule=(
            "cosine"
            if hasattr(noise_scheduler, "betas")
            and len(noise_scheduler.betas) > 0
            else "cosine"
        ),
        prediction_type=getattr(noise_scheduler, "prediction_type", "epsilon"),
    )
    ddim.set_timesteps(num_inference_steps)

    # ------------------------------------------------------------------
    # Derive the latent spatial/temporal dimensions from the VAE.
    # VideoVAE compresses by (T/4, H/8, W/8).
    # ------------------------------------------------------------------
    latent_channels: int = getattr(vae, "latent_channels", 4)
    latent_shape = vae.get_latent_shape((1, 3, num_frames, height, width))
    # latent_shape = (Z, T', H', W')
    _, latent_t, latent_h, latent_w = latent_shape

    # ------------------------------------------------------------------
    # Encode an empty prompt once for unconditional guidance.
    # ------------------------------------------------------------------
    uncond_out = text_encoder.encode([""])
    uncond_embeds = uncond_out["prompt_embeds"].to(device)  # [1, S, D]
    uncond_mask = uncond_out["attention_mask"].to(device)    # [1, S]

    videos: list[torch.Tensor] = []

    for prompt in prompts:
        logger.info("Generating sample for: %r", prompt)

        # 1) Encode the text prompt ----------------------------------------
        text_out = text_encoder.encode([prompt])
        text_embeds = text_out["prompt_embeds"].to(device)  # [1, S, D]
        text_mask = text_out["attention_mask"].to(device)    # [1, S]

        # Apply optional image conditioning --------------------------------
        cond_embeds = text_embeds
        cond_mask = text_mask
        uncond_embeds_cur = uncond_embeds
        uncond_mask_cur = uncond_mask

        if conditioning is not None and image_encoder is not None:
            # Use zero image embeddings for unconditional path
            dummy_img_embeds = torch.zeros(
                1,
                text_embeds.shape[1],
                text_embeds.shape[2],
                device=device,
            )
            cond_out = conditioning(
                text_embeds=text_embeds,
                text_mask=text_mask,
                image_embeds=dummy_img_embeds,
            )
            cond_embeds = cond_out["context"]
            cond_mask = cond_out["context_mask"]

            uncond_cond_out = conditioning(
                text_embeds=uncond_embeds,
                text_mask=uncond_mask,
                image_embeds=dummy_img_embeds,
            )
            uncond_embeds_cur = uncond_cond_out["context"]
            uncond_mask_cur = uncond_cond_out["context_mask"]

        # 2) Start from pure noise -----------------------------------------
        latents = torch.randn(
            1, latent_channels, latent_t, latent_h, latent_w,
            device=device,
        )

        # 3) DDIM denoising loop -------------------------------------------
        for i, t in enumerate(ddim.timesteps):
            t_tensor = torch.full(
                (1,), t.item(), device=device, dtype=torch.long,
            )

            # Classifier-free guidance: conditional + unconditional pass
            if guidance_scale > 1.0:
                noise_cond = model(
                    latents, t_tensor,
                    context=cond_embeds,
                    context_mask=cond_mask,
                )
                noise_uncond = model(
                    latents, t_tensor,
                    context=uncond_embeds_cur,
                    context_mask=uncond_mask_cur,
                )
                noise_pred = noise_uncond + guidance_scale * (
                    noise_cond - noise_uncond
                )
            else:
                noise_pred = model(
                    latents, t_tensor,
                    context=cond_embeds,
                    context_mask=cond_mask,
                )

            # Compute previous timestep
            prev_t = (
                ddim.timesteps[i + 1].item()
                if i < len(ddim.timesteps) - 1
                else 0
            )
            latents = ddim.step(
                noise_pred, t.item(), latents, prev_timestep=prev_t,
            )

        # 4) Decode latents → pixel space ----------------------------------
        decoded = vae.decode(latents)  # [1, C, T, H, W] in [-1, 1]
        # Map to [0, 1] for saving
        video = (decoded[0].clamp(-1, 1) + 1.0) / 2.0  # [C, T, H, W]
        videos.append(video.cpu())

    # Restore original training state.
    if was_training:
        model.train()

    return videos


# ---------------------------------------------------------------------------
# Video saving
# ---------------------------------------------------------------------------

def save_sample_videos(
    videos: list[torch.Tensor],
    prompts: list[str],
    save_dir: str | Path,
    step: int,
    fps: int = 8,
) -> list[Path]:
    """Save generated sample videos to disk.

    Each video is stored as an MP4 file named
    ``step{step:08d}_{index:02d}.mp4`` inside *save_dir*.

    Args:
        videos: List of video tensors ``[C, T, H, W]`` in ``[0, 1]``.
        prompts: Corresponding text prompts (logged alongside the file).
        save_dir: Directory to write videos into (created if needed).
        step: Current training step number (used in the filename).
        fps: Frames per second for the output videos.

    Returns:
        List of :class:`Path` objects pointing to the saved files.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []

    for idx, (video, prompt) in enumerate(zip(videos, prompts)):
        filename = f"step{step:08d}_{idx:02d}.mp4"
        output_path = save_dir / filename

        # tensor_to_frames expects [C, T, H, W] and returns [T, H, W, C] uint8
        # Our tensors are already in [0, 1], so we pass denormalize=False.
        frames = tensor_to_frames(video, denormalize=False)  # [T, H, W, C]

        save_video_frames(frames, output_path, fps=float(fps))
        logger.info(
            "Saved sample %d/%d  step=%d  path=%s  prompt=%r",
            idx + 1,
            len(videos),
            step,
            output_path,
            prompt,
        )

        saved_paths.append(output_path)

    return saved_paths
