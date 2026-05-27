"""Unified video generation inference pipeline.

Supports:
  - Text-to-video generation
  - Image-to-video generation (with reference image conditioning)
  - Classifier-free guidance
  - Negative prompts
  - Seed control for reproducibility
  - Configurable resolution, frame count, FPS
"""

from __future__ import annotations
import logging, time
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image

from models.vae.vae import VideoVAE
from models.dit.dit import VideoDiT
from models.text_encoder.t5_encoder import T5TextEncoder
from models.image_encoder.clip_encoder import CLIPImageEncoder
from models.conditioning.fusion import ConditioningFusion
from models.schedulers.ddim import DDIMScheduler
from utils.duration import normalize_num_frames, resolve_frame_count, split_frame_count
from utils.video_utils import VideoFrameWriter, tensor_to_frames, save_video_frames
from utils.memory import clear_gpu_cache, log_gpu_memory

logger = logging.getLogger(__name__)


class VideoPipeline:
    """Unified inference pipeline for video generation.

    Orchestrates: text encoding -> image encoding -> conditioning fusion ->
    DDIM denoising loop -> VAE decoding -> video output.

    Args:
        dit: Diffusion Transformer model.
        vae: Video VAE decoder.
        text_encoder: T5 text encoder.
        scheduler: DDIM noise scheduler.
        image_encoder: Optional CLIP image encoder.
        conditioning: Optional conditioning fusion module.
        device: Target device.
        dtype: Computation dtype.
    """

    def __init__(self, dit: VideoDiT, vae: VideoVAE, text_encoder: T5TextEncoder,
                 scheduler: DDIMScheduler, image_encoder: Optional[CLIPImageEncoder] = None,
                 conditioning: Optional[ConditioningFusion] = None,
                 device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.dit = dit.to(device=device).eval()
        self.vae = vae.to(device=device).eval()
        if hasattr(text_encoder, "device"):
            text_encoder.device = str(device)
        self.text_encoder = text_encoder
        self.scheduler = scheduler
        self.image_encoder = image_encoder
        if self.image_encoder is not None and hasattr(self.image_encoder, "device"):
            self.image_encoder.device = str(device)
        self.conditioning = conditioning.to(device).eval() if conditioning is not None else None
        self.device = torch.device(device)
        self.dtype = dtype

    @torch.no_grad()
    def generate(
        self, prompt: str, negative_prompt: str = "",
        num_frames: int = 16, height: int = 256, width: int = 256,
        num_inference_steps: int = 50, guidance_scale: float = 7.5,
        seed: Optional[int] = None, fps: float = 8.0,
        duration_seconds: Optional[float] = None,
        chunk_frames: Optional[int] = None,
        reference_image: Optional[Image.Image] = None,
        output_path: Optional[str] = None,
        callback: Optional[callable] = None,
    ) -> dict:
        """Generate a video from text and/or image input.

        Args:
            prompt: Text prompt describing the video.
            negative_prompt: What to avoid in generation.
            num_frames: Number of output frames when duration_seconds is not set.
            height: Video height in pixels.
            width: Video width in pixels.
            num_inference_steps: DDIM sampling steps.
            guidance_scale: CFG scale (1.0=no guidance, 7.5=default).
            seed: Random seed for reproducibility.
            fps: Output video FPS.
            duration_seconds: Optional duration in seconds. Valid range is 4 to 3600.
            chunk_frames: Optional frames per denoising chunk for long generation.
            reference_image: Optional reference image for I2V.
            output_path: Optional path to save video.
            callback: Optional callback(step, total_steps, latent).

        Returns:
            Dict with 'frames' (numpy), 'video_path' (if saved), 'latent'.
        """
        start_time = time.time()
        resolved_frames = resolve_frame_count(num_frames, duration_seconds, fps)
        if duration_seconds is not None:
            default_chunk_frames = max(num_frames, int(fps * 4))
            chunk_size = normalize_num_frames(chunk_frames or default_chunk_frames)
            if resolved_frames > chunk_size:
                return self._generate_chunked(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    total_frames=resolved_frames,
                    chunk_frames=chunk_size,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=seed,
                    fps=fps,
                    reference_image=reference_image,
                    output_path=output_path,
                    callback=callback,
                    start_time=start_time,
                )
        num_frames = resolved_frames
        logger.info(f"Generating: '{prompt}' | {num_frames}f @ {height}x{width} | "
                   f"{num_inference_steps} steps | cfg={guidance_scale}")

        # Set seed
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            torch.manual_seed(seed)

        # Encode text
        text_out = self.text_encoder.encode(
            [prompt],
            negative_prompts=[negative_prompt] if negative_prompt else None,
        )

        # Encode reference image if provided
        image_embeds = None
        if reference_image is not None and self.image_encoder is not None:
            image_embeds = self.image_encoder.encode_for_conditioning([reference_image])

        # Build conditioning
        use_fusion = self.conditioning is not None and image_embeds is not None
        if use_fusion:
            cond = self.conditioning(
                text_embeds=text_out["prompt_embeds"],
                text_mask=text_out["attention_mask"],
                image_embeds=image_embeds,
            )
            uncond = self.conditioning(
                text_embeds=text_out.get("negative_embeds", torch.zeros_like(text_out["prompt_embeds"])),
                text_mask=text_out.get("negative_mask", text_out["attention_mask"]),
                image_embeds=None,
                force_unconditional=True,
            )
        else:
            cond = {"context": text_out["prompt_embeds"], "context_mask": text_out["attention_mask"]}
            neg_embeds = text_out.get("negative_embeds", torch.zeros_like(text_out["prompt_embeds"]))
            uncond = {"context": neg_embeds, "context_mask": text_out["attention_mask"]}

        # Compute latent shape
        latent_shape = self.vae.get_latent_shape((num_frames, height, width))
        latent_shape = (1,) + latent_shape  # Add batch dim

        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps)

        # Build model function for DDIM — autocast handles mixed precision cleanly
        def model_fn(x_t, t, context=None, context_mask=None):
            with torch.autocast(self.device.type, dtype=self.dtype, enabled=(self.dtype != torch.float32)):
                return self.dit(x_t, t, context=context, context_mask=context_mask)

        # Build full conditioning dict
        full_cond = {
            "context": cond["context"],
            "context_mask": cond["context_mask"],
            "unconditional_context": uncond["context"],
            "unconditional_mask": uncond["context_mask"],
        }

        # DDIM sampling loop
        def step_callback(step, timestep, x_t):
            if callback:
                callback(step, num_inference_steps, x_t)
            if step % 10 == 0:
                logger.info(f"  Step {step}/{num_inference_steps} (t={timestep})")

        latent = self.scheduler.sample_loop(
            model_fn=model_fn, shape=latent_shape, device=self.device,
            conditioning=full_cond, guidance_scale=guidance_scale,
            generator=generator, callback=step_callback,
        )

        # Decode latent to video
        logger.info("Decoding latent to video...")
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=(self.dtype != torch.float32)):
            video = self.vae.decode(latent)  # [B, C, T, H, W]
        video = video.float()

        # Convert to frames
        frames = tensor_to_frames(video)  # [T, H, W, C] uint8

        result = {"frames": frames, "latent": latent.cpu()}

        # Save video
        if output_path:
            save_video_frames(frames, output_path, fps=fps)
            result["video_path"] = output_path

        elapsed = time.time() - start_time
        logger.info(f"Generation complete in {elapsed:.1f}s | {frames.shape[0]} frames")
        clear_gpu_cache()

        return result

    def _generate_chunked(
        self,
        prompt: str,
        negative_prompt: str,
        total_frames: int,
        chunk_frames: int,
        height: int,
        width: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: Optional[int],
        fps: float,
        reference_image: Optional[Image.Image],
        output_path: Optional[str],
        callback: Optional[callable],
        start_time: float,
    ) -> dict:
        """Generate long videos in VAE-safe chunks."""
        chunks = split_frame_count(total_frames, chunk_frames)
        logger.info(
            "Generating long video: %s frames in %s chunks of up to %s frames",
            total_frames,
            len(chunks),
            chunk_frames,
        )

        collected_frames: list[np.ndarray] = []
        last_latent = None
        writer = VideoFrameWriter(output_path, fps=fps) if output_path else None

        try:
            for chunk_idx, frames_in_chunk in enumerate(chunks):
                chunk_seed = None if seed is None else seed + chunk_idx
                logger.info(
                    "Generating chunk %s/%s (%s frames)",
                    chunk_idx + 1,
                    len(chunks),
                    frames_in_chunk,
                )
                result = self.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_frames=frames_in_chunk,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    seed=chunk_seed,
                    fps=fps,
                    duration_seconds=None,
                    chunk_frames=None,
                    reference_image=reference_image,
                    output_path=None,
                    callback=callback,
                )
                frames = result["frames"]
                last_latent = result.get("latent")
                if writer is not None:
                    writer.write(frames)
                else:
                    collected_frames.append(frames)
        finally:
            if writer is not None:
                writer.close()

        final_frames = (
            np.concatenate(collected_frames, axis=0)
            if collected_frames else None
        )
        elapsed = time.time() - start_time
        logger.info(
            "Long generation complete in %.1fs | %s frames | %s chunks",
            elapsed,
            total_frames,
            len(chunks),
        )

        result = {
            "frames": final_frames,
            "latent": last_latent,
            "num_frames": total_frames,
            "chunks": len(chunks),
        }
        if output_path:
            result["video_path"] = output_path
        clear_gpu_cache()
        return result
