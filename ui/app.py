"""Gradio Web UI for local video generation.

Provides a user-friendly interface with tabs for:
  - Text-to-Video generation
  - Image-to-Video generation
  - Settings and model configuration
"""

from __future__ import annotations
import sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr

logger = logging.getLogger(__name__)

# Global pipeline (lazy-loaded)
_pipeline = None


def _get_pipeline():
    """Lazy-load the generation pipeline."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from utils.config import load_config
    from models.vae.vae import VideoVAE
    from models.dit.dit import VideoDiT
    from models.text_encoder.t5_encoder import T5TextEncoder
    from models.image_encoder.clip_encoder import CLIPImageEncoder
    from models.conditioning.fusion import ConditioningFusion
    from models.schedulers.ddim import DDIMScheduler
    from inference.pipeline import VideoPipeline
    from utils.checkpoint import load_checkpoint

    config = load_config("configs/model/dit_small.yaml")
    vae_cfg = config.model.vae
    dit_cfg = config.model.dit

    vae = VideoVAE(in_channels=vae_cfg.in_channels, latent_channels=vae_cfg.latent_channels,
                   base_channels=vae_cfg.base_channels, channel_multipliers=vae_cfg.channel_multipliers)

    dit = VideoDiT(in_channels=vae_cfg.latent_channels, hidden_size=dit_cfg.hidden_size,
                   num_layers=dit_cfg.num_layers, num_heads=dit_cfg.num_heads,
                   mlp_ratio=dit_cfg.mlp_ratio, patch_size=tuple(dit_cfg.patch_size),
                   cross_attention_dim=config.model.text_encoder.hidden_size)

    vae_ckpt = Path("checkpoints/vae/vae_latest.pt")
    if vae_ckpt.exists():
        load_checkpoint(str(vae_ckpt), vae)

    text_encoder = T5TextEncoder(model_name=config.model.text_encoder.name,
                                  max_length=config.model.text_encoder.max_length,
                                  output_hidden_size=config.model.text_encoder.hidden_size)

    img_cfg = config.model.image_encoder
    image_encoder = CLIPImageEncoder(
        model_name=img_cfg.name,
        output_hidden_size=img_cfg.hidden_size,
    )
    conditioning = ConditioningFusion(
        hidden_size=config.model.text_encoder.hidden_size,
        text_dim=config.model.text_encoder.hidden_size,
        image_dim=img_cfg.hidden_size,
    )

    # Load checkpoints if available
    ckpt = Path("checkpoints/checkpoint_latest.pt")
    if ckpt.exists():
        load_checkpoint(
            str(ckpt), dit,
            extra_modules={"conditioning": conditioning},
        )

    scheduler = DDIMScheduler(num_timesteps=config.diffusion.num_timesteps,
                              beta_schedule=config.diffusion.beta_schedule,
                              prediction_type=config.diffusion.prediction_type)

    _pipeline = VideoPipeline(
        dit=dit, vae=vae, text_encoder=text_encoder, scheduler=scheduler,
        image_encoder=image_encoder, conditioning=conditioning,
    )
    return _pipeline


def generate_text_to_video(prompt, negative_prompt, duration, chunk_frames,
                           height, width, steps, cfg_scale, seed, fps):
    """Generate video from text prompt."""
    try:
        pipeline = _get_pipeline()
        output_path = f"output/ui_t2v_{seed}.mp4"
        Path("output").mkdir(exist_ok=True)
        result = pipeline.generate(
            prompt=prompt, negative_prompt=negative_prompt,
            num_frames=int(chunk_frames), duration_seconds=float(duration),
            chunk_frames=int(chunk_frames), height=int(height), width=int(width),
            num_inference_steps=int(steps), guidance_scale=float(cfg_scale),
            seed=int(seed), fps=float(fps), output_path=output_path,
        )
        return result.get("video_path", "Generation complete")
    except Exception as e:
        return f"Error: {str(e)}"


def generate_image_to_video(image, prompt, duration, chunk_frames,
                            height, width, steps, cfg_scale, seed, fps):
    """Generate video from reference image."""
    try:
        from PIL import Image
        if image is None:
            return "Please upload a reference image"
        pipeline = _get_pipeline()
        output_path = f"output/ui_i2v_{seed}.mp4"
        Path("output").mkdir(exist_ok=True)
        ref_image = Image.fromarray(image).convert("RGB")
        result = pipeline.generate(
            prompt=prompt, num_frames=int(chunk_frames),
            duration_seconds=float(duration), chunk_frames=int(chunk_frames),
            height=int(height), width=int(width), num_inference_steps=int(steps),
            guidance_scale=float(cfg_scale), seed=int(seed), fps=float(fps),
            reference_image=ref_image, output_path=output_path,
        )
        return result.get("video_path", "Generation complete")
    except Exception as e:
        return f"Error: {str(e)}"


def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(title="VideoGen - Local AI Video Generator", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎬 VideoGen — Local AI Video Generation\n"
                    "*Fully local text-to-video and image-to-video generation*")

        with gr.Tabs():
            # Text-to-Video Tab
            with gr.Tab("Text to Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t2v_prompt = gr.Textbox(label="Prompt", lines=3,
                                                placeholder="a golden retriever running in a meadow")
                        t2v_neg = gr.Textbox(label="Negative Prompt", lines=2,
                                            placeholder="blurry, low quality, static")
                        with gr.Row():
                            t2v_duration = gr.Slider(4, 3600, value=4, step=1, label="Duration (seconds)")
                            t2v_chunk_frames = gr.Slider(16, 128, value=32, step=4, label="Chunk Frames")
                            t2v_fps = gr.Slider(1, 30, value=8, step=1, label="FPS")
                        with gr.Row():
                            t2v_height = gr.Slider(64, 512, value=256, step=64, label="Height")
                            t2v_width = gr.Slider(64, 512, value=256, step=64, label="Width")
                        with gr.Row():
                            t2v_steps = gr.Slider(10, 200, value=50, step=5, label="Steps")
                            t2v_cfg = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="CFG Scale")
                        t2v_seed = gr.Number(value=42, label="Seed", precision=0)
                        t2v_btn = gr.Button("🎬 Generate Video", variant="primary")
                    with gr.Column(scale=1):
                        t2v_output = gr.Video(label="Generated Video")

                t2v_btn.click(generate_text_to_video,
                             inputs=[t2v_prompt, t2v_neg, t2v_duration, t2v_chunk_frames, t2v_height,
                                    t2v_width, t2v_steps, t2v_cfg, t2v_seed, t2v_fps],
                             outputs=t2v_output)

            # Image-to-Video Tab
            with gr.Tab("Image to Video"):
                with gr.Row():
                    with gr.Column(scale=1):
                        i2v_image = gr.Image(label="Reference Image", type="numpy")
                        i2v_prompt = gr.Textbox(label="Motion Prompt", lines=2,
                                                value="animate this image with gentle motion")
                        with gr.Row():
                            i2v_duration = gr.Slider(4, 3600, value=4, step=1, label="Duration (seconds)")
                            i2v_chunk_frames = gr.Slider(16, 128, value=32, step=4, label="Chunk Frames")
                            i2v_fps = gr.Slider(1, 30, value=8, step=1, label="FPS")
                        with gr.Row():
                            i2v_height = gr.Slider(64, 512, value=256, step=64, label="Height")
                            i2v_width = gr.Slider(64, 512, value=256, step=64, label="Width")
                        with gr.Row():
                            i2v_steps = gr.Slider(10, 200, value=50, step=5, label="Steps")
                            i2v_cfg = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="CFG Scale")
                        i2v_seed = gr.Number(value=42, label="Seed", precision=0)
                        i2v_btn = gr.Button("🖼️ Animate Image", variant="primary")
                    with gr.Column(scale=1):
                        i2v_output = gr.Video(label="Generated Video")

                i2v_btn.click(generate_image_to_video,
                             inputs=[i2v_image, i2v_prompt, i2v_duration, i2v_chunk_frames, i2v_height,
                                    i2v_width, i2v_steps, i2v_cfg, i2v_seed, i2v_fps],
                             outputs=i2v_output)

            # Info Tab
            with gr.Tab("About"):
                gr.Markdown("""
## Architecture
- **Diffusion Transformer (DiT)** with spatial-temporal attention
- **3D Causal VAE** for video compression (4×8×8)
- **T5-small** text encoder (local, no API)
- **CLIP ViT** image encoder for I2V conditioning
- **DDIM** sampler for fast inference

## Commands
```bash
# Train VAE
python scripts/train.py --mode vae

# Train DiT
python scripts/train.py --mode dit

# Generate video
python scripts/generate.py --mode t2v --prompt "your prompt"

# Prepare dataset
python scripts/prepare_dataset.py --input /path/to/videos --output data/
```
""")

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
