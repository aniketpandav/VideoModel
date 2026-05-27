# VideoGen — Local AI Video Generation Framework

A fully local, production-grade text-to-video and image-to-video generation system built from scratch using PyTorch, CUDA, and open-source libraries only. **No paid APIs. No cloud services. No closed-source models.**

## Architecture

```
┌─────────────────┐     ┌──────────────────┐
│   Text Prompt    │     │  Reference Image  │
└───────┬─────────┘     └───────┬──────────┘
        │                       │
   ┌────▼─────┐          ┌─────▼──────┐
   │ T5-small │          │  CLIP ViT  │
   │ Encoder  │          │  Encoder   │
   └────┬─────┘          └─────┬──────┘
        │                      │
   ┌────▼──────────────────────▼────┐
   │    Conditioning Fusion          │
   │    (Cross-Attention Context)    │
   └──────────────┬─────────────────┘
                  │
   ┌──────────────▼─────────────────┐
   │   Diffusion Transformer (DiT)   │
   │  ┌─────────────────────────┐   │
   │  │ N × DiTBlock            │   │
   │  │  ├─ AdaLN (timestep)    │   │
   │  │  ├─ Self-Attention (ST) │   │
   │  │  ├─ Cross-Attention     │   │
   │  │  └─ Feed-Forward        │   │
   │  └─────────────────────────┘   │
   └──────────────┬─────────────────┘
                  │
   ┌──────────────▼─────────────────┐
   │     3D Video VAE Decoder        │
   │     (Latent → Pixel Space)      │
   └──────────────┬─────────────────┘
                  │
           ┌──────▼──────┐
           │ Output Video │
           └─────────────┘
```

## Mathematical Foundations

### Diffusion Process

**Forward process** (adding noise):
```
q(x_t | x_0) = N(x_t; √ᾱ_t · x_0, (1-ᾱ_t) · I)
```

**Training objective** (noise prediction):
```
L = E_{t, x_0, ε} [ ||ε - ε_θ(x_t, t, c)||² ]
```

**Reverse process** (DDIM sampling):
```
x_{t-1} = √ᾱ_{t-1} · x̂_0 + √(1-ᾱ_{t-1}-σ²) · ε_θ + σ · z
```

**Classifier-Free Guidance**:
```
ε̃ = ε_uncond + s · (ε_cond - ε_uncond)
```

### VAE (Variational Autoencoder)

**ELBO** (Evidence Lower Bound):
```
L_VAE = E_q[log p(x|z)] - KL(q(z|x) || p(z))
```

**KL Divergence** (closed form for Gaussians):
```
KL = -0.5 · Σ(1 + log(σ²) - μ² - σ²)
```

### Attention

**Scaled Dot-Product Attention**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

With **3D RoPE** (Rotary Positional Embedding) for spatial-temporal position encoding.

---

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt

# Install project
pip install -e .
```

### 2. Prepare Dataset

```bash
# From raw videos
python scripts/prepare_dataset.py \
  --input /path/to/your/videos \
  --output data/ \
  --clip_duration 4 \
  --fps 8 \
  --height 256 \
  --width 256

# Or create manifest.csv manually:
# path,caption
# data/clips/video1.mp4,a cat walking on grass
# data/clips/video2.mp4,ocean waves at sunset
```

### 3. Train

```bash
# Step 1: Train VAE (encodes video to latent space)
python scripts/train.py --mode vae

# Step 2: Train DiT (the diffusion model)
python scripts/train.py --mode dit

# Or train from online sources with a safe synthetic fallback
python scripts/train.py --mode vae --online --limit 1000
python scripts/train.py --mode dit --online --limit 1000

# With custom config
python scripts/train.py --mode dit \
  --model_config configs/model/dit_small.yaml \
  --train_config configs/training/train_dit.yaml
```

### 4. Generate Videos

```bash
# Text-to-Video
python scripts/generate.py --mode t2v \
  --prompt "a golden retriever playing in snow" \
  --output output/dog.mp4 \
  --duration 4 --fps 8 --chunk_frames 32 --steps 50 --cfg 7.5

# Image-to-Video
python scripts/generate.py --mode i2v \
  --image reference.jpg \
  --prompt "animate with gentle motion" \
  --output output/animated.mp4 \
  --duration 4 --fps 8 --chunk_frames 32
```

`--duration` accepts 4 to 3600 seconds. Long videos are generated in
`--chunk_frames` chunks so memory use stays bounded; for best motion quality,
train VAE and DiT with `dataset.num_frames: 32` or larger at 8 FPS.

DiT training now enables first-frame reference conditioning by default, so
image-to-video checkpoints save both the DiT weights and the conditioning
projection needed at inference.

### 5. Launch Web UI

```bash
python ui/app.py
# Open http://localhost:7860
```

---

## Project Structure

```
├── configs/          # YAML configuration files
│   ├── model/        # Model architecture configs (small, base, large)
│   └── training/     # Training hyperparameter configs
├── models/           # Core model implementations
│   ├── vae/          # 3D Video VAE (encoder + decoder)
│   ├── dit/          # Diffusion Transformer (attention, blocks, embeddings)
│   ├── text_encoder/ # T5-based text encoder
│   ├── image_encoder/# CLIP image encoder
│   ├── conditioning/ # Cross-attention fusion
│   └── schedulers/   # DDPM/DDIM noise schedulers
├── datasets/         # Dataset loading and preprocessing
├── training/         # Training loop, losses, EMA, LoRA
├── inference/        # Generation pipelines (T2V, I2V, interpolation)
├── evaluation/       # Metrics (PSNR, SSIM, temporal consistency)
├── scripts/          # CLI entry points
├── ui/               # Gradio web interface
└── utils/            # Config, checkpoints, video I/O, memory utils
```

---

## Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| GPU | RTX 3060 (12GB) | RTX 4090 (24GB) | Multi-GPU A100 |
| RAM | 16GB | 32GB | 64GB+ |
| Storage | 50GB | 200GB | 1TB+ SSD |
| CUDA | 11.8+ | 12.1+ | 12.1+ |

### VRAM Estimates

| Mode | dit_small (512d/12L) | dit_base (1024d/24L) |
|------|---------------------|---------------------|
| Inference (16f, 256×256) | ~4GB | ~12GB |
| Training (bs=1) | ~8GB | ~20GB |
| Training (bs=4) | ~16GB | ~48GB (multi-GPU) |

### Training Time Estimates

| Dataset Size | GPU | dit_small | dit_base |
|-------------|-----|-----------|----------|
| 1K videos | RTX 4090 | ~2 hours | ~8 hours |
| 10K videos | RTX 4090 | ~20 hours | ~80 hours |
| 100K videos | 4× A100 | ~2 days | ~1 week |

---

## Advanced Features

### LoRA Fine-Tuning

```python
from training.lora import inject_lora, save_lora_weights

# Inject LoRA into DiT attention layers
lora_modules = inject_lora(dit, rank=8, alpha=1.0,
                           target_modules=["to_q", "to_v"])
# Train only LoRA params (< 1% of total)
# ... training loop ...
save_lora_weights(dit, "checkpoints/my_lora.pt")
```

### DreamBooth-Style Personalization
Train on 5-20 images/clips of a subject with a unique token:
```bash
# Prepare: put subject clips in data/subject/
# Train with low LR and few steps
python scripts/train.py --mode lora \
  --overrides training.learning_rate=1e-5 training.max_steps=500
```

### ControlNet-Style Conditioning
The `ConditioningFusion` module supports additional structural inputs:
- Edge maps (Canny)
- Depth maps
- Pose estimation
Extend `image_encoder` to encode these modalities.

### Frame Interpolation
```python
from inference.interpolation import interpolate_frames
smooth_frames = interpolate_frames(frames, factor=3)  # 8fps → 24fps
```

---

## Debugging

### Common Issues

**CUDA Out of Memory**:
- Reduce `batch_size` to 1
- Enable `gradient_checkpointing: true` in config
- Reduce resolution: `height: 128, width: 128`
- Use `mixed_precision: bf16`

**Training Loss Not Decreasing**:
- Check learning rate (try 1e-4 to 1e-5)
- Verify dataset loading (check `datasets/video_dataset.py` output)
- Ensure VAE is frozen during DiT training

**Generated Videos Are Noisy**:
- Increase `num_inference_steps` (try 100+)
- Adjust `guidance_scale` (try 5.0-15.0)
- The model needs sufficient training data and iterations

---

## Model Configurations

### dit_small (Development)
- Hidden: 512, Layers: 12, Heads: 8
- ~85M parameters
- Fits on 8GB VRAM

### dit_base (Production)
- Hidden: 1024, Layers: 24, Heads: 16
- ~600M parameters
- Requires 24GB+ VRAM

---

## License

This project is provided as-is for research and educational purposes.
All components use open-source, locally-run models only.
