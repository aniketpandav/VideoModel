# AI Video Generation Platform — Architecture Blueprint

> **Goal:** A startup-grade, open-source-only platform that produces Text-to-Video, Script-to-Video,
> Image+Prompt-to-Video, fully-automated faceless YouTube videos, and 4K cinematic AI video —
> from 4 seconds up to ~1 hour.
>
> **Strategy:** *Hybrid.* Ship the product **now** on best-in-class pretrained open models, while
> building an in-house training/fine-tuning pipeline **in parallel** to specialize models over time.

---

## 0. Reality Check (read this first)

A senior architect does not promise what physics and VRAM forbid. The hard truths that shape every
decision below:

| Claim people want | Reality (2026) | What we actually do |
|---|---|---|
| "Generate a coherent 1-hour 4K video in one pass" | **Impossible.** No open or closed model has the context length or VRAM. SOTA single-pass is ~5–10s at 480–720p. | **Chunked hierarchical generation + stitching + upscaling.** A 1-hour video = hundreds of 4–10s shots, planned by an LLM "director", anchored by reference images, blended at boundaries. |
| "Train my own video model from scratch on free GPUs" | **Not competitive.** A useful base video model needs thousands of GPU-hours on A100/H100 + millions of clips. Free tiers cap at ~16 GB T4/P100. | Free tier = **toy-scale pretraining for learning** + **LoRA/temporal/character fine-tuning** of existing bases. Real pretraining = rent spot A100s later. |
| "Native 4K diffusion" | Memory-prohibitive and rarely better. | Generate at model-native res (480–768p) → **spatial super-res (Real-ESRGAN/SUPIR, tiled)** → **temporal interpolation (RIFE) to 24/30/60 fps** → **color grade**. |
| "Perfect character/temporal consistency" | Best-effort. Drift is the central unsolved problem. | Reference-image anchoring (IP-Adapter/PuLID), last-frame chaining (I2V), latent overlap blending, FramePack-style context compression, an LLM "story memory". |

**Bottom line:** the platform is an *orchestration system* around diffusion backbones, not a single
magic model. Most engineering value is in the **director, consistency, long-video, and rendering**
layers — not the backbone itself.

---

## 1. System Overview

```
                         ┌──────────────────────────────────────────────────────────┐
                         │                    CONTROL PLANE                          │
                         │  REST API (FastAPI)  ·  Job Queue (Redis/RQ|Celery)       │
                         │  Auth · Billing · Project store (Postgres) · Asset store  │
                         └───────────────┬──────────────────────────────────────────┘
                                         │ job
        ┌────────────────────────────────┼────────────────────────────────────────────┐
        ▼                                ▼                                              ▼
┌───────────────┐            ┌──────────────────────────┐                  ┌────────────────────┐
│  IDEATION /   │            │   DIRECTOR / PLANNER      │                  │  ASSET GENERATION  │
│  SCRIPT LLM   │ ─script──▶ │  • script→scene parser    │ ─shot list──▶    │  GPU WORKERS       │
│ (Qwen/Llama)  │            │  • emotion/tone tagger    │                  │ ┌────────────────┐ │
└───────────────┘            │  • shot-list generator    │                  │ │ Keyframe (FLUX)│ │
                             │  • camera/pacing engine   │                  │ │ Video backbone │ │
                             │  • continuity "memory"    │◀──state/refs──▶  │ │  (Wan/CogX/LTX)│ │
                             └──────────────────────────┘                  │ │ Voice (TTS)    │ │
                                                                            │ │ Music (MusicGen)│ │
                                                                            │ └────────────────┘ │
                                                                            └─────────┬──────────┘
                                                                                      │ raw clips+audio
                                                                                      ▼
                                                                       ┌─────────────────────────┐
                                                                       │  POST / RENDER PIPELINE  │
                                                                       │  interp(RIFE)→upscale    │
                                                                       │  (ESRGAN/SUPIR)→grade    │
                                                                       │  →subtitles→mux(FFmpeg)  │
                                                                       └────────────┬─────────────┘
                                                                                    ▼
                                                            final.mp4 · thumbnail · SEO meta → (YouTube upload)
```

### Module → Implementation map

| Layer | Module | Primary OSS choice | Fallback / notes |
|---|---|---|---|
| Understanding | Text encoder | T5-XXL / UMT5 (ships with backbone) | CLIP for image-prompt fusion |
| Understanding | Script engine | Qwen2.5-7B / Llama-3.1-8B (Ollama/vLLM) | API LLM if allowed |
| Planning | Scene planner | LLM + Pydantic schema | rule-based fallback |
| Planning | Storyboard | FLUX.1-schnell / SDXL keyframes | SD3.5 |
| Vision | Image encoder | CLIP-ViT / SigLIP, IP-Adapter, PuLID | InstantID |
| Backbone | Video diffusion | **Wan 2.1**, **CogVideoX-5B**, **LTX-Video** | HunyuanVideo, Mochi-1, SVD (I2V) |
| Consistency | Temporal engine | latent overlap blend + I2V chaining | optical-flow warp (RAFT) |
| Consistency | Long-context memory | FramePack + LLM story state + ref bank | StreamingT2V |
| Smoothness | Frame interpolation | **RIFE** (Practical-RIFE) | FILM |
| Editing | Multi-shot transitions | FFmpeg xfade + LLM-chosen cuts | GL transitions |
| Audio | Voiceover | **Kokoro-82M**, XTTS-v2, F5-TTS | Piper (fast), Bark |
| Audio | Lip-sync (optional) | LatentSync / MuseTalk / Wav2Lip | SadTalker (talking head) |
| Audio | Music | MusicGen / Stable Audio Open / ACE-Step | royalty-free library |
| Audio | Sync + subtitles | **faster-whisper / WhisperX** (word ts) | aeneas forced-align |
| Finishing | Upscale to 4K | **Real-ESRGAN (tiled)**, SUPIR (quality) | CCSR, SwinIR |
| Finishing | Color/HDR grade | FFmpeg `lut3d` + OpenColorIO | film grain + bloom |
| Output | Thumbnail | FLUX + PIL text overlay | SDXL |

---

## 2. Hybrid Model Strategy

Two tracks run concurrently. **Track A pays the bills; Track B builds the moat.**

```
TRACK A  (Production, ship today)        TRACK B  (Training, build over months)
─────────────────────────────────       ──────────────────────────────────────
Use pretrained backbones as-is:          1. Toy pretrain (learning) — tiny DiT+VAE on UCF/WebVid subset
  Wan 2.1 / CogVideoX / LTX              2. LoRA fine-tune backbone for a STYLE (e.g. "cinematic noir")
Build orchestration around them:         3. Character-consistency LoRA (DreamBooth-style on ref set)
  director, consistency, long-video,     4. Temporal LoRA (motion module fine-tune)
  4K render, YouTube automation          5. Eventually: rent A100 spot → larger fine-tunes/distill
                                         Feed Track A's best outputs back as training data (flywheel).
```

**Why hybrid wins:** you are revenue-capable in weeks (Track A), and every video you render
becomes labeled training data (caption + script + shots) for Track B. The data flywheel is the
real startup asset, not the model weights.

### Backbone selection (the decision that matters most)

| Model | Params | Native res / len | Free-GPU friendly? | License | Best for |
|---|---|---|---|---|---|
| **LTX-Video 0.9** | ~2B DiT | 768×512, ~5–10s, very fast | ✅ T4/P100 w/ offload | Apache-2.0 (model) | speed, drafts, real-time-ish |
| **Wan 2.1-1.3B** | 1.3B | 480p, 5s | ✅ ~8 GB | Apache-2.0 | best quality/VRAM ratio |
| **Wan 2.1-14B** | 14B | 720p | ⚠️ needs offload+quant | Apache-2.0 | quality ceiling |
| **CogVideoX-5B** | 5B | 720×480, 6s, 49f | ✅ ~5–12 GB w/ offload+quant | custom (commercial-friendly) | strong T2V + I2V |
| **HunyuanVideo** | 13B | 720p | ❌ heavy (24 GB+ w/ quant) | custom | top quality, paid GPU |
| **Mochi-1** | 10B | 480p | ⚠️ | Apache-2.0 | motion quality |
| **SVD (img2vid)** | ~1.5B | 576×1024, 14/25f | ✅ | research/commercial-tiered | I2V chaining, B-roll |

> **Recommended default stack:** **LTX-Video** for fast drafts/previews → **Wan 2.1** (1.3B free, 14B paid)
> or **CogVideoX-5B** for final shots → **SVD** for cheap I2V B-roll chaining.
> Always verify each model's license against your commercial use before shipping.

---

## 3. Full AI Architecture (per-module deep dive)

### 3.1 Text encoder & script understanding
- **Text encoder** is bundled with the backbone (T5-XXL for CogVideoX, UMT5 for Wan). Don't reinvent;
  reuse the backbone's encoder so prompt embeddings match training distribution.
- **Script understanding engine** = an LLM that converts free-form scripts/ideas into a typed scene graph.
  Run locally with **Ollama** (Qwen2.5-7B / Llama-3.1-8B) — free, private, good enough for structure.

### 3.2 Scene planning & storyboard
- LLM emits a `StoryPlan` (Pydantic schema, §16.3): scenes → shots, each with prompt, duration, camera
  move, emotion, characters present, transition-in.
- **Storyboard generator** renders one **keyframe per shot** with FLUX.1-schnell (4-step, fast). Keyframes
  serve two purposes: (1) human/LLM review before expensive video gen, (2) **I2V anchor** for the backbone.

### 3.3 Image encoder & character consistency
- **IP-Adapter** (image prompt) + **PuLID/InstantID** (identity) injected into keyframe generation so the
  same character appears across shots. Maintain a per-project **reference bank** (face/style embeddings).
- For video, anchor each shot via **I2V** from its keyframe → the backbone inherits identity from the image.

### 3.4 Video backbone
- Diffusion Transformer (DiT) with 3D/causal attention over a spatiotemporal **latent** from a 3D-VAE.
  We *use* pretrained ones (Track A) and *train* a tiny one (Track B) — see §12.

### 3.5 Temporal consistency engine
Three stacked techniques (cheap → expensive):
1. **I2V chaining** — last frame of shot *n* becomes the conditioning image of shot *n+1*.
2. **Latent overlap blending** — generate shots with K overlapping latent frames; cross-fade in latent space.
3. **Optical-flow guidance (RAFT)** — warp + consistency loss for high-end shots (slow; reserve for hero shots).

### 3.6 Long-context memory module (the 1-hour trick)
```
LLM Story State (JSON)         Reference Bank (embeddings)        FramePack context
─────────────────────          ───────────────────────────       ──────────────────
characters[], locations[],     face/style/keyframe vectors        compress prior frames
plot beats, "last seen at",  + IP-Adapter anchors per char        into fixed-size context
emotional arc, time-of-day                                        → constant VRAM regardless
                                                                    of total length
```
- **FramePack** (frame-context packing) is the key enabler: it compresses the growing frame history into a
  fixed-length context so autoregressive long video runs in **~6 GB** VRAM. This is what makes "long video on
  free GPU" actually true.
- The LLM story state prevents *semantic* drift (plot/character), FramePack prevents *visual* drift,
  the reference bank prevents *identity* drift.

### 3.7 Frame interpolation
- **RIFE / Practical-RIFE**: generate at low fps (8–16) → interpolate ×2/×4 to 24/30/60 fps. Cuts generation
  cost 3–4× and yields smoother motion than native high-fps gen. FILM as a higher-quality fallback.

### 3.8 Multi-shot scene transitions
- LLM picks transition per cut (hard cut / crossfade / whip-pan / match-cut). Implemented with FFmpeg
  `xfade` filters (50+ presets). Match-cuts use the keyframe similarity to choose the cut point.

### 3.9–3.13 Audio, subtitles, voiceover, music, 4K upscale
- See §7 (4K), §8 (YouTube audio), §16 (code).

### Module summary table

| Module | Input | Output | VRAM | Latency note |
|---|---|---|---|---|
| Script LLM | idea/script | StoryPlan JSON | 6–16 GB (or API) | seconds |
| Keyframe gen | shot prompt + refs | PNG per shot | 6–12 GB | ~1–4 s/img (schnell) |
| Video backbone | keyframe + prompt | 4–10 s clip | 6–24 GB | 30 s–5 min/clip |
| RIFE interp | clip @ 8fps | clip @ 24–60fps | 2–4 GB | fast |
| Upscale 4K | 720p clip | 4K clip | tiled, 6–12 GB | slow (tiled) |
| TTS | text | wav | 2–6 GB | realtime-ish |
| Whisper subs | wav | word-timed ASS | 2–6 GB | fast (faster-whisper) |
| FFmpeg mux | all | final mp4 | CPU | fast |

---

## 4. Long-Video Generation Strategy (4 s → 1 hour)

```
1h target
   │  LLM Director splits into ACTS → SCENES → SHOTS (4–10s each)  → ~360–600 shots
   ▼
For each shot:
   keyframe(FLUX+IP-Adapter)  ──►  backbone I2V (8–16 fps)  ──►  RIFE→24/30/60fps
        ▲ identity from ref bank        ▲ last-frame chain          │
        └──────────── LLM story state + FramePack context ──────────┘
   ▼
Boundary blend (latent/pixel crossfade) → per-scene grade → upscale tiles → encode
   ▼
Concat all scenes (FFmpeg) + global music bed + ducking + subtitles → final 1h MP4
```

**Memory math:** never hold the whole video in VRAM/RAM. Stream shot-by-shot to disk as encoded
chunks, then do a final lossless `concat`. A 1-hour 4K H.265 file is ~3–8 GB on disk; intermediate
ProRes/lossless chunks are larger, so render to a fast scratch disk and delete chunks after concat.

**Quality vs. cost knobs:** draft pass with LTX at 480p/8fps for the *whole* video (human review),
then re-render only approved shots at final quality. This is how you keep a 1-hour render affordable.

---

## 5. Script-to-Video System

Pipeline: `raw script → segment → tag → shot-list → keyframes → clips → assemble`.

1. **Parse** long script into scenes (LLM + scene-break heuristics: `INT./EXT.`, blank-line breaks, time cues).
2. **Detect emotion & cinematic tone** per scene (LLM classifier → {mood, palette, pace, lens}).
3. **Generate matching shots** — map tone → prompt modifiers + camera move + shot size (WS/MS/CU).
4. **Cinematic transitions** — LLM chooses cut type from emotional delta between scenes.
5. **Visual consistency** — shared reference bank + style LoRA + fixed seed family per character/location.
6. **Timeline** — every shot gets `(start, end, layer)`; audio/subs/music align to this timeline.
7. **Multi-character planning** — track who's in frame; allocate IP-Adapter refs per character.
8. **Auto shot-list** — output a CSV/JSON shot list (see schema §16.3) the renderer consumes.

---

## 6. 4K Cinematic Rendering Pipeline

```
backbone clip (720×480, 8–16fps)
   │ 1. RIFE temporal interp ──────────► 24 / 30 / 60 fps
   │ 2. Spatial super-res (TILED):
   │       Real-ESRGAN x4  (fast, default)
   │       SUPIR / CCSR    (diffusion restore, hero shots only)
   │ 3. Detail/denoise pass (optional)
   │ 4. Color grade: FFmpeg lut3d (cinematic LUT) + film grain + subtle bloom + vignette
   │ 5. "HDR-style" tonemap (PQ/HLG optional; mostly stylized SDR grade)
   ▼
encode: HEVC (libx265) or AV1 (libsvtav1), yuv420p10le, CRF 16–20, 4K bitrate ladder
```

- **Native 4K**: avoid. Generating at 2160p is VRAM-prohibitive and quality gain is marginal vs. 720p→4K SR.
- **Latent upscaling**: optional middle path — upscale in latent space (×1.5–2) before VAE decode for sharper
  base, *then* ESRGAN. Costs VRAM; use only when you have headroom.
- **Tile-based rendering**: ESRGAN/SUPIR process the frame in overlapping tiles (e.g. 512² with 32px overlap)
  to fit 4K into 8–12 GB. Stitch with feathered seams.
- **fps support**: RIFE multiplier configurable (×2/×3/×4); target 24 (cinema), 30 (standard), 60 (smooth/Shorts).
- **Color/lighting**: LUT packs (Kodak/Fuji emulation), per-scene palette from the tone tagger;
  "realistic lighting" is achieved at *generation* time via prompt + depth-aware relighting, not post.

---

## 7. Faceless YouTube Automation Pipeline

```
idea → SEO title → script(LLM) → scene breakdown → B-roll(video gen) →
narration(TTS) → align(Whisper) → dynamic subtitles → motion gfx + Ken-Burns →
music bed + ducking → thumbnail → export → description + tags → (upload)
```

### Channel archetypes (presets)

| Archetype | Visual style | Voice | Pacing | Notes |
|---|---|---|---|---|
| **Documentary** | realistic B-roll, slow pans | calm, authoritative (XTTS) | slow, 0.9× | maps/archival-look stills + Ken Burns |
| **Educational** | diagrams + B-roll, on-screen text | clear, neutral (Kokoro) | medium | heavy motion-graphics layer |
| **Horror/storytelling** | dark, high-contrast, grain | breathy, low (Bark/XTTS) | very slow, tension cuts | LUT = teal/orange noir, ambient drones |
| **Finance/news** | charts, tickers, city B-roll | crisp, fast (Piper) | fast, punchy | data overlays, lower-thirds |

### Short-form vs long-form
- **Shorts (9:16, ≤60s, 60fps)**: 5–8 fast shots, big animated captions, hook in first 1.5s, loop-friendly end.
- **Long-form (16:9, 5–20 min)**: chaptered, intro/outro, B-roll under continuous narration, music bed.
- Same engine, different **format preset** (aspect, fps, caption style, shot density, music intensity).

### Faceless audio specifics
- **Narration**: TTS from script (Kokoro for speed/quality, XTTS for cloned/branded voice).
- **Dynamic subtitles**: WhisperX word-level timestamps → animated ASS karaoke captions (word pop/color).
- **Music + ducking**: MusicGen bed; FFmpeg `sidechaincompress` ducks music under narration automatically.

---

## 8. Fully Automated YouTube Pipeline (idea → upload)

| Step | Tool | Output |
|---|---|---|
| 1. Video idea | LLM (trend prompt) | topic + angle |
| 2. SEO title | LLM (CTR-optimized, ≤60 chars) | title variants |
| 3. Script | LLM (archetype-tuned) | timed script |
| 4. Scene breakdown | director (§5) | shot list JSON |
| 5. AI visuals | backbone + keyframes | clips |
| 6. Narration | TTS | voice.wav |
| 7. Subtitles | WhisperX → ASS | captions |
| 8. Transitions/edit | FFmpeg xfade | edited timeline |
| 9. Thumbnail | FLUX + PIL overlay | thumb.png |
| 10. Export MP4 | FFmpeg (H.265/AV1) | final.mp4 |
| 11. Description | LLM | desc.txt |
| 12. Tags | LLM + keyword tool | tags[] |
| 13. **Upload** | YouTube Data API v3 (`google-api-python-client`, OAuth) | published video |

> **Upload caution:** the YouTube Data API has strict quota (default 10k units/day; an upload costs ~1600).
> Respect platform ToS on automated/AI content (disclosure of synthetic media is now required). Build the
> uploader as an *opt-in human-in-the-loop* step by default — auto-publish is a policy risk, not a tech limit.

---

## 9. Advanced Cinematic AI Systems

| System | How | OSS hooks |
|---|---|---|
| Camera movement | prompt tokens ("slow dolly-in", "crane up") + backbone camera control; for stills use Ken-Burns `zoompan` | CameraCtrl/MotionCtrl LoRAs |
| AI-directed cinematography | LLM director assigns shot size, lens, angle, rule-of-thirds composition per beat | prompt templates |
| Story pacing engine | LLM maps emotional arc → shot duration & cut rhythm (tension = shorter cuts) | timeline weights |
| Emotional scene control | tone tagger → palette/lens/music intensity per scene | LUT + MusicGen params |
| Dynamic lighting | depth-aware relight (Depth-Anything-V2 + IC-Light) | IC-Light |
| Multi-angle | re-gen same beat with angle variants from same keyframe (I2V) | seed/camera variants |
| Depth-aware scene gen | Depth-Anything-V2 → ControlNet-depth for parallax/3D feel | ControlNet |
| AI editing | LLM picks best take, trims dead frames, beat-syncs cuts to music | scene-detect + FFmpeg |

---

## 10. Dataset Pipeline (Track B)

```
collect → filter(NSFW/quality) → dedup → scene-segment → frame+audio extract →
caption(VLM) → align caption↔clip → metadata → shard (WebDataset/Parquet)
```

| Stage | Tool | Notes |
|---|---|---|
| Collection | yt-dlp, public datasets (WebVid-derived, Panda-70M, OpenVid-1M) | **respect copyright/ToS & licenses** |
| NSFW/quality filter | LAION CLIP-NSFW, aesthetic predictor, blur/scene-cut filters | drop low-motion, watermark, static |
| Dedup | CLIP/pHash near-dup clustering | avoid train leakage |
| Scene segmentation | PySceneDetect | cut into single-shot clips |
| Frame/audio extract | FFmpeg / decord | fixed fps, keyframe sampling |
| Captioning | Qwen2-VL / CogVLM2-Video / LLaVA-Video | dense + short captions |
| Alignment | timestamp join (clip↔caption↔transcript via Whisper) | for script-caption pairs |
| Metadata | resolution, fps, motion score, aesthetic, tags | for curriculum/filtering |
| Sharding | **WebDataset** (.tar) or Parquet | streamable, cloud-friendly |
| Distributed preproc | Ray / Spark / `accelerate` multi-proc | scale to TBs |

> **Legal:** dataset sourcing is the #1 startup risk. Prefer permissively-licensed/public-domain sources,
> keep provenance metadata, and document it. This is a business decision, not just engineering.

---

## 11. Training Workflow (Track B)

| Stage | Goal | Method | Realistic on free GPU? |
|---|---|---|---|
| Base pretrain | learn video latent dynamics | 3D-VAE + DiT, flow-matching/DDPM | **toy-scale only** (learning) |
| Fine-tune | adapt base to domain | full or partial FT | small models / LoRA only |
| **LoRA** | cheap style/character/motion adapt | PEFT LoRA on attn/MLP | ✅ yes (best ROI) |
| Style transfer | "look" (noir, anime, doc) | LoRA on curated style set | ✅ |
| Character consistency | fixed identity | DreamBooth-style + reg images | ✅ |
| Temporal consistency | smoother motion | motion-module FT + flow loss | ⚠️ limited |
| Long-video | extend context | FramePack-style context training | ⚠️ research |
| Distributed | scale | **FSDP / DeepSpeed ZeRO-2/3**, `accelerate` | multi-GPU paid |
| Mixed precision | speed/VRAM | bf16/fp16 autocast | ✅ |
| Grad checkpointing | fit big models | `torch.utils.checkpoint` | ✅ essential |
| Checkpointing | resume/safety | safetensors + EMA, every N steps to durable store | ✅ |
| Evaluation | quality tracking | **FVD, CLIPSIM, VBench**, human A/B | ✅ |
| Benchmarking | speed/VRAM | tokens/s, sec/clip, peak VRAM | ✅ |

**Recommended near-term focus:** skip base pretraining (Track A covers production). Spend Track B effort on
**LoRA style + character-consistency** fine-tunes of Wan/CogVideoX — highest quality-per-GPU-hour and directly
improves the product.

---

## 12. Free GPU Providers & VRAM-Saving

| Provider | Free GPU | VRAM | Caps | Good for |
|---|---|---|---|---|
| **Kaggle** | 2× T4 *or* P100 | 16 GB (×2) | 30 h/week, 12h/session | LoRA training, batch inference |
| **Colab Free** | T4 (when available) | 16 GB | idle timeouts, no guarantee | demos, light inference |
| **Lightning AI** | free monthly credits | varies | Studios, persistent | dev + small training |
| **HF ZeroGPU** | A100 slices (Spaces) | shared | time-sliced, per-call | hosted demos/inference |
| **Modal** | free monthly credits | A10/A100 | serverless | bursty GPU jobs, API backend |
| **RunPod / Vast.ai** | (paid, cheap spot) | up to H100 | preemptible | real training/4K renders |

### VRAM-saving toolbox (apply in this order)
1. **`enable_model_cpu_offload()`** — biggest win; keeps only the active module on GPU.
2. **Sequential CPU offload** — for the tightest budgets (slower).
3. **VAE tiling + slicing** — decode large frames in tiles (`vae.enable_tiling()`).
4. **Quantization** — fp8 (optimum-quanto / torchao) or int8/nf4 (bitsandbytes) on the transformer.
5. **Mixed precision** — bf16 (Ampere+) / fp16 (T4).
6. **Gradient checkpointing** (training) + **8-bit optimizer** (bnb `AdamW8bit`).
7. **FramePack** for long video (constant VRAM).
8. **xFormers / Flash-Attention / PyTorch SDPA** memory-efficient attention.
9. **`torch.compile`** for throughput (PyTorch 2.x).
10. **Chunked generation** — never hold the whole video resident.

---

## 13. Production Engineering

### Folder structure (the repo we'll scaffold next)
```
video-platform/
├── README.md
├── BLUEPRINT.md                 # this file
├── pyproject.toml / requirements.txt
├── docker/                      # Dockerfile.gpu, Dockerfile.api, compose.yaml
├── k8s/                         # deployments, hpa, gpu nodepool, queue
├── configs/                     # hydra/yaml: model, render, channel presets
│   ├── models/ (wan.yaml, cogvideox.yaml, ltx.yaml)
│   ├── render/ (4k.yaml, shorts.yaml, longform.yaml)
│   └── channels/ (documentary.yaml, horror.yaml, finance.yaml)
├── platform/
│   ├── api/                     # FastAPI app, routers, schemas, auth
│   ├── queue/                   # RQ/Celery tasks, worker entrypoints
│   ├── director/                # script parser, scene/shot planner, pacing, memory
│   ├── generation/
│   │   ├── backbones/           # wan.py, cogvideox.py, ltx.py, svd.py (unified iface)
│   │   ├── keyframe.py          # FLUX/SDXL + IP-Adapter/PuLID
│   │   ├── consistency.py       # I2V chaining, latent blend, ref bank
│   │   └── longvideo.py         # FramePack orchestration
│   ├── audio/                   # tts.py, music.py, align.py (whisper), ducking
│   ├── render/                  # interp(rife), upscale(esrgan), grade(lut), ffmpeg
│   ├── youtube/                 # ideation, seo, thumbnail, uploader
│   └── common/                  # config, gpu_utils, logging, storage, video_utils
├── training/                    # datasets/, models/(vae,dit), train_*.py, lora.py, eval/
├── scripts/                     # cli: generate.py, make_short.py, prepare_dataset.py
├── notebooks/                   # kaggle_train.ipynb, colab_infer.ipynb
└── tests/
```

### Serving architecture
- **API**: FastAPI (async) — submit job → returns `job_id`; poll/stream status; fetch artifacts.
- **Queue**: Redis + **RQ** (simple) or **Celery** (heavier) → GPU workers pull jobs. Separate queues by
  GPU class (draft/T4 vs. final/A100).
- **Storage**: object store (S3/MinIO) for assets; Postgres for projects/jobs; Redis for queue + cache.
- **Workers**: stateless GPU containers, model weights baked or cached on a PVC; scale by queue depth.
- **Multi-stage rendering**: draft → review → final → upscale as distinct job types (different GPU tiers).

### Deployment
- **Docker**: CUDA base image, pinned torch/diffusers; separate `api` (CPU) and `worker` (GPU) images.
- **Kubernetes**: GPU node pool (nvidia device plugin), `HorizontalPodAutoscaler` on queue length
  (KEDA on Redis), node taints/tolerations for GPU, `nodeSelector` for GPU class. Use spot/preemptible
  GPU nodes for batch renders to cut cost.
- **Quantization/KV-cache**: fp8 transformer for inference; KV-cache + Flash-Attention for the LLM director
  (vLLM serves the LLM with paged KV-cache).

---

## 14. Phased Implementation Roadmap

| Phase | Deliverable | Track | Free-GPU? |
|---|---|---|---|
| **1. Dataset collection** | yt-dlp + public sets, provenance metadata | B | ✅ |
| **2. Preprocessing** | scene-cut, caption (VLM), WebDataset shards | B | ✅ (Kaggle/Ray) |
| **3. Base model** | *Track A:* wire pretrained backbones behind unified iface; *Track B:* toy DiT/VAE | A+B | A ✅ / B toy |
| **4. Temporal consistency** | I2V chaining + latent blend + FramePack long-video | A | ✅ |
| **5. Script-to-video** | LLM director, scene/shot schema, keyframes, assembly | A | ✅ |
| **6. 4K render** | RIFE → ESRGAN(tiled) → grade → encode | A | ✅ (tiled) |
| **7. YouTube automation** | idea→script→narration→subs→thumb→export→(upload) | A | ✅ |
| **8. Deploy & scale** | FastAPI + queue + Docker + K8s + autoscale | A | mixed |

**Suggested first 2 weeks (vertical slice):** Phase 5 (mini) + 7 → "text prompt → narrated, subtitled
9:16 short" running end-to-end on Kaggle. Everything else expands outward from that working slice.

---

## 15. Engineering Tradeoffs (decision log)

| Decision | Option A | Option B | Choice & why |
|---|---|---|---|
| Backbone | one big model | multi-model tiered | **Tiered** — draft cheap (LTX), final good (Wan/CogX) |
| Long video | bigger context model | chunked + FramePack | **Chunked** — only feasible path on real VRAM |
| 4K | native gen | gen-low + SR + interp | **SR+interp** — cheaper, equal/better quality |
| LLM | API | local (Ollama/vLLM) | **Local** — cost/privacy; API optional |
| Queue | Celery | RQ | **RQ** to start (simpler), Celery if needed |
| Train scope | pretrain | LoRA fine-tune | **LoRA first** — best ROI on free GPU |
| Consistency | optical-flow everywhere | tiered (chain→blend→flow) | **Tiered** — flow only for hero shots |
| Upload | auto-publish | human-in-loop | **Human-in-loop** — ToS/policy risk |

---

## 16. Representative Code Examples

> Illustrative and current-API-shaped; treat as scaffolding to harden, not copy-paste production.

### 16.1 Unified backbone inference (CogVideoX, VRAM-optimized)
```python
# platform/generation/backbones/cogvideox.py
import torch
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video

def load_t2v(model_id="THUDM/CogVideoX-5b"):
    pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()      # keep only active module on GPU
    pipe.vae.enable_tiling()             # decode big frames in tiles
    pipe.vae.enable_slicing()
    return pipe

@torch.inference_mode()
def generate(pipe, prompt, num_frames=49, steps=50, guidance=6.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    out = pipe(prompt=prompt, num_frames=num_frames,
               num_inference_steps=steps, guidance_scale=guidance, generator=g)
    return out.frames[0]   # list[PIL]

if __name__ == "__main__":
    frames = generate(load_t2v(), "a lone lighthouse in a storm, cinematic, slow dolly-in")
    export_to_video(frames, "shot.mp4", fps=8)
```

### 16.2 Long video via I2V chaining (shot-to-shot continuity)
```python
# platform/generation/longvideo.py  (chaining; swap in FramePack for constant-VRAM)
def render_sequence(i2v_pipe, shots, ref_bank):
    """shots: list of dicts {prompt, num_frames, transition}. Returns list of clip paths."""
    clips, prev_last = [], None
    for i, shot in enumerate(shots):
        init_img = prev_last or keyframe(shot["prompt"], ref_bank)   # anchor identity
        frames = i2v_pipe(image=init_img, prompt=shot["prompt"],
                          num_frames=shot["num_frames"]).frames[0]
        path = f"scratch/shot_{i:04d}.mp4"
        export_to_video(frames, path, fps=8)
        clips.append(path)
        prev_last = frames[-1]          # chain: last frame -> next shot's init
    return clips
```

### 16.3 Script parser → typed StoryPlan (LLM + Pydantic)
```python
# platform/director/schema.py
from pydantic import BaseModel
from typing import Literal

class Shot(BaseModel):
    prompt: str
    duration_s: float
    shot_size: Literal["WS","MS","CU","ECU"] = "MS"
    camera: str = "static"                      # "slow dolly-in", "pan left", ...
    emotion: str = "neutral"
    characters: list[str] = []
    transition_in: Literal["cut","crossfade","whip","match"] = "cut"

class Scene(BaseModel):
    title: str; mood: str; palette: str; shots: list[Shot]

class StoryPlan(BaseModel):
    title: str; aspect: Literal["16:9","9:16"]; fps: int; scenes: list[Scene]

# platform/director/parser.py
import json, ollama
SYS = "You are a film director. Output ONLY JSON matching the StoryPlan schema."
def parse_script(script: str, aspect="16:9", fps=24) -> StoryPlan:
    msg = f"{SYS}\nSchema fields: title,aspect,fps,scenes[{{title,mood,palette,shots[...]}}]\n\nSCRIPT:\n{script}"
    raw = ollama.chat(model="qwen2.5:7b", messages=[{"role":"user","content":msg}],
                      format="json")["message"]["content"]
    return StoryPlan(**json.loads(raw), aspect=aspect, fps=fps)
```

### 16.4 TTS narration (Kokoro — fast, high quality)
```python
# platform/audio/tts.py
import soundfile as sf
from kokoro import KPipeline   # pip install kokoro
def narrate(text, out="voice.wav", voice="af_heart", speed=1.0):
    pipe = KPipeline(lang_code="a")
    audio = []
    for _, _, chunk in pipe(text, voice=voice, speed=speed):
        audio.extend(chunk)
    sf.write(out, audio, 24000)
    return out
# Branded/cloned voice -> swap to XTTS-v2 (TTS lib) with a speaker_wav reference.
```

### 16.5 Word-level subtitles (faster-whisper → animated ASS)
```python
# platform/audio/align.py
from faster_whisper import WhisperModel
def words(wav):
    m = WhisperModel("large-v3", device="cuda", compute_type="float16")
    segs, _ = m.transcribe(wav, word_timestamps=True)
    return [(w.start, w.end, w.word) for s in segs for w in s.words]

def to_ass(words, path="subs.ass"):
    head = ("[Script Info]\nScriptType: v4.00+\n\n[V4+ Styles]\n"
            "Format: Name,Fontname,Fontsize,PrimaryColour,Bold,Alignment,MarginV\n"
            "Style: Pop,Montserrat,64,&H00FFFFFF,1,2,80\n\n"
            "[Events]\nFormat: Layer,Start,End,Style,Text\n")
    def t(s): h=int(s//3600); m=int(s%3600//60); return f"{h:d}:{m:02d}:{s%60:05.2f}"
    lines = [f"Dialogue: 0,{t(a)},{t(b)},Pop,{{\\fad(80,80)}}{w.strip()}" for a,b,w in words]
    open(path,"w",encoding="utf-8").write(head+"\n".join(lines)); return path
```

### 16.6 Frame interpolation + 4K upscale (tiled) + grade + encode (FFmpeg)
```python
# platform/render/finish.py
import subprocess
def finish_4k(in_mp4, out_mp4, lut="luts/cinematic.cube", target_fps=24):
    # 1) RIFE interpolation (CLI from Practical-RIFE) -> 2) ESRGAN tiled upscale handled upstream.
    # Here: grade + encode the already-interpolated/upscaled stream.
    vf = (f"minterpolate=fps={target_fps}:mi_mode=mci,"   # fallback interp if no RIFE
          f"lut3d='{lut}',"                               # cinematic color grade
          "noise=alls=6:allf=t,"                          # subtle film grain
          "vignette")
    subprocess.run([
        "ffmpeg","-y","-i",in_mp4,"-vf",vf,
        "-c:v","libx265","-crf","18","-pix_fmt","yuv420p10le",
        "-tag:v","hvc1","-preset","slow", out_mp4
    ], check=True)
```

### 16.7 Ken-Burns (auto zoom/pan) for still-based B-roll
```python
def ken_burns(img, out, dur=4, fps=30, w=3840, h=2160):
    import subprocess
    z = "min(zoom+0.0015,1.15)"   # slow zoom-in
    vf = f"scale={w*2}:-1,zoompan=z='{z}':d={dur*fps}:s={w}x{h}:fps={fps}"
    subprocess.run(["ffmpeg","-y","-loop","1","-i",img,"-t",str(dur),
                    "-vf",vf,"-c:v","libx264","-pix_fmt","yuv420p",out], check=True)
```

### 16.8 FastAPI + RQ job submission
```python
# platform/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from redis import Redis
from rq import Queue
from platform.queue.tasks import render_video

app = FastAPI()
q = Queue("render", connection=Redis())

class Req(BaseModel):
    prompt: str; format: str = "short"; quality: str = "draft"

@app.post("/v1/videos")
def create(req: Req):
    job = q.enqueue(render_video, req.prompt, req.format, req.quality, job_timeout=3600)
    return {"job_id": job.id, "status": "queued"}

@app.get("/v1/videos/{job_id}")
def status(job_id: str):
    job = q.fetch_job(job_id)
    return {"status": job.get_status(), "result": job.result}
```

### 16.9 GPU optimization (quantize transformer + compile)
```python
# platform/common/gpu_utils.py
import torch
from optimum.quanto import quantize, freeze, qfloat8
def optimize(pipe):
    quantize(pipe.transformer, weights=qfloat8); freeze(pipe.transformer)  # fp8 weights
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.transformer = torch.compile(pipe.transformer, mode="reduce-overhead")
    return pipe
```

### 16.10 LoRA fine-tune sketch (Track B, Accelerate)
```python
# training/lora.py  (style/character LoRA on a video DiT — sketch)
from peft import LoraConfig, get_peft_model
from accelerate import Accelerator
def add_lora(transformer, r=64, alpha=64):
    cfg = LoraConfig(r=r, lora_alpha=alpha,
                     target_modules=["to_q","to_k","to_v","to_out.0"])
    return get_peft_model(transformer, cfg)

def train(transformer, loader, steps=2000, lr=1e-4):
    acc = Accelerator(mixed_precision="bf16", gradient_accumulation_steps=4)
    transformer = add_lora(transformer); transformer.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(transformer.parameters(), lr=lr)
    transformer, opt, loader = acc.prepare(transformer, opt, loader)
    for step, batch in zip(range(steps), loader):
        with acc.accumulate(transformer):
            loss = diffusion_loss(transformer, batch)   # flow-matching / v-pred
            acc.backward(loss); opt.step(); opt.zero_grad()
        if step % 500 == 0 and acc.is_main_process:
            acc.unwrap_model(transformer).save_pretrained(f"ckpt/lora_{step}")
```

---

## 17. Verification Checklist (the "before finalizing" gate)

- [x] **All training stages covered** — pretrain, fine-tune, LoRA, style, character, temporal, long-video,
      distributed, mixed-precision, grad-checkpoint, checkpoint strategy, eval, benchmarking (§11).
- [x] **4K rendering pipeline complete** — interp → tiled SR → grade/HDR-style → encode, with native-vs-SR
      tradeoff and memory strategy (§6).
- [x] **YouTube automation complete** — idea → SEO → script → scenes → visuals → narration → subs →
      transitions → thumbnail → export → description → tags → upload (§7–8).
- [x] **T2V and I2V both implemented** — T2V backbone (§16.1) and I2V chaining/keyframe anchoring (§16.2, §3.3).
- [x] **Free-GPU compatibility** — provider matrix + VRAM toolbox + FramePack for long video (§12).
- [x] **Deployment pipeline** — FastAPI + queue + Docker + K8s + autoscale + quantization (§13).
- [x] **Long-video strategy** — chunked hierarchical gen + story memory + FramePack, with memory math (§4).

---

## 18. Immediate Next Steps

1. **Scaffold the repo** to the structure in §13 (runnable stubs: config, unified backbone iface, director
   schema, FFmpeg utils, FastAPI+RQ).
2. **Build the vertical slice**: `text prompt → narrated, subtitled 9:16 short` on Kaggle (LTX draft + Kokoro
   + WhisperX + FFmpeg).
3. **Expand outward**: add long-video (FramePack), 4K finish, channel presets, then deployment.

> Tell me which of these to start, and whether you want it tuned for a specific channel archetype
> (documentary / educational / horror / finance) — and I'll begin building.
