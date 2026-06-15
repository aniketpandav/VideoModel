"""Complete Kaggle notebook — LTX-Video LoRA on YouTube Trending data.

Copy each CELL block into a separate Kaggle notebook cell and run in order.

Requirements (set before running):
  - Accelerator : T4 x1  (GPU)
  - Internet    : ON
  - Session type: Notebook

Estimated runtime: ~4-5 hours total
  Cell 1 install   ~5  min
  Cell 2 clone     ~2  min
  Cell 3 dataset   ~1  min
  Cell 4 parse     ~1  min
  Cell 5 download  ~60 min  (300 clips × ~12s each via yt-dlp)
  Cell 6 prepare   ~30 min  (FFmpeg resize + black-frame filter)
  Cell 7 train     ~120 min (1000 steps, T4 16 GB)
  Cell 8 package   ~1  min
"""

# =============================================================================
# CELL 1 — Install all dependencies
# =============================================================================
# Paste this block into Cell 1 of your Kaggle notebook, then run it.
#
# !pip install -q \
#     yt-dlp \
#     "diffusers>=0.32" \
#     peft \
#     accelerate \
#     transformers \
#     sentencepiece \
#     opencv-python-headless \
#     imageio-ffmpeg \
#     kagglehub \
#     pyyaml \
#     "numba>=0.60.0,<0.62.0" \
#     "numba-cuda>=0.22.1,<0.23.0" \
#     "cuda-core>=0.3.0,<0.4.0"
#
# The last 3 lines pin Kaggle's pre-installed RAPIDS packages (dask-cuda, cuml,
# cudf) at the versions they expect. Without them pip upgrades numba/cuda-core
# and triggers resolver warnings — those are harmless (we never use RAPIDS),
# but pinning keeps the environment clean.


# =============================================================================
# CELL 2 — Clone repo & set working directory
# =============================================================================

import os
import subprocess

REPO_SLUG  = "aniketpandav/VideoModel"
CLONE_DIR  = "/kaggle/working/video-model"

# Add GITHUB_TOKEN as a Kaggle Secret (Notebook settings → Secrets → Add)
# if the repo is private.  For a public repo the token is optional.
token    = os.environ.get("GITHUB_TOKEN", "")
repo_url = f"https://{token}@github.com/{REPO_SLUG}.git" if token else f"https://github.com/{REPO_SLUG}.git"

if not os.path.isdir(CLONE_DIR):
    subprocess.run(["git", "clone", "--depth", "1", repo_url, CLONE_DIR], check=True)
else:
    print(f"Repo already cloned at {CLONE_DIR}, pulling latest …")
    subprocess.run(["git", "-C", CLONE_DIR, "pull", "--ff-only"], check=True)

os.chdir(CLONE_DIR)
print("Working dir:", os.getcwd())


# =============================================================================
# CELL 3 — Download YouTube Trending Video Dataset metadata
# =============================================================================

import kagglehub

# Download latest version
path = kagglehub.dataset_download("rsrishav/youtube-trending-video-dataset")

print("Path to dataset files:", path)


# =============================================================================
# CELL 4 — Parse CSV → extract video IDs and build captions from titles
# =============================================================================

import os
import json
import random
from pathlib import Path

import pandas as pd

# The dataset contains one CSV per country (US, GB, IN, CA, …)
csv_files = sorted(Path(path).glob("*.csv"))
print(f"Found {len(csv_files)} country CSVs: {[f.name for f in csv_files]}")

dfs = []
for csv_path in csv_files:
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", on_bad_lines="skip")
        dfs.append(df)
    except Exception as exc:
        print(f"  Skip {csv_path.name}: {exc}")

if not dfs:
    raise RuntimeError("No CSVs loaded — check dataset path above.")

combined = pd.concat(dfs, ignore_index=True)

# Keep only the columns we need
keep = [c for c in ["video_id", "title", "channelTitle"] if c in combined.columns]
df_clean = (
    combined[keep]
    .dropna(subset=["video_id", "title"])
    .drop_duplicates("video_id")
    .reset_index(drop=True)
)
print(f"Unique trending video IDs: {len(df_clean)}")

# Build a rich caption:  "Title, Channel Name"
def _build_caption(row):
    parts = [row["title"].strip()]
    if "channelTitle" in row and pd.notna(row.get("channelTitle")):
        parts.append(row["channelTitle"].strip())
    return ", ".join(parts)

df_clean["caption"] = df_clean.apply(_build_caption, axis=1)

# Sample N diverse videos (shuffle so we get different categories each run)
N_VIDEOS = 300
sample = df_clean.sample(min(N_VIDEOS, len(df_clean)), random_state=42).reset_index(drop=True)
print(f"Selected {len(sample)} videos for download")
print(sample[["video_id", "caption"]].head(5).to_string())

# Persist the full caption map so Cell 6 can read it even if Cell 5 is re-run
os.makedirs("data/raw", exist_ok=True)
caption_map = dict(zip(sample["video_id"].astype(str), sample["caption"]))
with open("data/raw/captions.json", "w", encoding="utf-8") as fh:
    json.dump(caption_map, fh, indent=2, ensure_ascii=False)
print(f"Caption map saved → data/raw/captions.json  ({len(caption_map)} entries)")


# =============================================================================
# CELL 5 — Download first 25 seconds of each trending video via yt-dlp
# =============================================================================

import shutil
import subprocess
from pathlib import Path

RAW_DIR = Path("data/raw/youtube")
RAW_DIR.mkdir(parents=True, exist_ok=True)

with open("data/raw/captions.json", encoding="utf-8") as fh:
    caption_map = json.load(fh)

# Resolve ffmpeg for yt-dlp's --download-sections trimming.
# Prefer the system ffmpeg (Kaggle ships /usr/bin/ffmpeg); the imageio_ffmpeg
# binary is a stripped wrapper that crashes (SIGSEGV) when yt-dlp calls it.
_FFMPEG_FOR_YTDLP = shutil.which("ffmpeg") or "ffmpeg"
print(f"yt-dlp will use ffmpeg: {_FFMPEG_FOR_YTDLP}")

# Locate the yt-dlp binary installed by pip (e.g. /usr/local/bin/yt-dlp).
# Do NOT use sys.executable + "-m yt_dlp": sys.executable may point to the
# system Python (/usr/bin/python3) while yt-dlp lives in a different prefix.
_YTDLP_BIN = shutil.which("yt-dlp")
if _YTDLP_BIN is None:
    raise RuntimeError(
        "yt-dlp binary not found on PATH. "
        "Make sure Cell 1 ran: !pip install -q yt-dlp"
    )
print(f"yt-dlp binary: {_YTDLP_BIN}")

TARGET_CLIPS = 150   # stop once we have this many successful downloads
MAX_ATTEMPTS = 600   # cap total attempts to avoid running forever

video_ids = list(caption_map.keys())
# Shuffle so re-runs pick different videos
import random; random.shuffle(video_ids)
print(f"Pool: {len(video_ids)} videos | target: {TARGET_CLIPS} clips | max attempts: {MAX_ATTEMPTS}")

ok_count = 0
fail_geo = 0
fail_other = 0

for i, vid_id in enumerate(video_ids):
    if ok_count >= TARGET_CLIPS or i >= MAX_ATTEMPTS:
        break

    out_stem = RAW_DIR / vid_id
    if any(out_stem.with_suffix(ext).exists() for ext in (".mp4", ".webm", ".mkv")):
        ok_count += 1
        continue

    cmd = [
        _YTDLP_BIN,
        f"https://www.youtube.com/watch?v={vid_id}",
        "-f", "best[height<=360]/bestvideo[height<=360]+bestaudio/best",
        "--download-sections", "*0:00-0:25",
        "--ffmpeg-location", _FFMPEG_FOR_YTDLP,
        "--geo-bypass",          # attempt to bypass country restrictions
        "--no-playlist",
        "--ignore-errors",
        "--quiet",
        "--no-warnings",
        "-o", str(RAW_DIR / f"{vid_id}.%(ext)s"),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if any(out_stem.with_suffix(ext).exists() for ext in (".mp4", ".webm", ".mkv")):
            ok_count += 1
        else:
            if "not made this video available in your country" in result.stderr:
                fail_geo += 1
            else:
                fail_other += 1
                # Show first non-geo failure for diagnosis
                if fail_other == 1:
                    print(f"  [first non-geo failure — vid={vid_id}]")
                    print("  stderr:", result.stderr.strip()[:400])
    except subprocess.TimeoutExpired:
        fail_other += 1

    if (i + 1) % 30 == 0:
        print(f"  tried={i+1} | ok={ok_count} | geo-blocked={fail_geo} | other-fail={fail_other}")

# Normalise extensions so Cell 6 only sees .mp4
for f in RAW_DIR.glob("*"):
    if f.suffix.lower() in (".webm", ".mkv"):
        f.rename(f.with_suffix(".mp4"))

total_dl = len(list(RAW_DIR.glob("*.mp4")))
print(f"\nDownload complete: {total_dl} clips in {RAW_DIR}")


# =============================================================================
# CELL 6 — Prepare data: resize → filter → write metadata JSONs + splits
# =============================================================================

import subprocess
import numpy as np
import cv2
from pathlib import Path

RAW_DIR    = Path("data/raw/youtube")
PROC_DIR   = Path("data/processed")
VID_DIR    = PROC_DIR / "videos"
META_DIR   = PROC_DIR / "metadata"
SPLIT_DIR  = PROC_DIR / "splits"
for d in (VID_DIR, META_DIR, SPLIT_DIR):
    d.mkdir(parents=True, exist_ok=True)

with open("data/raw/captions.json", encoding="utf-8") as fh:
    caption_map = json.load(fh)

# Processing parameters — must match configs/train_lora.yaml
RESOLUTION = 256
FPS        = 16
MIN_DUR    = 3.0   # seconds
MAX_DUR    = 26.0  # seconds (we downloaded 25s clips)

# Use FFmpeg bundled with imageio-ffmpeg (no system install needed)
try:
    import imageio_ffmpeg
    FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FFMPEG = "ffmpeg"
print(f"FFmpeg: {FFMPEG}")

raw_clips = sorted(RAW_DIR.glob("*.mp4"))
print(f"Raw clips to process: {len(raw_clips)}")

processed_paths = []
skipped = 0

for i, src in enumerate(raw_clips):
    vid_id = src.stem
    caption = caption_map.get(vid_id, f"a trending video clip")

    # ── Probe with OpenCV ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        cap.release()
        skipped += 1
        continue
    fps_src  = cap.get(cv2.CAP_PROP_FPS) or 25.0
    nframes  = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    duration = nframes / fps_src if fps_src > 0 else 0.0

    if not (MIN_DUR <= duration <= MAX_DUR) or w < 64 or h < 64:
        skipped += 1
        continue

    # ── FFmpeg: resize to 256×256 square with letterboxing, fix fps ───────
    clip_id  = f"{vid_id}_{i:05d}"
    out_vid  = VID_DIR  / f"{clip_id}.mp4"
    out_meta = META_DIR / f"{clip_id}.json"

    cmd = [
        FFMPEG, "-y", "-i", str(src),
        "-vf", (
            f"scale={RESOLUTION}:{RESOLUTION}:force_original_aspect_ratio=decrease,"
            f"pad={RESOLUTION}:{RESOLUTION}:(ow-iw)/2:(oh-ih)/2,"
            f"fps={FPS}"
        ),
        "-c:v", "libx264", "-crf", "23",
        "-an",           # strip audio — not needed for training
        str(out_vid),
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=60)
    if result.returncode != 0 or not out_vid.exists():
        skipped += 1
        continue

    # ── Black-frame guard ──────────────────────────────────────────────────
    cap2 = cv2.VideoCapture(str(out_vid))
    ok, frame = cap2.read()
    cap2.release()
    if not ok or frame.mean() < 8:
        out_vid.unlink(missing_ok=True)
        skipped += 1
        continue

    # ── Write metadata JSON (caption used during LoRA training) ───────────
    meta = {
        "id": clip_id,
        "source": src.name,
        "caption": caption,
        "tags": {"source": "youtube_trending"},
        "duration": round(duration, 2),
        "original_resolution": [w, h],
        "processed_resolution": [RESOLUTION, RESOLUTION],
        "fps": FPS,
    }
    with open(out_meta, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    processed_paths.append(str(out_vid))

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(raw_clips)} | kept={len(processed_paths)} | skipped={skipped}")

print(f"\nData preparation complete: {len(processed_paths)} clips ready")

if len(processed_paths) < 10:
    raise RuntimeError(
        f"Only {len(processed_paths)} clips passed filters — too few for training. "
        "Check that yt-dlp downloaded videos in Cell 5."
    )

# ── Train / val split (95 / 5) ─────────────────────────────────────────────
random.seed(42)
random.shuffle(processed_paths)
n_train = int(len(processed_paths) * 0.95)
train_paths = processed_paths[:n_train]
val_paths   = processed_paths[n_train:]

(SPLIT_DIR / "train.txt").write_text("\n".join(train_paths), encoding="utf-8")
(SPLIT_DIR / "val.txt").write_text("\n".join(val_paths),   encoding="utf-8")

print(f"Split: {len(train_paths)} train / {len(val_paths)} val")
print(f"Processed data → {PROC_DIR.resolve()}")
print("Ready to train ✓")


# =============================================================================
# CELL 7 — Train LTX-Video LoRA  (~120 min on T4)
# =============================================================================
# This launches train_lora.py which:
#   - Loads Lightricks/LTX-Video from HuggingFace (~8 GB download, cached)
#   - Freezes VAE + text encoder; only trains LoRA adapters on the transformer
#   - Uses YouTube titles (from metadata JSONs) as text conditioning
#   - Saves checkpoints to runs/lora/ every 250 steps
#   - Saves final adapter in diffusers format to runs/lora/last_lora/

import os
import shutil
import subprocess
from pathlib import Path

# Resolve the repo root: Cell 2 clones to /kaggle/working/video-model and
# calls os.chdir() there, but if cwd drifted back to /kaggle/working we
# find the script explicitly so the cell still works either way.
_REPO_ROOT = Path(os.getcwd())
_TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train_lora.py"
if not _TRAIN_SCRIPT.exists():
    _REPO_ROOT = Path("/kaggle/working/video-model")
    _TRAIN_SCRIPT = _REPO_ROOT / "scripts" / "train_lora.py"
if not _TRAIN_SCRIPT.exists():
    raise FileNotFoundError(
        f"train_lora.py not found. Expected it at {_TRAIN_SCRIPT}.\n"
        "Make sure Cell 2 cloned the repo and os.chdir() ran successfully."
    )
print(f"Repo root: {_REPO_ROOT}")

# Use the same Python that has the packages (shutil.which prefers /usr/local/bin)
_PYTHON = shutil.which("python3") or "python3"

result = subprocess.run(
    [_PYTHON, str(_TRAIN_SCRIPT), "--config", str(_REPO_ROOT / "configs/train_lora.yaml")],
    cwd=str(_REPO_ROOT),
    check=True,
)

print("\nTraining complete.")
print("Checkpoint: runs/lora/last_lora/")


# =============================================================================
# CELL 8 — Package weights for download
# =============================================================================
import os
import shutil
from pathlib import Path

# Ensure we're in the repo root (same as Cell 2 set)
_REPO_ROOT = Path("/kaggle/working/video-model")
if _REPO_ROOT.exists():
    os.chdir(_REPO_ROOT)

RUNS_DIR = Path("runs/lora")

# Search order: diffusers adapter dirs, then raw lora.pt checkpoints
adapter_dirs = sorted(RUNS_DIR.glob("step_*/adapter"))
lora_pts     = sorted(RUNS_DIR.glob("step_*/lora.pt"))

if adapter_dirs:
    LORA_DIR  = adapter_dirs[-1]          # latest step, diffusers format
    ZIP_SOURCE = (LORA_DIR.parent, LORA_DIR.name)
    print(f"Using adapter checkpoint: {LORA_DIR}")
elif lora_pts:
    LORA_DIR  = lora_pts[-1].parent       # e.g. step_000050/
    ZIP_SOURCE = (LORA_DIR.parent, LORA_DIR.name)
    print(f"Using lora.pt checkpoint: {LORA_DIR}")
else:
    # Print what IS in runs/ to help diagnose
    all_files = list(RUNS_DIR.rglob("*")) if RUNS_DIR.exists() else []
    print(f"Contents of {RUNS_DIR}: {[str(f) for f in all_files[:20]] or 'EMPTY / NOT FOUND'}")
    raise FileNotFoundError(
        "No LoRA checkpoint found in runs/lora/.\n"
        "Likely cause: training crashed before the first checkpoint (step 50).\n"
        "Re-run Cell 7 and check for errors — a CalledProcessError means "
        "the training script exited non-zero (dataset missing, OOM, import error)."
    )

out_zip = "/kaggle/working/lora_weights"
shutil.make_archive(out_zip, "zip", root_dir=str(ZIP_SOURCE[0]),
                    base_dir=ZIP_SOURCE[1])
zip_path = Path(f"{out_zip}.zip")
size_mb = zip_path.stat().st_size / 1024 ** 2

print(f"Packaged: {zip_path}  ({size_mb:.1f} MB)")
print()
print("Next steps:")
print("  1. Download lora_weights.zip from the Kaggle Output panel (right sidebar)")
print("  2. Unzip on your local machine")
print("  3. Set env vars and start the API:")
print("       $env:VDM_BACKBONE  = 'ltx'")
print("       $env:VDM_LORA_PATH = 'C:\\path\\to\\lora_weights'")
print("       $env:VDM_UPSCALE   = '4k'")
print("       .\\env\\Scripts\\python.exe -m uvicorn serve.api:app --port 8000")
