"""FastAPI video-generation service — production API.

Endpoints
---------
  POST   /v1/generate/text-to-video    Mode 1: text → video
  POST   /v1/generate/image-to-video   Mode 2: reference image(s) + text → video
  GET    /v1/jobs/{job_id}             Poll job status
  DELETE /v1/jobs/{job_id}             Cancel a queued/running job
  GET    /v1/jobs/{job_id}/download    Stream the completed video file
  GET    /v1/health                    Service health + backbone info

Legacy (backward compat):
  POST   /v1/videos                    Original single-endpoint request
  GET    /v1/videos/{job_id}           Legacy status poll
  GET    /v1/videos/{job_id}/file      Legacy download

Run:
    uvicorn serve.api:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .backbones import get_backbone
from .chunker import LongVideoChunker
from .pipeline import VideoPipeline
from .prompt_engine import PromptEngine, PromptValidationError, SafetyError
from .quality import QualityValidator
from .queue import AsyncJobQueue
from .render import encode_video, make_preview_gif, save_clip, to_uint8_clip
from .storage import StorageManager

log = logging.getLogger(__name__)

OUT_DIR = os.environ.get("VDM_OUT", "outputs")
MAX_RETRIES = int(os.environ.get("VDM_MAX_RETRIES", "3"))

# ---------------------------------------------------------------------------
# Global singletons (initialised in lifespan)
# ---------------------------------------------------------------------------
_backbone = None
_pipeline: Optional[VideoPipeline] = None
_prompt_engine: Optional[PromptEngine] = None
_validator: Optional[QualityValidator] = None
_storage: Optional[StorageManager] = None
_queue: Optional[AsyncJobQueue] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _backbone, _pipeline, _prompt_engine, _validator, _storage, _queue

    _storage = StorageManager(root=OUT_DIR)
    _prompt_engine = PromptEngine()
    _validator = QualityValidator(clip_check=False)
    _backbone = get_backbone()
    chunker = LongVideoChunker(_backbone, _prompt_engine)
    _pipeline = VideoPipeline(_backbone, chunker)
    _queue = AsyncJobQueue()
    _queue.start()
    log.info("Video platform ready. Backbone: %s", _backbone.name)

    yield

    # Graceful shutdown: drain queue
    if _queue:
        await _queue.shutdown(timeout=30.0)


app = FastAPI(
    title="Video Platform API",
    version="1.0.0",
    description="Enterprise-grade AI video generation — Mode 1 (T2V) and Mode 2 (I2V)",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class TextToVideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000,
                        examples=["a serene mountain lake at golden hour"])
    duration_seconds: float = Field(default=5.0, ge=2.0, le=3600.0)
    output_format: Literal["mp4", "mov", "avi", "mkv", "webm"] = "mp4"
    resolution: str = Field(default="512x512",
                             examples=["512x512", "1280x720", "704x480"])
    aspect_ratio: Literal["16:9", "9:16", "1:1", "4:3"] = "16:9"
    fps: int = Field(default=24, ge=8, le=60)
    quality: Literal["draft", "standard", "high", "cinematic"] = "standard"
    seed: Optional[int] = None
    enhance_prompt: bool = True


class ImageToVideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    reference_images: list[str] = Field(
        ..., min_length=1, max_length=8,
        description="Base64-encoded images (with or without data URI prefix) or raw base64 strings",
    )
    duration_seconds: float = Field(default=5.0, ge=2.0, le=3600.0)
    output_format: Literal["mp4", "mov", "avi", "mkv", "webm"] = "mp4"
    resolution: str = Field(default="512x512")
    fps: int = Field(default=24, ge=8, le=60)
    quality: Literal["draft", "standard", "high"] = "standard"
    seed: Optional[int] = None
    enhance_prompt: bool = True


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float
    output_url: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = 0
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    completed_at: Optional[str] = None
    mode: Optional[int] = None


# Legacy schema (kept for backward compat)
class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, examples=["a red square moving right"])
    steps: int = Field(50, ge=4, le=250)
    seed: int = 0
    fmt: str = Field("mp4", pattern="^(mp4|gif)$")


# ---------------------------------------------------------------------------
# Job execution handler (runs inside AsyncJobQueue worker)
# ---------------------------------------------------------------------------

async def _execute_job(
    enhanced_request,
    mode: int,
    job_id: str,
    cancel_event: asyncio.Event,
):
    """Async wrapper: runs blocking pipeline call in a thread executor."""
    loop = asyncio.get_event_loop()
    storage = _storage
    pipeline = _pipeline
    validator = _validator

    def _progress(pct: float):
        storage.jobs.update(job_id, progress=pct / 100.0, status="processing")

    def _run():
        for attempt in range(1, MAX_RETRIES + 1):
            if cancel_event.is_set():
                return None
            try:
                if mode == 1:
                    frames = pipeline.text_to_video(
                        enhanced_request, progress_callback=_progress
                    )
                else:
                    frames = pipeline.image_to_video(
                        enhanced_request, progress_callback=_progress
                    )

                storage.jobs.update(job_id, status="validating", progress=0.97)

                # Only validate resolution/duration when the backbone actually
                # supports dynamic sizing (ToyBackbone has fixed 32×32 / N frames).
                bb_caps = getattr(_backbone, "capabilities", {})
                dynamic_res = bb_caps.get("dynamic_resolution", True)
                dynamic_dur = bb_caps.get("dynamic_duration", True)

                result = validator.validate(
                    frames,
                    expected_resolution=enhanced_request.resolution if dynamic_res else None,
                    expected_duration_s=enhanced_request.duration_seconds if dynamic_dur else None,
                    expected_fps=enhanced_request.fps if dynamic_dur else None,
                    reference_images=enhanced_request.reference_images or None,
                )

                if not result.passed:
                    log.warning("Job %s validation failed (attempt %d): %s",
                                job_id, attempt, result.summary())
                    if attempt < MAX_RETRIES:
                        # Retry with a different seed
                        enhanced_request.seed = (enhanced_request.seed or 0) + attempt * 17
                        continue
                    log.warning("Job %s: all retries exhausted; delivering best-effort output",
                                job_id)

                return frames

            except Exception as exc:
                log.exception("Job %s attempt %d raised: %s", job_id, attempt, exc)
                if attempt >= MAX_RETRIES:
                    raise
        return None

    try:
        frames = await loop.run_in_executor(None, _run)
        if frames is None or cancel_event.is_set():
            storage.jobs.mark_cancelled(job_id)
            return

        # Encode video
        fmt = enhanced_request.output_format
        out_path = storage.files.output_path(job_id, fmt)
        encode_video(
            frames, out_path,
            fps=enhanced_request.fps,
            quality=enhanced_request.quality,
            fmt=fmt,
        )

        # Save preview GIF
        try:
            preview_path = storage.files.preview_path(job_id)
            make_preview_gif(frames, preview_path, fps=min(enhanced_request.fps, 10))
        except Exception as exc:
            log.warning("Preview GIF failed for job %s: %s", job_id, exc)

        # Save metadata
        storage.files.save_metadata(job_id, {
            "job_id": job_id, "mode": mode,
            "original_prompt": enhanced_request.original_prompt,
            "enhanced_prompt": enhanced_request.enhanced_prompt,
            "duration_seconds": enhanced_request.duration_seconds,
            "resolution": enhanced_request.resolution,
            "fps": enhanced_request.fps,
            "quality": enhanced_request.quality,
            "output_format": fmt,
            "frames": len(frames),
        })

        download_url = storage.files.download_url(job_id)
        storage.jobs.mark_complete(job_id, out_path, download_url)
        log.info("Job %s completed: %s", job_id, out_path)

    except asyncio.CancelledError:
        storage.jobs.mark_cancelled(job_id)
    except Exception as exc:
        log.exception("Job %s failed permanently", job_id)
        storage.jobs.mark_failed(job_id, f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Helper: get singleton or raise 503
# ---------------------------------------------------------------------------

def _require(singleton, name: str):
    if singleton is None:
        raise HTTPException(503, detail=f"{name} not initialised yet")
    return singleton


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/v1/health", tags=["system"])
@app.get("/health", tags=["system"])
def health():
    bb = _backbone
    info: dict = {"status": "ok", "backbone": "uninitialised"}
    if bb:
        info["backbone"] = bb.name
        info["trained_steps"] = getattr(bb, "step", None)
    else:
        info["status"] = "starting"
    return info


# ---------------------------------------------------------------------------
# Mode 1: Text → Video
# ---------------------------------------------------------------------------

@app.post("/v1/generate/text-to-video", tags=["generation"])
async def text_to_video(req: TextToVideoRequest):
    engine = _require(_prompt_engine, "PromptEngine")
    storage = _require(_storage, "StorageManager")
    queue = _require(_queue, "AsyncJobQueue")

    try:
        enhanced = engine.process_text_to_video(
            req.prompt,
            duration_seconds=req.duration_seconds,
            output_format=req.output_format,
            resolution=req.resolution,
            aspect_ratio=req.aspect_ratio,
            fps=req.fps,
            quality=req.quality,
            seed=req.seed,
            enhance_prompt=req.enhance_prompt,
        )
    except PromptValidationError as exc:
        raise HTTPException(422, detail=str(exc))
    except SafetyError as exc:
        raise HTTPException(400, detail=str(exc))

    job_id = uuid.uuid4().hex[:12]
    storage.jobs.create(job_id, mode=1, request_obj=req.model_dump())

    await queue.submit(
        _execute_job,
        enhanced, 1,
        job_id=job_id,
    )

    return {"job_id": job_id, "status": "queued",
            "enhanced_prompt": enhanced.enhanced_prompt}


# ---------------------------------------------------------------------------
# Mode 2: Reference Image(s) + Text → Video
# ---------------------------------------------------------------------------

@app.post("/v1/generate/image-to-video", tags=["generation"])
async def image_to_video(req: ImageToVideoRequest):
    engine = _require(_prompt_engine, "PromptEngine")
    storage = _require(_storage, "StorageManager")
    queue = _require(_queue, "AsyncJobQueue")

    try:
        enhanced = engine.process_image_to_video(
            req.prompt,
            req.reference_images,
            duration_seconds=req.duration_seconds,
            output_format=req.output_format,
            resolution=req.resolution,
            fps=req.fps,
            quality=req.quality,
            seed=req.seed,
            enhance_prompt=req.enhance_prompt,
        )
    except PromptValidationError as exc:
        raise HTTPException(422, detail=str(exc))
    except SafetyError as exc:
        raise HTTPException(400, detail=str(exc))

    job_id = uuid.uuid4().hex[:12]
    # Don't serialise raw image bytes into DB; store prompt + metadata only
    storage.jobs.create(job_id, mode=2, request_obj={
        "prompt": req.prompt,
        "num_reference_images": len(req.reference_images),
        "duration_seconds": req.duration_seconds,
        "output_format": req.output_format,
        "resolution": req.resolution,
        "fps": req.fps,
        "quality": req.quality,
    })

    await queue.submit(
        _execute_job,
        enhanced, 2,
        job_id=job_id,
    )

    return {"job_id": job_id, "status": "queued",
            "enhanced_prompt": enhanced.enhanced_prompt}


# ---------------------------------------------------------------------------
# Job management
# ---------------------------------------------------------------------------

@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
def get_job(job_id: str):
    storage = _require(_storage, "StorageManager")
    rec = storage.jobs.get(job_id)
    if rec is None:
        raise HTTPException(404, detail="Job not found")
    return rec.to_api_dict()


@app.delete("/v1/jobs/{job_id}", tags=["jobs"])
def cancel_job(job_id: str):
    storage = _require(_storage, "StorageManager")
    queue = _require(_queue, "AsyncJobQueue")

    rec = storage.jobs.get(job_id)
    if rec is None:
        raise HTTPException(404, detail="Job not found")
    if rec.status in ("completed", "failed", "cancelled"):
        raise HTTPException(409, detail=f"Job is already {rec.status}")

    cancelled = queue.cancel(job_id)
    if cancelled:
        storage.jobs.mark_cancelled(job_id)
    return {"job_id": job_id, "cancelled": cancelled}


@app.get("/v1/jobs/{job_id}/download", tags=["jobs"])
def download_job(job_id: str):
    storage = _require(_storage, "StorageManager")
    rec = storage.jobs.get(job_id)
    if rec is None:
        raise HTTPException(404, detail="Job not found")
    if rec.status != "completed":
        raise HTTPException(404, detail=f"Job not ready (status={rec.status})")
    if not rec.output_path or not os.path.exists(rec.output_path):
        raise HTTPException(404, detail="Output file not found on disk")

    ext = os.path.splitext(rec.output_path)[1].lower()
    media_map = {
        ".mp4": "video/mp4", ".mov": "video/quicktime",
        ".avi": "video/x-msvideo", ".mkv": "video/x-matroska",
        ".webm": "video/webm", ".gif": "image/gif",
    }
    media_type = media_map.get(ext, "application/octet-stream")
    filename = f"video_{job_id}{ext}"
    return FileResponse(rec.output_path, media_type=media_type,
                        filename=filename)


# ---------------------------------------------------------------------------
# Legacy /v1/videos endpoints (backward compatible)
# ---------------------------------------------------------------------------

# Re-use in-memory dict for legacy jobs so old callers don't break
_legacy_jobs: dict[str, dict] = {}


@app.post("/v1/videos", tags=["legacy"])
async def create_video_legacy(req: VideoRequest):
    """Legacy endpoint. New callers should use /v1/generate/text-to-video."""
    engine = _require(_prompt_engine, "PromptEngine")
    storage = _require(_storage, "StorageManager")
    queue = _require(_queue, "AsyncJobQueue")

    try:
        enhanced = engine.process_text_to_video(
            req.prompt, steps=req.steps if hasattr(req, "steps") else 50,
            output_format=req.fmt if req.fmt != "gif" else "mp4",
            seed=req.seed,
        )
    except (PromptValidationError, SafetyError) as exc:
        raise HTTPException(400, detail=str(exc))
    except TypeError:
        enhanced = engine.process_text_to_video(req.prompt, seed=req.seed)

    job_id = uuid.uuid4().hex[:12]
    _legacy_jobs[job_id] = {"status": "queued", "progress": 0, "created": time.time(),
                             "prompt": req.prompt}
    storage.jobs.create(job_id, mode=1, request_obj=req.model_dump())

    async def _legacy_handler(*args, job_id, cancel_event, **kwargs):
        _legacy_jobs[job_id]["status"] = "running"
        try:
            await _execute_job(enhanced, 1, job_id=job_id, cancel_event=cancel_event)
            rec = storage.jobs.get(job_id)
            if rec and rec.status == "completed":
                _legacy_jobs[job_id].update(
                    status="done", progress=100,
                    path=rec.output_path, url=f"/v1/videos/{job_id}/file"
                )
        except Exception as exc:
            _legacy_jobs[job_id].update(
                status="failed", error=f"{type(exc).__name__}: {exc}"
            )

    await queue.submit(_legacy_handler, job_id=job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/videos/{job_id}", tags=["legacy"])
def job_status_legacy(job_id: str):
    job = _legacy_jobs.get(job_id)
    if not job:
        # Fall through to new job store
        storage = _require(_storage, "StorageManager")
        rec = storage.jobs.get(job_id)
        if rec is None:
            raise HTTPException(404, "unknown job_id")
        return rec.to_api_dict()
    return {k: v for k, v in job.items() if k != "path"}


@app.get("/v1/videos/{job_id}/file", tags=["legacy"])
def job_file_legacy(job_id: str):
    job = _legacy_jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "not ready")
    path = job.get("path")
    if not path or not os.path.exists(path):
        raise HTTPException(404, "file not found")
    media = "image/gif" if path.endswith(".gif") else "video/mp4"
    return FileResponse(path, media_type=media)


# ---------------------------------------------------------------------------
# Web UI (static index.html if present)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index():
    here = os.path.dirname(os.path.abspath(__file__))
    index_path = os.path.join(here, "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            return f.read()
    return HTMLResponse(content=_default_ui(), status_code=200)


def _default_ui() -> str:
    return """<!DOCTYPE html>
<html><head><title>Video Platform</title></head>
<body style="font-family:monospace;padding:2rem">
<h1>Video Platform API</h1>
<p>See <a href="/docs">/docs</a> for the interactive API explorer.</p>
<ul>
  <li>POST /v1/generate/text-to-video — Mode 1: text → video</li>
  <li>POST /v1/generate/image-to-video — Mode 2: image(s) + text → video</li>
  <li>GET  /v1/jobs/{job_id} — poll status</li>
  <li>GET  /v1/jobs/{job_id}/download — download result</li>
</ul>
</body></html>"""
