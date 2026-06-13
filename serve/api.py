"""FastAPI video-generation service.

Async job model identical to the production contract (BLUEPRINT.md sec 16.8), but
backed by an in-process single-worker thread pool so it runs locally with zero infra.
Swap the executor for Redis/RQ and the toy backbone for LTX on a cloud GPU to go to
production — the HTTP contract does not change.

Run:
    uvicorn serve.api:app --reload --port 8000
Then open http://localhost:8000
"""
from __future__ import annotations

import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from .backbones import get_backbone
from .render import save_clip

OUT_DIR = os.environ.get("VDM_OUT", "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

app = FastAPI(title="Video Platform API", version="0.1.0")

# --- single-GPU job execution (cap concurrency at 1) ------------------------- #
_executor = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict] = {}
_lock = Lock()
_backbone = None  # lazy singleton


def _get_backbone():
    global _backbone
    if _backbone is None:
        _backbone = get_backbone()  # VDM_BACKBONE env (toy|ltx)
    return _backbone


class VideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1, examples=["a red square moving right"])
    steps: int = Field(50, ge=4, le=250)
    seed: int = 0
    fmt: str = Field("mp4", pattern="^(mp4|gif)$")


def _run_job(job_id: str, req: VideoRequest):
    with _lock:
        _jobs[job_id].update(status="running", progress=10)
    try:
        bb = _get_backbone()
        with _lock:
            _jobs[job_id]["progress"] = 30
        clip, fps = bb.generate(req.prompt, steps=req.steps, seed=req.seed)
        path = os.path.join(OUT_DIR, f"{job_id}.{req.fmt}")
        save_clip(clip, path, fps=fps)
        with _lock:
            _jobs[job_id].update(status="done", progress=100, path=path,
                                 url=f"/v1/videos/{job_id}/file")
    except Exception as e:  # surface the real error to the client
        with _lock:
            _jobs[job_id].update(status="failed", progress=100, error=f"{type(e).__name__}: {e}")


@app.get("/health")
def health():
    info = {"status": "ok", "backbone": os.environ.get("VDM_BACKBONE", "toy")}
    try:
        bb = _get_backbone()
        info["backbone"] = bb.name
        info["trained_steps"] = getattr(bb, "step", None)
    except Exception as e:
        info["status"] = "backbone_unavailable"
        info["detail"] = str(e)
    return info


@app.post("/v1/videos")
def create_video(req: VideoRequest):
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"status": "queued", "progress": 0, "created": time.time(),
                         "prompt": req.prompt}
    _executor.submit(_run_job, job_id, req)
    return {"job_id": job_id, "status": "queued"}


@app.get("/v1/videos/{job_id}")
def job_status(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "unknown job_id")
    return {k: v for k, v in job.items() if k != "path"}


@app.get("/v1/videos/{job_id}/file")
def job_file(job_id: str):
    with _lock:
        job = _jobs.get(job_id)
    if not job or job.get("status") != "done":
        raise HTTPException(404, "not ready")
    media = "image/gif" if job["path"].endswith(".gif") else "video/mp4"
    return FileResponse(job["path"], media_type=media)


@app.get("/", response_class=HTMLResponse)
def index():
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "static", "index.html"), encoding="utf-8") as f:
        return f.read()
