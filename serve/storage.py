"""Job metadata store (SQLite) + file storage (local filesystem).

Production swap-ins:
  - JobStore      → PostgreSQL-backed version with the same interface
  - FileStorage   → S3FileStorage that returns pre-signed URLs

Both are injected into the API layer via StorageManager so callers never
import concrete implementations directly.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_DDL = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id       TEXT PRIMARY KEY,
    mode         INTEGER NOT NULL,
    status       TEXT NOT NULL,
    request_json TEXT NOT NULL,
    output_path  TEXT,
    output_url   TEXT,
    error_msg    TEXT,
    retry_count  INTEGER DEFAULT 0,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL,
    completed_at REAL,
    progress     REAL DEFAULT 0.0
);
CREATE INDEX IF NOT EXISTS idx_jobs_status  ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at DESC);
"""


@dataclass
class JobRecord:
    job_id: str
    mode: int
    status: str
    request_json: str
    output_path: Optional[str] = None
    output_url: Optional[str] = None
    error_msg: Optional[str] = None
    retry_count: int = 0
    created_at: float = 0.0
    updated_at: float = 0.0
    completed_at: Optional[float] = None
    progress: float = 0.0

    def to_api_dict(self) -> dict:
        def _iso(ts):
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat() if ts else None

        return {
            "job_id": self.job_id,
            "mode": self.mode,
            "status": self.status,
            "progress": round(self.progress, 3),
            "output_url": self.output_url,
            "error": self.error_msg,
            "retry_count": self.retry_count,
            "created_at": _iso(self.created_at),
            "updated_at": _iso(self.updated_at),
            "completed_at": _iso(self.completed_at),
        }


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------

class JobStore:
    """SQLite-backed job metadata store.

    Thread-safe: each operation opens and immediately closes its own connection
    with a WAL-mode journal (allows concurrent reads + one writer at a time).
    """

    def __init__(self, db_path: str = "outputs/jobs.db"):
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._db_path = db_path
        with self._conn() as conn:
            conn.executescript(_DDL)
            conn.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self._db_path, timeout=15, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create(self, job_id: str, mode: int, request_obj) -> JobRecord:
        now = time.time()
        rj = (
            json.dumps(request_obj, default=str)
            if not isinstance(request_obj, str)
            else request_obj
        )
        rec = JobRecord(job_id=job_id, mode=mode, status="queued",
                        request_json=rj, created_at=now, updated_at=now)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (rec.job_id, rec.mode, rec.status, rec.request_json,
                 rec.output_path, rec.output_url, rec.error_msg,
                 rec.retry_count, rec.created_at, rec.updated_at,
                 rec.completed_at, rec.progress),
            )
        return rec

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        return JobRecord(**{k: row[k] for k in row.keys()})

    def update(self, job_id: str, **fields):
        fields["updated_at"] = time.time()
        cols = ", ".join(f"{k}=?" for k in fields)
        vals = list(fields.values()) + [job_id]
        with self._conn() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE job_id=?", vals)

    def mark_complete(self, job_id: str, output_path: str, output_url: str):
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status='completed', output_path=?, output_url=?, "
                "progress=1.0, updated_at=?, completed_at=? WHERE job_id=?",
                (output_path, output_url, now, now, job_id),
            )

    def mark_failed(self, job_id: str, error: str, retry_count: int = 0):
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status='failed', error_msg=?, retry_count=?, "
                "progress=0.0, updated_at=?, completed_at=? WHERE job_id=?",
                (error[:4000], retry_count, now, now, job_id),
            )

    def mark_cancelled(self, job_id: str):
        now = time.time()
        with self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status='cancelled', updated_at=?, completed_at=? WHERE job_id=?",
                (now, now, job_id),
            )


# ---------------------------------------------------------------------------
# File storage
# ---------------------------------------------------------------------------

class FileStorage:
    """Local filesystem storage for generated video outputs.

    Production: replace with S3FileStorage that returns pre-signed download URLs.
    """

    def __init__(self, root: str = "outputs"):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def job_dir(self, job_id: str) -> str:
        d = os.path.join(self.root, job_id)
        os.makedirs(d, exist_ok=True)
        return d

    def output_path(self, job_id: str, ext: str) -> str:
        return os.path.join(self.job_dir(job_id), f"output.{ext.lstrip('.')}")

    def chunk_dir(self, job_id: str) -> str:
        d = os.path.join(self.job_dir(job_id), "chunks")
        os.makedirs(d, exist_ok=True)
        return d

    def chunk_path(self, job_id: str, idx: int, ext: str = "mp4") -> str:
        return os.path.join(self.chunk_dir(job_id), f"chunk_{idx:04d}.{ext}")

    def preview_path(self, job_id: str) -> str:
        return os.path.join(self.job_dir(job_id), "preview.gif")

    def metadata_path(self, job_id: str) -> str:
        return os.path.join(self.job_dir(job_id), "metadata.json")

    def save_metadata(self, job_id: str, data: dict):
        with open(self.metadata_path(job_id), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def download_url(job_id: str) -> str:
        return f"/v1/jobs/{job_id}/download"


# ---------------------------------------------------------------------------
# Combined manager (injected into API layer)
# ---------------------------------------------------------------------------

class StorageManager:
    """Single entry-point for job metadata + file operations."""

    def __init__(self, root: str = "outputs"):
        self.jobs = JobStore(db_path=os.path.join(root, "jobs.db"))
        self.files = FileStorage(root=root)
