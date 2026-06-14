"""Async job queue for single-GPU local execution.

Design:
  - One asyncio.Queue feeds one persistent worker coroutine (sequential GPU use).
  - Each job gets a UUID, a cancel event, and a live status entry in `_slots`.
  - JobStore (storage.py) is the durable store; _slots is the fast in-memory view.

Production path:
  - Replace AsyncJobQueue.submit() with a Celery task dispatch.
  - The HTTP contract (job_id, status polling) does not change.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Awaitable, Callable, Optional

log = logging.getLogger(__name__)


class AsyncJobQueue:
    """Single-worker async queue designed for one-GPU servers.

    Usage:
        queue = AsyncJobQueue()
        # In FastAPI lifespan:
        queue.start()
        # Submit a coroutine:
        job_id = await queue.submit(my_handler, arg1, arg2, kwarg=val)
        # Cancel:
        queue.cancel(job_id)
    """

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._slots: dict[str, dict[str, Any]] = {}
        self._worker_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        """Start the background worker coroutine. Call once at app startup."""
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker(), name="job-worker")
            log.info("AsyncJobQueue: worker started")

    async def shutdown(self, timeout: float = 30.0):
        """Gracefully drain the queue and stop the worker."""
        if self._worker_task and not self._worker_task.done():
            try:
                await asyncio.wait_for(self._queue.join(), timeout=timeout)
            except asyncio.TimeoutError:
                log.warning("Queue drain timed out after %.0fs; cancelling worker", timeout)
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def submit(
        self,
        handler: Callable[..., Awaitable[None]],
        *args,
        job_id: str | None = None,
        **kwargs,
    ) -> str:
        """Enqueue a job. Returns job_id immediately (non-blocking).

        handler signature must accept keyword args:
            job_id: str
            cancel_event: asyncio.Event
        plus any positional/keyword args passed here.
        """
        job_id = job_id or uuid.uuid4().hex[:12]
        cancel_event = asyncio.Event()
        self._slots[job_id] = {
            "status": "queued",
            "cancel_event": cancel_event,
            "queued_at": time.time(),
            "error": None,
        }
        await self._queue.put((job_id, handler, args, kwargs, cancel_event))
        log.info("Job %s queued (queue depth=%d)", job_id, self._queue.qsize())
        return job_id

    def cancel(self, job_id: str) -> bool:
        """Request cancellation of a queued or running job."""
        slot = self._slots.get(job_id)
        if slot and slot["status"] in ("queued", "processing"):
            slot["cancel_event"].set()
            slot["status"] = "cancelled"
            log.info("Job %s cancelled", job_id)
            return True
        return False

    def get_slot(self, job_id: str) -> Optional[dict]:
        """Return in-memory status entry or None."""
        return self._slots.get(job_id)

    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    async def _worker(self):
        log.info("AsyncJobQueue worker ready")
        while True:
            try:
                job_id, handler, args, kwargs, cancel_event = await self._queue.get()
                slot = self._slots.get(job_id, {})

                if cancel_event.is_set():
                    slot["status"] = "cancelled"
                    self._queue.task_done()
                    continue

                slot["status"] = "processing"
                slot["started_at"] = time.time()
                log.info("Job %s started", job_id)

                try:
                    await handler(*args, job_id=job_id, cancel_event=cancel_event, **kwargs)
                    slot["status"] = "completed"
                    log.info("Job %s completed in %.1fs",
                             job_id, time.time() - slot.get("started_at", time.time()))
                except asyncio.CancelledError:
                    slot["status"] = "cancelled"
                    log.info("Job %s cancelled during execution", job_id)
                except Exception as exc:
                    slot["status"] = "failed"
                    slot["error"] = f"{type(exc).__name__}: {exc}"
                    log.exception("Job %s failed", job_id)
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                log.info("AsyncJobQueue worker shutting down")
                break
            except Exception as exc:
                log.exception("Unexpected error in job worker: %s", exc)
