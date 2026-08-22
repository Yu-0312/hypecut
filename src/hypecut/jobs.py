"""A small, dependency-free job queue for the web UI.

Deliberately not Celery. A single background worker thread with a bounded
queue covers the self-hosted single-box case that open-source users
actually run, and keeps `docker run` to one container. The interface
(:class:`JobStore`) is narrow enough that swapping in Redis/RQ later is a
contained change.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import Any

__all__ = ["JobStatus", "Job", "JobStore"]


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    filename: str
    source: str
    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    output: str | None = None
    plan: dict[str, Any] | None = None
    error: str | None = None
    options: dict[str, Any] = field(default_factory=dict)

    def public(self) -> dict[str, Any]:
        data = asdict(self)
        data["status"] = self.status.value
        # Never leak server-side paths to the browser.
        data.pop("source", None)
        data["output"] = bool(self.output)
        return data


class JobStore:
    """Thread-safe job registry plus a worker pool."""

    def __init__(self, data_dir: Path, workers: int = 1, max_jobs: int = 200) -> None:
        self.data_dir = Path(data_dir)
        self.uploads = self.data_dir / "uploads"
        self.outputs = self.data_dir / "outputs"
        for d in (self.uploads, self.outputs):
            d.mkdir(parents=True, exist_ok=True)

        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._queue: Queue[str] = Queue()
        self._max_jobs = max_jobs
        self._stopping = threading.Event()
        self._threads = [
            threading.Thread(target=self._worker, name=f"hypecut-worker-{i}", daemon=True)
            for i in range(max(1, workers))
        ]
        for t in self._threads:
            t.start()

    # ---------------------------------------------------------------- public

    def submit(self, filename: str, source: Path, options: dict[str, Any]) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], filename=filename, source=str(source), options=options)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._evict_locked()
        self._queue.put(job.id)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        with self._lock:
            ids = self._order[-limit:][::-1]
            return [self._jobs[i] for i in ids if i in self._jobs]

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job and job.status is JobStatus.QUEUED:
                job.status = JobStatus.CANCELLED
                job.message = "cancelled"
                return True
        return False

    def shutdown(self, timeout: float = 5.0) -> None:
        self._stopping.set()
        for t in self._threads:
            t.join(timeout=timeout)

    # ---------------------------------------------------------------- worker

    def _worker(self) -> None:
        while not self._stopping.is_set():
            try:
                job_id = self._queue.get(timeout=0.5)
            except Empty:
                continue
            job = self.get(job_id)
            if job is None or job.status is JobStatus.CANCELLED:
                continue
            try:
                self._process(job)
            except Exception:  # pragma: no cover - defensive
                job.status = JobStatus.FAILED
                job.error = traceback.format_exc(limit=4)
                job.finished_at = time.time()
            finally:
                self._queue.task_done()

    def _process(self, job: Job) -> None:
        from .config import load_config
        from .pipeline import analyze, render_plan

        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        job.message = "starting"

        def progress(fraction: float, message: str) -> None:
            job.progress = round(max(0.0, min(1.0, fraction)), 4)
            job.message = message

        try:
            cfg = load_config(job.options.get("profile") or None)
            overrides = _options_to_overrides(job.options)
            if overrides:
                cfg = cfg.merged(overrides)

            plan = analyze(job.source, cfg, progress=lambda p, m: progress(p * 0.55, m))
            job.plan = plan.to_dict()
            if not plan.segments:
                raise RuntimeError(
                    "No highlights found. Lower the sensitivity percentile and retry."
                )

            dest = self.outputs / f"{job.id}.mp4"
            out, sidecar = render_plan(
                plan, dest, cfg, progress=lambda p, m: progress(0.55 + p * 0.45, m)
            )
            job.output = str(out)
            job.plan = plan.to_dict()
            if sidecar:
                Path(sidecar).write_text(
                    json.dumps(job.plan, indent=2, ensure_ascii=False), encoding="utf-8"
                )
            job.status = JobStatus.DONE
            job.progress = 1.0
            job.message = "done"
        except Exception as exc:
            job.status = JobStatus.FAILED
            job.error = str(exc)
            job.message = "failed"
            raise
        finally:
            job.finished_at = time.time()

    # ---------------------------------------------------------------- private

    def _evict_locked(self) -> None:
        while len(self._order) > self._max_jobs:
            old = self._order.pop(0)
            job = self._jobs.pop(old, None)
            if job is None:
                continue
            for path in (job.source, job.output):
                if path:
                    Path(path).unlink(missing_ok=True)


def _options_to_overrides(options: dict[str, Any]) -> dict[str, Any]:
    seg: dict[str, Any] = {}
    for key in ("max_clips", "min_duration", "max_duration", "percentile"):
        if options.get(key) is not None:
            seg[key] = options[key]
    if options.get("target_duration") is not None:
        target = float(options["target_duration"])
        seg["target_duration"] = target if target > 0 else None
    if options.get("snap_to_shots") is not None:
        seg["snap_to_shots"] = bool(options["snap_to_shots"])
    if options.get("trim_to_silence") is not None:
        seg["trim_to_silence"] = bool(options["trim_to_silence"])

    render: dict[str, Any] = {}
    reframe: dict[str, Any] = {}
    if options.get("reframe"):
        reframe["mode"] = str(options["reframe"])
    if options.get("reframe_track") is not None:
        reframe["track"] = bool(options["reframe_track"])
    if options.get("react_to_facecam") is not None:
        reframe["react_to_facecam"] = bool(options["react_to_facecam"])
    if reframe:
        render["reframe"] = reframe

    out: dict[str, Any] = {}
    if seg:
        out["segments"] = seg
    if render:
        out["render"] = render
    if options.get("refiners"):
        out["refiners"] = list(options["refiners"])
    if options.get("weights"):
        out["signals"] = {"weights": dict(options["weights"])}
    return out
