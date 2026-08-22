"""FastAPI application: upload a VOD, watch it process, download the reel.

Endpoints
---------
``POST /api/jobs``            multipart upload, returns a job id
``GET  /api/jobs``            recent jobs
``GET  /api/jobs/{id}``       status, progress and the cut list
``GET  /api/jobs/{id}/reel``  the rendered mp4
``GET  /api/jobs/{id}/plan``  the JSON cut list
``GET  /api/meta``            registered signals, refiners and limits
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .. import __version__
from ..jobs import JobStore
from ..reframe import MODES as REFRAME_MODES

DATA_DIR = Path(os.environ.get("HYPECUT_DATA_DIR", "./hypecut-data")).resolve()
MAX_UPLOAD_MB = int(os.environ.get("HYPECUT_MAX_UPLOAD_MB", "4096"))
WORKERS = int(os.environ.get("HYPECUT_WORKERS", "1"))
ALLOWED_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".ts", ".m4v"}

STATIC_DIR = Path(__file__).parent / "static"

store = JobStore(DATA_DIR, workers=WORKERS)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    yield
    store.shutdown()


app = FastAPI(title="HypeCut", version=__version__, lifespan=lifespan)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    from ..refine import available_refiners
    from ..refine import load_plugins as lrp
    from ..signals import available_signals
    from ..signals import load_plugins as lsp

    lsp()
    lrp()
    return {
        "version": __version__,
        "signals": available_signals(),
        "refiners": available_refiners(),
        "max_upload_mb": MAX_UPLOAD_MB,
        "allowed_suffixes": sorted(ALLOWED_SUFFIXES),
        "reframe_modes": list(REFRAME_MODES),
        "profiles": sorted(p.name for p in Path("configs").glob("*.yaml"))
        if Path("configs").is_dir()
        else [],
    }


@app.post("/api/jobs")
async def create_job(
    file: UploadFile = File(...),
    target_duration: float = Form(120.0),
    max_clips: int = Form(20),
    percentile: float = Form(92.0),
    min_duration: float = Form(4.0),
    max_duration: float = Form(20.0),
    refiners: str = Form("diversity,pacing"),
    profile: str = Form(""),
    reframe: str = Form("off"),
    reframe_track: bool = Form(False),
    snap_to_shots: bool = Form(True),
) -> JSONResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if reframe not in REFRAME_MODES:
        raise HTTPException(
            status_code=422, detail=f"Unknown reframe mode {reframe!r}; expected {REFRAME_MODES}"
        )
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {suffix!r}. Allowed: {sorted(ALLOWED_SUFFIXES)}",
        )

    safe_name = Path(file.filename or "upload").name
    dest = store.uploads / f"{os.urandom(6).hex()}{suffix}"
    limit = MAX_UPLOAD_MB * 1024 * 1024
    written = 0
    with dest.open("wb") as fh:
        while chunk := await file.read(1024 * 1024):
            written += len(chunk)
            if written > limit:
                fh.close()
                dest.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413, detail=f"File exceeds {MAX_UPLOAD_MB} MB limit"
                )
            fh.write(chunk)

    options = {
        "target_duration": target_duration,
        "max_clips": max_clips,
        "percentile": percentile,
        "min_duration": min_duration,
        "max_duration": max_duration,
        "refiners": [r.strip() for r in refiners.split(",") if r.strip()],
        "profile": profile or None,
        "reframe": reframe,
        "reframe_track": reframe_track,
        "snap_to_shots": snap_to_shots,
    }
    job = store.submit(safe_name, dest, options)
    return JSONResponse({"id": job.id, "status": job.status.value}, status_code=202)


@app.get("/api/jobs")
def list_jobs(limit: int = 25) -> dict[str, Any]:
    return {"jobs": [j.public() for j in store.list(limit=limit)]}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    return job.public()


@app.get("/api/jobs/{job_id}/plan")
def get_plan(job_id: str) -> JSONResponse:
    job = store.get(job_id)
    if job is None or job.plan is None:
        raise HTTPException(status_code=404, detail="No plan for this job yet")
    return JSONResponse(job.plan)


@app.get("/api/jobs/{job_id}/reel")
def get_reel(job_id: str) -> FileResponse:
    job = store.get(job_id)
    if job is None or not job.output or not Path(job.output).exists():
        raise HTTPException(status_code=404, detail="Reel not ready")
    stem = Path(job.filename).stem or "highlights"
    return FileResponse(job.output, media_type="video/mp4", filename=f"{stem}_highlights.mp4")


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    job = store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job")
    if store.cancel(job_id):
        return {"cancelled": True}
    return {"cancelled": False, "status": job.status.value}


def cleanup_data_dir() -> None:  # pragma: no cover - operational helper
    """Delete every upload and output. Used by tests and `make clean`."""
    shutil.rmtree(DATA_DIR, ignore_errors=True)


__all__ = ["app", "store", "cleanup_data_dir"]

if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
    print(json.dumps({"data_dir": str(DATA_DIR)}))
