"""Hardening of the self-hosted web API: auth, profile paths, input bounds, disk cleanup."""

from __future__ import annotations

import importlib
import time
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from hypecut.jobs import Job, JobStatus, JobStore  # noqa: E402

TINY = ("clip.mp4", b"\x00" * 16, "video/mp4")


def _client(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("HYPECUT_DATA_DIR", str(tmp_path / "data"))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    from hypecut.web import app as app_module

    importlib.reload(app_module)
    return app_module


@pytest.fixture()
def web(tmp_path, monkeypatch):
    monkeypatch.delenv("HYPECUT_TOKEN", raising=False)
    module = _client(tmp_path, monkeypatch)
    with TestClient(module.app) as c:
        yield c
    module.store.shutdown(timeout=2)


def test_profile_must_be_a_shipped_profile(web):
    for bad in ("/etc/passwd", "../pyproject.toml", "configs/../pyproject.toml", "nope"):
        res = web.post("/api/jobs", files={"file": TINY}, data={"profile": bad})
        assert res.status_code == 422, bad
        assert "Unknown profile" in res.json()["detail"]


def test_profile_accepts_name_or_listed_path(web):
    profiles = web.get("/api/meta").json()["profiles"]
    if not profiles:
        pytest.skip("no shipped profiles found")
    first = profiles[0]
    for value in (first["name"], first["path"]):
        res = web.post("/api/jobs", files={"file": TINY}, data={"profile": value, "refiners": ""})
        assert res.status_code == 202, res.text


@pytest.mark.parametrize(
    "field,value",
    [
        ("percentile", "150"),
        ("target_duration", "-5"),
        ("min_duration", "0"),
        ("clips_per_reel", "0"),
        ("max_clips", "-1"),
    ],
)
def test_numeric_inputs_are_bounded(web, field, value):
    res = web.post("/api/jobs", files={"file": TINY}, data={field: value})
    assert res.status_code == 422


def test_min_duration_cannot_exceed_max(web):
    res = web.post(
        "/api/jobs", files={"file": TINY}, data={"min_duration": "30", "max_duration": "10"}
    )
    assert res.status_code == 422


def test_token_protects_everything_but_meta_and_login(tmp_path, monkeypatch):
    module = _client(tmp_path, monkeypatch, HYPECUT_TOKEN="s3cret")
    try:
        with TestClient(module.app) as c:
            assert c.get("/api/meta").status_code == 200
            assert c.get("/api/jobs").status_code == 401
            assert c.get("/").status_code == 401
            assert c.get("/api/jobs", headers={"Authorization": "Bearer wrong"}).status_code == 401
            assert c.get("/api/jobs", headers={"Authorization": "Bearer s3cret"}).status_code == 200
            assert c.get("/login?token=wrong", follow_redirects=False).status_code == 401
            res = c.get("/login?token=s3cret", follow_redirects=False)
            assert res.status_code == 303
            assert c.get("/api/jobs").status_code == 200  # cookie set by /login
    finally:
        module.store.shutdown(timeout=2)


def _finished_job(store: JobStore, job_id: str, finished_at: float) -> Job:
    source = store.uploads / f"{job_id}.mp4"
    source.write_bytes(b"src")
    parts = [store.outputs / f"{job_id}.mp4", store.outputs / f"{job_id}_2.mp4"]
    for p in parts:
        p.write_bytes(b"reel")
    sidecar = store.outputs / f"{job_id}.hypecut.json"
    sidecar.write_text("{}")
    job = Job(id=job_id, filename="x.mp4", source=str(source), options={})
    job.status = JobStatus.DONE
    job.parts = [str(p) for p in parts]
    job.output = job.parts[0]
    job.finished_at = finished_at
    with store._lock:
        store._jobs[job_id] = job
        store._order.append(job_id)
    return job


def test_eviction_removes_every_part_and_sidecar(tmp_path):
    store = JobStore(tmp_path, workers=1, max_jobs=1)
    try:
        _finished_job(store, "old", time.time())
        _finished_job(store, "new", time.time())
        with store._lock:
            store._evict_locked()
        assert store.get("old") is None
        leftovers = sorted(p.name for p in Path(tmp_path).rglob("old*"))
        assert leftovers == [], leftovers
        assert store.get("new") is not None
    finally:
        store.shutdown(timeout=2)


def test_ttl_sweep_drops_finished_jobs(tmp_path):
    store = JobStore(tmp_path, workers=1, ttl_seconds=60)
    try:
        _finished_job(store, "stale", time.time() - 3600)
        _finished_job(store, "fresh", time.time())
        store._maybe_sweep(every=0)
        assert store.get("stale") is None
        assert store.get("fresh") is not None
        assert not list(Path(tmp_path).rglob("stale*"))
    finally:
        store.shutdown(timeout=2)


def test_failure_details_do_not_leak_server_paths(tmp_path):
    store = JobStore(tmp_path, workers=1)
    try:
        bogus = store.uploads / "missing.mp4"
        job = store.submit("missing.mp4", bogus, {"refiners": []})
        deadline = time.time() + 30
        while (
            store.get(job.id).status in (JobStatus.QUEUED, JobStatus.RUNNING)
            and time.time() < deadline
        ):
            time.sleep(0.05)
        public = store.get(job.id).public()
        assert public["status"] == "failed"
        assert "Traceback" not in (public["error"] or "")
    finally:
        store.shutdown(timeout=2)
