"""Web API tests. Skipped unless the [web] extra is installed."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tests.conftest import requires_ffmpeg  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPECUT_DATA_DIR", str(tmp_path / "data"))
    import importlib

    from hypecut.web import app as app_module

    importlib.reload(app_module)
    with TestClient(app_module.app) as c:
        yield c
    app_module.store.shutdown(timeout=2)


def test_meta_lists_signals_and_refiners(client):
    data = client.get("/api/meta").json()
    assert "audio_rms" in data["signals"]
    assert "diversity" in data["refiners"]
    assert data["max_upload_mb"] > 0


def test_index_serves_the_ui(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "HypeCut" in res.text


def test_rejects_unsupported_file_type(client):
    res = client.post("/api/jobs", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert res.status_code == 415


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404


@requires_ffmpeg
def test_upload_runs_a_job_to_completion(client, sample_vod):
    with open(sample_vod, "rb") as fh:
        res = client.post(
            "/api/jobs",
            files={"file": ("sample.mp4", fh, "video/mp4")},
            data={"target_duration": "15", "percentile": "88", "refiners": ""},
        )
    assert res.status_code == 202
    job_id = res.json()["id"]

    deadline = time.time() + 180
    status = "queued"
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        status = payload["status"]
        if status in {"done", "failed"}:
            break
        time.sleep(1)

    assert status == "done", payload.get("error")
    assert "source" not in payload  # server paths must never reach the browser

    plan = client.get(f"/api/jobs/{job_id}/plan").json()
    assert plan["segments"]

    reel = client.get(f"/api/jobs/{job_id}/reel")
    assert reel.status_code == 200
    assert reel.headers["content-type"] == "video/mp4"


def test_meta_lists_the_variant_presets(client):
    assert "vertical" in client.get("/api/meta").json()["variants"]


def test_unknown_variant_is_rejected(client, tmp_path):
    fake = tmp_path / "x.mp4"
    fake.write_bytes(b"\x00" * 64)
    with open(fake, "rb") as fh:
        res = client.post(
            "/api/jobs", files={"file": ("x.mp4", fh, "video/mp4")}, data={"also": "sideways"}
        )
    assert res.status_code == 422
    assert "sideways" in res.json()["detail"]


@requires_ffmpeg
def test_variant_renders_are_downloadable(client, sample_vod):
    import time

    with open(sample_vod, "rb") as fh:
        res = client.post(
            "/api/jobs",
            files={"file": ("sample.mp4", fh, "video/mp4")},
            data={"target_duration": "10", "percentile": "88", "refiners": "", "also": "vertical"},
        )
    assert res.status_code == 202
    job_id = res.json()["id"]

    deadline, payload = time.time() + 240, {}
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"done", "failed"}:
            break
        time.sleep(1)
    assert payload["status"] == "done", payload.get("error")
    assert payload["variants"] == ["vertical"]

    reel = client.get(f"/api/jobs/{job_id}/reel/vertical")
    assert reel.status_code == 200
    assert reel.headers["content-type"] == "video/mp4"
    assert client.get(f"/api/jobs/{job_id}/reel/square").status_code == 404
