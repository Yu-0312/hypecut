"""End-to-end tests. These need ffmpeg and are skipped without it."""

from __future__ import annotations

import json

import pytest

from hypecut import analyze, load_config, run
from hypecut.config import Config
from tests.conftest import EXCITING_WINDOWS, requires_ffmpeg


@requires_ffmpeg
def test_probe_reads_duration_and_audio(sample_vod):
    from hypecut.ffmpeg import probe

    info = probe(sample_vod)
    assert info.has_audio is True
    assert 40 < info.duration < 50
    assert info.width == 320 and info.height == 180


@requires_ffmpeg
def test_analyze_lands_clips_on_the_loud_stretches(sample_vod):
    cfg = Config().merged(
        {"segments": {"percentile": 88, "min_duration": 3.0, "target_duration": 30.0}}
    )
    plan = analyze(sample_vod, cfg)
    assert plan.segments, "expected at least one highlight"

    for lo, hi in EXCITING_WINDOWS:
        centre = (lo + hi) / 2
        assert any(s.start <= centre <= s.end for s in plan.segments), (
            f"no clip covers the loud stretch at {centre}s"
        )


@requires_ffmpeg
def test_analyze_never_exceeds_the_clip_budget(sample_vod):
    cfg = Config().merged({"segments": {"max_clips": 1, "percentile": 85}})
    plan = analyze(sample_vod, cfg)
    assert len(plan.segments) <= 1


@requires_ffmpeg
def test_run_produces_a_playable_reel(sample_vod, tmp_path):
    from hypecut.ffmpeg import probe

    out = tmp_path / "reel.mp4"
    cfg = Config().merged(
        {
            "segments": {"percentile": 88, "target_duration": 20.0},
            "render": {"fade": 0.1, "write_chapters": True},
        }
    )
    result = run(sample_vod, out, cfg)

    assert out.exists() and out.stat().st_size > 1000
    reel = probe(out)
    assert reel.has_audio
    assert reel.duration == pytest.approx(result.plan.total_duration, abs=1.5)

    sidecar = out.with_suffix(".hypecut.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["segments"] and "config" in payload
    assert out.with_suffix(".edl").exists()


@requires_ffmpeg
def test_silent_video_still_works(silent_vod, tmp_path):
    """A video with no audio must fall back to visual signals, not crash."""
    cfg = Config().merged({"segments": {"percentile": 80, "target_duration": 8.0}})
    plan = analyze(silent_vod, cfg)
    assert plan.segments
    assert all(t.name in {"scene_change", "motion", "roi_activity", "flash"} for t in plan.tracks)

    out = tmp_path / "silent_reel.mp4"
    run(silent_vod, out, cfg)
    assert out.exists()


@requires_ffmpeg
def test_refiners_change_the_ordering_but_keep_the_contract(sample_vod):
    base = analyze(sample_vod, Config().merged({"segments": {"percentile": 85}}))
    refined = analyze(
        sample_vod,
        Config().merged({"segments": {"percentile": 85}, "refiners": ["diversity", "pacing"]}),
    )
    assert refined.segments
    assert all(s.duration > 0 for s in refined.segments)
    assert len(refined.segments) <= len(base.segments) + 2


@requires_ffmpeg
def test_shipped_profiles_run_end_to_end(sample_vod):
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted(root.glob("*.yaml")):
        cfg = load_config(path).merged({"segments": {"percentile": 80}, "refiners": []})
        analyze(sample_vod, cfg)  # must not raise


@requires_ffmpeg
def test_clips_land_on_the_synthetic_shot_boundaries(sample_vod):
    """The fixture is concatenated at 15/20/35/40 s — real cuts to snap to."""
    cfg = Config().merged(
        {"segments": {"percentile": 85, "min_duration": 3.0, "target_duration": 40.0}}
    )
    plan = analyze(sample_vod, cfg)
    assert plan.segments

    cuts = [15.0, 20.0, 35.0, 40.0]
    snapped = [s for s in plan.segments if s.meta.get("snapped")]
    assert snapped, "expected at least one edge to reach a boundary"
    for seg in snapped:
        for edge in ("start", "end"):
            if edge in seg.meta["snapped"]:
                value = getattr(seg, edge)
                assert min(abs(value - c) for c in cuts) < 0.2, (
                    f"{edge} at {value:.2f}s is not on a cut"
                )


@requires_ffmpeg
def test_snapping_can_be_switched_off(sample_vod):
    cfg = Config().merged({"segments": {"percentile": 85, "snap_to_shots": False}})
    plan = analyze(sample_vod, cfg)
    assert all("snapped" not in s.meta for s in plan.segments)


@requires_ffmpeg
@pytest.mark.parametrize("mode", ["crop", "stack", "blur_pad"])
def test_vertical_reel_renders_at_the_requested_size(sample_vod, tmp_path, mode):
    from hypecut.ffmpeg import probe

    out = tmp_path / f"{mode}.mp4"
    cfg = Config().merged(
        {
            "segments": {"percentile": 88, "target_duration": 12.0},
            "render": {
                "fade": 0.1,
                "reframe": {"mode": mode, "width": 540, "height": 960, "track": mode == "crop"},
            },
        }
    )
    result = run(sample_vod, out, cfg)

    reel = probe(out)
    assert (reel.width, reel.height) == (540, 960)
    assert all(s.meta.get("reframe", {}).get("mode") == mode for s in result.plan.segments)
