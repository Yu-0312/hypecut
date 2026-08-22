"""The agent-facing surface: plan round-trips, contact sheets, catalogues."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hypecut import Config, analyze
from hypecut.cli import _describe_profiles
from hypecut.plan import plan_from_dict
from tests.conftest import requires_ffmpeg


def test_profiles_are_described_by_their_own_first_comment():
    """The summary lives in the profile, so it cannot drift from it."""
    found = _describe_profiles(Path(__file__).resolve().parents[1] / "configs")
    assert found, "no profiles found"
    names = {item["name"] for item in found}
    assert {"default", "sports-broadcast", "shorts"} <= names
    for item in found:
        assert item["summary"], f"{item['name']} has no leading comment to describe it"
        assert not item["summary"].startswith("#")


def test_signals_catalogue_is_serialisable():
    from hypecut.refine import available_refiners
    from hypecut.signals import available_signals

    payload = json.loads(
        json.dumps({"signals": available_signals(), "refiners": available_refiners()})
    )
    assert "crowd_roar" in payload["signals"]
    assert "diversity" in payload["refiners"]
    assert all(payload["signals"].values()), "every signal needs a description"


# ------------------------------------------------------------------ plan I/O


@requires_ffmpeg
def test_a_plan_round_trips_through_json(sample_vod, tmp_path):
    from hypecut.pipeline import render_plan

    cfg = Config().merged({"segments": {"percentile": 88, "target_duration": 15.0}})
    plan = analyze(sample_vod, cfg)
    out, sidecar = render_plan(plan, tmp_path / "reel.mp4", cfg)
    assert sidecar is not None

    reloaded, reloaded_cfg = plan_from_dict(json.loads(sidecar.read_text()))

    assert len(reloaded.segments) == len(plan.segments)
    for before, after in zip(plan.segments, reloaded.segments, strict=True):
        assert after.start == pytest.approx(before.start, abs=1e-3)
        assert after.end == pytest.approx(before.end, abs=1e-3)
    assert reloaded_cfg.segments.percentile == 88


@requires_ffmpeg
def test_an_edited_plan_renders_what_was_asked_for(sample_vod, tmp_path):
    from hypecut.ffmpeg import probe
    from hypecut.pipeline import render_plan

    cfg = Config().merged({"segments": {"percentile": 88, "target_duration": 15.0}})
    plan = analyze(sample_vod, cfg)
    _, sidecar = render_plan(plan, tmp_path / "reel.mp4", cfg)

    payload = json.loads(sidecar.read_text())
    payload["segments"] = payload["segments"][:1]
    payload["segments"][0]["start"] = round(payload["segments"][0]["start"] + 1.0, 3)
    payload["segments"][0]["end"] = round(payload["segments"][0]["start"] + 5.0, 3)

    edited, edited_cfg = plan_from_dict(payload)
    out, _ = render_plan(edited, tmp_path / "edited.mp4", edited_cfg, write_sidecar=False)

    assert probe(out).duration == pytest.approx(5.0, abs=0.6)


@requires_ffmpeg
def test_plan_times_are_clamped_to_the_source(sample_vod, tmp_path):
    """Edited plans are untrusted input; the numbers become ffmpeg seeks."""
    payload = {"source": str(sample_vod), "segments": [{"start": -50.0, "end": 99999.0}]}
    plan, _ = plan_from_dict(payload)
    assert plan.segments[0].start == 0.0
    assert plan.segments[0].end == pytest.approx(plan.info.duration, abs=1e-6)


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"segments": [{"start": 0, "end": 1}]}, "no source"),
        ({"source": "/nope/missing.mp4", "segments": [{"start": 0, "end": 1}]}, "not found"),
    ],
)
def test_plan_rejects_unusable_input(payload, match):
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        plan_from_dict(payload)


@requires_ffmpeg
def test_plan_rejects_a_collapsed_segment(sample_vod):
    with pytest.raises(ValueError, match="after clamping"):
        plan_from_dict({"source": str(sample_vod), "segments": [{"start": 5.0, "end": 5.01}]})


@requires_ffmpeg
def test_plan_rejects_a_non_finite_time(sample_vod):
    with pytest.raises(ValueError, match="non-finite"):
        plan_from_dict(
            {"source": str(sample_vod), "segments": [{"start": 0.0, "end": float("inf")}]}
        )


@requires_ffmpeg
def test_plan_accepts_a_relocated_source(sample_vod, tmp_path):
    payload = {"source": "/somewhere/else.mp4", "segments": [{"start": 1.0, "end": 6.0}]}
    plan, _ = plan_from_dict(payload, source=sample_vod)
    assert plan.info.path == str(sample_vod)


# ------------------------------------------------------------- contact sheet


@requires_ffmpeg
def test_contact_sheet_samples_the_whole_video_without_a_plan(sample_vod, tmp_path):
    from hypecut.contact import contact_sheet
    from hypecut.ffmpeg import probe

    info = probe(sample_vod)
    dest, index = contact_sheet(info, tmp_path / "sheet.png", count=6, columns=3)

    assert dest.exists() and dest.stat().st_size > 1000
    assert len(index) == 6
    times = [entry["time"] for entry in index]
    assert times == sorted(times)
    assert all(0 <= t <= info.duration for t in times)


@requires_ffmpeg
def test_contact_sheet_follows_a_plan_and_reports_why_each_clip_was_picked(sample_vod, tmp_path):
    from hypecut.contact import contact_sheet

    plan = analyze(sample_vod, Config().merged({"segments": {"percentile": 88}}))
    dest, index = contact_sheet(plan.info, tmp_path / "plan.png", segments=plan.segments)

    assert dest.exists()
    assert len(index) == len(plan.segments)
    assert {"start", "end", "score", "top_signal"} <= set(index[0])


@requires_ffmpeg
def test_contact_sheet_does_not_pad_a_short_grid(sample_vod, tmp_path):
    """One clip should make a one-tile sheet, not a row of four."""
    from hypecut.contact import contact_sheet
    from hypecut.ffmpeg import probe

    info = probe(sample_vod)
    single, _ = contact_sheet(info, tmp_path / "one.png", count=1, columns=4, tile_width=200)
    wide, _ = contact_sheet(info, tmp_path / "four.png", count=4, columns=4, tile_width=200)

    from PIL import Image  # noqa: PLC0415 - only needed for this assertion

    assert Image.open(single).width < Image.open(wide).width
