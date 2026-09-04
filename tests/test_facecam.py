"""Locating the facecam from the footage, and wiring the box through."""

from __future__ import annotations

import numpy as np
import pytest

from hypecut.config import DEFAULT_FACECAM_BOX, ReframeConfig
from hypecut.facecam import locate_facecam
from hypecut.reframe import geometry_filters, plan_reframe
from hypecut.types import AnalysisContext, Candidate, VideoInfo

# Where the synthetic webcam sits, in 0-1 coordinates.
TRUTH = [0.70, 0.06, 0.94, 0.30]


def _ctx_with_webcam(*, frames: int = 240, seed: int = 4) -> AnalysisContext:
    """A static game frame with a corner webcam that never quite holds still.

    The rest of the frame is a still HUD punctuated by three brief flashes —
    high energy, low duty cycle, and in the wrong corner: exactly the
    decoys a real stream layout puts next to a webcam.
    """
    rng = np.random.default_rng(seed)
    h, w = 54, 96
    gray = np.full((frames, h, w), 40, dtype=np.float32)

    x0, y0, x1, y1 = (int(round(v * m)) for v, m in zip(TRUTH, (w, h, w, h), strict=True))
    for i in range(frames):
        wobble = rng.normal(0, 6.0, size=(y1 - y0, x1 - x0))
        gray[i, y0:y1, x0:x1] = 120 + wobble

    for at in (30, 110, 190):
        gray[at : at + 2, 8:20, 8:24] = 220  # a flash in the top-left

    return AnalysisContext(
        info=VideoInfo("fake.mp4", frames / 10.0, 30.0, 1280, 720, True),
        grid_fps=10.0,
        times=np.arange(frames) / 10.0,
        gray=np.clip(gray, 0, 255).astype(np.uint8),
    )


def _overlap(a: list[float], b: list[float]) -> float:
    """Intersection-over-union of two 0-1 boxes."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union else 0.0


def test_locate_finds_the_webcam_box():
    found = locate_facecam(_ctx_with_webcam())
    assert found is not None
    assert _overlap(found["box"], TRUTH) > 0.5, f"{found['box']} vs {TRUTH}"
    assert found["confidence"] > 0.5


def test_locate_refuses_footage_without_a_webcam():
    """Rare flashes are not a webcam, however bright they are."""
    frames = 240
    rng = np.random.default_rng(9)
    gray = np.full((frames, 54, 96), 40, dtype=np.float32)
    for at in (30, 80, 130, 180):
        gray[at : at + 2, 10:22, 10:30] = 230
    ctx = AnalysisContext(
        info=VideoInfo("fake.mp4", frames / 10.0, 30.0, 1280, 720, True),
        grid_fps=10.0,
        times=np.arange(frames) / 10.0,
        gray=np.clip(gray + rng.normal(0, 0.4, gray.shape), 0, 255).astype(np.uint8),
    )
    assert locate_facecam(ctx) is None


def test_locate_needs_enough_footage():
    ctx = _ctx_with_webcam(frames=20)
    assert locate_facecam(ctx) is None


def test_plan_reframe_stamps_the_resolved_box():
    ctx = _ctx_with_webcam()
    cfg = ReframeConfig(mode="stack", facecam="auto")
    seg = Candidate(2.0, 12.0, 0.9)

    plan_reframe(ctx, [seg], cfg)

    plan = seg.meta["reframe"]
    assert _overlap(plan["facecam"], TRUTH) > 0.5


def test_plan_reframe_falls_back_to_the_default_box_when_nothing_is_found():
    ctx = _ctx_with_webcam()
    ctx.gray = np.full_like(ctx.gray, 40)  # nothing moves at all
    cfg = ReframeConfig(mode="stack", facecam="auto")
    seg = Candidate(2.0, 12.0, 0.9)

    plan_reframe(ctx, [seg], cfg)

    assert seg.meta["reframe"]["facecam"] == pytest.approx(DEFAULT_FACECAM_BOX)


def test_detection_runs_once_across_variants():
    """The base plan and a variant share one pass — locate is cached."""
    ctx = _ctx_with_webcam()
    calls = {"n": 0}
    original = locate_facecam

    def counting(c):
        calls["n"] += 1
        return original(c)

    import hypecut.reframe as r

    r.locate_facecam = counting
    try:
        plan_reframe(ctx, [Candidate(2.0, 12.0, 0.9)], ReframeConfig(mode="stack", facecam="auto"))
        plan_reframe(
            ctx,
            [Candidate(2.0, 12.0, 0.9)],
            ReframeConfig(mode="stack", facecam="auto"),
            key="reframe:vertical",
        )
    finally:
        r.locate_facecam = original
    assert calls["n"] == 1


def test_render_uses_the_stamped_box_not_the_config():
    """A sidecar round-trip keeps the detected crop without re-detecting."""
    ctx = _ctx_with_webcam()
    cfg = ReframeConfig(mode="stack", facecam="auto")
    seg = Candidate(2.0, 12.0, 0.9)
    plan_reframe(ctx, [seg], cfg)

    info = VideoInfo("fake.mp4", 30.0, 30.0, 1920, 1080, True)
    filters = geometry_filters(info, seg, ReframeConfig(mode="stack", facecam="auto"))

    chain = " ".join(filters)
    # The filter must crop the facecam pane at the box the analysis stamped
    # into the clip — the config still says "auto", which is not a box.
    x0 = seg.meta["reframe"]["facecam"][0]
    assert f"iw*{x0:g}" in chain
    assert x0 > 0.5, "the detected webcam sits right of centre"


def test_render_without_a_plan_falls_back_to_the_default_box():
    """A hand-built plan that skipped planning must not crash on 'auto'."""
    info = VideoInfo("fake.mp4", 30.0, 30.0, 1920, 1080, True)
    seg = Candidate(2.0, 12.0, 0.9)
    filters = geometry_filters(info, seg, ReframeConfig(mode="stack", facecam="auto"))
    assert filters and "iw*0" in " ".join(filters)


def test_react_crop_uses_the_detected_box():
    ctx = _ctx_with_webcam()
    cfg = ReframeConfig(mode="crop", facecam="auto", react_to_facecam=True, track=False)
    seg = Candidate(2.0, 12.0, 0.9)

    plan_reframe(ctx, [seg], cfg)

    plan = seg.meta["reframe"]
    assert _overlap(plan["facecam"], TRUTH) > 0.5
    # A still crop centred on a right-side facecam sits right of centre.
    assert plan["x"] > 0.5
