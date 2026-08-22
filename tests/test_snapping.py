"""Shot-boundary detection and edge snapping."""

from __future__ import annotations

import numpy as np
import pytest

from hypecut.config import SegmentConfig
from hypecut.snapping import boundary_strength, find_boundaries, snap_segments
from hypecut.types import AnalysisContext, Candidate, VideoInfo


def _gray_with_cuts(seconds: float, cuts: list[float], grid_fps: float = 10.0) -> np.ndarray:
    """Frames that are constant within a shot and jump at each cut."""
    n = int(seconds * grid_fps)
    frames = np.zeros((n, 8, 12), dtype=np.uint8)
    level, edges = 20, [int(c * grid_fps) for c in cuts]
    for i in range(n):
        if i in edges:
            level = 220 if level < 128 else 20
        # A little texture so a real cut is not the only non-zero difference.
        frames[i] = np.clip(level + (i % 3), 0, 255)
    return frames


def _ctx(gray: np.ndarray, grid_fps: float = 10.0, fps: float = 30.0) -> AnalysisContext:
    n = gray.shape[0]
    return AnalysisContext(
        info=VideoInfo("fake.mp4", n / grid_fps, fps, 1280, 720, True),
        grid_fps=grid_fps,
        times=np.arange(n) / grid_fps,
        gray=gray,
    )


def test_boundary_strength_is_relative_to_the_local_baseline():
    """A cut inside busy footage must still stand out from that footage."""
    rng = np.random.default_rng(0)
    busy = rng.integers(0, 60, size=(200, 8, 12)).astype(np.uint8)
    busy[120:] = busy[120:] + 180  # a hard cut into a much brighter shot
    strength = boundary_strength(busy)
    assert strength.shape == (200,)
    assert strength[120] == strength.max()


def test_find_boundaries_locates_each_cut():
    gray = _gray_with_cuts(20.0, [5.0, 12.0, 16.0])
    found = find_boundaries(gray, grid_fps=10.0)
    for cut in (5.0, 12.0, 16.0):
        assert np.min(np.abs(found - cut)) <= 0.15, f"missed the cut at {cut}s"


def test_find_boundaries_returns_nothing_for_a_continuous_shot():
    gray = np.full((150, 8, 12), 100, dtype=np.uint8)
    assert find_boundaries(gray, grid_fps=10.0).size == 0


def test_snap_moves_edges_onto_the_cuts():
    gray = _gray_with_cuts(30.0, [10.0, 20.0])
    ctx = _ctx(gray)
    cfg = SegmentConfig(snap_fine=False, min_duration=3.0, max_duration=25.0, pre_roll=3.0)
    seg = Candidate(8.4, 21.6, 0.9, meta={"peak_time": 15.0})

    snap_segments(ctx, [seg], cfg)

    assert seg.start == pytest.approx(10.0, abs=0.15)
    assert seg.end == pytest.approx(20.0, abs=0.15)
    assert set(seg.meta["snapped"]) == {"start", "end"}


def test_snap_never_crosses_the_peak():
    """The moment being kept is the one thing an edge may not eat."""
    gray = _gray_with_cuts(30.0, [10.0, 20.0])
    ctx = _ctx(gray)
    cfg = SegmentConfig(snap_fine=False, min_duration=3.0, max_duration=25.0)
    # Peak sits before the 20 s cut, so the in-point must not jump to it.
    seg = Candidate(18.5, 26.0, 0.9, meta={"peak_time": 19.0})

    snap_segments(ctx, [seg], cfg)

    assert seg.start <= 19.0
    assert seg.end >= 19.0


def test_snap_rejects_moves_that_break_the_length_budget():
    gray = _gray_with_cuts(30.0, [10.0, 20.0])
    ctx = _ctx(gray)
    # A snap to 10.0 would leave a 2 s clip; min_duration forbids it.
    cfg = SegmentConfig(snap_fine=False, min_duration=8.0, max_duration=25.0)
    seg = Candidate(8.4, 12.0, 0.9, meta={"peak_time": 11.0})

    snap_segments(ctx, [seg], cfg)

    assert seg.start == pytest.approx(8.4)
    assert "snapped" not in seg.meta


def test_snap_is_a_no_op_when_disabled_or_without_frames():
    gray = _gray_with_cuts(30.0, [10.0])
    seg = Candidate(8.4, 15.0, 0.9, meta={"peak_time": 12.0})

    snap_segments(_ctx(gray), [seg], SegmentConfig(snap_to_shots=False))
    assert seg.start == pytest.approx(8.4)

    ctx = _ctx(gray)
    ctx.gray = None
    snap_segments(ctx, [seg], SegmentConfig(snap_fine=False))
    assert seg.start == pytest.approx(8.4)
