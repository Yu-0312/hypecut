"""Shot-boundary detection and edge snapping."""

from __future__ import annotations

import numpy as np
import pytest

from hypecut.config import SegmentConfig
from hypecut.snapping import (
    boundary_strength,
    find_boundaries,
    find_dissolves,
    find_wipes,
    snap_segments,
)
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


def _gray_with_dissolve(
    seconds: float = 20.0, dissolve: tuple[float, float] = (8.0, 9.5), grid_fps: float = 10.0
) -> np.ndarray:
    """Two textured shots joined by a linear crossfade."""
    n = int(seconds * grid_fps)
    h, w = 12, 16
    rng = np.random.default_rng(5)
    shot_a = rng.integers(0, 255, size=(h, w)).astype(np.float64)
    shot_b = rng.integers(0, 255, size=(h, w)).astype(np.float64)

    i0, i1 = int(dissolve[0] * grid_fps), int(dissolve[1] * grid_fps)
    frames = np.zeros((n, h, w), dtype=np.uint8)
    for i in range(n):
        if i < i0:
            frame = shot_a
        elif i >= i1:
            frame = shot_b
        else:
            alpha = (i - i0) / max(1, i1 - i0)
            frame = shot_a * (1 - alpha) + shot_b * alpha
        frames[i] = np.clip(frame, 0, 255).astype(np.uint8)
    return frames


def test_find_dissolves_locates_a_crossfade():
    gray = _gray_with_dissolve(dissolve=(8.0, 9.5))
    found = find_dissolves(gray, grid_fps=10.0)
    assert len(found) == 1
    start, end = found[0]
    assert start == pytest.approx(8.0, abs=0.3)
    assert end == pytest.approx(9.5, abs=0.3)


def test_find_dissolves_ignores_a_hard_cut():
    """A cut is one huge frame; the run is too short to be a transition."""
    gray = _gray_with_cuts(20.0, [10.0])
    assert find_dissolves(gray, grid_fps=10.0) == []


def test_find_dissolves_ignores_sustained_motion():
    """A pan is sustained and spike-free too — contrast is what separates them."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, size=(12, 60)).astype(np.uint8)
    gray = np.stack([np.roll(base, shift=i, axis=1)[:, :16] for i in range(200)])
    assert find_dissolves(gray, grid_fps=10.0) == []


def test_static_footage_does_not_produce_phantom_cuts():
    """With a near-zero baseline, flicker is thousands of times the median."""
    gray = np.full((200, 12, 16), 120, dtype=np.uint8)
    gray[70] = 121  # one level of compression flicker
    gray[130] = 119
    assert find_boundaries(gray, grid_fps=10.0).size == 0


def test_snap_lands_on_the_far_side_of_a_dissolve_for_the_in_point():
    gray = _gray_with_dissolve(dissolve=(8.0, 9.5))
    ctx = _ctx(gray, fps=30.0)
    cfg = SegmentConfig(snap_fine=False, min_duration=3.0, max_duration=25.0)
    seg = Candidate(8.6, 16.0, 0.9, meta={"peak_time": 13.0})

    snap_segments(ctx, [seg], cfg)

    assert seg.start == pytest.approx(9.5, abs=0.3), "should open on the incoming shot"
    assert seg.meta["snap_kind"]["start"] == "dissolve"


def test_snap_lands_on_the_near_side_of_a_dissolve_for_the_out_point():
    gray = _gray_with_dissolve(seconds=24.0, dissolve=(14.0, 15.5))
    ctx = _ctx(gray, fps=30.0)
    cfg = SegmentConfig(snap_fine=False, min_duration=3.0, max_duration=25.0)
    seg = Candidate(6.0, 14.8, 0.9, meta={"peak_time": 10.0})

    snap_segments(ctx, [seg], cfg)

    assert seg.end == pytest.approx(14.0, abs=0.3), "should leave before the mix starts"
    assert seg.meta["snap_kind"]["end"] == "dissolve"


def test_dissolve_snapping_can_be_switched_off():
    gray = _gray_with_dissolve(dissolve=(8.0, 9.5))
    ctx = _ctx(gray, fps=30.0)
    cfg = SegmentConfig(snap_fine=False, snap_to_dissolves=False, min_duration=3.0)
    seg = Candidate(8.6, 16.0, 0.9, meta={"peak_time": 13.0})

    snap_segments(ctx, [seg], cfg)

    assert seg.meta.get("snap_kind", {}).get("start") != "dissolve"


def _gray_with_wipe(
    seconds: float = 20.0, wipe: tuple[float, float] = (8.0, 9.5), grid_fps: float = 10.0
) -> np.ndarray:
    """Two textured shots joined by a left-to-right slide of the incoming one."""
    n = int(seconds * grid_fps)
    h, w = 12, 32
    rng = np.random.default_rng(7)
    shot_a = rng.integers(0, 255, size=(h, w)).astype(np.float64)
    shot_b = rng.integers(0, 255, size=(h, w)).astype(np.float64)

    i0, i1 = int(wipe[0] * grid_fps), int(wipe[1] * grid_fps)
    frames = np.zeros((n, h, w), dtype=np.uint8)
    for i in range(n):
        if i < i0:
            frame = shot_a
        elif i >= i1:
            frame = shot_b
        else:
            front = int((i - i0) / max(1, i1 - i0) * w)  # incoming shot's right edge
            frame = shot_a
            if front > 0:
                frame = np.concatenate([shot_b[:, :front], shot_a[:, front:]], axis=1)
        frames[i] = np.clip(frame, 0, 255).astype(np.uint8)
    return frames


def test_find_wipes_locates_a_slide():
    gray = _gray_with_wipe(wipe=(8.0, 9.5))
    found = find_wipes(gray, grid_fps=10.0)
    assert len(found) == 1
    start, end = found[0]
    assert start == pytest.approx(8.0, abs=0.3)
    assert end == pytest.approx(9.5, abs=0.3)


def test_find_dissolves_does_not_claim_a_wipe():
    """The two classifiers split the same candidate run, not both claim it."""
    gray = _gray_with_wipe(wipe=(8.0, 9.5))
    assert find_dissolves(gray, grid_fps=10.0) == []


def test_find_wipes_ignores_a_pan():
    """A pan is sustained and spike-free too — but nothing sweeps across it."""
    rng = np.random.default_rng(1)
    base = rng.integers(0, 255, size=(12, 60)).astype(np.uint8)
    gray = np.stack([np.roll(base, shift=i, axis=1)[:, :32] for i in range(200)])
    assert find_wipes(gray, grid_fps=10.0) == []


def test_find_wipes_ignores_sustained_scattered_motion():
    """Busy gameplay changes a lot everywhere, in no direction — not a wipe."""
    rng = np.random.default_rng(3)
    gray = rng.integers(0, 255, size=(200, 12, 32)).astype(np.uint8)
    assert find_wipes(gray, grid_fps=10.0) == []


def test_snap_lands_on_the_far_side_of_a_wipe_for_the_in_point():
    gray = _gray_with_wipe(wipe=(8.0, 9.5))
    ctx = _ctx(gray, fps=30.0)
    cfg = SegmentConfig(snap_fine=False, min_duration=3.0, max_duration=25.0)
    seg = Candidate(8.6, 16.0, 0.9, meta={"peak_time": 13.0})

    snap_segments(ctx, [seg], cfg)

    assert seg.start == pytest.approx(9.5, abs=0.3), "should open on the incoming shot"
    assert seg.meta["snap_kind"]["start"] == "wipe"


def test_a_hard_cut_beats_a_dissolve_edge_at_the_same_distance():
    """Ties go to the more certain boundary."""
    from hypecut.snapping import _merge_boundaries, _nearest

    boundaries = _merge_boundaries(np.array([10.0]), [(10.0, "dissolve")])
    found = _nearest(boundaries, target=10.0, window=1.0, lo=0.0, hi=20.0)
    assert found == (10.0, "cut")


def test_snap_protects_a_whole_rally_not_just_its_loudest_frame():
    """The guard is the event span, not the peak — that is the point of it.

    A long exchange whose maximum lands early would, under a peak-only guard,
    let the in-point be dragged to that maximum and delete the rest.
    """
    gray = _gray_with_cuts(40.0, [12.0, 30.0])
    ctx = _ctx(gray, fps=30.0)
    cfg = SegmentConfig(snap_fine=False, min_duration=4.0, max_duration=30.0)
    rally = Candidate(
        10.0,
        32.0,
        0.9,
        # Loudest frame at 14 s, but the exchange itself runs 13 s -> 28 s.
        meta={"peak_time": 14.0, "event_start": 13.0, "event_end": 28.0},
    )

    snap_segments(ctx, [rally], cfg)

    assert rally.start <= 13.0, "must not trim into the rally"
    assert rally.end >= 28.0, "must not end before the rally does"


def test_snap_falls_back_to_the_peak_without_event_bounds():
    """Hand-built clips carry no span; behaviour there is unchanged."""
    gray = _gray_with_cuts(30.0, [10.0, 20.0])
    ctx = _ctx(gray, fps=30.0)
    cfg = SegmentConfig(snap_fine=False, min_duration=3.0, max_duration=25.0)
    seg = Candidate(8.4, 21.6, 0.9, meta={"peak_time": 15.0})

    snap_segments(ctx, [seg], cfg)

    assert seg.start == pytest.approx(10.0, abs=0.15)
    assert seg.end == pytest.approx(20.0, abs=0.15)


def test_a_truncated_clip_can_still_snap_its_end():
    """max_duration may cut mid-event; landing on a cut still beats a raw frame."""
    gray = _gray_with_cuts(40.0, [12.0, 24.0])
    ctx = _ctx(gray, fps=30.0)
    cfg = SegmentConfig(snap_fine=False, min_duration=4.0, max_duration=30.0)
    seg = Candidate(
        11.0, 24.4, 0.9, meta={"peak_time": 15.0, "event_start": 13.0, "event_end": 35.0}
    )

    snap_segments(ctx, [seg], cfg)

    assert seg.end == pytest.approx(24.0, abs=0.15)
