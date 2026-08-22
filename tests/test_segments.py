"""Unit tests for the curve -> clip-list logic. No ffmpeg needed."""

from __future__ import annotations

import numpy as np

from hypecut.config import SegmentConfig
from hypecut.fusion import fuse, robust_z, smooth
from hypecut.segments import build_candidates, find_regions, merge, select
from hypecut.types import Candidate, SignalTrack


def test_find_regions_detects_contiguous_runs():
    curve = np.array([0, 0, 1, 1, 0, 0, 1, 0], dtype=float)
    assert find_regions(curve, 0.5) == [(2, 4), (6, 7)]


def test_find_regions_handles_all_below_threshold():
    assert find_regions(np.zeros(10), 0.5) == []


def test_robust_z_is_outlier_resistant():
    values = np.concatenate([np.zeros(99), [1000.0]])
    z = robust_z(values, clip=4.0)
    assert z.max() == 4.0  # the spike is clipped, not allowed to flatten the rest
    assert abs(z[:99].mean()) < 1e-9


def test_smooth_preserves_length_and_mean():
    values = np.random.default_rng(0).normal(size=200)
    out = smooth(values, 11)
    assert out.shape == values.shape
    assert abs(out.mean() - values.mean()) < 0.1


def test_fuse_returns_unit_range():
    tracks = [
        SignalTrack("a", np.sin(np.linspace(0, 20, 300)), weight=1.0),
        SignalTrack("b", np.cos(np.linspace(0, 20, 300)), weight=0.5),
    ]
    curve = fuse(tracks, grid_fps=10.0, smooth_seconds=0.5)
    assert curve.shape == (300,)
    assert 0.0 <= curve.min() <= curve.max() <= 1.0


def test_build_candidates_centres_short_regions_on_the_peak():
    times = np.arange(600) / 10.0
    curve = np.zeros(600)
    curve[300:303] = 1.0  # a 0.3 s spike at t = 30 s
    cfg = SegmentConfig(min_duration=6.0, pre_roll=0.0, post_roll=0.0, percentile=99.0)
    cands = build_candidates(curve, times, cfg, grid_fps=10.0, duration=60.0)
    assert len(cands) == 1
    assert cands[0].duration >= 6.0 - 1e-6
    centre = (cands[0].start + cands[0].end) / 2
    assert abs(centre - 30.0) < 1.5


def test_merge_joins_near_neighbours_and_caps_duration():
    cfg = SegmentConfig(merge_gap=2.0, max_duration=10.0)
    merged = merge(
        [
            Candidate(0.0, 5.0, 0.5, meta={"peak_time": 2.0}),
            Candidate(6.0, 20.0, 0.9, meta={"peak_time": 12.0}),
            Candidate(100.0, 105.0, 0.7),
        ],
        cfg,
    )
    assert len(merged) == 2
    assert merged[0].duration <= 10.0 + 1e-6
    assert merged[1].start == 100.0


def test_select_respects_budget_and_returns_timeline_order():
    cfg = SegmentConfig(max_clips=10, target_duration=20.0)
    chosen = select(
        [Candidate(50.0, 60.0, 0.9), Candidate(10.0, 20.0, 0.8), Candidate(80.0, 90.0, 0.7)], cfg
    )
    assert sum(c.duration for c in chosen) <= 20.0
    assert [c.start for c in chosen] == sorted(c.start for c in chosen)


def test_select_always_returns_something_when_budget_is_tiny():
    cfg = SegmentConfig(target_duration=1.0)
    chosen = select([Candidate(0.0, 30.0, 0.9)], cfg)
    assert len(chosen) == 1  # a too-small budget must not yield an empty reel
