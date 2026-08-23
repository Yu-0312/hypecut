"""Unit tests for the curve -> clip-list logic. No ffmpeg needed."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_select_returns_timeline_order_and_budgets_each_reel():
    """The budget is per reel now, so a third clip spills instead of vanishing."""
    cfg = SegmentConfig(max_clips=10, target_duration=20.0)
    chosen = select(
        [Candidate(50.0, 60.0, 0.9), Candidate(10.0, 20.0, 0.8), Candidate(80.0, 90.0, 0.7)], cfg
    )
    assert [c.start for c in chosen] == sorted(c.start for c in chosen)
    assert len(chosen) == 3, "nothing should be discarded — max_clips allows all three"

    reels: dict[int, float] = {}
    for clip in chosen:
        reels[clip.meta["reel"]] = reels.get(clip.meta["reel"], 0.0) + clip.duration
    assert list(reels) == [1, 2]
    assert all(total <= 20.0 for total in reels.values())


def test_max_clips_discards_by_score_not_by_position():
    """The cap is the only thing that throws work away, and it keeps the best."""
    cfg = SegmentConfig(max_clips=2, target_duration=None)
    chosen = select(
        [Candidate(10.0, 20.0, 0.3), Candidate(50.0, 60.0, 0.9), Candidate(80.0, 90.0, 0.7)], cfg
    )
    assert [c.start for c in chosen] == [50.0, 80.0]


def test_a_long_video_spills_into_several_reels_in_time_order():
    cfg = SegmentConfig(max_clips=0, target_duration=None, clips_per_reel=10)
    chosen = select([Candidate(i * 30.0, i * 30.0 + 8.0, 0.5) for i in range(23)], cfg)

    reels: dict[int, list[float]] = {}
    for clip in chosen:
        reels.setdefault(clip.meta["reel"], []).append(clip.start)

    assert [len(v) for v in reels.values()] == [10, 10, 3], "10 per reel, remainder last"
    assert max(reels[1]) < min(reels[2]) < max(reels[2]) < min(reels[3]), "chronological parts"
    assert [c.meta["rank"] for c in chosen[:11]] == [*range(1, 11), 1], "rank restarts per reel"


def test_select_always_returns_something_when_budget_is_tiny():
    cfg = SegmentConfig(target_duration=1.0)
    chosen = select([Candidate(0.0, 30.0, 0.9)], cfg)
    assert len(chosen) == 1  # a too-small budget must not yield an empty reel


def test_candidates_record_the_event_span():
    times = np.arange(600) / 10.0
    curve = np.zeros(600)
    curve[200:260] = 1.0  # a six-second exchange
    cfg = SegmentConfig(percentile=99.0, min_duration=1.0, pre_roll=3.0, post_roll=2.0)

    cand = build_candidates(curve, times, cfg, grid_fps=10.0, duration=60.0)[0]

    assert cand.protected() == (pytest.approx(20.0, abs=0.2), pytest.approx(26.0, abs=0.2))
    assert cand.start < cand.protected()[0], "the roll sits outside the event"
    assert cand.end > cand.protected()[1]


def test_protected_falls_back_to_the_peak():
    assert Candidate(0.0, 10.0, 1.0, meta={"peak_time": 4.0}).protected() == (4.0, 4.0)
    # No metadata at all: the midpoint is the only defensible answer.
    assert Candidate(0.0, 10.0, 1.0).protected() == (5.0, 5.0)


def test_merge_unions_the_event_spans():
    cfg = SegmentConfig(merge_gap=2.0, max_duration=60.0)
    merged = merge(
        [
            Candidate(0.0, 10.0, 0.5, meta={"event_start": 3.0, "event_end": 7.0}),
            Candidate(11.0, 20.0, 0.9, meta={"event_start": 14.0, "event_end": 18.0}),
        ],
        cfg,
    )
    assert len(merged) == 1
    assert merged[0].protected() == (3.0, 18.0)


def test_merge_keeps_the_event_when_it_has_to_truncate():
    """The clamp holds the exchange, not an arbitrary window round the peak."""
    cfg = SegmentConfig(merge_gap=2.0, max_duration=12.0)
    merged = merge(
        [
            Candidate(
                0.0, 10.0, 0.5, meta={"peak_time": 1.0, "event_start": 6.0, "event_end": 9.0}
            ),
            Candidate(
                11.0, 22.0, 0.9, meta={"peak_time": 20.0, "event_start": 14.0, "event_end": 17.0}
            ),
        ],
        cfg,
    )
    assert merged[0].duration <= 12.0 + 1e-6
    assert merged[0].start <= 6.0, "the first event must survive the clamp"
