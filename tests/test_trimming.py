"""Silence-aware edge trimming."""

from __future__ import annotations

import numpy as np
import pytest

from hypecut.config import SegmentConfig
from hypecut.trimming import find_pause, level_db, silence_mask, trim_segments
from hypecut.types import AnalysisContext, Candidate, VideoInfo

SR = 16_000
GRID = 10.0


def _ctx_from_layout(layout: list[tuple[str, float]]) -> AnalysisContext:
    """Build a context whose audio alternates between sound and silence."""
    rng = np.random.default_rng(0)
    chunks = []
    for kind, seconds in layout:
        n = int(seconds * SR)
        if kind == "sound":
            chunks.append(rng.normal(0, 0.15, n).astype(np.float32))
        elif kind == "loud":
            chunks.append(rng.normal(0, 0.5, n).astype(np.float32))
        else:
            chunks.append(np.zeros(n, dtype=np.float32))
    audio = np.concatenate(chunks)
    duration = audio.size / SR
    steps = int(duration * GRID)
    return AnalysisContext(
        info=VideoInfo("fake.mp4", duration, 30.0, 1280, 720, True),
        grid_fps=GRID,
        times=np.arange(steps) / GRID,
        gray=np.full((steps, 8, 12), 40, dtype=np.uint8),
        audio=audio,
        audio_sr=SR,
    )


def test_level_db_separates_sound_from_silence():
    ctx = _ctx_from_layout([("sound", 2.0), ("gap", 2.0)])
    level = level_db(ctx)
    assert level[:20].mean() - level[25:35].mean() > 30


def test_silence_mask_uses_the_clips_own_level():
    """A quiet clip and a loud one both have a gap between talking and not."""
    for kind in ("sound", "loud"):
        ctx = _ctx_from_layout([(kind, 2.0), ("gap", 1.0), (kind, 2.0)])
        mask = silence_mask(level_db(ctx), 0, 50, drop_db=14.0)
        assert mask is not None
        # Layout is 2 s sound, 1 s gap, 2 s sound → the gap is steps 20-30.
        assert mask[21:29].all(), f"{kind}: the gap should read as silence"
        assert not mask[:15].any(), f"{kind}: speech should not read as silence"


def test_silence_mask_declines_on_continuous_sound():
    ctx = _ctx_from_layout([("sound", 5.0)])
    assert silence_mask(level_db(ctx), 0, 50, drop_db=14.0) is None


def test_silence_mask_declines_on_continuous_quiet():
    ctx = _ctx_from_layout([("gap", 5.0)])
    assert silence_mask(level_db(ctx), 0, 50, drop_db=14.0) is None


def test_find_pause_returns_the_requested_end_of_the_gap():
    mask = np.zeros(100, dtype=bool)
    mask[40:50] = True  # a one-second pause from 4.0 s to 5.0 s
    kwargs = {"grid_fps": GRID, "window": 3.0, "min_silence": 0.3, "lo": 0.0, "hi": 10.0}
    assert find_pause(mask, target=4.5, prefer="start", **kwargs) == pytest.approx(4.0)
    assert find_pause(mask, target=4.5, prefer="end", **kwargs) == pytest.approx(5.0)


def test_find_pause_ignores_gaps_that_are_too_short():
    mask = np.zeros(100, dtype=bool)
    mask[40:42] = True  # 0.2 s — a breath, not a pause
    assert (
        find_pause(
            mask,
            grid_fps=GRID,
            target=4.0,
            window=3.0,
            min_silence=0.5,
            lo=0.0,
            hi=10.0,
            prefer="start",
        )
        is None
    )


def test_trim_moves_both_edges_into_the_pauses():
    ctx = _ctx_from_layout(
        [("sound", 5.0), ("gap", 2.0), ("loud", 6.0), ("gap", 2.0), ("sound", 5.0)]
    )
    cfg = SegmentConfig(trim_to_silence=True, min_duration=4.0, max_duration=20.0, silence_pad=0.1)
    # Edges sitting a second inside the neighbouring speech, as a fixed roll leaves them.
    seg = Candidate(6.0, 14.0, 0.9, meta={"peak_time": 10.0})

    trim_segments(ctx, [seg], cfg)

    assert seg.start == pytest.approx(6.9, abs=0.2), "should land just before the loud part"
    assert seg.end == pytest.approx(13.1, abs=0.2), "should land just after it"
    assert set(seg.meta["trimmed"]) == {"start", "end"}


def test_trim_never_overrides_a_snapped_edge():
    """A hard cut is evidence; a pause is a guess. The cut has to win."""
    ctx = _ctx_from_layout([("sound", 5.0), ("gap", 2.0), ("loud", 6.0), ("gap", 2.0)])
    cfg = SegmentConfig(trim_to_silence=True, min_duration=4.0)
    seg = Candidate(6.0, 14.0, 0.9, meta={"peak_time": 10.0, "snapped": {"start": -0.4}})

    trim_segments(ctx, [seg], cfg)

    assert seg.start == pytest.approx(6.0), "the snapped edge must be left alone"
    assert "start" not in seg.meta.get("trimmed", {})


def test_trim_respects_the_length_budget():
    ctx = _ctx_from_layout(
        [("sound", 5.0), ("gap", 2.0), ("loud", 6.0), ("gap", 2.0), ("sound", 5.0)]
    )
    cfg = SegmentConfig(trim_to_silence=True, min_duration=7.5, max_duration=20.0)
    seg = Candidate(6.0, 14.0, 0.9, meta={"peak_time": 10.0})

    trim_segments(ctx, [seg], cfg)

    assert seg.duration >= 7.5 or "trimmed" not in seg.meta


def test_trim_records_whether_the_clip_ends_in_silence():
    """The renderer lengthens the audio fade for clips cut off mid-sound."""
    # Layout: sound 0-5, gap 5-7, loud 7-13, gap 13-16.
    ctx = _ctx_from_layout([("sound", 5.0), ("gap", 2.0), ("loud", 6.0), ("gap", 3.0)])
    # Rolls pinned small so the edges cannot travel and reach a pause on their own.
    cfg = SegmentConfig(
        trim_to_silence=True, min_duration=4.0, silence_window=0.1, pre_roll=0.1, post_roll=0.1
    )

    ending_quiet = Candidate(8.0, 14.5, 0.9, meta={"peak_time": 10.0})
    trim_segments(ctx, [ending_quiet], cfg)
    assert ending_quiet.meta["ends_in_silence"] is True

    ending_loud = Candidate(8.0, 11.0, 0.9, meta={"peak_time": 9.0})
    trim_segments(ctx, [ending_loud], cfg)
    assert ending_loud.meta["ends_in_silence"] is False


def test_trim_is_a_no_op_without_audio_or_when_disabled():
    ctx = _ctx_from_layout([("sound", 4.0), ("gap", 2.0), ("sound", 4.0)])
    seg = Candidate(3.0, 8.0, 0.9, meta={"peak_time": 5.0})

    trim_segments(ctx, [seg], SegmentConfig(trim_to_silence=False))
    assert seg.start == pytest.approx(3.0)

    ctx.audio = None
    trim_segments(ctx, [seg], SegmentConfig(trim_to_silence=True))
    assert seg.start == pytest.approx(3.0)


def test_trimming_never_moves_an_edge_into_a_neighbouring_clip():
    """Same invariant as the snapper, and it can break the same way.

    The travel allowed here is `max(silence_window, pre_roll)`, wider than the
    gap `merge` guarantees between clips, and each segment is trimmed knowing
    nothing about the next. An in-point that lands behind the previous
    out-point puts the same source seconds in the reel twice.
    """
    ctx = _ctx_from_layout(
        [("sound", 6.0), ("gap", 2.0), ("sound", 4.0), ("gap", 2.0), ("sound", 6.0)]
    )
    cfg = SegmentConfig(min_duration=2.0, max_duration=20.0, pre_roll=4.0, post_roll=4.0)
    first = Candidate(2.0, 9.0, 0.9, meta={"peak_time": 4.0})
    second = Candidate(10.0, 18.0, 0.9, meta={"peak_time": 15.0})

    trim_segments(ctx, [first, second], cfg)

    assert second.start >= first.end, (
        f"clip 2 opens at {second.start:.2f} but clip 1 runs to {first.end:.2f}"
    )
    assert first.start < first.end and second.start < second.end
