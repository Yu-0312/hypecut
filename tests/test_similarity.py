"""Visual de-duplication, and the replay rule that shapes it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hypecut.refine.similarity import Similarity, cosine, descriptor
from hypecut.types import AnalysisContext, Candidate, VideoInfo


def _ctx(frames: np.ndarray, grid_fps: float = 10.0) -> AnalysisContext:
    info = VideoInfo(
        path="x.mp4",
        duration=frames.shape[0] / grid_fps,
        fps=30.0,
        width=frames.shape[2],
        height=frames.shape[1],
        has_audio=False,
    )
    times = np.arange(frames.shape[0], dtype=np.float64) / grid_fps
    return AnalysisContext(info=info, grid_fps=grid_fps, times=times, gray=frames)


def _blank(n: int) -> np.ndarray:
    return np.full((n, 54, 96), 40, dtype=np.uint8)


def _moving_box(frames: np.ndarray, t0: int, t1: int, row: int, col0: int) -> None:
    """Draw a bright box sliding right — one 'play' with a distinct shape."""
    for step, index in enumerate(range(t0, t1)):
        col = col0 + step * 2
        frames[index, row : row + 8, col : col + 8] = 230


def _seg(start: float, end: float, score: float = 0.5) -> Candidate:
    seg = Candidate(start, end, score)
    seg.meta.update(peak_time=(start + end) / 2, event_start=start, event_end=end)
    return seg


# ------------------------------------------------------------- the descriptor


def test_the_same_action_in_the_same_place_matches_itself():
    frames = _blank(600)
    _moving_box(frames, 100, 130, row=10, col0=10)
    _moving_box(frames, 400, 430, row=10, col0=10)
    ctx = _ctx(frames)

    a = descriptor(ctx, _seg(10.0, 13.0))
    b = descriptor(ctx, _seg(40.0, 43.0))
    assert cosine(a, b) > 0.95


def test_the_same_action_elsewhere_in_the_frame_does_not_match():
    """Motion shape, not motion amount — otherwise every busy clip is a dupe."""
    frames = _blank(600)
    _moving_box(frames, 100, 130, row=8, col0=8)
    _moving_box(frames, 400, 430, row=40, col0=60)
    ctx = _ctx(frames)

    a = descriptor(ctx, _seg(10.0, 13.0))
    b = descriptor(ctx, _seg(40.0, 43.0))
    assert cosine(a, b) < 0.5


def test_a_still_clip_refuses_to_be_compared():
    """A normalised noise vector correlates with anything; that invents dupes."""
    ctx = _ctx(_blank(300))
    assert descriptor(ctx, _seg(5.0, 8.0)) is None


def test_a_descriptor_survives_a_brightness_change():
    """Same play, brighter stadium. Zero-mean plus normalise should absorb it."""
    frames = _blank(600)
    _moving_box(frames, 100, 130, row=10, col0=10)
    _moving_box(frames, 400, 430, row=10, col0=10)
    frames[380:460] = np.clip(frames[380:460].astype(np.int16) + 60, 0, 255).astype(np.uint8)
    ctx = _ctx(frames)

    a = descriptor(ctx, _seg(10.0, 13.0))
    b = descriptor(ctx, _seg(40.0, 43.0))
    assert cosine(a, b) > 0.9


# ------------------------------------------------------------- the replay rule


def _run(frames: np.ndarray, segments: list[Candidate], **params) -> list[Candidate]:
    refiner = Similarity(**params)
    refiner.ctx = _ctx(frames)
    return refiner.refine(refiner.ctx.info, segments)


def test_a_replay_is_kept_and_grouped_with_what_it_replays():
    """The user's rule: seeing the goal again from another angle is editing."""
    frames = _blank(900)
    _moving_box(frames, 100, 130, row=10, col0=10)  # the goal
    _moving_box(frames, 200, 230, row=10, col0=10)  # the replay, 10s later
    goal, replay = _seg(10.0, 13.0, 0.9), _seg(20.0, 23.0, 0.6)

    _run(frames, [goal, replay], replay_window=90.0)

    assert replay.score == pytest.approx(0.6), "a replay must not be penalised"
    assert goal.meta["moment"] == replay.meta["moment"], "and it belongs to the same moment"


def test_the_same_thing_happening_again_much_later_is_penalised():
    frames = _blank(3000)
    _moving_box(frames, 100, 130, row=10, col0=10)
    _moving_box(frames, 2000, 2030, row=10, col0=10)
    first, again = _seg(10.0, 13.0, 0.9), _seg(200.0, 203.0, 0.6)

    _run(frames, [first, again], replay_window=90.0)

    assert again.score < 0.6, "the weaker of two identical moments is demoted"
    assert first.score == pytest.approx(0.9), "the better take keeps its score"
    assert again.meta["repeat_penalty"] > 0
    assert first.meta["moment"] != again.meta["moment"]


def test_different_plays_far_apart_are_left_alone():
    frames = _blank(3000)
    _moving_box(frames, 100, 130, row=8, col0=8)
    _moving_box(frames, 2000, 2030, row=40, col0=60)
    first, other = _seg(10.0, 13.0, 0.9), _seg(200.0, 203.0, 0.6)

    _run(frames, [first, other], replay_window=90.0)

    assert other.score == pytest.approx(0.6)
    assert "repeat_penalty" not in other.meta


def test_a_goal_its_replay_and_a_third_angle_all_share_one_moment():
    """Chained: the first and last are 40s apart, still one event."""
    frames = _blank(1200)
    for start in (100, 300, 500):
        _moving_box(frames, start, start + 30, row=10, col0=10)
    clips = [_seg(10.0, 13.0, 0.9), _seg(30.0, 33.0, 0.7), _seg(50.0, 53.0, 0.6)]

    _run(frames, clips, replay_window=25.0)

    assert len({c.meta["moment"] for c in clips}) == 1
    assert all("repeat_penalty" not in c.meta for c in clips)


def test_it_does_nothing_without_frames():
    """Driven outside the pipeline there is no context; degrade, do not crash."""
    clips = [_seg(10.0, 13.0, 0.9), _seg(200.0, 203.0, 0.6)]
    out = Similarity().refine(
        VideoInfo(path="x", duration=300, fps=30, width=96, height=54, has_audio=False), clips
    )
    assert [c.score for c in out] == [0.9, 0.6]


# ------------------------------------------------------------- the whole way


@pytest.mark.skipif(not __import__("shutil").which("ffmpeg"), reason="ffmpeg not installed")
def test_a_repeat_is_demoted_through_the_real_pipeline(repeat_vod):
    """Unit tests prove the rule; this proves the rule survives the pipeline.

    Three things had to line up for this to work, and each was wrong first
    time: the refiner needs the decoded frames (it gets them from `ctx`), the
    descriptor has to describe motion rather than appearance, and `merge` has
    to carry the marker across a join or the cut list cannot explain itself.
    """
    from hypecut import load_config
    from hypecut.pipeline import analyze
    from tests.conftest import REPEAT_AGAIN, REPEAT_DIFFERENT, REPEAT_FIRST

    root = Path(__file__).resolve().parents[1] / "configs"
    cfg = load_config(root / "default.yaml").merged(
        {"segments": {"percentile": 88, "min_prominence": 0}}
    )
    plan = analyze(repeat_vod, cfg)

    def covering(moment: tuple[float, float]) -> Candidate | None:
        mid = sum(moment) / 2
        return next((s for s in plan.segments if s.start <= mid <= s.end), None)

    first, again = covering(REPEAT_FIRST), covering(REPEAT_AGAIN)
    different = covering(REPEAT_DIFFERENT)
    assert None not in (first, again, different), "all three moments should be found"

    # Exactly one of the pair is demoted, and which one is not fixed: by the
    # time this refiner runs, `diversity` and `pacing` have already moved the
    # scores, and the rule is to keep whichever take is better *then*. Assert
    # the contract, not the winner.
    penalised = [c for c in (first, again) if c.meta.get("repeat_penalty")]
    assert len(penalised) == 1, "one of two identical runs, not both and not neither"
    assert first.meta["moment"] != again.meta["moment"], "far apart is not a replay"
    assert not different.meta.get("repeat_penalty"), (
        "the control moves the other way across the frame and is nobody's repeat"
    )
