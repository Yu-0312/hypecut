"""Sports signals and the reaction-lag offset."""

from __future__ import annotations

import numpy as np
import pytest

from hypecut.config import SegmentConfig
from hypecut.segments import build_candidates
from hypecut.signals import get_signal
from hypecut.types import AnalysisContext, VideoInfo

SR = 16_000
GRID = 10.0


def _audio_ctx(build, seconds: float = 30.0) -> AnalysisContext:
    """A context whose audio is produced by ``build(t)`` over ``seconds``."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    audio = build(t).astype(np.float32)
    steps = int(seconds * GRID)
    return AnalysisContext(
        info=VideoInfo("fake.mp4", seconds, 30.0, 1280, 720, True),
        grid_fps=GRID,
        times=np.arange(steps) / GRID,
        audio=audio,
        audio_sr=SR,
    )


def _peak_time(values: np.ndarray) -> float:
    return float(np.argmax(values)) / GRID


# --------------------------------------------------------------- crowd_roar


def test_crowd_roar_prefers_a_plateau_over_a_spike():
    """The whole point: a roar holds, a bang does not."""
    rng = np.random.default_rng(3)

    def build(t):
        bed = rng.normal(0, 0.02, t.size)
        spike = ((t >= 8.0) & (t < 8.4)) * rng.normal(0, 0.6, t.size)  # loud, brief
        roar = ((t >= 18.0) & (t < 24.0)) * rng.normal(0, 0.25, t.size)  # quieter, held
        return bed + spike + roar

    values = get_signal("crowd_roar")(sustain_seconds=2.0).compute(_audio_ctx(build))

    assert 18.0 <= _peak_time(values) <= 24.0, "should peak inside the sustained roar"
    spike_window = values[75:90]
    roar_window = values[185:235]
    assert roar_window.max() > spike_window.max(), "the brief spike must not win"


def test_crowd_roar_is_flat_on_uniform_noise():
    rng = np.random.default_rng(4)
    values = get_signal("crowd_roar")().compute(_audio_ctx(lambda t: rng.normal(0, 0.05, t.size)))
    assert float(np.ptp(values)) < 1.0, "no roar, no structure"


def test_crowd_roar_prefers_continuous_noise_over_speech_that_breathes():
    """Duration, not frequency, is what separates a crowd from a commentator.

    A voice fundamental sits inside the crowd band, so no band choice can
    tell them apart. A speaker pauses between phrases; a stadium does not.
    """
    rng = np.random.default_rng(5)

    def build(t):
        bed = rng.normal(0, 0.02, t.size)
        # "Commentary": loud, but with a gap every second.
        talking = (t >= 6.0) & (t < 14.0) & (np.sin(2 * np.pi * 1.0 * t) > -0.2)
        # "Crowd": quieter, but unbroken.
        roaring = (t >= 20.0) & (t < 27.0)
        return bed + talking * rng.normal(0, 0.5, t.size) + roaring * rng.normal(0, 0.2, t.size)

    values = get_signal("crowd_roar")(sustain_seconds=2.0).compute(_audio_ctx(build))
    assert 20.0 <= _peak_time(values) <= 27.0


# ------------------------------------------------------------------ whistle


def test_whistle_fires_on_a_narrowband_tone_and_not_on_noise():
    rng = np.random.default_rng(6)

    def build(t):
        bed = rng.normal(0, 0.05, t.size)
        blast = ((t >= 12.0) & (t < 12.6)) * 0.4 * np.sin(2 * np.pi * 3500 * t)
        return bed + blast

    values = get_signal("whistle")().compute(_audio_ctx(build))

    assert 11.9 <= _peak_time(values) <= 12.7
    assert values.max() > 20 * float(np.median(values))


def test_whistle_ignores_broadband_noise_in_its_own_band():
    """Loudness in the band is not enough — the energy has to be concentrated."""
    rng = np.random.default_rng(7)

    def build(t):
        bed = rng.normal(0, 0.05, t.size)
        # Hiss, deliberately loud, spread across the whistle band.
        hiss = ((t >= 12.0) & (t < 14.0)) * rng.normal(0, 0.5, t.size)
        return bed + hiss

    values = get_signal("whistle")().compute(_audio_ctx(build))
    assert values.max() < 8 * float(np.median(values))


# --------------------------------------------------------------- roi_change


def _video_ctx(gray: np.ndarray) -> AnalysisContext:
    n = gray.shape[0]
    return AnalysisContext(
        info=VideoInfo("fake.mp4", n / GRID, 30.0, 1280, 720, True),
        grid_fps=GRID,
        times=np.arange(n) / GRID,
        gray=gray,
    )


def test_roi_change_catches_a_scoreboard_flip():
    n, h, w = 100, 54, 96
    rng = np.random.default_rng(8)
    gray = np.zeros((n, h, w), dtype=np.uint8)
    for i in range(n):
        gray[i] = rng.integers(40, 60, size=(h, w))  # gently noisy pitch
        gray[i, 2:8, 2:14] = 200  # the score bug, steady...
    gray[60:, 2:8, 2:14] = 90  # ...until it flips

    values = get_signal("roi_change")(box=[0.0, 0.0, 0.16, 0.16]).compute(_video_ctx(gray))
    assert _peak_time(values) == pytest.approx(6.0, abs=0.2)


def test_roi_change_ignores_a_camera_cut():
    """A cut moves the box and everything else; subtracting the frame cancels it."""
    n, h, w = 100, 54, 96
    gray = np.full((n, h, w), 40, dtype=np.uint8)
    gray[50:] = 200  # whole-frame change, box included

    values = get_signal("roi_change")(box=[0.0, 0.0, 0.16, 0.16]).compute(_video_ctx(gray))
    assert values.max() == pytest.approx(0.0, abs=1e-6)


def test_roi_change_ignores_action_outside_the_box():
    n, h, w = 100, 54, 96
    gray = np.full((n, h, w), 40, dtype=np.uint8)
    for i in range(n):
        gray[i, 20:40, (i % 70) : (i % 70) + 12] = 220  # a ball crossing the pitch

    values = get_signal("roi_change")(box=[0.0, 0.0, 0.16, 0.16]).compute(_video_ctx(gray))
    assert values.max() == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------- reaction_lag


def _curve_with_burst(n: int = 600, burst: slice = slice(300, 340)) -> np.ndarray:
    curve = np.zeros(n)
    curve[burst] = 1.0
    return curve


def test_reaction_lag_moves_the_in_point_earlier():
    times = np.arange(600) / GRID
    curve = _curve_with_burst()
    base = SegmentConfig(pre_roll=3.0, post_roll=2.0, percentile=99.0, min_duration=1.0)
    lagged = SegmentConfig(
        pre_roll=3.0, post_roll=2.0, percentile=99.0, min_duration=1.0, reaction_lag=2.5
    )

    a = build_candidates(curve, times, base, grid_fps=GRID, duration=60.0)[0]
    b = build_candidates(curve, times, lagged, grid_fps=GRID, duration=60.0)[0]

    assert b.start == pytest.approx(a.start - 2.5, abs=1e-6)


def test_reaction_lag_leaves_the_out_point_alone():
    """The roar and the celebration are worth keeping; only the start moves."""
    times = np.arange(600) / GRID
    curve = _curve_with_burst()
    base = SegmentConfig(pre_roll=3.0, post_roll=2.0, percentile=99.0, min_duration=1.0)
    lagged = SegmentConfig(
        pre_roll=3.0, post_roll=2.0, percentile=99.0, min_duration=1.0, reaction_lag=2.5
    )

    a = build_candidates(curve, times, base, grid_fps=GRID, duration=60.0)[0]
    b = build_candidates(curve, times, lagged, grid_fps=GRID, duration=60.0)[0]

    assert b.end == pytest.approx(a.end, abs=1e-6)


def test_reaction_lag_records_both_the_moment_and_the_reaction():
    times = np.arange(600) / GRID
    cfg = SegmentConfig(percentile=99.0, min_duration=1.0, reaction_lag=2.5)
    cand = build_candidates(_curve_with_burst(), times, cfg, grid_fps=GRID, duration=60.0)[0]

    assert cand.meta["reaction_time"] == pytest.approx(30.0, abs=0.2)
    assert cand.meta["peak_time"] == pytest.approx(27.5, abs=0.2)
    assert cand.start <= cand.meta["peak_time"] <= cand.end


def test_reaction_lag_defaults_to_zero_and_records_nothing():
    times = np.arange(600) / GRID
    cfg = SegmentConfig(percentile=99.0, min_duration=1.0)
    cand = build_candidates(_curve_with_burst(), times, cfg, grid_fps=GRID, duration=60.0)[0]
    assert "reaction_time" not in cand.meta


def test_reaction_lag_cannot_push_a_clip_before_the_start_of_the_video():
    times = np.arange(600) / GRID
    curve = np.zeros(600)
    curve[5:15] = 1.0  # a moment right at the top of the file
    cfg = SegmentConfig(percentile=99.0, min_duration=1.0, pre_roll=3.0, reaction_lag=5.0)
    cand = build_candidates(curve, times, cfg, grid_fps=GRID, duration=60.0)[0]
    assert cand.start >= 0.0
    assert cand.meta["peak_time"] >= 0.0
