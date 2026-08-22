"""Signals for broadcast and amateur sports footage.

Sports breaks three assumptions the gameplay signals were built on.

**The reaction is not the moment.** A goal is silent; the roar arrives a
second or two later and then *holds* for five or ten. `audio_transient`,
which rewards sudden change, fires on the onset of the roar and then loses
interest exactly when the crowd is loudest. What matters here is the
plateau, not the edge — see :class:`CrowdRoar`. (The offset between the
moment and its reaction is handled separately, by ``segments.reaction_lag``.)

**The scoreboard is the ground truth.** A score changing is not a proxy for
something important happening; it *is* the thing happening. But a scoreboard
digit is a few dozen pixels, invisible to any whole-frame measure — and the
naive fix, watching a region for change, fires on every camera cut too.
:class:`RoiChange` measures the region *against the rest of the frame*
instead, which is what makes it specific.

**The referee tells you where the boundaries are.** A whistle is a near-pure
tone in a narrow band, trivially separable from crowd noise and commentary,
and it marks the start and end of almost every phase of play.
"""

from __future__ import annotations

import numpy as np

from ..types import AnalysisContext
from .base import Signal, register

__all__ = ["CrowdRoar", "Whistle", "RoiChange"]


@register("crowd_roar")
class CrowdRoar(Signal):
    """Sustained elevated crowd noise, measured as a plateau rather than a spike.

    The trick is a rolling *minimum*. A transient — a ball hitting a post, a
    door slamming, a clipped microphone — cannot survive a minimum taken over
    two seconds, while a genuine roar can, because a roar is loud for its
    whole duration. Subtracting a long-window median then asks the only
    question that matters: is this stretch louder than this match's normal?

    Params
    ------
    sustain_seconds: how long the noise must hold up (default 2.0).
    baseline_seconds: window for "this match's normal" (default 30.0).
    low_hz / high_hz: crowd band (default 100-1500 Hz), where a stadium puts
        most of its energy.

    Note what this does *not* do: it cannot separate a crowd from a
    commentator on frequency. A human voice fundamental sits at 85-255 Hz,
    squarely inside the crowd band, and no band choice fixes that. What
    separates them is duration — a commentator breathes and a crowd does
    not, so a sentence dies under a two-second rolling minimum while a roar
    survives it. Widen ``sustain_seconds`` if commentary is still leaking in.
    """

    description = "Sustained crowd noise — the roar that follows a goal, not the hit."
    requires_audio = True

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        sustain = max(1, int(round(float(self.params.get("sustain_seconds", 2.0)) * ctx.grid_fps)))
        baseline = max(
            3, int(round(float(self.params.get("baseline_seconds", 30.0)) * ctx.grid_fps))
        )
        low = float(self.params.get("low_hz", 100.0))
        high = float(self.params.get("high_hz", 1500.0))

        frames = ctx.audio_frames()
        n_fft = frames.shape[1]
        if n_fft < 8:
            return np.zeros(ctx.n)

        window = np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frames * window, axis=1))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / ctx.audio_sr)
        band = (freqs >= low) & (freqs <= high)
        if not band.any():
            return np.zeros(ctx.n)

        energy = np.log1p(spec[:, band].sum(axis=1))
        return _rolling(energy, sustain, np.min) - _rolling(energy, baseline, np.median)


@register("whistle")
class Whistle(Signal):
    """Referee whistles: a near-pure tone in a narrow high band.

    Two things have to be true at once. The band has to hold a real share of
    the frame's energy — otherwise faint background tones count — and the
    energy inside it has to be concentrated in one bin rather than spread,
    which is what separates a whistle from crowd noise that happens to reach
    the same frequencies. Multiplying the two means neither alone is enough.

    Params
    ------
    low_hz / high_hz: whistle band (default 2000-5000 Hz, covering pea and
        pealess whistles and their first harmonic).
    """

    description = "Referee whistle — a narrowband tone burst above the crowd."
    requires_audio = True

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        low = float(self.params.get("low_hz", 2000.0))
        high = float(self.params.get("high_hz", 5000.0))

        frames = ctx.audio_frames()
        n_fft = frames.shape[1]
        if n_fft < 8:
            return np.zeros(ctx.n)

        window = np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frames * window, axis=1))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / ctx.audio_sr)
        band = (freqs >= low) & (freqs <= high)
        if band.sum() < 4:
            return np.zeros(ctx.n)

        in_band = spec[:, band]
        peakiness = in_band.max(axis=1) / (in_band.mean(axis=1) + 1e-9)
        share = in_band.sum(axis=1) / (spec.sum(axis=1) + 1e-9)
        return peakiness * share


@register("roi_change")
class RoiChange(Signal):
    """Change confined to a small region — a scoreboard, a clock, a counter.

    ``roi_activity`` asks "is this region busy?", which a camera cut answers
    yes to as loudly as a goal does. This asks the better question: is this
    region changing *more than the rest of the frame*? A score bug flipping
    from 1 to 2 moves a few dozen pixels and nothing else; a cut moves
    everything. Subtracting the global difference leaves the first and
    cancels the second, which is the whole idea.

    Params
    ------
    box: ``[x0, y0, x1, y1]`` in 0-1 coordinates. There is no useful default
        across sports and broadcasters — find yours with ``hypecut analyze``
        and adjust.
    """

    description = "Change isolated to a small box — scoreboards, clocks, counters."
    requires_video = True

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        assert ctx.gray is not None
        box = self.params.get("box", [0.0, 0.0, 0.35, 0.18])
        f = ctx.gray.astype(np.float32, copy=False)
        if f.shape[0] < 2:
            return np.zeros(ctx.n)

        h, w = f.shape[1], f.shape[2]
        x0, x1 = sorted((int(np.clip(box[0], 0, 1) * w), int(np.clip(box[2], 0, 1) * w)))
        y0, y1 = sorted((int(np.clip(box[1], 0, 1) * h), int(np.clip(box[3], 0, 1) * h)))
        x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)

        diff = np.abs(np.diff(f, axis=0))
        roi = diff[:, y0:y1, x0:x1].mean(axis=(1, 2))
        whole = diff.mean(axis=(1, 2))
        isolated = np.maximum(roi - whole, 0.0)
        return np.concatenate([isolated[:1], isolated])


def _rolling(values: np.ndarray, window: int, fn) -> np.ndarray:
    """Apply ``fn`` over a centred sliding window, with edge padding."""
    window = max(1, int(window))
    if window <= 1 or values.size == 0:
        return values
    window = min(window, values.size)
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    strides = np.lib.stride_tricks.sliding_window_view(padded, window)
    return fn(strides, axis=1)[: values.size]
