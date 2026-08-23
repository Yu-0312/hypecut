"""Turn many raw signals into one excitement curve.

The rule is deliberately simple and inspectable: robustly normalise each
track, clip the tails, weight, sum, smooth. Every step is reversible in
your head, which matters when a user asks "why did it pick *that* clip?" —
:func:`explain` answers that from the same numbers.
"""

from __future__ import annotations

import numpy as np

from .types import SignalTrack

__all__ = ["robust_z", "smooth", "fuse", "explain", "prominence"]


def robust_z(values: np.ndarray, clip: float = 4.0) -> np.ndarray:
    """Median/MAD normalisation, resistant to a few huge outliers.

    Standard z-scoring lets one explosion flatten the rest of the video;
    the median absolute deviation keeps the ordinary range readable.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return v
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = mad * 1.4826 if mad > 1e-9 else (float(v.std()) or 1.0)
    return np.clip((v - med) / scale, -clip, clip)


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with edge padding."""
    window = int(max(1, window))
    if window <= 1 or values.size == 0:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def fuse(
    tracks: list[SignalTrack], *, grid_fps: float, smooth_seconds: float = 1.5, clip: float = 4.0
) -> np.ndarray:
    """Weighted sum of normalised tracks, smoothed, rescaled to 0-1."""
    if not tracks:
        return np.zeros(0, dtype=np.float64)

    length = max(t.values.shape[0] for t in tracks)
    total_weight = sum(abs(t.weight) for t in tracks) or 1.0
    acc = np.zeros(length, dtype=np.float64)
    for t in tracks:
        z = robust_z(t.values, clip=clip)
        if z.shape[0] < length:
            z = np.pad(z, (0, length - z.shape[0]), mode="edge")
        acc += t.weight * z
    acc /= total_weight

    acc = smooth(acc, int(round(smooth_seconds * grid_fps)))
    lo, hi = float(acc.min()), float(acc.max())
    if hi - lo < 1e-9:
        return np.zeros_like(acc)
    return (acc - lo) / (hi - lo)


def prominence(tracks: list[SignalTrack], *, grid_fps: float, smooth_seconds: float = 1.5) -> float:
    """How far the best moment stands above this video's own background.

    This is the one number in the pipeline that means something across
    videos, and it exists to answer a question the rest of the design
    cannot: *is there anything here at all?*

    Everything downstream is relative. :func:`fuse` min-max rescales the
    curve to 0-1 and :func:`build_candidates` thresholds at a percentile of
    that, so by construction some fraction of every video clears the bar.
    Feed in three hours of an idle lobby and you get back a confident reel of
    its least-boring moments. The relative design is right — a quiet VOD and
    a loud one should both yield reels — but it cannot distinguish a quiet
    video from an empty one.

    So: measure each signal in its own raw units, before any normalisation,
    as the distance from its median to its smoothed peak in MAD-scale units.
    Being a ratio it carries no units and needs no calibration corpus, and
    the whole-video maximum is taken over the *smoothed* track so a single
    corrupt frame cannot pass for a highlight. The strongest signal wins
    rather than the average: one detector finding something is enough, and a
    goal does not stop being a goal because the other seven signals slept
    through it.

    A ratio alone is not enough, though, and the failure is worth stating
    because it is the same one that produced phantom shot boundaries. In
    footage that never changes, ``scene_change`` has a MAD near zero, so
    codec flicker of five hundredths of a luma level divides out to "nine
    times the usual" and looks exactly like a cut. Each signal therefore
    declares a ``noise_floor`` in its own units, and a track whose peak does
    not rise that far above its median does not get a vote. Signals whose
    output has no physical unit declare no floor and are judged on the ratio
    alone.
    """
    best = 0.0
    window = int(round(max(0.0, smooth_seconds) * max(grid_fps, 1e-6)))
    for track in tracks:
        if track.weight <= 0:
            continue
        values = np.asarray(track.values, dtype=np.float64)
        if values.size == 0:
            continue
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)))
        scale = mad * 1.4826 if mad > 1e-9 else float(values.std())
        if scale <= 1e-9:
            # A dead-flat track carries no information either way. It is not
            # evidence of emptiness — a muted stream has a silent audio
            # track and may still be full of highlights — so it abstains.
            continue
        peak = float(smooth(values, window).max())
        rise = peak - med
        if rise < track.noise_floor:
            continue
        best = max(best, rise / scale)
    return best


def explain(
    tracks: list[SignalTrack], start_idx: int, end_idx: int, clip: float = 4.0
) -> dict[str, float]:
    """Per-signal mean contribution over a window, for clip provenance.

    The numbers returned are what the UI shows as "why this clip": each
    signal's normalised average across the segment, so a viewer can see
    whether a clip was picked for its audio spike or its screen motion.
    """
    out: dict[str, float] = {}
    for t in tracks:
        z = robust_z(t.values, clip=clip)
        window = z[start_idx:end_idx]
        if window.size:
            out[t.name] = float(window.mean())
    return out
