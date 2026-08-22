"""Turn many raw signals into one excitement curve.

The rule is deliberately simple and inspectable: robustly normalise each
track, clip the tails, weight, sum, smooth. Every step is reversible in
your head, which matters when a user asks "why did it pick *that* clip?" —
:func:`explain` answers that from the same numbers.
"""

from __future__ import annotations

import numpy as np

from .types import SignalTrack

__all__ = ["robust_z", "smooth", "fuse", "explain"]


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
