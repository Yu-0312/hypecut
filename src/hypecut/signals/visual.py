"""Frame-derived signals: cuts, motion, flashes and HUD regions.

All of these read the tiny grayscale plane already on the context, so the
whole set costs a few hundred milliseconds for an hour of footage.
"""

from __future__ import annotations

import numpy as np

from ..types import AnalysisContext
from .base import Signal, register

__all__ = ["SceneChange", "Motion", "Flash", "RoiActivity"]


def _frames(ctx: AnalysisContext) -> np.ndarray:
    assert ctx.gray is not None
    return ctx.gray.astype(np.float32, copy=False)


@register("scene_change")
class SceneChange(Signal):
    """Mean absolute difference between consecutive frames.

    Spikes on hard cuts, respawns, scoreboard pop-ups and killcams — the
    structural boundaries an editor would also notice.
    """

    description = "Frame-to-frame difference — cuts, respawns, killcams."
    requires_video = True

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        f = _frames(ctx)
        if f.shape[0] < 2:
            return np.zeros(ctx.n)
        diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
        return np.concatenate([diff[:1], diff])


@register("motion")
class Motion(Signal):
    """Spatial variance of the frame difference — how *much* of the screen moves.

    A camera flick moves everything; a menu cursor moves nothing. Weighting
    by coverage separates real action from idle UI churn.

    Params
    ------
    threshold: per-pixel difference counted as motion (default 6).
    """

    description = "Fraction of the frame in motion — action density."
    requires_video = True

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        thr = float(self.params.get("threshold", 6.0))
        f = _frames(ctx)
        if f.shape[0] < 2:
            return np.zeros(ctx.n)
        d = np.abs(np.diff(f, axis=0))
        coverage = (d > thr).mean(axis=(1, 2))
        return np.concatenate([coverage[:1], coverage])


@register("flash")
class Flash(Signal):
    """Global brightness jumps — explosions, flashbangs, ult animations."""

    description = "Global luminance jumps — explosions, flashes, ultimates."
    requires_video = True

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        f = _frames(ctx)
        if f.shape[0] < 2:
            return np.zeros(ctx.n)
        mean = f.mean(axis=(1, 2))
        jump = np.abs(np.diff(mean))
        return np.concatenate([jump[:1], jump])


@register("roi_activity")
class RoiActivity(Signal):
    """Activity inside a fixed region of interest, e.g. the kill feed.

    This is the cheapest way to encode game-specific knowledge without OCR:
    most shooters draw the kill feed in the top-right corner, so churn in
    that rectangle is close to a kill counter. The rectangle is expressed in
    normalised coordinates so it survives any resolution.

    Params
    ------
    box: ``[x0, y0, x1, y1]`` in 0-1 coordinates (default top-right corner).
    threshold: per-pixel difference counted as activity (default 8).
    """

    description = "Change inside a normalised ROI box — kill feed / scoreboard."
    requires_video = True

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        box = self.params.get("box", [0.62, 0.03, 0.99, 0.30])
        thr = float(self.params.get("threshold", 8.0))
        f = _frames(ctx)
        if f.shape[0] < 2:
            return np.zeros(ctx.n)
        h, w = f.shape[1], f.shape[2]
        x0 = int(np.clip(box[0], 0, 1) * w)
        y0 = int(np.clip(box[1], 0, 1) * h)
        x1 = max(x0 + 1, int(np.clip(box[2], 0, 1) * w))
        y1 = max(y0 + 1, int(np.clip(box[3], 0, 1) * h))
        roi = f[:, y0:y1, x0:x1]
        d = np.abs(np.diff(roi, axis=0))
        activity = (d > thr).mean(axis=(1, 2))
        return np.concatenate([activity[:1], activity])
