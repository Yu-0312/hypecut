"""Move clip edges onto real shot boundaries.

The single most visible difference between an auto-cut reel and a hand-cut
one is where the edges land. A clip that starts three frames into a
continuous shot looks *sliced*; the same clip started half a second earlier,
on the cut the game or the streamer's scene switcher already made, looks
*edited*. Nothing about the content changed — only the edge did.

So this runs after selection: find the hard cuts, then let every edge travel
up to a couple of seconds to reach one. Two guards keep it honest:

* an edge never crosses the clip's peak (that is the moment being kept), and
* a snap that would violate the length budget is rejected rather than
  clamped, because a clip that is suddenly a second under ``min_duration``
  is a worse outcome than an unsnapped edge.

Detection is coarse-then-fine. The 10 Hz analysis frames are already in
memory, so candidate boundaries cost nothing to find; each accepted edge is
then re-examined at the source frame rate over a one-second window, which
turns a ±50 ms answer into a frame-exact one for the price of decoding a few
dozen tiny frames.
"""

from __future__ import annotations

import numpy as np

from . import ffmpeg as ff
from .config import SegmentConfig
from .types import AnalysisContext, Candidate

__all__ = ["boundary_strength", "find_boundaries", "refine_boundary", "snap_segments"]


def boundary_strength(gray: np.ndarray) -> np.ndarray:
    """Per-frame cut likelihood: frame difference over its local baseline.

    Raw frame difference is useless as an absolute measure — a chaotic
    teamfight differs more frame-to-frame than a hard cut in a menu does.
    Dividing by a running median turns it into "how unusual is this
    difference *for this stretch of video*", which is what a cut actually is.
    """
    if gray is None or gray.shape[0] < 3:
        return np.zeros(0, dtype=np.float64)

    f = gray.astype(np.float32, copy=False)
    diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2)).astype(np.float64)
    diff = np.concatenate([diff[:1], diff])

    # Local baseline over ~4 s, wide enough to span a fight but not a scene.
    window = min(diff.size, 41)
    if window >= 3:
        pad = window // 2
        padded = np.pad(diff, (pad, pad), mode="edge")
        strides = np.lib.stride_tricks.sliding_window_view(padded, window)
        baseline = np.median(strides, axis=1)[: diff.size]
    else:  # pragma: no cover - only for pathologically short inputs
        baseline = np.full_like(diff, np.median(diff))

    return diff / np.maximum(baseline, 1e-6)


def find_boundaries(
    gray: np.ndarray, grid_fps: float, *, ratio: float = 2.5, min_gap: float = 0.5
) -> np.ndarray:
    """Times (seconds) of likely shot boundaries.

    ``ratio`` is how many times its local baseline a frame difference must
    reach. 2.5 is deliberately permissive: a false boundary costs at most a
    slightly different edge, while a missed one costs the whole feature.
    """
    strength = boundary_strength(gray)
    if strength.size == 0:
        return np.zeros(0, dtype=np.float64)

    peaks = np.flatnonzero(
        (strength >= ratio)
        & (strength >= np.concatenate([strength[:1], strength[:-1]]))
        & (strength >= np.concatenate([strength[1:], strength[-1:]]))
    )
    if peaks.size == 0:
        return np.zeros(0, dtype=np.float64)

    # Keep the strongest peak within each min_gap cluster.
    kept: list[int] = []
    gap = max(1, int(round(min_gap * grid_fps)))
    for idx in peaks[np.argsort(strength[peaks])[::-1]]:
        if all(abs(int(idx) - k) >= gap for k in kept):
            kept.append(int(idx))
    return np.sort(np.asarray(kept, dtype=np.float64)) / grid_fps


def refine_boundary(
    path: str,
    at: float,
    *,
    source_fps: float,
    window: float = 0.5,
    size: tuple[int, int] = (96, 54),
) -> float:
    """Re-locate a boundary at the source frame rate within ±``window``.

    The coarse pass can only ever be as precise as the analysis grid (100 ms
    at the default 10 Hz), which is two or three frames of visible slop on a
    hard cut. Decoding a one-second window at native rate fixes that for a
    few milliseconds of work per edge.
    """
    fps = source_fps if source_fps and source_fps > 0 else 30.0
    start = max(0.0, at - window)
    try:
        frames = ff.decode_gray_frames(
            path, fps=fps, width=size[0], height=size[1], start=start, duration=window * 2
        )
    except ff.FFmpegError:  # pragma: no cover - defensive; keep the coarse answer
        return at
    if frames.shape[0] < 3:
        return at

    # The largest difference is between frames k and k+1, so the incoming shot
    # begins at k+1. Rounding to the frame *after* the change is deliberate:
    # being one frame late is invisible, while being one frame early shows a
    # flash of the outgoing shot, which is exactly the artefact this is
    # supposed to remove. Expect the result to be accurate to ±1 frame.
    diff = np.abs(np.diff(frames.astype(np.float32), axis=0)).mean(axis=(1, 2))
    return start + (int(np.argmax(diff)) + 1) / fps


def snap_segments(
    ctx: AnalysisContext, segments: list[Candidate], cfg: SegmentConfig
) -> list[Candidate]:
    """Move each clip's edges to the nearest shot boundary, in place."""
    if not segments or not cfg.snap_to_shots or ctx.gray is None:
        return segments

    boundaries = find_boundaries(ctx.gray, ctx.grid_fps)
    if boundaries.size == 0:
        return segments

    duration = ctx.info.duration

    # An edge may always travel at least as far as the roll that placed it.
    # The roll is a guess about how much wind-up a moment needs; a real shot
    # boundary inside that span is better information than the guess, and with
    # a 3 s pre-roll a fixed 2 s window could never reach the cut that started
    # the scene.
    start_window = max(cfg.snap_window, cfg.pre_roll)
    end_window = max(cfg.snap_window, cfg.post_roll)

    for seg in segments:
        peak = float(seg.meta.get("peak_time", (seg.start + seg.end) / 2))
        moved: dict[str, float] = {}

        # In-point: anywhere in the window, up to the peak itself. It may look
        # wrong to let the start move forward until almost no wind-up is left,
        # but a hard cut between the old start and the peak means that wind-up
        # belonged to a different scene — keeping it would open the clip on
        # unrelated footage. The peak is the hard stop; nothing may cross it.
        new_start = _nearest(boundaries, seg.start, start_window, lo=0.0, hi=peak)
        if new_start is not None and cfg.min_duration <= seg.end - new_start <= cfg.max_duration:
            if cfg.snap_fine:
                new_start = refine_boundary(ctx.info.path, new_start, source_fps=ctx.info.fps)
            moved["start"] = round(new_start - seg.start, 3)
            seg.start = max(0.0, new_start)

        # Out-point: not a mirror image. Here ``snap_guard`` does apply — an
        # end that lands right on the peak would cut the payoff off mid-beat,
        # and unlike the in-point there is no "wrong scene" argument for it.
        new_end = _nearest(boundaries, seg.end, end_window, lo=peak + cfg.snap_guard, hi=duration)
        if new_end is not None and cfg.min_duration <= new_end - seg.start <= cfg.max_duration:
            if cfg.snap_fine:
                new_end = refine_boundary(ctx.info.path, new_end, source_fps=ctx.info.fps)
            moved["end"] = round(new_end - seg.end, 3)
            seg.end = min(duration, new_end)

        if moved:
            seg.meta["snapped"] = moved
    return segments


def _nearest(
    boundaries: np.ndarray, target: float, window: float, *, lo: float, hi: float
) -> float | None:
    """Closest boundary to ``target`` within ``window`` and inside [lo, hi]."""
    if hi <= lo:
        return None
    mask = (np.abs(boundaries - target) <= window) & (boundaries >= lo) & (boundaries <= hi)
    if not mask.any():
        return None
    options = boundaries[mask]
    return float(options[int(np.argmin(np.abs(options - target)))])
