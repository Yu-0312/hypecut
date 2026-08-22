"""Move clip edges into the pauses between sounds.

Shot snapping fixed the visual half of edge placement. This is the audio
half, and it matters for the footage snapping cannot help: a locked-off
talking-head stream has no cuts at all, so every edge lands wherever the
post-roll happened to put it — routinely three words into a sentence.

The rule is simple and the same at both ends: an edge belongs in a gap, not
in the middle of a sound. Find the nearest pause, land just inside it.

**A real cut always wins.** This only ever considers edges that found no shot
boundary. A hard cut is unambiguous evidence about where a moment ends; a
pause is a good guess. Running both and letting the second overwrite the
first would mean the weaker signal decides, which is backwards.

Everything here reads the audio already decoded onto the analysis grid, so
the whole pass costs a few milliseconds per clip.
"""

from __future__ import annotations

import numpy as np

from .config import SegmentConfig
from .types import AnalysisContext, Candidate

__all__ = ["level_db", "silence_mask", "ends_quiet", "find_pause", "trim_segments"]


def level_db(ctx: AnalysisContext) -> np.ndarray:
    """Per-grid-step loudness in dBFS."""
    frames = ctx.audio_frames()
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
    return 20.0 * np.log10(np.maximum(rms, 1e-7))


def silence_mask(level: np.ndarray, i0: int, i1: int, drop_db: float) -> np.ndarray | None:
    """Boolean "this step is a pause" over the whole track, keyed to one clip.

    The threshold is relative to *this clip's* own speech level, not a global
    one: a whispered aside and a shouted play have nothing in common in dBFS,
    but both have a 14 dB gap between talking and not talking.

    Returns ``None`` when the clip has no usable contrast — either almost
    nothing is below the threshold (continuous sound) or almost everything is
    (continuous quiet). In both cases there is no pause to find, and guessing
    would move edges for no reason.
    """
    window = level[i0:i1]
    if window.size < 4:
        return None

    reference = float(np.percentile(window, 75))
    mask = level < (reference - drop_db)

    quiet_share = float(mask[i0:i1].mean())
    if quiet_share < 0.02 or quiet_share > 0.7:
        return None
    return mask


def ends_quiet(level: np.ndarray, i0: int, i1: int, end_idx: int, drop_db: float) -> bool:
    """Is the clip's last moment a quiet one?

    Answered separately from :func:`silence_mask` on purpose. A clip sitting
    entirely inside continuous sound has no pauses to snap to, but it is
    exactly the clip that most needs the longer audio fade — so "no usable
    pauses" must not also mean "no answer here".
    """
    window = level[i0:i1]
    if window.size < 2:
        return False
    return bool(
        level[int(np.clip(end_idx, 0, level.size - 1))] < np.percentile(window, 75) - drop_db
    )


def find_pause(
    mask: np.ndarray,
    grid_fps: float,
    *,
    target: float,
    window: float,
    min_silence: float,
    lo: float,
    hi: float,
    prefer: str,
) -> float | None:
    """Nearest usable pause to ``target``, in seconds.

    ``prefer`` selects which end of the pause is returned: ``"start"`` for an
    out-point (stop as the sound stops) and ``"end"`` for an in-point (start
    as the sound resumes).
    """
    runs = _silent_runs(mask, min_length=max(1, int(round(min_silence * grid_fps))))
    if not runs:
        return None

    best: float | None = None
    best_distance = window
    for run_start, run_end in runs:
        edge = (run_start if prefer == "start" else run_end) / grid_fps
        if not (lo <= edge <= hi):
            continue
        distance = abs(edge - target)
        if distance <= best_distance:
            best, best_distance = edge, distance
    return best


def trim_segments(
    ctx: AnalysisContext, segments: list[Candidate], cfg: SegmentConfig
) -> list[Candidate]:
    """Nudge un-snapped clip edges into nearby pauses, in place."""
    if not segments or not cfg.trim_to_silence:
        return segments
    if ctx.audio is None or ctx.audio.size == 0:
        return segments

    level = level_db(ctx)
    duration = ctx.info.duration

    # Same reasoning as the snapper: an edge may travel at least as far as the
    # roll that placed it, because that roll was only ever a guess.
    start_window = max(cfg.silence_window, cfg.pre_roll)
    end_window = max(cfg.silence_window, cfg.post_roll)

    for seg in segments:
        snapped = seg.meta.get("snapped") or {}
        peak = float(seg.meta.get("peak_time", (seg.start + seg.end) / 2))
        i0 = int(np.clip(seg.start * ctx.grid_fps, 0, level.size - 1))
        i1 = int(np.clip(seg.end * ctx.grid_fps, i0 + 1, level.size))

        mask = silence_mask(level, i0, i1, cfg.silence_drop_db)
        if mask is None:
            # No pauses to work with, but the fade hint is still knowable.
            seg.meta["ends_in_silence"] = ends_quiet(
                level, i0, i1, int(seg.end * ctx.grid_fps), cfg.silence_drop_db
            )
            continue

        moved: dict[str, float] = {}

        if "start" not in snapped:
            # In-point: land where sound resumes, minus a little breathing room.
            edge = find_pause(
                mask,
                ctx.grid_fps,
                target=seg.start,
                window=start_window,
                min_silence=cfg.min_silence,
                lo=max(0.0, seg.start - start_window),
                hi=min(peak, seg.start + start_window),
                prefer="end",
            )
            if edge is not None:
                new_start = max(0.0, edge - cfg.silence_pad)
                if cfg.min_duration <= seg.end - new_start <= cfg.max_duration:
                    moved["start"] = round(new_start - seg.start, 3)
                    seg.start = new_start

        if "end" not in snapped:
            # Out-point: land where sound stops, plus the same breathing room.
            edge = find_pause(
                mask,
                ctx.grid_fps,
                target=seg.end,
                window=end_window,
                min_silence=cfg.min_silence,
                lo=peak + cfg.snap_guard,
                hi=min(duration, seg.end + end_window),
                prefer="start",
            )
            if edge is not None:
                new_end = min(duration, edge + cfg.silence_pad)
                if cfg.min_duration <= new_end - seg.start <= cfg.max_duration:
                    moved["end"] = round(new_end - seg.end, 3)
                    seg.end = new_end

        if moved:
            seg.meta["trimmed"] = moved

        # Whether the clip fades out of sound or gets cut off mid-noise is
        # useful downstream: the renderer lengthens the audio fade for the
        # latter so a hard stop does not read as a dropout.
        end_idx = int(np.clip(seg.end * ctx.grid_fps, 0, mask.size - 1))
        seg.meta["ends_in_silence"] = bool(mask[end_idx])

    return segments


def _silent_runs(mask: np.ndarray, min_length: int) -> list[tuple[int, int]]:
    """Index ranges of runs of ``True`` at least ``min_length`` long."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends, strict=True) if e - s >= min_length]
