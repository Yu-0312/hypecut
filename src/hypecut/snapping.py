"""Move clip edges onto real shot boundaries.

The single most visible difference between an auto-cut reel and a hand-cut
one is where the edges land. A clip that starts three frames into a
continuous shot looks *sliced*; the same clip started half a second earlier,
on the cut the game or the streamer's scene switcher already made, looks
*edited*. Nothing about the content changed — only the edge did.

So this runs after selection: find the hard cuts, then let every edge travel
up to a couple of seconds to reach one. Two guards keep it honest:

* an edge never crosses into the clip's *event* — the above-threshold span the
  clip was built around, not merely its loudest frame. For a goal the two are
  the same thing; for a twenty-second rally the loudest frame can be the third
  shot, and a guard placed there would let the first third be trimmed away.
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
from .segments import out_point_floor
from .types import AnalysisContext, Candidate

__all__ = [
    "boundary_strength",
    "find_boundaries",
    "find_dissolves",
    "find_wipes",
    "refine_boundary",
    "snap_segments",
]


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
    gray: np.ndarray,
    grid_fps: float,
    *,
    ratio: float = 2.5,
    min_gap: float = 0.5,
    min_diff: float = 1.5,
) -> np.ndarray:
    """Times (seconds) of likely shot boundaries.

    ``ratio`` is how many times its local baseline a frame difference must
    reach. 2.5 is deliberately permissive: a false boundary costs at most a
    slightly different edge, while a missed one costs the whole feature.

    ``min_diff`` is the absolute floor, in 0-255 luma levels, and it is doing
    real work. On footage that holds perfectly still — a menu, a paused game,
    a static title card — the local baseline collapses to nearly zero, and
    then a single frame of compression flicker is *hundreds of times* the
    baseline. Relative evidence alone would call that a shot change. A
    difference of half a level is not a cut whatever the ratio says.

    The floor is set low on purpose: a cut between two dark scenes can be
    worth only two or three levels, so anything higher would start missing
    real boundaries in night maps and menus.
    """
    strength = boundary_strength(gray)
    if strength.size == 0:
        return np.zeros(0, dtype=np.float64)

    raw = np.abs(np.diff(gray.astype(np.float32), axis=0)).mean(axis=(1, 2))
    raw = np.concatenate([raw[:1], raw]).astype(np.float64)

    peaks = np.flatnonzero(
        (strength >= ratio)
        & (raw >= min_diff)
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


def find_dissolves(
    gray: np.ndarray,
    grid_fps: float,
    *,
    min_length: float = 0.3,
    max_length: float = 2.5,
    accumulation: float = 2.0,
    flatness: float = 3.0,
    min_diff: float = 1.5,
) -> list[tuple[float, float]]:
    """``(start, end)`` times of gradual transitions — dissolves and fades.

    A hard cut is one enormous frame difference. A dissolve spreads the same
    total change across a second or two, so no single frame stands out and the
    peak detector above walks straight past it. The classic answer is
    twin-comparison: look at the difference *accumulated* over a window instead
    of the maximum within it, and treat a window whose total is large while its
    peak is not as a gradual transition.

    That alone would also flag a camera pan, which is likewise sustained change
    with no spike. The discriminator is spatial contrast: during a dissolve the
    frame is a blend of two images, so its standard deviation dips below both
    neighbours — a pan's does not. Requiring that dip is what keeps this from
    firing on every swing of the mouse.

    Parameters
    ----------
    accumulation: how many times the baseline a frame difference must reach to
        count as part of a transition.
    flatness: reject a run whose peak is this many times its own mean — that
        is a hard cut, already handled by :func:`find_boundaries`.
    min_diff: absolute floor in 0-255 luma levels, for the same reason as in
        :func:`find_boundaries`.
    """
    if gray is None or gray.shape[0] < 5:
        return []

    f = gray.astype(np.float32, copy=False)
    diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2)).astype(np.float64)
    spread = f.std(axis=(1, 2)).astype(np.float64)[: diff.size]
    if diff.size < 4:
        return []

    # The contrast context either side of a run is as wide as the longest run
    # the scan can accept — the same window the original twin-comparison used.
    hi_steps = max(3, int(round(max_length * grid_fps)))

    found: list[tuple[float, float]] = []
    for i, j in _gradual_runs(
        diff,
        grid_fps,
        min_length=min_length,
        max_length=max_length,
        accumulation=accumulation,
        flatness=flatness,
        min_diff=min_diff,
    ):
        # Contrast has to dip inside the run relative to its surroundings: a
        # blend of two images is flatter than either, while a camera pan (also
        # sustained, also spike-free) keeps its contrast. A wipe keeps its
        # contrast too, which is why it is invisible here and needs
        # :func:`find_wipes`.
        outside = np.concatenate([spread[max(0, i - hi_steps) : i], spread[j : j + hi_steps]])
        if outside.size == 0 or float(spread[i:j].min()) >= float(np.median(outside)) * 0.92:
            continue
        found.append((i / grid_fps, j / grid_fps))
    return found


def find_wipes(
    gray: np.ndarray,
    grid_fps: float,
    *,
    min_length: float = 0.3,
    max_length: float = 2.5,
    accumulation: float = 2.0,
    min_diff: float = 1.5,
    band_share: float = 0.45,
    travel: float = 0.5,
    coherence: float = 0.7,
) -> list[tuple[float, float]]:
    """``(start, end)`` times of wipes and slides — transitions with a moving front.

    A wipe keeps its contrast throughout, so the contrast-dip test that
    isolates dissolves never fires on one, and no single frame is a spike, so
    the cut detector walks past it too. What a wipe *does* have is structure:
    at every moment the change is dominated by a narrow band — the front —
    and that band crosses the frame in one direction.

    Each frame gets a front: the lines (columns or rows) whose change reaches
    half the frame's peak. Three tests separate a wipe from the other
    sustained-change stretches (pans, busy gameplay) that share its profile:

    * **The change is a front, not a field.** The band stays narrow — its
      mean width is at most ``band_share`` of the frame. In a pan every
      column moves a similar amount, so the "band" is the whole frame; the
      same goes for a fade.
    * **The front travels.** Its centroid must cross at least ``travel`` of
      the frame over the run, and do so in one direction — net travel over
      total path length at least ``coherence``. Gameplay's centroid
      wanders; a wipe's marches.
    * **The front is strong.** Frames vote on the geometry only while their
      front is both absolutely strong (``2 * min_diff``) and within the
      run's own league (30% of its strongest frame), so quiet frames at the
      edges of the sweep cannot skew its shape.

    The test is run on both axes and the stronger one wins, so vertical
    wipes and horizontal slides are found alike.

    Candidate runs are scanned on the *front-peak* trace — the per-frame
    strongest line — rather than the frame-mean difference the dissolve
    scan uses. A wipe's front is by far the highest-contrast thing moving
    while it sweeps, so the transition reads as an isolated run even when
    both of its shots keep moving on their own; on a frame-mean trace it
    would drown in the surrounding footage's own run. A three-frame
    smoothing of the trace bridges one-frame dips; a hard cut survives as a
    three-frame spike and is rejected by the length test.

    ``accumulation`` sets how many times the footage's own busy-time front
    strength a run must reach; ``min_length``/``max_length`` bound the sweep
    duration as in :func:`find_dissolves`.
    """
    if gray is None or gray.shape[0] < 5:
        return []

    f = gray.astype(np.float32, copy=False)
    delta = np.abs(np.diff(f, axis=0))  # (T-1, H, W)
    if delta.shape[0] < 4:
        return []

    h, w = delta.shape[1], delta.shape[2]
    lo = max(2, int(round(min_length * grid_fps)))
    hi = max(lo + 1, int(round(max_length * grid_fps)))

    found: list[tuple[float, float]] = []
    for axis, span in ((0, w), (1, h)):
        # Mean over the axis perpendicular to the sweep: a horizontal wipe
        # summarises each *column*, a vertical one each row.
        spatial = 1 if axis == 0 else 2
        profile = delta.mean(axis=spatial)  # (T-1, span): change per line
        peak = profile.max(axis=1).astype(np.float64)  # per-frame front strength

        # The front must be both absolutely strong and unusual for this
        # footage. The baseline is the 75th percentile of the trace, not the
        # median: a video that is half static has a median of zero, which
        # would make any multiplier vacuous, while the p75 still tracks how
        # busy the busy parts are. Smoothing bridges one-frame dips.
        kernel = np.ones(3) / 3.0
        smooth = np.convolve(peak, kernel, mode="same")
        smooth[0], smooth[-1] = peak[0], peak[-1]
        baseline = float(np.percentile(peak, 75))
        threshold = max(2 * min_diff, baseline * (accumulation + 0.5))

        best: tuple[float, tuple[int, int]] | None = None
        for i, j in _runs(smooth >= threshold):
            if not (lo <= j - i <= hi):
                continue

            vote = peak[i:j] >= max(2 * min_diff, 0.3 * float(peak[i:j].max()))
            if vote.sum() < 3:
                continue

            # The front: lines within half the frame's own peak. Diffuse
            # change spread over many lines does not survive the 50% bar.
            window = profile[i:j]
            band = window >= (0.5 * window.max(axis=1))[:, None]
            width = band.sum(axis=1) / span
            if float(width[vote].mean()) > band_share:
                continue

            lines = np.arange(span, dtype=np.float64)
            mass = np.maximum((window * band).sum(axis=1), 1e-6)
            centroid = (window * band * lines).sum(axis=1) / mass

            c = centroid[vote]
            path = float(np.abs(np.diff(c)).sum())
            net = abs(float(c[-1] - c[0]))
            if net < travel * span or (net / path if path > 1e-6 else 0.0) < coherence:
                continue

            score = net * (2.0 - float(width[vote].mean()))
            if best is None or score > best[0]:
                best = (score, (i, j))
        if best is not None:
            i, j = best[1]
            found.append((i / grid_fps, j / grid_fps))
    return sorted(found)


def _gradual_runs(
    diff: np.ndarray,
    grid_fps: float,
    *,
    min_length: float,
    max_length: float,
    accumulation: float,
    flatness: float,
    min_diff: float,
) -> list[tuple[int, int]]:
    """Index spans of sustained, spike-free change — the candidates both the
    dissolve and the wipe detectors classify in their own way.

    Too short is a hard cut; too long is simply a busy stretch of video, and
    calling that a transition would let an edge land anywhere inside it. A
    spike inside the run means a cut that :func:`find_boundaries` already
    handled, not a gradual one.
    """
    baseline = float(np.median(diff))
    threshold = max(baseline * accumulation, min_diff)

    lo = max(2, int(round(min_length * grid_fps)))
    hi = max(lo + 1, int(round(max_length * grid_fps)))

    out: list[tuple[int, int]] = []
    for i, j in _runs(diff >= threshold):
        if not (lo <= j - i <= hi):
            continue
        window = diff[i:j]
        if float(window.max()) > flatness * float(window.mean()):
            continue
        out.append((i, j))
    return out


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges of contiguous ``True`` values (end-exclusive)."""
    if mask.size == 0:
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(np.int8))
    return list(
        zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist(), strict=True)
    )


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

    cuts = find_boundaries(ctx.gray, ctx.grid_fps)
    dissolves = find_dissolves(ctx.gray, ctx.grid_fps) if cfg.snap_to_dissolves else []
    wipes = find_wipes(ctx.gray, ctx.grid_fps) if cfg.snap_to_dissolves else []

    # A gradual transition has two useful landing points, and which one is
    # right depends on the edge: an in-point wants the far side, so the clip
    # opens on the incoming shot rather than on the mix; an out-point wants the
    # near side, so it leaves before the picture starts dissolving away. A
    # wipe is the same shape of answer — its landing points are where the sweep
    # begins and where it completes — so both kinds share the machinery.
    # A transition is not spike-free at 10 Hz — some of its frames clear the
    # cut detector's bar on their own. Those are not separate boundaries, they
    # are the middle of one, and letting an edge land there would put the clip
    # in the mix between two shots. The transition is the better description
    # of that stretch, so cuts inside one are dropped.
    transitions = dissolves + wipes
    cuts = _outside_spans(cuts, transitions)
    in_points = _merge_boundaries(
        cuts, [(end, "dissolve") for _, end in dissolves] + [(end, "wipe") for _, end in wipes]
    )
    out_points = _merge_boundaries(
        cuts,
        [(start, "dissolve") for start, _ in dissolves] + [(start, "wipe") for start, _ in wipes],
    )
    if not in_points[0].size and not out_points[0].size:
        return segments

    duration = ctx.info.duration

    # An edge may always travel at least as far as the roll that placed it.
    # The roll is a guess about how much wind-up a moment needs; a real shot
    # boundary inside that span is better information than the guess, and with
    # a 3 s pre-roll a fixed 2 s window could never reach the cut that started
    # the scene.
    start_window = max(cfg.snap_window, cfg.pre_roll)
    end_window = max(cfg.snap_window, cfg.post_roll)

    for index, seg in enumerate(segments):
        # The event, not the padding. For an instant this is a single point and
        # the guards below behave exactly as they did when that was all there
        # was; for a rally or a teamfight it is the whole exchange, and the
        # difference matters — the loudest frame of a twenty-second rally can
        # be its third shot, and nothing should be allowed to trim to there.
        event_lo, _ = seg.protected()

        # An edge may never cross into a neighbour. `merge` leaves clips more
        # than `merge_gap` apart, but the travel allowed here is larger than
        # that gap — 3 s of start_window against a 2 s default gap — and each
        # segment is snapped in isolation, so nothing else stops one clip's
        # in-point from landing behind the previous clip's out-point. It shows
        # up when the earlier clip's snap is rejected on length while the later
        # one's is accepted: only the later edge moves, and it moves backwards
        # past the earlier end. The reel then plays the same source seconds
        # twice, and the sidecar and EDL faithfully record the overlap.
        #
        # The previous segment is already final here; the next one is not, so
        # its *current* start is the ceiling. Both bounds hold whether or not a
        # snap happens, which is what makes the result non-overlapping rather
        # than merely usually non-overlapping.
        neighbour_lo = segments[index - 1].end if index else 0.0
        neighbour_hi = segments[index + 1].start if index + 1 < len(segments) else duration

        moved: dict[str, float] = {}
        kinds: dict[str, str] = {}

        # In-point: anywhere in the window, up to where the event begins. It may
        # look wrong to let the start move forward until almost no wind-up is
        # left, but a hard cut between the old start and the event means that
        # wind-up belonged to a different scene — keeping it would open the clip
        # on unrelated footage. The event is the hard stop.
        found = _nearest(in_points, seg.start, start_window, lo=neighbour_lo, hi=event_lo)
        if found is not None and cfg.min_duration <= seg.end - found[0] <= cfg.max_duration:
            new_start, kind = found
            # Sub-frame precision only means something for a hard cut; a
            # dissolve has no single frame to be exact about.
            if cfg.snap_fine and kind == "cut":
                new_start = _refined_or(
                    new_start,
                    refine_boundary(ctx.info.path, new_start, source_fps=ctx.info.fps),
                    lo=neighbour_lo,
                    hi=event_lo,
                    fixed_edge=seg.end,
                    is_start=True,
                    cfg=cfg,
                )
            moved["start"] = round(new_start - seg.start, 3)
            kinds["start"] = kind
            seg.start = max(neighbour_lo, new_start)

        # Out-point: not a mirror image. An end landing inside the event would
        # cut the payoff off mid-beat, and unlike the in-point there is no
        # "wrong scene" argument for doing so.
        floor = out_point_floor(seg, cfg.snap_guard)
        found = _nearest(out_points, seg.end, end_window, lo=floor, hi=neighbour_hi)
        if found is not None and cfg.min_duration <= found[0] - seg.start <= cfg.max_duration:
            new_end, kind = found
            if cfg.snap_fine and kind == "cut":
                new_end = _refined_or(
                    new_end,
                    refine_boundary(ctx.info.path, new_end, source_fps=ctx.info.fps),
                    lo=floor,
                    hi=neighbour_hi,
                    fixed_edge=seg.start,
                    is_start=False,
                    cfg=cfg,
                )
            moved["end"] = round(new_end - seg.end, 3)
            kinds["end"] = kind
            seg.end = min(neighbour_hi, new_end)

        if moved:
            seg.meta["snapped"] = moved
            seg.meta["snap_kind"] = kinds
    return segments


def _refined_or(
    coarse: float,
    refined: float,
    *,
    lo: float,
    hi: float,
    fixed_edge: float,
    is_start: bool,
    cfg: SegmentConfig,
) -> float:
    """``refined`` if it still satisfies every guard, otherwise ``coarse``.

    The guards above are checked against the coarse landing point, but the
    value actually assigned is the refined one — and `refine_boundary` returns
    the strongest frame difference anywhere within half a second of where it
    was pointed, with no preference for the point it started from. On fast-cut
    footage that is a *different* boundary: a camera cut 0.3 s after a replay
    wipe, an explosion flash beside the edge. So the refined value can open a
    clip inside the event the window existed to protect, or leave it short of
    `min_duration`, both silently.

    Twenty milliseconds of extra precision is not worth a clip that starts in
    the wrong place, so a refined value that fails re-validation is discarded
    rather than clamped: the coarse boundary is still a real one.
    """
    if not lo <= refined <= hi:
        return coarse
    length = fixed_edge - refined if is_start else refined - fixed_edge
    if not cfg.min_duration <= length <= cfg.max_duration:
        return coarse
    return refined


def _outside_spans(times: np.ndarray, spans: list[tuple[float, float]]) -> np.ndarray:
    """Drop the times that fall inside one of ``spans`` — endpoints included.

    A transition contributes both of its edges as landing points, so a cut on
    a span's own boundary is redundant with it. Steep transitions — a fast
    wipe, a short dissolve — routinely clear the cut detector's bar on their
    final step, and letting that frame pose as a hard cut would land the edge
    mid-sweep under a "cut" label; the transition is the better description
    of the stretch it belongs to.
    """
    if times.size == 0 or not spans:
        return times
    keep = np.ones(times.shape, dtype=bool)
    for start, end in spans:
        keep &= ~((times >= start) & (times <= end))
    return times[keep]


def _merge_boundaries(
    cuts: np.ndarray, gradual: list[tuple[float, str]]
) -> tuple[np.ndarray, list[str]]:
    """One sorted list of landing points, each tagged ``cut``/``dissolve``/``wipe``."""
    times = list(cuts.tolist()) + [t for t, _ in gradual]
    kinds = ["cut"] * int(cuts.size) + [kind for _, kind in gradual]
    if not times:
        return np.zeros(0, dtype=np.float64), []
    order = np.argsort(np.asarray(times, dtype=np.float64))
    return np.asarray(times, dtype=np.float64)[order], [kinds[i] for i in order]


def _nearest(
    boundaries: tuple[np.ndarray, list[str]], target: float, window: float, *, lo: float, hi: float
) -> tuple[float, str] | None:
    """Closest landing point to ``target`` within ``window`` and inside [lo, hi].

    Ties go to a hard cut: when a dissolve edge and a cut are equally close,
    the cut is the more certain boundary of the two.
    """
    times, kinds = boundaries
    if hi <= lo or times.size == 0:
        return None
    mask = (np.abs(times - target) <= window) & (times >= lo) & (times <= hi)
    if not mask.any():
        return None
    idx = np.flatnonzero(mask)
    distances = np.abs(times[idx] - target)
    best = distances.min()
    tied = idx[distances <= best + 1e-9]
    for i in tied:
        if kinds[int(i)] == "cut":
            return float(times[int(i)]), "cut"
    return float(times[int(tied[0])]), kinds[int(tied[0])]
