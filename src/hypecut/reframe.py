"""Turn a landscape capture into a vertical (or square) frame.

A 16:9 gameplay clip pasted into a 9:16 slot loses two thirds of its height
to black. The three ways out, all implemented here:

``crop``
    Take a 9:16 slice of the source and throw the sides away. Highest impact
    per pixel — the action fills the screen — and the reason this module
    needs analysis at all: *which* slice depends on where the action is.

``stack``
    Facecam on top, gameplay below. The layout most gaming Shorts use,
    because it keeps the reaction and the play in one frame.

``blur_pad``
    Whole frame, scaled to width, over a blurred blow-up of itself. Loses
    nothing, wastes half the screen. The safe default for footage where the
    important thing might be anywhere — scoreboards, minimaps, UI.

The crop centre is decided during analysis (where the decoded frames live)
and stored on the clip, so rendering stays a pure function of the plan and
the sidecar JSON records exactly how each clip was framed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .config import DEFAULT_FACECAM_BOX, ReframeConfig
from .facecam import locate_facecam
from .types import AnalysisContext, Candidate, VideoInfo

__all__ = ["action_track", "plan_reframe", "geometry_filters", "MODES"]

MODES = ("off", "crop", "stack", "blur_pad")

Progress = Callable[[float, str], None]

#: Where the auto-detected box is cached on the analysis context, so several
#: variants share one detection pass instead of each paying for its own.
_FACECAM_CACHE_KEY = "facecam_auto"


def action_track(ctx: AnalysisContext, seg: Candidate, cfg: ReframeConfig) -> list[float]:
    """Horizontal centre of the action over a clip, as fractions of width.

    Motion energy per column, summed over rows, gives a mass distribution
    across the frame; its centroid is where things are happening. It is a
    crude estimator and that is fine — the crop is 56% of the frame wide, so
    it only has to be right to within a few percent to keep the action in
    shot, and being crude keeps it free.
    """
    if ctx.gray is None or ctx.gray.shape[0] < 2:
        return [0.5]

    i0 = int(max(0, np.searchsorted(ctx.times, seg.start)))
    i1 = int(min(ctx.times.size, np.searchsorted(ctx.times, seg.end) + 1))
    frames = ctx.gray[i0:i1]
    if frames.shape[0] < 2:
        return [0.5]

    diff = np.abs(np.diff(frames.astype(np.float32), axis=0))  # (T-1, H, W)
    energy = diff.sum(axis=1)  # (T-1, W)
    total = energy.sum(axis=1)
    columns = np.arange(energy.shape[1], dtype=np.float64)
    centre = np.where(
        total > 1e-6,
        (energy * columns).sum(axis=1) / np.maximum(total, 1e-6),
        energy.shape[1] / 2.0,
    ) / max(1, energy.shape[1] - 1)

    if cfg.react_to_facecam:
        centre = _bias_toward_facecam(centre, diff, cfg)

    centre = _moving_average(centre, int(round(cfg.smooth_seconds * ctx.grid_fps)))
    return _limit_pan(centre, max_step=cfg.max_pan / max(ctx.grid_fps, 1e-6)).tolist()


def plan_reframe(
    ctx: AnalysisContext,
    segments: list[Candidate],
    cfg: ReframeConfig,
    key: str = "reframe",
    *,
    progress: Progress | None = None,
) -> list[Candidate]:
    """Annotate each clip with how it should be reframed.

    ``key`` lets several framings of the same clip coexist, which is what
    makes one analysis serve a landscape reel and a vertical cutdown at once.
    """
    if cfg.mode == "off":
        return segments
    if cfg.mode not in MODES:
        raise ValueError(f"Unknown reframe mode {cfg.mode!r}. Expected one of {MODES}.")

    # `facecam: auto` is resolved once per analysis and cached on the context,
    # so the base render and every variant share one detection pass. The box
    # is stamped into each clip's reframe plan: that is what makes a render
    # from the sidecar reproduce the same crop without re-locating anything.
    resolved = cfg
    if cfg.facecam == "auto":
        found = _locate_cached(ctx, progress)
        box = found["box"] if found else list(DEFAULT_FACECAM_BOX)
        resolved = replace(cfg, facecam=list(box))

    for seg in segments:
        plan: dict[str, object] = {"mode": cfg.mode}
        if cfg.facecam == "auto":
            plan["facecam"] = list(resolved.facecam)
        if cfg.mode == "crop":
            track = action_track(ctx, seg, resolved)
            if cfg.track and len(track) > 1:
                plan["keyframes"] = _keyframes(track, seg.duration, cfg.keyframes)
            else:
                # A still crop, placed at the median of where the action was.
                # Panning reads as camera drift on gameplay footage; unless the
                # user asks for it, holding the frame is the better default.
                plan["x"] = round(float(np.median(track)), 4)
        seg.meta[key] = plan
    return segments


def _locate_cached(ctx: AnalysisContext, progress: Progress | None) -> dict[str, object] | None:
    """Detect the facecam once, reporting what was found the first time."""
    if _FACECAM_CACHE_KEY not in ctx.extras:
        found = locate_facecam(ctx)
        ctx.extras[_FACECAM_CACHE_KEY] = found
        if found:
            box = ", ".join(f"{v:.2f}" for v in found["box"])  # type: ignore[attr-defined]
            if progress:
                progress(0.0, f"facecam located at [{box}] (confidence {found['confidence']})")
        elif progress:
            progress(0.0, "no facecam found with usable confidence — using the default box")
    return ctx.extras[_FACECAM_CACHE_KEY]


def geometry_filters(
    info: VideoInfo, seg: Candidate, cfg: ReframeConfig, key: str = "reframe"
) -> list[str]:
    """ffmpeg filter chain that reframes one clip. Empty when disabled.

    Filters must contain no whitespace: the whole chain is passed as a single
    ``-vf`` argument, and expressions carrying commas are single-quoted so
    ffmpeg does not read them as filter separators.
    """
    if cfg.mode == "off":
        return []

    plan = seg.meta.get(key) or {"mode": cfg.mode}
    mode = str(plan.get("mode", cfg.mode))
    out_w, out_h = _even(cfg.width), _even(cfg.height)

    # The box stamped during analysis wins; the config is the fallback for a
    # plan that never went through planning. "auto" cannot be resolved here —
    # there are no decoded frames at render time — so it falls back to the
    # default rather than guessing.
    box = plan.get("facecam") or cfg.facecam
    if not isinstance(box, list):
        box = list(DEFAULT_FACECAM_BOX)

    if mode == "blur_pad":
        return [
            f"split[bgsrc][fgsrc];"
            f"[bgsrc]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
            f"crop={out_w}:{out_h},gblur=sigma={cfg.blur_sigma:g}[bg];"
            f"[fgsrc]scale={out_w}:-2[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,setsar=1"
        ]

    if mode == "stack":
        top_h = _even(out_h * min(max(cfg.facecam_share, 0.05), 0.9))
        bottom_h = _even(out_h - top_h)
        return [
            f"split[fcsrc][gpsrc];"
            f"[fcsrc]{_box_crop(box)},{_fit(out_w, top_h)}[top];"
            f"[gpsrc]{_box_crop(cfg.gameplay)},{_fit(out_w, bottom_h)}[bot];"
            f"[top][bot]vstack=inputs=2,setsar=1"
        ]

    # mode == "crop"
    crop_w, crop_h = _crop_box(info, out_w / out_h)
    span = max(0, info.width - crop_w)
    if "keyframes" in plan and span > 0:
        expr = _pan_expression(plan["keyframes"], span, crop_w, info.width)  # type: ignore[arg-type]
        x = f"'{expr}'"
    else:
        centre = float(plan.get("x", 0.5))  # type: ignore[arg-type]
        x = str(int(round(min(max(centre * info.width - crop_w / 2, 0), span))))
    return [
        f"crop=w={crop_w}:h={crop_h}:x={x}:y=(ih-{crop_h})/2",
        f"scale={out_w}:{out_h}",
        "setsar=1",
    ]


# --------------------------------------------------------------------- private


def _bias_toward_facecam(centre: np.ndarray, diff: np.ndarray, cfg: ReframeConfig) -> np.ndarray:
    """Pull the crop toward the facecam during the moments it comes alive.

    The reaction *is* the highlight as often as the play is, and a vertical
    crop that never shows the streamer's face throws half of it away. But
    holding on the facecam for a whole clip is worse than never cutting to it,
    so the pull is gated on the facecam actually being busy — measured against
    that clip's own median activity there, since a webcam that is always
    slightly noisy would otherwise read as a permanent reaction.

    No detector is involved: the box comes from the profile. That keeps this
    dependency-free and, more importantly, correct for whatever layout the
    streamer actually uses, which no general face model can assume.
    """
    h, w = diff.shape[1], diff.shape[2]
    x0, y0, x1, y1 = (float(v) for v in cfg.facecam)
    cx0, cx1 = sorted((int(np.clip(x0, 0, 1) * w), int(np.clip(x1, 0, 1) * w)))
    cy0, cy1 = sorted((int(np.clip(y0, 0, 1) * h), int(np.clip(y1, 0, 1) * h)))
    if cx1 - cx0 < 1 or cy1 - cy0 < 1:
        return centre

    activity = diff[:, cy0:cy1, cx0:cx1].mean(axis=(1, 2))

    # The resting level, not the median. A reaction that fills most of the clip
    # would drag the median up into itself and then measure as "normal" — the
    # same mistake as normalising a signal against a window containing the
    # thing you are trying to detect. A low percentile estimates the webcam's
    # idle noise instead, which is what "busy" should be compared against.
    baseline = float(np.percentile(activity, 25))
    if baseline <= 1e-6:
        baseline = float(np.mean(activity)) * 0.5
    if baseline <= 1e-6:
        return centre

    hot = activity > baseline * cfg.react_threshold
    if not hot.any():
        return centre

    face_x = ((cx0 + cx1) / 2.0) / max(1, w - 1)
    weight = float(np.clip(cfg.react_weight, 0.0, 1.0))
    out = centre.copy()
    out[hot] = (1.0 - weight) * out[hot] + weight * face_x
    return out


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window <= 1 or values.size == 0:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")[: values.size]


def _limit_pan(values: np.ndarray, max_step: float) -> np.ndarray:
    """Cap how far the crop centre may move between grid steps.

    Smoothing alone still lets the centre teleport when the action jumps
    across the screen; a velocity cap turns that into a slow push, which is
    what a camera operator would do and what a viewer can follow.
    """
    if values.size == 0 or max_step <= 0:
        return values
    out = values.copy()
    for i in range(1, out.size):
        delta = float(np.clip(out[i] - out[i - 1], -max_step, max_step))
        out[i] = out[i - 1] + delta
    return out


def _keyframes(track: list[float], duration: float, count: int) -> list[list[float]]:
    """Sample the centre track down to a handful of (time, x) pairs."""
    count = max(2, int(count))
    arr = np.asarray(track, dtype=np.float64)
    if arr.size <= count:
        times = np.linspace(0.0, duration, arr.size)
        return [[round(float(t), 3), round(float(x), 4)] for t, x in zip(times, arr, strict=True)]
    idx = np.linspace(0, arr.size - 1, count).round().astype(int)
    times = np.linspace(0.0, duration, count)
    return [[round(float(t), 3), round(float(arr[i]), 4)] for t, i in zip(times, idx, strict=True)]


def _pan_expression(keyframes: list[list[float]], span: int, crop_w: int, src_w: int) -> str:
    """Piecewise-linear ffmpeg expression for the crop's x over time."""
    points = [(float(t), float(min(max(x * src_w - crop_w / 2, 0), span))) for t, x in keyframes]
    if len(points) == 1:
        return f"{points[0][1]:.1f}"

    expr = f"{points[-1][1]:.1f}"
    for (t0, x0), (t1, x1) in reversed(list(zip(points, points[1:], strict=False))):
        dt = max(t1 - t0, 1e-3)
        lerp = f"{x0:.1f}+({x1:.1f}-{x0:.1f})*(t-{t0:.3f})/{dt:.3f}"
        expr = f"if(lt(t,{t1:.3f}),{lerp},{expr})"
    return expr


def _crop_box(info: VideoInfo, aspect: float) -> tuple[int, int]:
    """Largest ``aspect``-shaped rectangle that fits inside the source."""
    w, h = max(info.width, 2), max(info.height, 2)
    crop_w = _even(min(w, h * aspect))
    crop_h = _even(min(h, crop_w / aspect))
    return crop_w, crop_h


def _box_crop(box: list[float]) -> str:
    x0, y0, x1, y1 = (float(v) for v in box)
    x0, x1 = sorted((min(max(x0, 0.0), 1.0), min(max(x1, 0.0), 1.0)))
    y0, y1 = sorted((min(max(y0, 0.0), 1.0), min(max(y1, 0.0), 1.0)))
    bw = max(x1 - x0, 0.05)
    bh = max(y1 - y0, 0.05)
    return f"crop=w=iw*{bw:g}:h=ih*{bh:g}:x=iw*{x0:g}:y=ih*{y0:g}"


def _fit(width: int, height: int) -> str:
    return f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height}"


def _even(value: float) -> int:
    """Round to an even integer — h.264 chroma subsampling requires it."""
    return max(2, int(round(value / 2)) * 2)
