"""End-to-end orchestration: video in, highlight reel out.

The whole flow in one place, so a reader can follow it top to bottom:

    decode once  ->  run signals  ->  fuse  ->  propose  ->  refine
                 ->  merge  ->  select  ->  render

Analysis (:func:`analyze`) and rendering (:func:`render_plan`) are split so
the web UI can show the proposed cut list, let the user drop a clip, and
only then spend the encode.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import ffmpeg as ff
from .config import Config, load_config
from .fusion import fuse, prominence
from .refine import build_refiners
from .refine import load_plugins as load_refiner_plugins
from .reframe import plan_reframe
from .render import render_reel, write_edl
from .segments import build_candidates, merge, select
from .signals import build_signals
from .signals import load_plugins as load_signal_plugins
from .snapping import snap_segments
from .trimming import trim_segments
from .types import AnalysisContext, HighlightPlan, SignalTrack, VideoInfo

__all__ = [
    "analyze",
    "render_plan",
    "render_variants",
    "render_reels",
    "reel_path",
    "run",
    "PipelineResult",
    "ReelOutput",
    "Progress",
]


def _plan_key(variant: str | None) -> str:
    """Where a variant's framing decisions live on each clip's metadata."""
    return "reframe" if not variant else f"reframe:{variant}"


Progress = Callable[[float, str], None]


@dataclass
class ReelOutput:
    """One rendered reel: the file, its cut list, and its other framings."""

    path: Path
    sidecar: Path | None = None
    #: Extra aspect-ratio renders of this same reel, keyed by variant name.
    variants: dict[str, Path] = field(default_factory=dict)
    clips: int = 0


@dataclass
class PipelineResult:
    """What a full run produces."""

    plan: HighlightPlan
    output: Path | None
    sidecar: Path | None
    elapsed: float
    #: Extra aspect-ratio renders of the same plan, keyed by variant name.
    variants: dict[str, Path] = field(default_factory=dict)
    #: Every reel this run produced. A long video with many highlights yields
    #: more than one; ``output`` is the first of them. Empty when nothing was
    #: found, in which case ``plan.empty_reason`` says why.
    reels: list[ReelOutput] = field(default_factory=list)


def _noop(_p: float, _m: str) -> None:
    return None


def analyze(
    source: str | Path, config: Config | None = None, *, progress: Progress | None = None
) -> HighlightPlan:
    """Decode, score and propose highlight segments. No encoding happens here."""
    progress = progress or _noop
    cfg = config or load_config()
    load_signal_plugins()
    load_refiner_plugins()

    progress(0.02, "probing")
    info = ff.probe(source)
    ctx = _build_context(info, cfg)

    progress(0.35, "running signals")
    tracks = _run_signals(ctx, cfg)
    if not tracks:
        raise RuntimeError(
            "No usable signals for this input — it has neither decodable video "
            "frames nor an audio track."
        )

    progress(0.55, "fusing")
    curve = fuse(tracks, grid_fps=cfg.signals.grid_fps, smooth_seconds=cfg.signals.smooth_seconds)

    # Before proposing anything, ask whether this video contains anything at
    # all. Every threshold below is relative to this video's own distribution,
    # so a percentile will always find something — including in three hours of
    # an idle lobby. This is the one check that can answer "no".
    strength = prominence(
        tracks, grid_fps=cfg.signals.grid_fps, smooth_seconds=cfg.signals.smooth_seconds
    )
    empty = HighlightPlan(
        info=info,
        segments=[],
        curve=curve,
        times=ctx.times,
        tracks=tracks,
        prominence=strength,
        min_prominence=cfg.segments.min_prominence,
    )
    if cfg.segments.min_prominence > 0 and strength < cfg.segments.min_prominence:
        progress(1.0, empty.empty_reason)
        return empty

    progress(0.62, "proposing clips")
    candidates = build_candidates(
        curve,
        ctx.times,
        cfg.segments,
        grid_fps=cfg.signals.grid_fps,
        duration=info.duration,
        tracks=tracks,
    )

    progress(0.72, "refining")
    for refiner in build_refiners(cfg.refiners, cfg.refiner_params):
        ok, reason = refiner.available()
        if not ok:
            progress(0.72, f"skipping {refiner.name}: {reason}")
            continue
        # Hand over the decoded frames. Most refiners ignore this; the ones
        # that compare candidates visually would otherwise have to shell out
        # to ffmpeg and decode the video a second time.
        refiner.ctx = ctx
        candidates = refiner.refine(info, candidates)

    progress(0.85, "selecting")
    segments = select(merge(candidates, cfg.segments), cfg.segments)

    # Both of these need the decoded frames, and both are decisions about the
    # cut rather than about the encode — so they belong here, on the analysis
    # side, and travel to the renderer inside each clip's metadata.
    if cfg.segments.snap_to_shots:
        progress(0.88, "snapping to shot boundaries")
        segments = snap_segments(ctx, segments, cfg.segments)
    if cfg.segments.trim_to_silence:
        # Strictly after snapping: this only touches edges no cut claimed.
        progress(0.92, "trimming to pauses")
        segments = trim_segments(ctx, segments, cfg.segments)
    # Every framing this run will produce is planned here, while the frames
    # are still in memory. That is the whole point of variants: one decode,
    # one set of cut decisions, several aspect ratios out.
    for variant in [None, *sorted(cfg.variants)]:
        reframe = cfg.render_for(variant).reframe
        if reframe.mode == "off":
            continue
        progress(0.95, f"planning reframe ({variant or 'base'})")
        segments = plan_reframe(ctx, segments, reframe, key=_plan_key(variant), progress=progress)

    return HighlightPlan(
        info=info,
        segments=segments,
        curve=curve,
        times=ctx.times,
        tracks=tracks,
        prominence=strength,
        min_prominence=cfg.segments.min_prominence,
    )


def render_plan(
    plan: HighlightPlan,
    output: str | Path,
    config: Config | None = None,
    *,
    progress: Progress | None = None,
    write_sidecar: bool = True,
    variant: str | None = None,
) -> tuple[Path, Path | None]:
    """Encode a plan's segments to ``output``; optionally write the JSON sidecar."""
    progress = progress or _noop
    cfg = config or load_config()
    out = render_reel(
        plan.info,
        plan.segments,
        output,
        cfg.render_for(variant),
        progress=progress,
        plan_key=_plan_key(variant),
    )

    sidecar: Path | None = None
    if write_sidecar:
        sidecar = out.with_suffix(".hypecut.json")
        payload = plan.to_dict()
        payload["config"] = cfg.to_dict()
        sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        write_edl(plan.segments, out.with_suffix(".edl"), fps=plan.info.fps or 30.0)
    return out, sidecar


def render_variants(
    plan: HighlightPlan, output: str | Path, config: Config, *, progress: Progress | None = None
) -> dict[str, Path]:
    """Render the base output plus every configured variant from one plan.

    Variant files sit next to the base one with the variant name appended,
    so ``reel.mp4`` gains ``reel_vertical.mp4``. Encoding is the only work
    repeated — the decode, the scoring and the cut decisions are shared.
    """
    progress = progress or _noop
    output = Path(output)
    names: list[str | None] = [None, *sorted(config.variants)]
    outputs: dict[str, Path] = {}

    for index, variant in enumerate(names):
        dest = (
            output
            if variant is None
            else output.with_name(f"{output.stem}_{variant}{output.suffix}")
        )
        lo, hi = index / len(names), (index + 1) / len(names)
        out, _ = render_plan(
            plan,
            dest,
            config,
            progress=lambda p, m, lo=lo, hi=hi: progress(lo + (hi - lo) * p, m),
            write_sidecar=variant is None,
            variant=variant,
        )
        outputs[variant or "base"] = out
    return outputs


def reel_path(output: Path, index: int, total: int) -> Path:
    """Where reel ``index`` of ``total`` goes. One reel keeps the plain name."""
    if total <= 1:
        return output
    return output.with_name(f"{output.stem}.part{index}{output.suffix}")


def render_reels(
    plan: HighlightPlan,
    output: str | Path,
    config: Config | None = None,
    *,
    progress: Progress | None = None,
    write_sidecar: bool = True,
) -> list[ReelOutput]:
    """Render every reel in a plan, with its variants and its own cut list.

    A plan holds one flat, chronological list of clips carrying reel numbers;
    this is where that becomes files. One reel keeps the requested filename,
    several become ``reel.part1.mp4``, ``reel.part2.mp4`` and so on. Each part
    gets its own sidecar and EDL describing only that part, because a cut list
    that does not match its video is worse than no cut list at all.
    """
    progress = progress or _noop
    cfg = config or load_config()
    output = Path(output)
    groups = plan.reels()
    outputs: list[ReelOutput] = []

    for index, segments in enumerate(groups, start=1):
        dest = reel_path(output, index, len(groups))
        part = replace(plan, segments=segments)
        lo, hi = (index - 1) / len(groups), index / len(groups)

        def scaled(p: float, m: str, lo: float = lo, hi: float = hi) -> None:
            progress(lo + (hi - lo) * max(0.0, min(1.0, p)), m)

        if cfg.variants:
            rendered = render_variants(part, dest, cfg, progress=scaled)
            path = rendered["base"]
            sidecar = path.with_suffix(".hypecut.json")
            outputs.append(
                ReelOutput(
                    path=path,
                    sidecar=sidecar if sidecar.exists() else None,
                    variants={k: v for k, v in rendered.items() if k != "base"},
                    clips=len(segments),
                )
            )
        else:
            path, sidecar = render_plan(
                part, dest, cfg, progress=scaled, write_sidecar=write_sidecar
            )
            outputs.append(ReelOutput(path=path, sidecar=sidecar, clips=len(segments)))

    return outputs


def run(
    source: str | Path,
    output: str | Path,
    config: Config | None = None,
    *,
    progress: Progress | None = None,
) -> PipelineResult:
    """Analyse then render, in one call."""
    started = time.time()
    cfg = config or load_config()

    def stage(lo: float, hi: float) -> Progress:
        def inner(p: float, msg: str) -> None:
            if progress:
                progress(lo + (hi - lo) * max(0.0, min(1.0, p)), msg)

        return inner

    plan = analyze(source, cfg, progress=stage(0.0, 0.6))

    # Nothing found is a result, not a failure: the caller decides whether to
    # complain (one file the user asked about) or move on (a batch).
    if not plan.segments:
        return PipelineResult(plan=plan, output=None, sidecar=None, elapsed=time.time() - started)

    reels = render_reels(plan, output, cfg, progress=stage(0.6, 1.0))
    first = reels[0]
    return PipelineResult(
        plan=plan,
        output=first.path,
        sidecar=first.sidecar,
        elapsed=time.time() - started,
        variants=dict(first.variants),
        reels=reels,
    )


def _build_context(info: VideoInfo, cfg: Config) -> AnalysisContext:
    s = cfg.signals
    n = max(1, int(round(info.duration * s.grid_fps)))
    times = np.arange(n, dtype=np.float64) / s.grid_fps

    gray = None
    try:
        gray = ff.decode_gray_frames(
            info.path, fps=s.grid_fps, width=s.frame_width, height=s.frame_height
        )
        if gray.shape[0]:
            n = min(n, gray.shape[0]) if gray.shape[0] < n else n
    except ff.FFmpegError:
        gray = None

    audio = None
    if info.has_audio:
        try:
            audio = ff.decode_audio(info.path, sr=s.audio_sr)
        except ff.FFmpegError:
            audio = None

    times = times[:n]
    if gray is not None and gray.shape[0] > n:
        gray = gray[:n]

    return AnalysisContext(
        info=info, grid_fps=s.grid_fps, times=times, gray=gray, audio=audio, audio_sr=s.audio_sr
    )


def _run_signals(ctx: AnalysisContext, cfg: Config) -> list[SignalTrack]:
    tracks: list[SignalTrack] = []
    for sig in build_signals(cfg.signals.enabled, cfg.signals.params):
        if not sig.applicable(ctx):
            continue
        weight = float(cfg.signals.weights.get(sig.name, 1.0))
        if weight == 0:
            continue
        tracks.append(sig.track(ctx, weight=weight))
    return tracks
