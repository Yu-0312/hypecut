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
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import ffmpeg as ff
from .config import Config, load_config
from .fusion import fuse
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

__all__ = ["analyze", "render_plan", "render_variants", "run", "PipelineResult", "Progress"]


def _plan_key(variant: str | None) -> str:
    """Where a variant's framing decisions live on each clip's metadata."""
    return "reframe" if not variant else f"reframe:{variant}"


Progress = Callable[[float, str], None]


@dataclass
class PipelineResult:
    """What a full run produces."""

    plan: HighlightPlan
    output: Path | None
    sidecar: Path | None
    elapsed: float
    #: Extra aspect-ratio renders of the same plan, keyed by variant name.
    variants: dict[str, Path] = field(default_factory=dict)


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
        segments = plan_reframe(ctx, segments, reframe, key=_plan_key(variant))

    return HighlightPlan(info=info, segments=segments, curve=curve, times=ctx.times, tracks=tracks)


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

    if cfg.variants:
        outputs = render_variants(plan, output, cfg, progress=stage(0.6, 1.0))
        base = outputs["base"]
        sidecar = base.with_suffix(".hypecut.json")
        return PipelineResult(
            plan=plan,
            output=base,
            sidecar=sidecar if sidecar.exists() else None,
            elapsed=time.time() - started,
            variants={name: path for name, path in outputs.items() if name != "base"},
        )

    out, sidecar = render_plan(plan, output, cfg, progress=stage(0.6, 1.0))
    return PipelineResult(plan=plan, output=out, sidecar=sidecar, elapsed=time.time() - started)


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
