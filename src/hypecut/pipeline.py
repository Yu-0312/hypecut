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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ffmpeg as ff
from .config import Config, load_config
from .fusion import fuse
from .refine import build_refiners
from .refine import load_plugins as load_refiner_plugins
from .render import render_reel, write_edl
from .segments import build_candidates, merge, select
from .signals import build_signals
from .signals import load_plugins as load_signal_plugins
from .types import AnalysisContext, HighlightPlan, SignalTrack, VideoInfo

__all__ = ["analyze", "render_plan", "run", "PipelineResult", "Progress"]

Progress = Callable[[float, str], None]


@dataclass
class PipelineResult:
    """What a full run produces."""

    plan: HighlightPlan
    output: Path | None
    sidecar: Path | None
    elapsed: float


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

    return HighlightPlan(info=info, segments=segments, curve=curve, times=ctx.times, tracks=tracks)


def render_plan(
    plan: HighlightPlan,
    output: str | Path,
    config: Config | None = None,
    *,
    progress: Progress | None = None,
    write_sidecar: bool = True,
) -> tuple[Path, Path | None]:
    """Encode a plan's segments to ``output``; optionally write the JSON sidecar."""
    progress = progress or _noop
    cfg = config or load_config()
    out = render_reel(plan.info, plan.segments, output, cfg.render, progress=progress)

    sidecar: Path | None = None
    if write_sidecar:
        sidecar = out.with_suffix(".hypecut.json")
        payload = plan.to_dict()
        payload["config"] = cfg.to_dict()
        sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        write_edl(plan.segments, out.with_suffix(".edl"), fps=plan.info.fps or 30.0)
    return out, sidecar


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
