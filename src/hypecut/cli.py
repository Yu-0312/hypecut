"""Command-line interface.

    hypecut cut vod.mp4 -o reel.mp4 --profile configs/valorant.yaml
    hypecut analyze vod.mp4 --json plan.json
    hypecut signals
    hypecut serve --port 8000

Built on argparse so the core install stays at numpy + PyYAML.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .config import Config, load_config
from .ffmpeg import FFmpegError, FFmpegNotFound
from .reframe import MODES as REFRAME_MODES

__all__ = ["main", "build_parser", "VARIANT_PRESETS"]

#: Named extra renders for ``--also``. Each is a partial ``render`` override
#: applied on top of whatever the profile already says.
VARIANT_PRESETS: dict[str, dict[str, object]] = {
    "vertical": {"reframe": {"mode": "crop", "width": 1080, "height": 1920, "track": True}},
    "square": {"reframe": {"mode": "crop", "width": 1080, "height": 1080, "track": True}},
    "stack": {"reframe": {"mode": "stack", "width": 1080, "height": 1920}},
    "blurred": {"reframe": {"mode": "blur_pad", "width": 1080, "height": 1920}},
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypecut", description="Automatic highlight reels for gameplay, esports and sports."
    )
    parser.add_argument("--version", action="version", version=f"hypecut {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("source", help="input video file")
        p.add_argument("-p", "--profile", help="YAML/JSON profile path")
        p.add_argument("--max-clips", type=int, help="hard cap on clip count")
        p.add_argument(
            "--target",
            type=float,
            metavar="SECONDS",
            help="target total reel duration (0 disables the budget)",
        )
        p.add_argument("--min-duration", type=float, help="minimum clip length")
        p.add_argument("--max-duration", type=float, help="maximum clip length")
        p.add_argument(
            "--percentile",
            type=float,
            help="excitement percentile that counts as a highlight (default 92)",
        )
        p.add_argument(
            "--refiner",
            action="append",
            default=None,
            metavar="NAME",
            help="enable a refiner (repeatable): diversity, pacing, clip_rerank, ...",
        )
        p.add_argument(
            "--no-snap",
            action="store_true",
            help="do not move clip edges onto nearby shot boundaries",
        )
        p.add_argument(
            "--snap-window",
            type=float,
            metavar="SECONDS",
            help="how far a clip edge may travel to reach a cut (default 2)",
        )
        p.add_argument(
            "--reframe",
            choices=REFRAME_MODES,
            help="reframe for vertical: crop (motion-centred), stack (facecam+game), blur_pad",
        )
        p.add_argument(
            "--vertical", action="store_true", help="shorthand for --reframe crop at 1080x1920"
        )
        p.add_argument(
            "--reframe-track",
            action="store_true",
            help="let the vertical crop pan to follow the action instead of holding still",
        )
        p.add_argument(
            "--no-trim",
            action="store_true",
            help="do not move un-snapped clip edges into nearby pauses",
        )
        p.add_argument(
            "--facecam",
            metavar="X0,Y0,X1,Y1",
            help="facecam box in 0-1 coordinates, e.g. 0,0,0.26,0.3 (stack mode, --react)",
        )
        p.add_argument(
            "--react",
            action="store_true",
            help="in crop mode, pull the frame toward the facecam when it is busy",
        )
        p.add_argument(
            "--also",
            action="append",
            default=None,
            metavar="VARIANT",
            choices=sorted(VARIANT_PRESETS),
            help=(
                "render an extra aspect ratio from the same analysis (repeatable): "
                + ", ".join(sorted(VARIANT_PRESETS))
            ),
        )
        p.add_argument("-q", "--quiet", action="store_true")

    cut = sub.add_parser("cut", help="analyse and render a highlight reel")
    add_common(cut)
    cut.add_argument("-o", "--output", default=None, help="output video path")
    cut.add_argument("--width", type=int, help="output width")
    cut.add_argument("--height", type=int, help="output height")
    cut.add_argument("--crf", type=int, help="x264 quality (lower = better)")
    cut.add_argument("--no-sidecar", action="store_true", help="skip JSON/EDL output")

    batch = sub.add_parser("batch", help="cut every video in a folder")
    add_common(batch)
    batch.add_argument("-o", "--output-dir", default=None, help="where reels are written")
    batch.add_argument(
        "--pattern",
        action="append",
        default=None,
        metavar="GLOB",
        help="which files to pick up (repeatable, default: common video extensions)",
    )
    batch.add_argument("--recursive", action="store_true", help="descend into subfolders")
    batch.add_argument(
        "--overwrite", action="store_true", help="re-cut files whose reel already exists"
    )

    ana = sub.add_parser("analyze", help="propose clips without encoding")
    add_common(ana)
    ana.add_argument("--json", dest="json_out", help="write the plan to this path")

    sub.add_parser("signals", help="list available signals and refiners")

    serve = sub.add_parser("serve", help="run the web UI")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.add_argument("--data-dir", default=None, help="where uploads and reels are stored")

    return parser


def _config_from_args(args: argparse.Namespace) -> Config:
    cfg = load_config(args.profile) if getattr(args, "profile", None) else load_config()
    seg: dict[str, object] = {}
    if args.max_clips is not None:
        seg["max_clips"] = args.max_clips
    if args.target is not None:
        seg["target_duration"] = args.target if args.target > 0 else None
    if args.min_duration is not None:
        seg["min_duration"] = args.min_duration
    if args.max_duration is not None:
        seg["max_duration"] = args.max_duration
    if args.percentile is not None:
        seg["percentile"] = args.percentile
    if args.no_snap:
        seg["snap_to_shots"] = False
    if args.snap_window is not None:
        seg["snap_window"] = args.snap_window
    if args.no_trim:
        seg["trim_to_silence"] = False

    render: dict[str, object] = {}
    for key in ("width", "height", "crf"):
        value = getattr(args, key, None)
        if value is not None:
            render[key] = value

    reframe: dict[str, object] = {}
    if args.vertical and not args.reframe:
        reframe["mode"] = "crop"
    if args.reframe:
        reframe["mode"] = args.reframe
    if args.reframe_track:
        reframe["track"] = True
    if args.facecam:
        reframe["facecam"] = _parse_box(args.facecam)
    if args.react:
        reframe["react_to_facecam"] = True
        reframe.setdefault("mode", "crop")
    if reframe:
        # When reframing is on it owns the output geometry, so --width/--height
        # have to land there instead of on the (now unused) scale/pad path.
        for key in ("width", "height"):
            if render.pop(key, None) is not None:
                reframe[key] = getattr(args, key)
        render["reframe"] = reframe

    if args.also:
        overrides_variants = {name: VARIANT_PRESETS[name] for name in dict.fromkeys(args.also)}
    else:
        overrides_variants = {}

    overrides: dict[str, object] = {}
    if overrides_variants:
        overrides["variants"] = overrides_variants
    if seg:
        overrides["segments"] = seg
    if render:
        overrides["render"] = render
    if args.refiner:
        overrides["refiners"] = args.refiner
    return cfg.merged(overrides) if overrides else cfg


BATCH_PATTERNS = ("*.mp4", "*.mkv", "*.mov", "*.webm", "*.avi", "*.flv", "*.ts", "*.m4v")


def _collect(root: Path, patterns: list[str] | None, recursive: bool) -> list[Path]:
    """Every video under ``root`` matching ``patterns``, de-duplicated and sorted."""
    globber = root.rglob if recursive else root.glob
    found: set[Path] = set()
    for pattern in patterns or BATCH_PATTERNS:
        found.update(p for p in globber(pattern) if p.is_file())
    return sorted(found)


def _run_batch(args: argparse.Namespace, cfg: Config, report) -> int:
    """Cut every video in a folder, carrying on past individual failures.

    A batch that aborts on the first bad file is useless for the job people
    actually have — a directory of recordings, one of which is truncated. Each
    failure is reported and counted; the exit code reflects whether any failed.
    """
    from .pipeline import run as run_one

    root = Path(args.source)
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 2

    sources = _collect(root, args.pattern, args.recursive)
    if not sources:
        print(f"No videos found under {root}", file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else root / "highlights"
    out_dir.mkdir(parents=True, exist_ok=True)
    sources = [s for s in sources if out_dir not in s.parents]

    done: list[tuple[Path, Path]] = []
    skipped: list[Path] = []
    failed: list[tuple[Path, str]] = []

    for index, source in enumerate(sources, start=1):
        dest = out_dir / f"{source.stem}_highlights.mp4"
        if dest.exists() and not args.overwrite:
            skipped.append(source)
            continue
        if not args.quiet:
            print(f"\n[{index}/{len(sources)}] {source.name}", file=sys.stderr)
        try:
            result = run_one(source, dest, cfg, progress=report)
            done.append((source, result.output or dest))
        except KeyboardInterrupt:  # pragma: no cover - user abort
            print("\ninterrupted", file=sys.stderr)
            return 130
        except Exception as exc:
            # One bad file must not end the run. Flatten the message: ffmpeg
            # errors put the useful part on the *second* line, so keeping only
            # the first would report "ffprobe failed" and nothing about why.
            reason = " ".join(str(exc).split())
            failed.append((source, reason[:160]))

    print(f"\n{len(done)} cut, {len(skipped)} skipped, {len(failed)} failed -> {out_dir}")
    for source, dest in done:
        print(f"  ok      {source.name} -> {dest.name}")
    for source in skipped:
        print(f"  skip    {source.name} (already cut; --overwrite to redo)")
    for source, why in failed:
        print(f"  failed  {source.name}: {why}", file=sys.stderr)
    return 1 if failed else 0


def _parse_box(text: str) -> list[float]:
    """Parse ``x0,y0,x1,y1`` in normalised 0-1 coordinates."""
    try:
        values = [float(part) for part in text.replace(" ", "").split(",")]
    except ValueError as exc:
        raise ValueError(f"--facecam expects four numbers, got {text!r}") from exc
    if len(values) != 4 or not all(0.0 <= v <= 1.0 for v in values):
        raise ValueError(f"--facecam expects four 0-1 values as X0,Y0,X1,Y1, got {text!r}")
    return values


def _progress(quiet: bool):
    state = {"last": 0.0}

    def report(fraction: float, message: str) -> None:
        if quiet:
            return
        now = time.time()
        if fraction < 1.0 and now - state["last"] < 0.2:
            return
        state["last"] = now
        bar = int(fraction * 30)
        sys.stderr.write(
            f"\r[{'#' * bar}{'.' * (30 - bar)}] {fraction * 100:5.1f}%  {message[:40]:<40}"
        )
        sys.stderr.flush()
        if fraction >= 1.0:
            sys.stderr.write("\n")

    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "signals":
        from .refine import available_refiners
        from .refine import load_plugins as lrp
        from .signals import available_signals
        from .signals import load_plugins as lsp

        lsp()
        lrp()
        print("Signals (stage 1 — cheap, run over the whole video):")
        for name, desc in available_signals().items():
            print(f"  {name:<18} {desc}")
        print("\nRefiners (stage 2 — run only on candidates):")
        for name, desc in available_refiners().items():
            print(f"  {name:<18} {desc}")
        return 0

    if args.command == "serve":
        try:
            import uvicorn
        except ModuleNotFoundError:
            print("The web UI needs extra packages: pip install 'hypecut[web]'", file=sys.stderr)
            return 2
        import os

        if args.data_dir:
            os.environ["HYPECUT_DATA_DIR"] = args.data_dir
        uvicorn.run("hypecut.web.app:app", host=args.host, port=args.port, reload=args.reload)
        return 0

    from .pipeline import analyze, render_plan, render_variants

    try:
        cfg = _config_from_args(args)
        report = _progress(args.quiet)

        if args.command == "batch":
            return _run_batch(args, cfg, report)

        if args.command == "analyze":
            plan = analyze(args.source, cfg, progress=report)
            report(1.0, "done")
            payload = json.dumps(plan.to_dict(), indent=2, ensure_ascii=False)
            if args.json_out:
                Path(args.json_out).write_text(payload, encoding="utf-8")
                if not args.quiet:
                    print(f"Wrote {args.json_out}")
            else:
                print(payload)
            _summarise(plan, args.quiet)
            return 0

        source = Path(args.source)
        output = (
            Path(args.output) if args.output else source.with_name(f"{source.stem}_highlights.mp4")
        )
        plan = analyze(source, cfg, progress=lambda p, m: report(p * 0.6, m))
        if not plan.segments:
            print(
                "No highlights found. Try --percentile 85 or a different profile.", file=sys.stderr
            )
            return 1
        if cfg.variants:
            outputs = render_variants(
                plan, output, cfg, progress=lambda p, m: report(0.6 + p * 0.4, m)
            )
            out = outputs["base"]
            sidecar = out.with_suffix(".hypecut.json")
            sidecar = sidecar if sidecar.exists() else None
        else:
            outputs = {}
            out, sidecar = render_plan(
                plan,
                output,
                cfg,
                progress=lambda p, m: report(0.6 + p * 0.4, m),
                write_sidecar=not args.no_sidecar,
            )
        _summarise(plan, args.quiet)
        if not args.quiet:
            print(f"\nReel:    {out}")
            for name, path in outputs.items():
                if name != "base":
                    print(f"  +{name:<9} {path}")
            if sidecar:
                print(f"Cutlist: {sidecar}")
        return 0

    except FFmpegNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (FFmpegError, ValueError, RuntimeError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:  # pragma: no cover
        print("\ninterrupted", file=sys.stderr)
        return 130


def _summarise(plan, quiet: bool) -> None:
    if quiet:
        return
    print(
        f"\n{len(plan.segments)} clips, {plan.total_duration:.1f}s reel "
        f"from {plan.info.duration:.1f}s source "
        f"({plan.total_duration / max(plan.info.duration, 1e-9) * 100:.1f}% kept)"
    )
    snapped = sum(1 for s in plan.segments if s.meta.get("snapped"))
    trimmed = sum(1 for s in plan.segments if s.meta.get("trimmed"))
    total = len(plan.segments)
    if snapped:
        print(f"{snapped}/{total} clips moved onto a shot boundary")
    if trimmed:
        print(f"{trimmed}/{total} clips had an edge moved into a pause")
    for seg in plan.segments:
        top = max(seg.reasons, key=seg.reasons.get) if seg.reasons else "-"
        notes = []
        if seg.meta.get("snapped"):
            notes.append(
                "snap " + " ".join(f"{k}{v:+.2f}s" for k, v in seg.meta["snapped"].items())
            )
        if seg.meta.get("trimmed"):
            notes.append(
                "pause " + " ".join(f"{k}{v:+.2f}s" for k, v in seg.meta["trimmed"].items())
            )
        if seg.meta.get("reframe"):
            plan_ = seg.meta["reframe"]
            notes.append(str(plan_.get("mode")) + ("/pan" if "keyframes" in plan_ else ""))
        suffix = f"  [{'; '.join(notes)}]" if notes else ""
        print(
            f"  {_hhmmss(seg.start)}–{_hhmmss(seg.end)}  score {seg.score:.3f}  "
            f"top signal: {top}{suffix}"
        )


def _hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
