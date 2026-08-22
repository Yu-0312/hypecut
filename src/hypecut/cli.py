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

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypecut", description="Automatic highlight reels for gameplay and esports VODs."
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
        p.add_argument("-q", "--quiet", action="store_true")

    cut = sub.add_parser("cut", help="analyse and render a highlight reel")
    add_common(cut)
    cut.add_argument("-o", "--output", default=None, help="output video path")
    cut.add_argument("--width", type=int, help="output width")
    cut.add_argument("--height", type=int, help="output height")
    cut.add_argument("--crf", type=int, help="x264 quality (lower = better)")
    cut.add_argument("--no-sidecar", action="store_true", help="skip JSON/EDL output")

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

    render: dict[str, object] = {}
    for key in ("width", "height", "crf"):
        value = getattr(args, key, None)
        if value is not None:
            render[key] = value

    overrides: dict[str, object] = {}
    if seg:
        overrides["segments"] = seg
    if render:
        overrides["render"] = render
    if args.refiner:
        overrides["refiners"] = args.refiner
    return cfg.merged(overrides) if overrides else cfg


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

    from .pipeline import analyze, render_plan

    try:
        cfg = _config_from_args(args)
        report = _progress(args.quiet)

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
    for seg in plan.segments:
        top = max(seg.reasons, key=seg.reasons.get) if seg.reasons else "-"
        print(
            f"  {_hhmmss(seg.start)}–{_hhmmss(seg.end)}  score {seg.score:.3f}  top signal: {top}"
        )


def _hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
