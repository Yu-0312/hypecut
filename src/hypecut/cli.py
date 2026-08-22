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

    sig = sub.add_parser("signals", help="list available signals and refiners")
    sig.add_argument("--json", action="store_true", help="machine-readable catalogue")

    prof = sub.add_parser("profiles", help="list the shipped profiles and what each is for")
    prof.add_argument("--json", action="store_true", help="machine-readable listing")
    prof.add_argument("--dir", default="configs", help="where to look for profiles")

    sheet = sub.add_parser(
        "contact-sheet", help="one labelled grid of frames — what an agent looks at"
    )
    sheet.add_argument("source", help="input video file")
    sheet.add_argument("-o", "--output", default="contact-sheet.png")
    sheet.add_argument("--plan", help="cut list; without one, sample the whole video evenly")
    sheet.add_argument("--count", type=int, default=12, help="tiles when sampling evenly")
    sheet.add_argument("--columns", type=int, default=4)
    sheet.add_argument("--json", action="store_true", help="print the tile index as JSON")

    rp = sub.add_parser("render", help="render an edited cut list")
    rp.add_argument("plan", help="a .hypecut.json cut list, edited or not")
    rp.add_argument("-o", "--output", default=None, help="output video path")
    rp.add_argument("--source", default=None, help="override the video the plan points at")
    rp.add_argument("--also", action="append", default=None, choices=sorted(VARIANT_PRESETS))
    rp.add_argument("-q", "--quiet", action="store_true")

    lab = sub.add_parser("label", help="draft an answer key from a generous first pass")
    lab.add_argument("source", help="input video file")
    lab.add_argument(
        "-o", "--output", default=None, help="labels file (default: <video>.labels.yaml)"
    )
    lab.add_argument("-p", "--profile", help="profile to propose with")
    lab.add_argument(
        "--percentile",
        type=float,
        default=80.0,
        help="deliberately low: over-propose, let the human throw things out (default 80)",
    )
    lab.add_argument("--max-clips", type=int, default=40)
    lab.add_argument("--annotator", default="", help="who marked it — scores are per-annotator")
    lab.add_argument("--no-sheet", action="store_true", help="skip the contact sheet")
    lab.add_argument("-q", "--quiet", action="store_true")

    ev = sub.add_parser("eval", help="score a profile against one or more answer keys")
    ev.add_argument("labels", nargs="+", help="labels files written by `hypecut label`")
    ev.add_argument("-p", "--profile", action="append", default=None, help="profile(s) to score")
    ev.add_argument("--json", action="store_true", help="machine-readable results")
    ev.add_argument("-q", "--quiet", action="store_true")

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


def _describe_profiles(root: Path) -> list[dict[str, str]]:
    """Name + one-line purpose for every profile in a directory.

    The summary is the profile's own first comment line. Keeping it there
    rather than in a table somewhere means it cannot drift from the file it
    describes, and a contributor adding a profile writes the description
    without being told to.
    """
    out: list[dict[str, str]] = []
    for path in sorted(root.glob("*.yaml")):
        summary = ""
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                summary = stripped.lstrip("# ").strip()
                break
            if stripped:
                break
        out.append({"name": path.stem, "path": str(path), "summary": summary})
    return out


def _run_contact_sheet(args: argparse.Namespace) -> int:
    from .contact import contact_sheet
    from .ffmpeg import probe

    segments = None
    if args.plan:
        from .plan import load_plan

        plan, _ = load_plan(args.plan, source=args.source if hasattr(args, "source") else None)
        segments = plan.segments
        info = plan.info
    else:
        info = probe(args.source)

    dest, index = contact_sheet(
        info, args.output, segments=segments, count=args.count, columns=args.columns
    )
    if args.json:
        print(json.dumps({"sheet": str(dest), "tiles": index}, indent=2, ensure_ascii=False))
    else:
        print(f"Wrote {dest} ({len(index)} tiles)")
        for entry in index:
            print(f"  {entry['tile']:02d}  {_hhmmss(float(entry['time']))}")
    return 0


def _run_render(args: argparse.Namespace, report) -> int:
    from .pipeline import render_plan, render_variants
    from .plan import load_plan

    plan, cfg = load_plan(args.plan, source=args.source)
    if args.also:
        cfg = cfg.merged(
            {"variants": {name: VARIANT_PRESETS[name] for name in dict.fromkeys(args.also)}}
        )

    output = (
        Path(args.output) if args.output else Path(args.plan).with_suffix("").with_suffix(".mp4")
    )
    if cfg.variants:
        outputs = render_variants(plan, output, cfg, progress=report)
        out = outputs["base"]
    else:
        outputs = {}
        out, _ = render_plan(plan, output, cfg, progress=report, write_sidecar=False)

    if not args.quiet:
        print(f"\n{len(plan.segments)} clips, {plan.total_duration:.1f}s reel")
        print(f"Reel:    {out}")
        for name, path in outputs.items():
            if name != "base":
                print(f"  +{name:<9} {path}")
    return 0


def _run_label(args: argparse.Namespace, report) -> int:
    from .contact import contact_sheet
    from .evaluation import Labels, write_labels
    from .pipeline import analyze

    cfg = load_config(args.profile) if args.profile else load_config()
    cfg = cfg.merged(
        {
            "segments": {
                "percentile": args.percentile,
                "max_clips": args.max_clips,
                "target_duration": None,
            },
            "refiners": [],
        }
    )
    plan = analyze(args.source, cfg, progress=report)
    if not plan.segments:
        print("Nothing proposed — try a lower --percentile.", file=sys.stderr)
        return 1

    source = Path(args.source)
    dest = Path(args.output) if args.output else source.with_suffix(".labels.yaml")

    draft = [
        {
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "keep": None,
            "why": max(seg.reasons, key=seg.reasons.get) if seg.reasons else "",
        }
        for seg in plan.segments
    ]
    labels = Labels(
        video=str(source),
        highlights=[],
        annotator=args.annotator,
        notes="draft — set keep: true/false, and add anything that was missed",
        profile=args.profile or "",
    )
    write_labels(labels, dest, draft=draft)

    sheet = None
    if not args.no_sheet:
        sheet, _ = contact_sheet(plan.info, dest.with_suffix(".png"), segments=plan.segments)

    if not args.quiet:
        print(f"\nDrafted {len(draft)} proposals -> {dest}")
        if sheet:
            print(f"Contact sheet         -> {sheet}  (one tile per proposal, in order)")
        print("\nNext: mark keep: true/false, add anything missed, then")
        print(f"  hypecut eval {dest} --profile <profile>")
    return 0


def _run_eval(args: argparse.Namespace) -> int:
    from .evaluation import load_labels, score_plan
    from .pipeline import analyze

    profiles: list[str | None] = list(args.profile) if args.profile else [None]
    results: dict[str, list[dict[str, object]]] = {}

    for profile in profiles:
        cfg = load_config(profile) if profile else load_config()
        name = Path(profile).stem if profile else "default"
        rows: list[dict[str, object]] = []
        for labels_path in args.labels:
            labels = load_labels(labels_path)
            plan = analyze(labels.video, cfg)
            score = score_plan(labels, [(s.start, s.end) for s in plan.segments])
            rows.append(score.to_dict())
        results[name] = rows

    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    print(
        f"{'profile':<20} {'clips':>6} {'found':>12} {'prec':>6} {'recall':>7} {'F1':>6} {'cover':>6}"
    )
    for name, rows in results.items():
        clips = sum(int(r["clips"]) for r in rows)
        found = sum(int(r["found"]) for r in rows)
        labelled = sum(int(r["labelled"]) for r in rows)
        precision = _mean(float(r["precision"]) for r in rows)
        recall = found / labelled if labelled else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        coverage = _mean(float(r["coverage"]) for r in rows)
        print(
            f"{name:<20} {clips:>6} {found:>5}/{labelled:<6} "
            f"{precision:>6.2f} {recall:>7.2f} {f1:>6.2f} {coverage:>6.2f}"
        )

    if not args.quiet:
        for name, rows in results.items():
            missed = [(r["video"], m) for r in rows for m in r["missed"]]
            if missed:
                print(f"\n{name} missed:")
                for video, item in missed[:12]:
                    label = f"  {item['label']}" if item.get("label") else ""
                    print(
                        f"  {_hhmmss(float(item['start']))}-{_hhmmss(float(item['end']))}"
                        f"  {Path(str(video)).name}{label}"
                    )
                if len(missed) > 12:
                    print(f"  ... and {len(missed) - 12} more")
    return 0


def _mean(values) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


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
        if args.json:
            print(
                json.dumps(
                    {"signals": available_signals(), "refiners": available_refiners()},
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        print("Signals (stage 1 — cheap, run over the whole video):")
        for name, desc in available_signals().items():
            print(f"  {name:<18} {desc}")
        print("\nRefiners (stage 2 — run only on candidates):")
        for name, desc in available_refiners().items():
            print(f"  {name:<18} {desc}")
        return 0

    if args.command == "profiles":
        found = _describe_profiles(Path(args.dir))
        if args.json:
            print(json.dumps(found, indent=2, ensure_ascii=False))
            return 0
        if not found:
            print(f"No profiles found in {args.dir}/", file=sys.stderr)
            return 1
        for item in found:
            print(f"  {item['name']:<22} {item['summary']}")
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
        if args.command == "contact-sheet":
            return _run_contact_sheet(args)
        if args.command == "label":
            return _run_label(args, _progress(args.quiet))
        if args.command == "eval":
            return _run_eval(args)
        if args.command == "render":
            return _run_render(args, _progress(args.quiet))

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
