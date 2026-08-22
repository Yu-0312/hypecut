"""Cut the clips and stitch the reel.

Two-pass by design: every segment is first re-encoded to an identical
intermediate (same codec, resolution, sample rate, timebase), then joined
with the concat demuxer. Cutting straight from the source with stream copy
is faster but lands on keyframes, which is exactly the wrong trade for a
highlight reel — a clip that starts 1.8 s late has already missed the shot.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from .config import RenderConfig
from .ffmpeg import cmd, require_ffmpeg, run
from .reframe import geometry_filters
from .types import Candidate, VideoInfo

__all__ = ["render_reel", "write_chapters_file", "write_edl"]

Progress = Callable[[float, str], None]


def render_reel(
    info: VideoInfo,
    segments: list[Candidate],
    output: str | Path,
    cfg: RenderConfig,
    *,
    progress: Progress | None = None,
    workdir: str | Path | None = None,
) -> Path:
    """Encode ``segments`` from ``info.path`` into a single file at ``output``."""
    require_ffmpeg()
    if not segments:
        raise ValueError("No segments to render — nothing scored above threshold.")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    tmp_root = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="hypecut-"))
    tmp_root.mkdir(parents=True, exist_ok=True)
    owned = workdir is None
    parts: list[Path] = []

    try:
        for idx, seg in enumerate(segments):
            part = tmp_root / f"part_{idx:04d}.mp4"
            _encode_segment(info, seg, part, cfg)
            parts.append(part)
            if progress:
                progress((idx + 1) / (len(segments) + 1), f"clip {idx + 1}/{len(segments)}")

        _concat(parts, output, cfg, info, segments)
        if progress:
            progress(1.0, "done")
        return output
    finally:
        if owned:
            shutil.rmtree(tmp_root, ignore_errors=True)


def _encode_segment(info: VideoInfo, seg: Candidate, dest: Path, cfg: RenderConfig) -> None:
    filters = []
    if cfg.reframe.mode != "off":
        # Reframing decides the whole geometry; a scale/pad on top of it would
        # only undo the crop it just made.
        filters.extend(geometry_filters(info, seg, cfg.reframe))
    elif cfg.width and cfg.height:
        filters.append(
            f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=decrease,"
            f"pad={cfg.width}:{cfg.height}:-1:-1:color=black"
        )
    if cfg.fps:
        filters.append(f"fps={cfg.fps}")
    if cfg.fade > 0 and seg.duration > cfg.fade * 2:
        out_at = max(0.0, seg.duration - cfg.fade)
        filters.append(f"fade=t=in:st=0:d={cfg.fade:.3f}")
        filters.append(f"fade=t=out:st={out_at:.3f}:d={cfg.fade:.3f}")
    filters.append("setpts=PTS-STARTPTS")

    afilters = []
    if cfg.normalize_audio:
        afilters.append("dynaudnorm=f=200:g=5")
    # The video fade stays constant so the reel keeps one visual rhythm, but the
    # audio fade adapts: a clip that ends in a pause needs almost none, while one
    # cut off mid-sound needs a longer ramp or the stop reads as a dropout.
    a_fade = cfg.fade if seg.meta.get("ends_in_silence") else min(cfg.fade * 2.5, 0.6)
    if a_fade > 0 and seg.duration > a_fade * 2:
        out_at = max(0.0, seg.duration - a_fade)
        afilters.append(f"afade=t=in:st=0:d={min(cfg.fade, a_fade):.3f}")
        afilters.append(f"afade=t=out:st={out_at:.3f}:d={a_fade:.3f}")
    afilters.append("asetpts=PTS-STARTPTS")

    # -ss before -i seeks fast; -accurate_seek keeps the cut frame-exact, which
    # matters more than the speed: a clip that starts on the wrong keyframe has
    # already missed the moment.
    args = cmd(
        "ffmpeg -v error -nostdin -y -accurate_seek -ss {start} -t {dur} -i {src}",
        start=f"{seg.start:.3f}",
        dur=f"{seg.duration:.3f}",
        src=info.path,
    )
    if not info.has_audio:
        # Silent source: synthesise silence so every part has the same stream
        # layout and the concat demuxer stays happy.
        args += cmd(
            "-f lavfi -t {dur} -i anullsrc=channel_layout=stereo:sample_rate=48000",
            dur=f"{seg.duration:.3f}",
        )
    args += cmd(
        "-map 0:v:0 -map {amap} -vf {vf} -c:v {vcodec} -crf {crf} -preset {preset} "
        "-pix_fmt yuv420p -c:a {acodec} -b:a {abitrate} -ar 48000 -ac 2",
        amap="0:a:0" if info.has_audio else "1:a:0",
        vf=",".join(filters),
        vcodec=cfg.video_codec,
        crf=str(cfg.crf),
        preset=cfg.preset,
        acodec=cfg.audio_codec,
        abitrate=cfg.audio_bitrate,
    )
    if info.has_audio:
        args += ["-af", ",".join(afilters)]
    args += ["-shortest", "-movflags", "+faststart", str(dest)]
    run(args)


def _concat(
    parts: Iterable[Path],
    output: Path,
    cfg: RenderConfig,
    info: VideoInfo,
    segments: list[Candidate],
) -> None:
    parts = list(parts)
    listing = output.parent / f".{output.stem}.concat.txt"
    listing.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in parts), encoding="utf-8"
    )
    args = cmd("ffmpeg -v error -nostdin -y -f concat -safe 0 -i {listing}", listing=str(listing))
    metadata: list[str] = []
    if cfg.write_chapters and segments:
        chapters = output.parent / f".{output.stem}.chapters.txt"
        write_chapters_file(segments, chapters)
        args += ["-i", str(chapters), "-map_metadata", "1"]
        metadata = [str(chapters)]
    # Map only the concatenated A/V; without this the ffmetadata input would
    # also be muxed in as a stray data stream.
    args += ["-map", "0:v:0", "-map", "0:a:0?"]
    args += ["-c", "copy", "-movflags", "+faststart", str(output)]
    try:
        run(args)
    finally:
        listing.unlink(missing_ok=True)
        for m in metadata:
            Path(m).unlink(missing_ok=True)


def write_chapters_file(segments: list[Candidate], dest: str | Path) -> Path:
    """Emit an ffmetadata chapter list so players show one marker per clip."""
    dest = Path(dest)
    lines = [";FFMETADATA1"]
    cursor = 0.0
    for idx, seg in enumerate(segments, start=1):
        start_ms = int(cursor * 1000)
        cursor += seg.duration
        end_ms = int(cursor * 1000)
        top = max(seg.reasons, key=seg.reasons.get) if seg.reasons else "highlight"
        lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={start_ms}",
            f"END={end_ms}",
            f"title=#{idx} {top} @ {_hhmmss(seg.start)}",
        ]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def write_edl(segments: list[Candidate], dest: str | Path, fps: float = 30.0) -> Path:
    """Emit a CMX3600 EDL so the cut can be opened in a real NLE.

    HypeCut's job is to find the moments; the user may well want to finish
    the edit in Resolve or Premiere. Handing over an EDL makes the tool a
    first pass rather than a black box.
    """
    dest = Path(dest)
    lines = ["TITLE: HYPECUT REEL", "FCM: NON-DROP FRAME", ""]
    record = 0.0
    for idx, seg in enumerate(segments, start=1):
        lines.append(
            f"{idx:03d}  AX       AA/V  C        "
            f"{_tc(seg.start, fps)} {_tc(seg.end, fps)} "
            f"{_tc(record, fps)} {_tc(record + seg.duration, fps)}"
        )
        lines.append(f"* FROM CLIP NAME: {Path(dest).stem}")
        record += seg.duration
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def _hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _tc(seconds: float, fps: float) -> str:
    total = int(round(seconds * fps))
    f = int(fps) or 30
    return f"{total // (3600 * f):02d}:{(total // (60 * f)) % 60:02d}:{(total // f) % 60:02d}:{total % f:02d}"
