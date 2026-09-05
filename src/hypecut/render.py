"""Cut the clips and stitch the reel.

Two-pass by design: every segment is first re-encoded to an identical
intermediate (same codec, resolution, sample rate, timebase), then joined
with the concat demuxer. Cutting straight from the source with stream copy
is faster but lands on keyframes, which is exactly the wrong trade for a
highlight reel — a clip that starts 1.8 s late has already missed the shot.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path

from .config import RenderConfig
from .ffmpeg import cmd, require_ffmpeg, run
from .reframe import geometry_filters
from .types import Candidate, VideoInfo

__all__ = [
    "render_reel",
    "audio_filters",
    "measure_loudness",
    "plan_loudness_gains",
    "write_chapters_file",
    "write_edl",
]

Progress = Callable[[float, str], None]


def render_reel(
    info: VideoInfo,
    segments: list[Candidate],
    output: str | Path,
    cfg: RenderConfig,
    *,
    progress: Progress | None = None,
    workdir: str | Path | None = None,
    plan_key: str = "reframe",
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
        gains = plan_loudness_gains(
            info, segments, cfg, progress=(lambda p, m: progress(p * 0.25, m)) if progress else None
        )
        for idx, (seg, gain) in enumerate(zip(segments, gains, strict=True)):
            part = tmp_root / f"part_{idx:04d}.mp4"
            _encode_segment(info, seg, part, cfg, plan_key, gain)
            if gain:
                seg.meta["loudness_gain_db"] = round(gain, 2)
            parts.append(part)
            if progress:
                progress(
                    0.25 + 0.75 * (idx + 1) / (len(segments) + 1), f"clip {idx + 1}/{len(segments)}"
                )

        _concat(parts, output, cfg, info, segments)
        if progress:
            progress(1.0, "done")
        return output
    finally:
        if owned:
            shutil.rmtree(tmp_root, ignore_errors=True)


def audio_filters(seg: Candidate, cfg: RenderConfig, gain_db: float = 0.0) -> list[str]:
    """The audio chain for one clip, with an optional matching gain in front.

    Shared by the measurement pass and the encode so the two see the same
    signal: measuring the raw segment and then applying a compressor would
    give a gain computed for audio that no longer exists.
    """
    out: list[str] = []
    if abs(gain_db) > 0.05:
        out.append(f"volume={gain_db:.2f}dB")
    if cfg.normalize_audio:
        out.append("dynaudnorm=f=200:g=5")
    # The video fade stays constant so the reel keeps one visual rhythm, but the
    # audio fade adapts: a clip that ends in a pause needs almost none, while one
    # cut off mid-sound needs a longer ramp or the stop reads as a dropout.
    a_fade = cfg.fade if seg.meta.get("ends_in_silence") else min(cfg.fade * 2.5, 0.6)
    if a_fade > 0 and seg.duration > a_fade * 2:
        out_at = max(0.0, seg.duration - a_fade)
        out.append(f"afade=t=in:st=0:d={min(cfg.fade, a_fade):.3f}")
        out.append(f"afade=t=out:st={out_at:.3f}:d={a_fade:.3f}")
    out.append("asetpts=PTS-STARTPTS")
    return out


def measure_loudness(info: VideoInfo, seg: Candidate, cfg: RenderConfig) -> float | None:
    """Integrated loudness of one clip in LUFS, or ``None`` if unmeasurable.

    Decodes audio only, through the same filter chain the encode will use.
    """
    args = cmd(
        "ffmpeg -v info -nostdin -y -accurate_seek -ss {start} -t {dur} -i {src} "
        "-map 0:a:0 -af {af} -f null -",
        start=f"{seg.start:.3f}",
        dur=f"{seg.duration:.3f}",
        src=info.path,
        af=",".join([*audio_filters(seg, cfg), "loudnorm=print_format=json"]),
    )
    proc = subprocess.run(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    text = proc.stderr.decode("utf-8", "replace")
    match = re.search(r'"input_i"\s*:\s*"(-?[\d.]+|-inf)"', text)
    if not match or match.group(1) == "-inf":
        return None
    return float(match.group(1))


def plan_loudness_gains(
    info: VideoInfo,
    segments: list[Candidate],
    cfg: RenderConfig,
    *,
    progress: Progress | None = None,
) -> list[float]:
    """Per-clip gain in dB that brings the reel to a consistent loudness.

    Two-pass, because there is no other way: integrated loudness is a property
    of a whole clip, so you cannot know the right gain until you have heard it
    all. ``dynaudnorm`` alone does not solve this — it evens out dynamics
    *inside* a clip and says nothing about how two clips compare, which is
    exactly the artefact people notice when a reel jumps in volume halfway.

    ``loudness_match`` is deliberately not forced to 1.0. Matching every clip
    to the same number makes a quiet moment and a stadium roar equally loud,
    which is technically correct and editorially wrong. At 0.9 the spread
    collapses to a tenth of what it was — inaudible as a jump, still audible
    as character.
    """
    if not info.has_audio or cfg.loudness_match <= 0:
        return [0.0] * len(segments)

    gains: list[float] = []
    for idx, seg in enumerate(segments):
        measured = measure_loudness(info, seg, cfg)
        if measured is None or measured < -50.0:
            # Silence, or close enough. Lifting it would only amplify the
            # noise floor into something audible.
            gains.append(0.0)
        else:
            wanted = (cfg.loudness_target - measured) * cfg.loudness_match
            gains.append(float(max(-cfg.loudness_max_gain, min(cfg.loudness_max_gain, wanted))))
        if progress:
            progress((idx + 1) / len(segments), f"measuring {idx + 1}/{len(segments)}")
    return gains


def _encode_segment(
    info: VideoInfo,
    seg: Candidate,
    dest: Path,
    cfg: RenderConfig,
    plan_key: str = "reframe",
    gain_db: float = 0.0,
) -> None:
    filters = []
    if cfg.reframe.mode != "off":
        # Reframing decides the whole geometry; a scale/pad on top of it would
        # only undo the crop it just made.
        filters.extend(geometry_filters(info, seg, cfg.reframe, plan_key))
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

    afilters = audio_filters(seg, cfg, gain_db)

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


def write_edl(
    segments: list[Candidate],
    dest: str | Path,
    fps: float = 30.0,
    *,
    source_name: str | None = None,
) -> Path:
    """Emit a CMX3600 EDL so the cut can be opened in a real NLE.

    HypeCut's job is to find the moments; the user may well want to finish
    the edit in Resolve or Premiere. Handing over an EDL makes the tool a
    first pass rather than a black box.

    ``source_name`` names the media each event is cut *from*. It matters
    because the first pair of timecodes on every event line is source
    timecode: an NLE reads the clip name to decide what to relink those
    timecodes against. Naming the rendered reel there — which is what this
    did — pointed the source timecodes at the output file, where they mean
    nothing. It falls back to the EDL's own stem only so the signature stays
    compatible for anyone calling this directly.
    """
    dest = Path(dest)
    clip_name = source_name or dest.stem
    lines = ["TITLE: HYPECUT REEL", "FCM: NON-DROP FRAME", ""]
    record = 0.0
    for idx, seg in enumerate(segments, start=1):
        lines.append(
            f"{idx:03d}  AX       AA/V  C        "
            f"{_tc(seg.start, fps)} {_tc(seg.end, fps)} "
            f"{_tc(record, fps)} {_tc(record + seg.duration, fps)}"
        )
        lines.append(f"* FROM CLIP NAME: {clip_name}")
        record += seg.duration
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def _hhmmss(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _tc(seconds: float, fps: float) -> str:
    """CMX3600 non-drop timecode for a point in time.

    Two different rates are at work here, and conflating them is what made
    this wrong. The *frame number* is real time times the real rate — 29.97
    for any console capture, camcorder or broadcast source, all of which are
    30000/1001. The clock that displays that frame number counts at the
    *nominal* rate: 30 for 29.97 material, 24 for 23.976. That mismatch, 3.6
    seconds per hour, is precisely what "non-drop" means.

    Truncating the real rate to get the divisor put 29 frames in a timecode
    second of 29.97 footage, so the clock gained about two minutes an hour and
    the EDL could not be conformed against its source at all. Every fixture in
    the suite is 15 fps, where truncation and rounding agree — which is why
    nothing caught it.
    """
    base = int(round(fps)) or 30
    total = int(round(seconds * fps))
    return (
        f"{total // (3600 * base):02d}:{(total // (60 * base)) % 60:02d}:"
        f"{(total // base) % 60:02d}:{total % base:02d}"
    )
