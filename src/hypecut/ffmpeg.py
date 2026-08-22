"""Thin, well-behaved wrappers around the ffmpeg/ffprobe binaries.

HypeCut shells out rather than binding libav: it keeps the install to
``pip install hypecut`` plus a system ffmpeg, and it means any codec the
user's ffmpeg supports, HypeCut supports.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .types import VideoInfo

__all__ = [
    "FFmpegNotFound",
    "FFmpegError",
    "require_ffmpeg",
    "probe",
    "decode_audio",
    "decode_gray_frames",
    "run",
    "cmd",
]


class FFmpegNotFound(RuntimeError):
    """Raised when ffmpeg or ffprobe is not on PATH."""


class FFmpegError(RuntimeError):
    """Raised when an ffmpeg invocation exits non-zero."""


def require_ffmpeg() -> None:
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise FFmpegNotFound(
            f"Missing required binaries: {', '.join(missing)}. "
            "Install ffmpeg (https://ffmpeg.org/download.html) and retry."
        )


def run(args: list[str], *, capture: bool = False) -> bytes:
    """Run an ffmpeg-family command, raising a readable error on failure."""
    proc = subprocess.run(
        args, stdout=subprocess.PIPE if capture else subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    if proc.returncode != 0:
        tail = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-12:]
        raise FFmpegError(f"`{args[0]}` failed (exit {proc.returncode}):\n" + "\n".join(tail))
    return proc.stdout or b""


def cmd(template: str, **subs: str) -> list[str]:
    """Build an argv from a whitespace-separated template with ``{name}`` slots.

    ffmpeg invocations are long and read badly as one-token-per-line lists.
    Writing them the way they appear in documentation keeps them reviewable.

    Splitting happens *before* substitution, which is the point: a value can
    contain spaces (a path, a filter string) and still lands as exactly one
    argv entry. There is no shell involved, so nothing here is quoting-
    sensitive. Placeholders may sit inside a larger token, so filter
    expressions like ``color=s=320x180:d={dur}`` work.

        cmd("ffmpeg -i {src} -vn -ar {sr} -f f32le -", src=path, sr="16000")
    """
    pattern = re.compile(r"\{(\w+)\}")

    def fill(token: str) -> str:
        return pattern.sub(lambda m: subs[m.group(1)], token)

    return [fill(tok) for tok in template.split()]


def probe(path: str | Path) -> VideoInfo:
    """Read duration, fps and geometry without decoding the file."""
    require_ffmpeg()
    path = str(path)
    raw = run(
        cmd("ffprobe -v error -print_format json -show_format -show_streams {src}", src=path),
        capture=True,
    )
    data = json.loads(raw or b"{}")
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise FFmpegError(f"No video stream found in {path!r}")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    duration = _first_float(data.get("format", {}).get("duration"), video.get("duration"))
    if duration <= 0:
        # Some containers lie; fall back to counting.
        duration = _count_duration(path)

    return VideoInfo(
        path=path,
        duration=duration,
        fps=_parse_rate(video.get("avg_frame_rate") or video.get("r_frame_rate")),
        width=int(video.get("width") or 0),
        height=int(video.get("height") or 0),
        has_audio=has_audio,
    )


def decode_audio(path: str | Path, sr: int = 16_000) -> np.ndarray:
    """Decode the first audio stream to mono float32 at ``sr`` Hz."""
    raw = run(
        cmd(
            "ffmpeg -v error -nostdin -i {src} -vn -map 0:a:0 -ac 1 -ar {sr} -f f32le -",
            src=str(path),
            sr=str(sr),
        ),
        capture=True,
    )
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)


def decode_gray_frames(
    path: str | Path,
    *,
    fps: float,
    width: int = 96,
    height: int = 54,
    start: float | None = None,
    duration: float | None = None,
) -> np.ndarray:
    """Decode the video as tiny grayscale frames sampled at ``fps``.

    A 96x54 luma plane is ~5 KB per frame, so an hour of footage at 10 Hz
    fits comfortably in memory (~180 MB) while still carrying enough detail
    for scene-change, motion and region-of-interest signals.

    ``start`` / ``duration`` decode only a window, which is how the shot
    snapper affords a second pass at the source frame rate: a two-second
    window at 60 fps is 120 frames, not an hour of them.
    """
    seek = ""
    if start is not None:
        seek += f"-accurate_seek -ss {max(0.0, start):.3f} "
    if duration is not None:
        seek += f"-t {max(0.01, duration):.3f} "
    raw = run(
        cmd(
            f"ffmpeg -v error -nostdin {seek}-i {{src}} -an -map 0:v:0 -vf {{vf}} "
            "-pix_fmt gray -f rawvideo -",
            src=str(path),
            vf=f"fps={fps},scale={width}:{height}:flags=fast_bilinear",
        ),
        capture=True,
    )
    frame_bytes = width * height
    n = len(raw) // frame_bytes
    if n == 0:
        return np.zeros((0, height, width), dtype=np.uint8)
    arr = np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8)
    return arr.reshape(n, height, width)


def _parse_rate(value: str | None) -> float:
    if not value:
        return 0.0
    if "/" in value:
        num, _, den = value.partition("/")
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        except ValueError:
            return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _first_float(*values: object) -> float:
    for v in values:
        try:
            f = float(v)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if f > 0:
            return f
    return 0.0


def _count_duration(path: str) -> float:
    raw = run(
        cmd(
            "ffprobe -v error -select_streams v:0 -count_packets "
            "-show_entries stream=nb_read_packets,avg_frame_rate -print_format json {src}",
            src=path,
        ),
        capture=True,
    )
    data = json.loads(raw or b"{}")
    stream = (data.get("streams") or [{}])[0]
    packets = float(stream.get("nb_read_packets") or 0)
    rate = _parse_rate(stream.get("avg_frame_rate"))
    return packets / rate if rate else 0.0
