"""Shared fixtures.

The heavy fixture builds a small synthetic VOD with three known "exciting"
stretches, which lets the end-to-end tests assert on *where* clips land
rather than just that something was produced.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from hypecut.ffmpeg import cmd

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None

requires_ffmpeg = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")

# (start, end) seconds of the loud/busy stretches built by `sample_vod`.
EXCITING_WINDOWS = [(15.0, 20.0), (35.0, 40.0)]


def _run(template: str, **subs: str) -> None:
    subprocess.run(
        cmd(template, **subs), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )


@pytest.fixture(scope="session")
def sample_vod(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 45 s clip: quiet/static, then loud/busy, twice."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")

    d = tmp_path_factory.mktemp("vod")
    parts: list[Path] = []

    def boring(idx: int, dur: int) -> Path:
        p = d / f"b{idx}.mp4"
        _run(
            "ffmpeg -v error -y -f lavfi -i color=c=#181820:s=320x180:r=15:d={d} "
            "-f lavfi -i anoisesrc=c=pink:r=48000:a=0.01:d={d} "
            "-c:v libx264 -preset ultrafast -crf 32 -pix_fmt yuv420p "
            "-c:a aac -b:a 64k -shortest {out}",
            d=str(dur),
            out=str(p),
        )
        return p

    def exciting(idx: int, dur: int) -> Path:
        p = d / f"e{idx}.mp4"
        _run(
            "ffmpeg -v error -y -f lavfi -i testsrc2=s=320x180:r=15:d={d} "
            "-f lavfi -i sine=frequency={freq}:r=48000:d={d} "
            "-filter_complex [1:a]volume=6.0[a] -map 0:v -map [a] "
            "-c:v libx264 -preset ultrafast -crf 32 -pix_fmt yuv420p "
            "-c:a aac -b:a 64k -shortest {out}",
            d=str(dur),
            freq=str(400 + idx * 150),
            out=str(p),
        )
        return p

    parts = [boring(0, 15), exciting(0, 5), boring(1, 15), exciting(1, 5), boring(2, 5)]
    listing = d / "list.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")

    out = d / "sample.mp4"
    _run(
        "ffmpeg -v error -y -f concat -safe 0 -i {listing} -c copy {out}",
        listing=str(listing),
        out=str(out),
    )
    return out


@pytest.fixture(scope="session")
def silent_vod(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A video with no audio track at all — the awkward input."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    d = tmp_path_factory.mktemp("silent")
    out = d / "silent.mp4"
    _run(
        "ffmpeg -v error -y -f lavfi -i testsrc2=s=320x180:r=15:d=20 "
        "-c:v libx264 -preset ultrafast -crf 32 -pix_fmt yuv420p -an {out}",
        out=str(out),
    )
    return out
