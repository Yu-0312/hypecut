#!/usr/bin/env python3
"""Build a synthetic test VOD with known highlight moments.

Useful for trying HypeCut without hunting for footage, and for eyeballing a
change to the scoring: the loud/busy stretches are at known timestamps, so a
regression is obvious.

    python scripts/make_sample.py /tmp/sample.mp4
    hypecut cut /tmp/sample.mp4 -o /tmp/reel.mp4 --percentile 88
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from hypecut.ffmpeg import cmd

# (kind, seconds) — "exciting" stretches are loud and full of motion.
LAYOUT = [
    ("boring", 25),
    ("exciting", 8),
    ("boring", 25),
    ("exciting", 8),
    ("boring", 25),
    ("exciting", 8),
]


def _run(template: str, **subs: str) -> None:
    subprocess.run(
        cmd(template, **subs), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )


def build(dest: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="hypecut-sample-") as tmp:
        d = Path(tmp)
        parts: list[Path] = []
        cursor = 0.0
        for idx, (kind, dur) in enumerate(LAYOUT):
            part = d / f"{idx:02d}.mp4"
            if kind == "boring":
                _run(
                    "ffmpeg -v error -y -f lavfi -i color=c=#181820:s=640x360:r=30:d={d} "
                    "-f lavfi -i anoisesrc=c=pink:r=48000:a=0.01:d={d} "
                    "-c:v libx264 -preset ultrafast -crf 30 -pix_fmt yuv420p "
                    "-c:a aac -b:a 96k -shortest {out}",
                    d=str(dur),
                    out=str(part),
                )
            else:
                print(f"  highlight at {cursor:.0f}s–{cursor + dur:.0f}s")
                _run(
                    "ffmpeg -v error -y -f lavfi -i testsrc2=s=640x360:r=30:d={d} "
                    "-f lavfi -i sine=frequency={freq}:r=48000:d={d} "
                    "-filter_complex [1:a]volume=6.0[a] -map 0:v -map [a] "
                    "-c:v libx264 -preset ultrafast -crf 30 -pix_fmt yuv420p "
                    "-c:a aac -b:a 96k -shortest {out}",
                    d=str(dur),
                    freq=str(300 + idx * 120),
                    out=str(part),
                )
            parts.append(part)
            cursor += dur

        listing = d / "list.txt"
        listing.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
        dest.parent.mkdir(parents=True, exist_ok=True)
        _run(
            "ffmpeg -v error -y -f concat -safe 0 -i {listing} -c copy {out}",
            listing=str(listing),
            out=str(dest),
        )
    return dest


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "sample.mp4")
    print(f"Building {out} ({sum(d for _, d in LAYOUT)}s)…")
    build(out)
    print(f"Done: {out}")
