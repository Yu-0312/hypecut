"""Contact sheets — one image that shows an agent what the footage is.

An agent driving HypeCut hits a wall immediately: the cut list says
``roi_activity 2.1`` and it has no idea whether that means a kill feed, a
scoreboard, or a lighting change, because it has never seen the video. Text
cannot answer "is this broadcast football or a first-person shooter, and
which corner is the score bug in".

So: sample frames, label each with its index and timestamp, tile them into
one image. One image rather than N files because a vision model reasons far
better about a grid it can compare across than about a stream of separate
pictures, and because it is one attachment instead of twenty.

Two modes, for the two questions an agent actually asks:

* no plan — sample evenly across the whole video. *What kind of footage is
  this?* This is the first call, before any profile has been chosen.
* with a plan — one frame per proposed clip, at its peak. *Did the detector
  pick the right moments?* This is the review call.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from .ffmpeg import cmd, has_filter, require_ffmpeg, run
from .types import Candidate, VideoInfo

__all__ = ["contact_sheet", "sample_times"]


def sample_times(info: VideoInfo, segments: list[Candidate] | None, count: int) -> list[float]:
    """Timestamps to sample: clip peaks if there is a plan, else an even sweep."""
    if segments:
        return [float(seg.meta.get("peak_time", (seg.start + seg.end) / 2)) for seg in segments]
    count = max(1, count)
    # Inset by half a step so the first tile is not a black leader frame and
    # the last is not past the end of a slightly-short file.
    step = info.duration / count
    return [step * (i + 0.5) for i in range(count)]


def contact_sheet(
    info: VideoInfo,
    dest: str | Path,
    *,
    segments: list[Candidate] | None = None,
    count: int = 12,
    columns: int = 4,
    tile_width: int = 480,
    label: bool = True,
) -> tuple[Path, list[dict[str, object]]]:
    """Write a labelled grid of frames. Returns the path and the tile index.

    The index is returned rather than only burned into the image so the caller
    can still work if the labels failed to render — a system with no usable
    font is a real possibility, and losing the labels should not mean losing
    the mapping from tile to timestamp.
    """
    require_ffmpeg()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    times = sample_times(info, segments, count)
    if not times:
        raise ValueError("Nothing to sample — the video has no duration and no segments.")

    tile_height = max(2, int(round(tile_width * info.height / max(info.width, 1) / 2)) * 2)
    font = _font_file() if label else None

    index: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="hypecut-sheet-") as tmp:
        work = Path(tmp)
        for position, at in enumerate(times, start=1):
            frame = work / f"{position:03d}.png"
            filters = [f"scale={tile_width}:{tile_height}"]
            if font:
                caption = f"{position:02d}  {_hhmmss(at)}"
                # The caption sits at the *bottom*. The top-left corner is
                # where broadcasters put the score bug and games put the kill
                # feed — the exact thing an agent is reading the sheet to find
                # — so a label there would cover the answer.
                filters.append(
                    "drawbox=x=0:y=ih-34:w=iw:h=34:color=black@0.55:t=fill,"
                    f"drawtext=fontfile={font}:text='{caption}'"
                    ":x=10:y=h-27:fontsize=22:fontcolor=white"
                )
            run(
                cmd(
                    "ffmpeg -v error -nostdin -y -accurate_seek -ss {at} -i {src} "
                    "-frames:v 1 -vf {vf} {out}",
                    at=f"{max(0.0, at):.3f}",
                    src=info.path,
                    vf=",".join(filters),
                    out=str(frame),
                )
            )
            entry: dict[str, object] = {"tile": position, "time": round(at, 3)}
            if segments:
                seg = segments[position - 1]
                entry.update(
                    start=round(seg.start, 3),
                    end=round(seg.end, 3),
                    score=round(seg.score, 4),
                    top_signal=max(seg.reasons, key=seg.reasons.get) if seg.reasons else None,
                )
            index.append(entry)

        # Never more columns than tiles, or ffmpeg pads the grid with blank
        # cells and a one-clip sheet comes out four tiles wide.
        columns = max(1, min(columns, len(times)))
        rows = max(1, -(-len(times) // columns))
        run(
            cmd(
                "ffmpeg -v error -nostdin -y -i {pattern} -vf tile={cols}x{rows}:padding=6:"
                "margin=6:color=#12141a -frames:v 1 {out}",
                pattern=str(work / "%03d.png"),
                cols=str(columns),
                rows=str(rows),
                out=str(dest),
            )
        )
    return dest, index


def _font_file() -> str | None:
    """A usable TTF, or None. Labels are a nicety; their absence is survivable.

    Two things have to be true to burn a caption in, and they fail
    independently. There has to be a font, and the ffmpeg on PATH has to have
    been built with ``drawtext`` — which needs libfreetype at compile time and
    which Homebrew's macOS bottle does without. Checking only for the font
    found a perfectly good Arial on a machine whose ffmpeg could not draw with
    it, and turned an optional label into a crash.
    """
    import shutil
    import subprocess

    if not has_filter("drawtext"):
        return None

    if shutil.which("fc-match"):
        try:
            found = subprocess.run(
                ["fc-match", "-f", "%{file}", "DejaVu Sans"], capture_output=True, timeout=5
            ).stdout.decode()
            if found and Path(found).exists():
                return found
        except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment dependent
            pass
    for guess in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(guess).exists():
            return guess
    return None


def _hhmmss(seconds: float) -> str:
    s = int(max(0.0, seconds))
    return f"{s // 3600:02d}\\:{(s % 3600) // 60:02d}\\:{s % 60:02d}"
