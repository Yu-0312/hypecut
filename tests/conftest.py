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


# Pauses built into `talk_vod`, in seconds. No hard cuts anywhere in it: a
# locked-off camera is exactly the footage shot snapping cannot help with.
TALK_PAUSES = [(6.0, 7.5), (16.0, 17.5), (26.0, 28.0)]
TALK_LOUD = (17.5, 26.0)


@pytest.fixture(scope="session")
def talk_vod(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A static frame over speech-like audio with pauses at known times."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    np = pytest.importorskip("numpy")

    d = tmp_path_factory.mktemp("talk")
    sr, rng = 48_000, np.random.default_rng(11)
    layout = [
        ("talk", 6.0),
        ("gap", 1.5),
        ("talk", 8.5),
        ("gap", 1.5),
        ("loud", 8.5),
        ("gap", 2.0),
        ("talk", 6.0),
    ]
    chunks = []
    cursor = 0.0
    for kind, seconds in layout:
        n = int(seconds * sr)
        if kind == "gap":
            chunks.append(np.zeros(n, dtype=np.float32))
        elif kind == "talk":
            chunks.append(rng.normal(0, 0.04, n).astype(np.float32))
        else:
            t = np.arange(n) / sr + cursor
            chunks.append((0.5 * np.sin(2 * np.pi * 520 * t)).astype(np.float32))
        cursor += seconds

    raw = d / "audio.f32"
    raw.write_bytes(np.concatenate(chunks).tobytes())

    out = d / "talk.mp4"
    _run(
        "ffmpeg -v error -y -f f32le -ar 48000 -ac 1 -i {raw} "
        "-f lavfi -i color=c=#1a1a24:s=320x180:r=15 -shortest "
        "-c:v libx264 -preset ultrafast -crf 32 -pix_fmt yuv420p -c:a aac -b:a 96k {out}",
        raw=str(raw),
        out=str(out),
    )
    return out


# Landmarks in `sports_vod`, in seconds.
SPORT_GOAL = 26.0  # the moment itself — silent
SPORT_ROAR = (28.0, 36.0)  # the crowd reaction that follows it
SPORT_WHISTLE = 8.0
SPORT_DECOY = 17.0  # a loud but brief shout that must not win


@pytest.fixture(scope="session")
def sports_vod(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Broadcast-style sport: crowd bed, whistle, a silent goal, then a roar."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    np = pytest.importorskip("numpy")

    d = tmp_path_factory.mktemp("sport")
    sr, seconds = 48_000, 50.0
    n = int(sr * seconds)
    t = np.arange(n) / sr
    rng = np.random.default_rng(23)

    bed = np.convolve(rng.normal(0, 0.03, n), np.ones(60) / 60, mode="same")
    blast = (t >= SPORT_WHISTLE) & (t < SPORT_WHISTLE + 0.7)
    bed[blast] += 0.35 * np.sin(2 * np.pi * 3500 * t[blast])
    decoy = (t >= SPORT_DECOY) & (t < SPORT_DECOY + 0.4)
    bed[decoy] += rng.normal(0, 0.45, int(decoy.sum()))

    roar = np.zeros(n)
    lo, hi = SPORT_ROAR
    window = (t >= lo) & (t < hi)
    envelope = np.clip((t[window] - lo) / 1.5, 0, 1) * np.clip((hi - t[window]) / 2.0, 0, 1)
    roar[window] = rng.normal(0, 0.30, int(window.sum())) * envelope
    roar = np.convolve(roar, np.ones(40) / 40, mode="same")

    raw = d / "audio.f32"
    raw.write_bytes((bed + roar).astype(np.float32).tobytes())

    out = d / "sport.mp4"
    _run(
        "ffmpeg -v error -y -f lavfi -i color=c=#1d5c2a:s=480x270:r=15:d=50 "
        "-f lavfi -i color=c=white:s=14x14:r=15:d=50 "
        "-f lavfi -i color=c=#f0f0f0:s=10x18:r=15:d=50 "
        "-f f32le -ar 48000 -ac 1 -i {raw} "
        "-filter_complex {graph} -map [v] -map 3:a -shortest "
        "-c:v libx264 -preset ultrafast -crf 30 -pix_fmt yuv420p -c:a aac -b:a 96k {out}",
        raw=str(raw),
        out=str(out),
        # A ball wandering the pitch, plus a score digit that jumps once, a
        # second after the goal — a scoreboard update, isolated to its corner.
        graph=(
            "[0:v][1:v]overlay=x='240+210*sin(t*0.7)':y='135+75*cos(t*0.9)'[a];"
            f"[a][2:v]overlay=x='30+if(gte(t,{SPORT_GOAL + 1.0}),14,0)':y=12[v]"
        ),
    )
    return out


# Landmarks in `repeat_vod`, in seconds. The first and third runs are the same
# action in the same place; the middle one moves the other way, diagonally.
REPEAT_FIRST = (20.0, 26.0)
REPEAT_DIFFERENT = (110.0, 116.0)
REPEAT_AGAIN = (200.0, 206.0)


@pytest.fixture(scope="session")
def repeat_vod(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Four minutes with the same action happening twice, three minutes apart.

    Built to exercise the one case `similarity` exists for and `diversity`
    cannot see: two moments that are far enough apart to look unrelated in
    time and are in fact the same thing again. The middle event is a control —
    equally loud, visually different — so a refiner that simply penalises
    everything fails this fixture rather than passing it by accident.
    """
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("repeat") / "repeat.mp4"
    a0, a1 = REPEAT_FIRST
    b0, _b1 = REPEAT_DIFFERENT
    c0, c1 = REPEAT_AGAIN
    graph = (
        f"[1:a]volume='1+9*(between(t,{a0},{a1})+between(t,{b0},{b0 + 6})"
        f"+between(t,{c0},{c1}))':eval=frame[a];"
        "color=c=white:s=70x70:d=240[b];"
        f"[0:v][b]overlay=x='if(between(t,{a0},{a1}),40+(t-{a0})*60,"
        f"if(between(t,{c0},{c1}),40+(t-{c0})*60,"
        f"if(between(t,{b0},{b0 + 6}),380-(t-{b0})*40,-100)))':"
        f"y='if(between(t,{b0},{b0 + 6}),20+(t-{b0})*22,150)'[v]"
    )
    _run(
        "ffmpeg -v error -y -f lavfi -i color=c=0x14331f:s=480x270:r=25:d=240 "
        "-f lavfi -i anoisesrc=d=240:c=pink:a=0.02:r=16000 "
        "-filter_complex {graph} -map [v] -map [a] -shortest "
        "-c:v libx264 -preset ultrafast -crf 28 -pix_fmt yuv420p -c:a aac {out}",
        graph=graph,
        out=str(out),
    )
    return out


@pytest.fixture(scope="session")
def boring_vod(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A minute of nothing: a static scene and room tone.

    Not a broken file and not a silent one — a perfectly valid recording of an
    idle lobby, an unattended camera, a stream left running. This is the input
    every relative threshold in the pipeline gets wrong, because a percentile
    of a flat curve still selects its top slice.
    """
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")
    out = tmp_path_factory.mktemp("boring") / "boring.mp4"
    _run(
        "ffmpeg -v error -y -f lavfi -i color=c=0x203040:s=320x180:r=15:d=60 "
        "-f lavfi -i sine=frequency=110:sample_rate=16000:duration=60 "
        "-filter_complex [0:v]noise=alls=8:allf=t+u[v];[1:a]volume=0.05[a] "
        "-map [v] -map [a] -shortest "
        "-c:v libx264 -preset ultrafast -crf 30 -pix_fmt yuv420p -c:a aac {out}",
        out=str(out),
    )
    return out
