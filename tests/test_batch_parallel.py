"""Parallel batch — cutting a folder with several worker processes."""

from __future__ import annotations

import subprocess

import pytest

from hypecut.cli import main
from hypecut.ffmpeg import cmd

from .conftest import HAS_FFMPEG

pytestmark = pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed")


def _make_video(path, seconds: int = 12) -> None:
    """A short clip with one unmistakable highlight in the middle.

    The footage has to *contain an event*, not merely be non-silent. Every
    threshold below `min_prominence` is relative to the video it is given, so
    a clip of one constant tone over one uniform test pattern has nothing for
    the emptiness check to find and is correctly judged empty — which is what
    the previous fixture was. It cut on Linux only because the AAC encoder's
    own noise, 0.06 of a log-energy unit, divided by a near-zero MAD, looked
    to `prominence` like a 405x onset; the same tone at a different bitrate
    scores 0.9 and comes back empty. A quiet bed with a loud stretch in the
    middle gives the batch something real to cut whatever ffmpeg is on PATH.

    These tests are about the process pool, not the detector, so the footage
    they feed it should not be a borderline case for the detector.
    """
    loud_from, loud_to = seconds / 3.0, seconds * 2.0 / 3.0
    subprocess.run(
        cmd(
            "ffmpeg -v error -y -f lavfi -i testsrc2=s=320x180:r=15:d={d} "
            "-f lavfi -i anoisesrc=c=pink:r=48000:a=0.02:d={d} "
            "-filter_complex [1:a]volume='1+24*between(t,{lo},{hi})':eval=frame[a] "
            "-map 0:v -map [a] "
            "-c:v libx264 -preset ultrafast -crf 32 -pix_fmt yuv420p "
            "-c:a aac -b:a 64k {out}",
            d=str(seconds),
            lo=f"{loud_from:.2f}",
            hi=f"{loud_to:.2f}",
            out=str(path),
        ),
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


@pytest.fixture
def folder(tmp_path):
    for name in ("a.mp4", "b.mp4", "c.mp4"):
        _make_video(tmp_path / name)
    return tmp_path


def test_parallel_batch_cuts_every_file(folder, tmp_path):
    out = tmp_path / "reels"
    code = main(["batch", str(folder), "-o", str(out), "--workers", "2", "-q"])
    assert code == 0
    reels = sorted(p.name for p in out.glob("*_highlights.mp4"))
    assert reels == ["a_highlights.mp4", "b_highlights.mp4", "c_highlights.mp4"]


def test_parallel_batch_still_skips_existing_reels(folder, tmp_path):
    out = tmp_path / "reels"
    main(["batch", str(folder), "-o", str(out), "--workers", "2", "-q"])
    code = main(["batch", str(folder), "-o", str(out), "--workers", "2", "-q"])
    assert code == 0
    assert len(list(out.glob("*.mp4"))) == 3, "no reel should be re-cut"


def test_single_worker_and_parallel_agree_on_outputs(folder, tmp_path):
    out1, out2 = tmp_path / "one", tmp_path / "many"
    main(["batch", str(folder), "-o", str(out1), "-q"])
    main(["batch", str(folder), "-o", str(out2), "--workers", "2", "-q"])
    assert sorted(p.name for p in out1.glob("*.mp4")) == sorted(p.name for p in out2.glob("*.mp4"))
    for path in out2.glob("*.mp4"):
        assert path.stat().st_size > 0


def test_batch_reports_a_bad_file_and_carries_on(folder, tmp_path):
    (folder / "broken.mp4").write_bytes(b"not a video at all")
    out = tmp_path / "reels"
    code = main(["batch", str(folder), "-o", str(out), "--workers", "2"])
    assert code == 1
    assert len(list(out.glob("*_highlights.mp4"))) == 3, "the good files still get cut"
