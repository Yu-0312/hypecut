"""The CMX3600 EDL — the handover to a real NLE.

An EDL that cannot be conformed against its source is worse than no EDL, so
these tests are about the two things an NLE actually reads: the timecodes and
the name of the media they refer to.
"""

from __future__ import annotations

from hypecut.render import write_edl
from hypecut.types import Candidate

NTSC_30 = 30000 / 1001  # 29.97 — every console capture, camcorder and broadcast source
NTSC_24 = 24000 / 1001  # 23.976


def _events(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln[:3].isdigit()]


def test_timecode_counts_at_the_nominal_rate_not_the_truncated_one(tmp_path):
    """29.97 material is displayed on a 30-frame clock — that is what non-drop means.

    Truncating the rate instead of rounding it put 29 frames in a timecode
    second, so the clock gained about two minutes an hour: one hour into the
    source read as 01:02:00:12. Nothing caught it because every fixture in the
    suite runs at 15 fps, where truncation and rounding agree.
    """
    one_hour_in = Candidate(3600.0, 3610.0, 0.9)
    text = write_edl([one_hour_in], tmp_path / "reel.edl", fps=NTSC_30).read_text()

    source_in = _events(text)[0].split()[4]
    hours, minutes, _, frames = (int(p) for p in source_in.split(":"))
    assert (hours, minutes) == (0, 59), f"one hour in reads as {source_in}"
    assert frames < 30, "a non-drop frame count never reaches the nominal rate"


def test_timecode_is_exact_on_an_integer_rate(tmp_path):
    text = write_edl([Candidate(3600.0, 3610.0, 0.9)], tmp_path / "r.edl", fps=30.0).read_text()
    assert _events(text)[0].split()[4] == "01:00:00:00"


def test_every_supported_rate_stays_within_a_frame_of_real_time(tmp_path):
    """Non-drop drifts by design, but only by the 1000/1001 pulldown factor."""
    for fps, nominal in ((NTSC_30, 30), (NTSC_24, 24), (30.0, 30), (25.0, 25), (60.0, 60)):
        text = write_edl([Candidate(600.0, 610.0, 0.9)], tmp_path / "x.edl", fps=fps).read_text()
        h, m, s, f = (int(p) for p in _events(text)[0].split()[4].split(":"))
        shown = h * 3600 + m * 60 + s + f / nominal
        assert abs(shown - 600.0 * min(1.0, fps / nominal)) < 1.0 / nominal


def test_the_clip_name_is_the_source_media_not_the_rendered_reel(tmp_path):
    """The first timecode pair is *source* timecode.

    An NLE relinks it against whatever `FROM CLIP NAME` names, so naming the
    reel there pointed the source timecodes at the output file, where they
    mean nothing.
    """
    text = write_edl(
        [Candidate(10.0, 20.0, 0.9)],
        tmp_path / "match_highlights.part1.edl",
        fps=30.0,
        source_name="match.mp4",
    ).read_text()

    assert "* FROM CLIP NAME: match.mp4" in text
    assert "part1" not in text


def test_record_times_run_continuously_from_zero(tmp_path):
    """The reel is the clips back to back: each record-in is the previous out."""
    segments = [Candidate(100.0, 104.0, 0.9), Candidate(200.0, 206.0, 0.9)]
    text = write_edl(segments, tmp_path / "reel.edl", fps=30.0).read_text()

    rows = [ln.split() for ln in _events(text)]
    assert [r[6] for r in rows] == ["00:00:00:00", "00:00:04:00"]
    assert [r[7] for r in rows] == ["00:00:04:00", "00:00:10:00"]
