"""Tests for the ffmpeg wrapper helpers that don't need the binary."""

from __future__ import annotations

import pytest

from hypecut.ffmpeg import FFmpegError, _first_float, _parse_rate, cmd


def test_cmd_splits_before_substituting_so_paths_with_spaces_survive():
    args = cmd("ffmpeg -i {src} -f null -", src="/tmp/my videos/clip 1.mp4")
    assert args == ["ffmpeg", "-i", "/tmp/my videos/clip 1.mp4", "-f", "null", "-"]


def test_cmd_fills_placeholders_inside_a_token():
    args = cmd("ffmpeg -vf fps={fps},scale={w}:{h} -", fps="10", w="96", h="54")
    assert args[2] == "fps=10,scale=96:54"


def test_cmd_leaves_unbraced_text_alone():
    assert cmd("ffmpeg -filter_complex [1:a]volume=6.0[a]")[-1] == "[1:a]volume=6.0[a]"


def test_cmd_raises_on_a_missing_substitution():
    with pytest.raises(KeyError):
        cmd("ffmpeg -i {src}")


@pytest.mark.parametrize(
    ("value", "expected"),
    [("30000/1001", pytest.approx(29.97, abs=0.01)), ("25", 25.0), ("0/0", 0.0), (None, 0.0)],
)
def test_parse_rate(value, expected):
    assert _parse_rate(value) == expected


def test_first_float_skips_junk_and_zeroes():
    assert _first_float(None, "N/A", "0", "12.5") == 12.5
    assert _first_float(None, "nope") == 0.0


def test_ffmpeg_error_is_a_runtime_error():
    assert issubclass(FFmpegError, RuntimeError)
