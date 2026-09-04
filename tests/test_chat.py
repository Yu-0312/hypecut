"""The chat-log signal — message rate as a highlight track."""

from __future__ import annotations

import json

import numpy as np
import pytest

from hypecut.signals.chat import ChatRate
from hypecut.types import AnalysisContext, VideoInfo


def _ctx(tmp_path, video_name: str = "match.mp4") -> AnalysisContext:
    steps = 300  # 30 s at 10 Hz
    path = tmp_path / video_name
    path.write_bytes(b"")  # the video itself is never read
    return AnalysisContext(
        info=VideoInfo(str(path), 30.0, 30.0, 1280, 720, True),
        grid_fps=10.0,
        times=np.arange(steps) / 10.0,
    )


def _burst(seconds: float, count: int) -> list[float]:
    return [seconds + i * 0.01 for i in range(count)]


def test_jsonl_log_with_second_offsets(tmp_path):
    log = tmp_path / "match.chat.jsonl"
    lines = [f'{{"ts": {t}, "user": "u", "text": "PogChamp"}}' for t in _burst(10.0, 40)]
    log.write_text("\n".join(lines), encoding="utf-8")

    sig = ChatRate()
    values = sig.compute(_ctx(tmp_path))

    assert sig.applicable(_ctx(tmp_path))
    assert values[100:110].sum() == 40, "the burst lands in the right second"
    assert values[150:200].sum() == 0


def test_plain_text_clock_timestamps(tmp_path):
    log = tmp_path / "match.chat.txt"
    log.write_text(
        "[00:00:02] hello\n[00:00:02] hi\n[00:00:02.500] o/\nno timestamp line\n[0:12] late\n",
        encoding="utf-8",
    )

    sig = ChatRate()
    values = sig.compute(_ctx(tmp_path))

    assert values[20:26].sum() == 3
    assert values[120].sum() == 1


def test_twitchdownloader_json_format(tmp_path):
    log = tmp_path / "match.chat.json"
    comments = [{"contentOffsetSeconds": t, "message": {"body": "gg"}} for t in _burst(22.0, 25)]
    log.write_text(json.dumps({"comments": comments}), encoding="utf-8")

    sig = ChatRate()
    values = sig.compute(_ctx(tmp_path))

    assert values[220:226].sum() == 25


def test_iso_timestamps_are_relative_to_the_first_message(tmp_path):
    log = tmp_path / "match.chat.jsonl"
    log.write_text(
        '{"created_at": "2026-09-05T20:00:00Z", "text": "hi"}\n'
        '{"created_at": "2026-09-05T20:00:05Z", "text": "hello"}\n'
        '{"created_at": "2026-09-05T20:00:05Z", "text": "!!"}\n',
        encoding="utf-8",
    )

    values = ChatRate().compute(_ctx(tmp_path))

    assert values[0] == 1
    assert values[50:52].sum() == 2


def test_explicit_log_path_beats_the_sibling(tmp_path):
    sibling = tmp_path / "match.chat.jsonl"
    sibling.write_text('{"ts": 1.0}', encoding="utf-8")
    elsewhere = tmp_path / "other.jsonl"
    elsewhere.write_text('{"ts": 20.0}\n{"ts": 20.1}', encoding="utf-8")

    values = ChatRate(log=str(elsewhere)).compute(_ctx(tmp_path))

    # Only the explicit log counts; the sibling is ignored.
    assert values[10] == 0
    assert values[200:202].sum() == 2


def test_no_log_means_not_applicable(tmp_path):
    ctx = _ctx(tmp_path, video_name="lonely.mp4")
    with pytest.warns(UserWarning, match="no chat log"):
        assert not ChatRate().applicable(ctx)


def test_the_burst_stands_above_the_background(tmp_path):
    """Prominence: a real chat pile-up is an event, scattered chatter is not."""
    from hypecut.fusion import prominence

    log = tmp_path / "match.chat.jsonl"
    timestamps = [i * 0.9 for i in range(30)]  # ~1 msg / 0.9 s everywhere
    timestamps += _burst(21.0, 60)  # and a real pile-up at 21 s
    log.write_text("\n".join(f'{{"ts": {t}}}' for t in timestamps), encoding="utf-8")

    sig = ChatRate()
    ctx = _ctx(tmp_path)
    track = sig.track(ctx, weight=1.2)

    strength = prominence([track], grid_fps=10.0, smooth_seconds=1.5)
    assert strength > 1.0, "the pile-up must stand out from the background rate"


def test_cli_chat_flag_runs_the_signal(tmp_path, sample_vod, monkeypatch):
    """--chat wires the log into the analysis, and the plan shows the signal."""
    import json as js

    from hypecut.cli import main

    log = tmp_path / "chat.jsonl"
    # Steady chatter across the 45 s fixture, so analysis has something to fuse.
    log.write_text("\n".join(f'{{"ts": {t / 10.0:.1f}}}' for t in range(450)), encoding="utf-8")
    plan_path = tmp_path / "plan.json"
    code = main(["analyze", str(sample_vod), "--chat", str(log), "--json", str(plan_path), "-q"])
    assert code == 0
    plan = js.loads(plan_path.read_text())
    assert "chat_rate" in plan["signals"]
