"""Word-boundary trimming — pauses from transcribed word timings."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from hypecut.config import SegmentConfig
from hypecut.trimming import trim_segments
from hypecut.types import AnalysisContext, Candidate, VideoInfo

SR = 16_000
GRID = 10.0
DURATION = 10.0

# Words the fake model "hears". The gaps between them are the answer key:
# 0.0-1.0 lead-in, 2.4-6.0 mid-sentence pause, 8.2-10.0 tail.
WORDS = [(1.0, 1.5), (1.6, 2.4), (6.0, 6.8), (7.0, 8.2)]


def _install_fake_whisper(monkeypatch: pytest.MonkeyPatch, count: dict | None = None) -> None:
    """Stand in for faster_whisper with fixed word timings."""

    class FakeWord:
        def __init__(self, start: float, end: float) -> None:
            self.start, self.end = start, end

    class FakeSegment:
        def __init__(self, words) -> None:
            self.words = words

    class FakeModel:
        def __init__(self, model_size: str, **kwargs) -> None:
            assert model_size == "base"
            if count is not None:
                count["n"] += 1

        def transcribe(self, path, **kwargs):
            assert kwargs.get("word_timestamps") is True
            return iter([FakeSegment([FakeWord(s, e) for s, e in WORDS])]), None

    fake = types.ModuleType("faster_whisper")
    fake.WhisperModel = FakeModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)


def _continuous_sound_ctx() -> AnalysisContext:
    """Uniform sound — the case the loudness heuristic gets wrong.

    A level-based pause finder finds nothing here (no contrast between
    speech and pause), which is exactly the slow-speaker failure word
    timings exist to fix.
    """
    rng = np.random.default_rng(2)
    audio = rng.normal(0, 0.12, int(DURATION * SR)).astype(np.float32)
    steps = int(DURATION * GRID)
    return AnalysisContext(
        info=VideoInfo("fake.mp4", DURATION, 30.0, 1280, 720, True),
        grid_fps=GRID,
        times=np.arange(steps) / GRID,
        gray=np.full((steps, 8, 12), 40, dtype=np.uint8),
        audio=audio,
        audio_sr=SR,
    )


def _layout_ctx(layout: list[tuple[str, float]]) -> AnalysisContext:
    rng = np.random.default_rng(0)
    chunks = []
    for kind, seconds in layout:
        n = int(seconds * SR)
        chunks.append(rng.normal(0, 0.15, n).astype(np.float32) if kind == "sound" else np.zeros(n))
    audio = np.concatenate(chunks)
    duration = audio.size / SR
    steps = int(duration * GRID)
    return AnalysisContext(
        info=VideoInfo("fake.mp4", duration, 30.0, 1280, 720, True),
        grid_fps=GRID,
        times=np.arange(steps) / GRID,
        gray=np.full((steps, 8, 12), 40, dtype=np.uint8),
        audio=audio,
        audio_sr=SR,
    )


def test_word_gaps_lists_every_gap_including_lead_in_and_tail(monkeypatch):
    from hypecut.transcribe import word_gaps

    _install_fake_whisper(monkeypatch)
    assert word_gaps("fake.mp4", duration=DURATION) == [
        (0.0, 1.0),
        (1.5, 1.6),
        (2.4, 6.0),
        (6.8, 7.0),
        (8.2, DURATION),
    ]


def test_word_gaps_without_duration_has_no_tail(monkeypatch):
    from hypecut.transcribe import word_gaps

    _install_fake_whisper(monkeypatch)
    assert word_gaps("fake.mp4", duration=None) == [(0.0, 1.0), (1.5, 1.6), (2.4, 6.0), (6.8, 7.0)]


def test_trim_lands_edges_on_word_gaps(monkeypatch):
    """The loudness path finds nothing here; word timings still place edges."""
    _install_fake_whisper(monkeypatch)
    ctx = _continuous_sound_ctx()
    cfg = SegmentConfig(trim_to_silence=True, use_asr_words=True, min_duration=2.0, silence_pad=0.1)
    seg = Candidate(1.2, 7.5, 0.9, meta={"peak_time": 6.5})

    trim_segments(ctx, [seg], cfg)

    # The in-point closes on the lead-in gap's end (speech starts at 1.0);
    # the mid-sentence gap at 2.4-6.0 is out of its travel window. The
    # out-point lands where the last word ends, in the tail.
    assert seg.start == pytest.approx(1.0 - 0.1, abs=0.15)
    assert seg.end == pytest.approx(8.2 + 0.1, abs=0.15)
    assert set(seg.meta["trimmed"]) == {"start", "end"}
    assert seg.meta["ends_in_silence"] is True


def test_trim_transcribes_once_and_caches(monkeypatch):
    _install_fake_whisper(monkeypatch, count := {"n": 0})
    ctx = _continuous_sound_ctx()
    cfg = SegmentConfig(trim_to_silence=True, use_asr_words=True, min_duration=2.0)

    trim_segments(ctx, [Candidate(1.2, 7.5, 0.9, meta={"peak_time": 6.5})], cfg)
    trim_segments(ctx, [Candidate(3.0, 9.0, 0.9, meta={"peak_time": 6.5})], cfg)

    assert count["n"] == 1, "transcription must run once per video, not per clip"


def test_trim_falls_back_to_loudness_without_the_extra(monkeypatch):
    """No faster_whisper: warn, and behave exactly as before."""
    monkeypatch.setitem(sys.modules, "faster_whisper", None)
    ctx = _layout_ctx([("sound", 5.0), ("gap", 2.0), ("sound", 6.0)])
    cfg = SegmentConfig(trim_to_silence=True, use_asr_words=True, min_duration=4.0)
    seg = Candidate(6.0, 11.5, 0.9, meta={"peak_time": 9.0})

    with pytest.warns(UserWarning, match="use_asr_words"):
        trim_segments(ctx, [seg], cfg)

    # The loudness path handled it: the start moved to the 5-7 s gap's end.
    assert "trimmed" in seg.meta
    assert seg.start == pytest.approx(7.0 - 0.12, abs=0.15)


def test_trim_without_asr_words_is_unchanged(monkeypatch):
    """The default path never touches transcription."""
    _install_fake_whisper(monkeypatch, count := {"n": 0})
    ctx = _layout_ctx([("sound", 5.0), ("gap", 2.0), ("sound", 6.0)])
    cfg = SegmentConfig(trim_to_silence=True, min_duration=4.0)
    seg = Candidate(6.0, 11.5, 0.9, meta={"peak_time": 9.0})

    trim_segments(ctx, [seg], cfg)
    assert count["n"] == 0
    assert "trimmed" in seg.meta
