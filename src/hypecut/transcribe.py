"""Word timings from speech recognition.

Silence-aware trimming finds pauses from loudness, which answers "is
anyone making a sound" — not "are they between words". A slow speaker with
no real gaps between sentences gets cut mid-word all the same, because the
level never drops far enough for long enough.

When the ``[asr]`` extra is installed, this module supplies the *known*
pauses instead: every gap between one transcribed word and the next is a
place an edge can land without clipping anything. It is opt-in because it
transcribes the whole video once — a cost only worth paying when the
loudness heuristic is visibly choosing wrong.
"""

from __future__ import annotations

__all__ = ["word_gaps", "TranscribeError"]


class TranscribeError(RuntimeError):
    """Raised when transcription produced nothing usable."""


def word_gaps(
    path: str, *, model_size: str = "base", duration: float | None = None, min_gap: float = 0.05
) -> list[tuple[float, float]]:
    """``(start, end)`` of every silence between consecutive words.

    The spans cover lead-in before the first word and the tail after the
    last one, so an edge has somewhere to go at the very start and very end
    of the recording too. ``duration`` closes the final gap; without it the
    tail is simply absent and the out-point falls back to whatever else it
    can find.

    ``min_gap`` here only discards sub-frame slivers; the meaningful bar is
    the caller's ``min_silence``, applied later when a pause is chosen.
    """
    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"word-boundary trimming needs `pip install hypecut[asr]` (missing {exc.name})"
        ) from exc

    model = WhisperModel(model_size, compute_type="int8")
    segments, _ = model.transcribe(str(path), word_timestamps=True, vad_filter=True, beam_size=1)

    words: list[tuple[float, float]] = []
    for seg in segments:
        for word in seg.words or ():
            start, end = float(word.start), float(word.end)
            if end > start:
                words.append((start, end))
    words.sort()

    gaps: list[tuple[float, float]] = []
    if not words:
        return gaps

    if words[0][0] >= min_gap:
        gaps.append((0.0, words[0][0]))
    for (_, prev_end), (next_start, _) in zip(words, words[1:], strict=False):
        if next_start - prev_end >= min_gap:
            gaps.append((prev_end, next_start))
    if duration and duration - words[-1][1] >= min_gap:
        gaps.append((words[-1][1], float(duration)))
    return gaps
