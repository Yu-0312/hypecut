"""Core data types shared across the pipeline.

Everything that crosses a module boundary in HypeCut is one of these.
Keeping them dependency-free (stdlib + numpy) is deliberate: signal plugins
and refiners can be written without importing the rest of the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["VideoInfo", "AnalysisContext", "SignalTrack", "Candidate", "HighlightPlan"]


@dataclass(frozen=True)
class VideoInfo:
    """Container-level facts about the input, from ffprobe."""

    path: str
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 16 / 9


@dataclass
class AnalysisContext:
    """Everything a signal needs, decoded exactly once.

    The whole pipeline works on a single uniform time grid running at
    ``grid_fps`` (default 10 Hz). Every signal returns one value per grid
    step, which makes fusion a plain weighted sum instead of a resampling
    problem.
    """

    info: VideoInfo
    grid_fps: float
    times: np.ndarray  # (T,) seconds, the analysis grid
    gray: np.ndarray | None = None  # (T, h, w) uint8 downscaled luma frames
    audio: np.ndarray | None = None  # (N,) mono float32 in [-1, 1]
    audio_sr: int = 16_000
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return int(self.times.shape[0])

    def audio_frames(self) -> np.ndarray:
        """Audio reshaped to ``(T, samples_per_grid_step)``, zero-padded."""
        if self.audio is None:
            return np.zeros((self.n, 1), dtype=np.float32)
        hop = max(1, int(round(self.audio_sr / self.grid_fps)))
        need = self.n * hop
        buf = self.audio
        if buf.shape[0] < need:
            buf = np.pad(buf, (0, need - buf.shape[0]))
        return buf[:need].reshape(self.n, hop)


@dataclass
class SignalTrack:
    """One signal's output over the analysis grid."""

    name: str
    values: np.ndarray  # (T,) raw, un-normalised
    weight: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    """A proposed highlight segment."""

    start: float
    end: float
    score: float
    reasons: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def overlaps(self, other: Candidate) -> bool:
        return self.start < other.end and other.start < self.end

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "score": round(self.score, 4),
            "reasons": {k: round(v, 4) for k, v in self.reasons.items()},
            "meta": self.meta,
        }


@dataclass
class HighlightPlan:
    """The full result of analysis: what to cut, and why."""

    info: VideoInfo
    segments: list[Candidate]
    curve: np.ndarray  # (T,) fused excitement score
    times: np.ndarray  # (T,)
    tracks: list[SignalTrack] = field(default_factory=list)

    @property
    def total_duration(self) -> float:
        return sum(s.duration for s in self.segments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.info.path,
            "source_duration": round(self.info.duration, 3),
            "reel_duration": round(self.total_duration, 3),
            "segments": [s.to_dict() for s in self.segments],
            "signals": [t.name for t in self.tracks],
        }
