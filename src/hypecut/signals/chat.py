"""The chat-log signal.

Twitch chat message rate is close to a free human-labelled highlight track:
thousands of viewers spam exactly when something happens, and the timestamps
arrive in a log nobody had to annotate. This signal reads that log and turns
it into one more detector — no model, no decode, one file read.

The log is found from the video's own name unless configured: a recording
``match.mp4`` looks for ``match.chat.jsonl``, ``match.chat.json``,
``match.chat.txt``, ``match.chat.log`` and ``match.jsonl`` next to it. The
formats below are the ones real logs come in:

* **JSON lines** — one JSON object per message, timestamp under any of
  ``contentOffsetSeconds``, ``content_offset_seconds``, ``offset``, ``ts``,
  ``t``, ``time``, ``seconds`` (seconds, ``HH:MM:SS``, or an ISO timestamp).
* **TwitchDownloader JSON** — the whole-file ``{"comments": [...]}`` format
  produced by TwitchDownloaderCLI.
* **Plain text** — lines beginning ``[HH:MM:SS]``, ``HH:MM:SS`` or ``MM:SS``.

Absolute ISO timestamps are measured relative to the *first message in the
log*, which assumes the log starts with the VOD — the common case, and
worth knowing when it is not.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np

from ..types import AnalysisContext
from .base import Signal, register

__all__ = ["ChatRate"]

#: Suffixes probed next to the video when ``params["log"]`` is empty.
SIBLING_SUFFIXES = (".chat.jsonl", ".chat.json", ".chat.txt", ".chat.log", ".jsonl")

_TIME_KEYS = (
    "contentOffsetSeconds",
    "content_offset_seconds",
    "offset",
    "ts",
    "t",
    "time",
    "seconds",
    "createdAt",
    "created_at",
)

_CLOCK = re.compile(r"^\[?(\d{1,2}:)?(\d{1,2}):(\d{2})(?:\.(\d{1,3}))?\]?")


@register("chat_rate")
class ChatRate(Signal):
    """Message rate from a Twitch-style chat log.

    Params
    ------
    log: path to the chat log. Empty means "look next to the video" — see
        the module docstring for the names tried.
    smooth_seconds: extra smoothing of the rate, on top of what fusion does
        anyway (default 0 — off).
    """

    description = (
        "Chat message rate from a log file (params: log) — the audience "
        "labelling the highlights for you."
    )
    # Messages per grid step. A burst that never sustains two messages a
    # second is chatter, not a moment.
    noise_floor = 0.2

    def __init__(self, **params) -> None:
        super().__init__(**params)
        self._timestamps: list[float] | None = None

    def applicable(self, ctx: AnalysisContext) -> bool:
        if self._resolve_log(ctx) is None:
            import warnings

            warnings.warn(
                f"chat_rate is enabled but no chat log was found for {Path(ctx.info.path).name} "
                "(looked for params: log, then sibling files " + ", ".join(SIBLING_SUFFIXES) + ")",
                stacklevel=2,
            )
            return False
        return True

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        timestamps = self._timestamps or self._parse(ctx)
        rate = np.zeros(ctx.n)
        if timestamps:
            idx = np.clip((np.asarray(timestamps) * ctx.grid_fps).astype(np.int64), 0, ctx.n - 1)
            np.add.at(rate, idx, 1.0)
        seconds = float(self.params.get("smooth_seconds", 0.0))
        if seconds > 0:
            from ..fusion import smooth

            rate = smooth(rate, max(1, int(round(seconds * ctx.grid_fps))))
        return rate

    # ------------------------------------------------------------------ parse

    def _resolve_log(self, ctx: AnalysisContext) -> Path | None:
        explicit = str(self.params.get("log") or "").strip()
        if explicit:
            path = Path(explicit)
            return path if path.exists() else None
        stem = Path(ctx.info.path)
        for suffix in SIBLING_SUFFIXES:
            candidate = stem.with_name(stem.stem + suffix)
            if candidate.exists():
                return candidate
        return None

    def _parse(self, ctx: AnalysisContext) -> list[float]:
        path = self._resolve_log(ctx)
        if path is None:
            self._timestamps = []
            return []

        text = path.read_text(encoding="utf-8", errors="replace")
        # Whole-file JSON first: a parse of the full text only succeeds for
        # that format, since JSONL and plain logs are not one JSON value.
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if data is not None:
            entries = data if isinstance(data, list) else _comments_of(data)
        else:
            entries = _entries_from_lines(text)
        self._timestamps = _seconds(entries)
        return self._timestamps


def _entries_from_lines(text: str) -> list[object]:
    """Time candidates from JSONL or plain-text logs."""
    out: list[object] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                out.append(record)
                continue
        clock = _CLOCK.match(line)
        if clock:
            out.append(clock.group(0).strip("[]"))
    return out


def _comments_of(data: dict) -> list[object]:
    """The message list of a whole-file log, e.g. TwitchDownloader's."""
    comments = data.get("comments") if isinstance(data, dict) else None
    return comments if isinstance(comments, list) else []


def _seconds(entries: list[object]) -> list[float]:
    """Normalise every entry to seconds from the start of the video."""
    values: list[float] = []
    base: datetime | None = None
    for entry in entries:
        raw = _raw_time(entry)
        if raw is None:
            continue
        if isinstance(raw, int | float):
            values.append(float(raw))
            continue
        clock = _CLOCK.match(str(raw))
        if clock:
            values.append(_clock_seconds(clock))
            continue
        moment = _iso(str(raw))
        if moment is not None:
            if base is None:
                base = moment
            values.append((moment - base).total_seconds())
    return [v for v in values if v >= 0]


def _raw_time(entry: object) -> object:
    if isinstance(entry, str):
        clock = _CLOCK.match(entry.strip())
        return clock.group(0).strip("[]") if clock else None
    if not isinstance(entry, dict):
        return None
    for key in _TIME_KEYS:
        if key in entry:
            return entry[key]
    return None


def _clock_seconds(match: re.Match) -> float:
    hours = float(match.group(1)[:-1]) if match.group(1) else 0.0
    minutes = float(match.group(2))
    seconds = float(match.group(3))
    fraction = float(f"0.{match.group(4)}") if match.group(4) else 0.0
    return hours * 3600 + minutes * 60 + seconds + fraction


def _iso(text: str) -> datetime | None:
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
