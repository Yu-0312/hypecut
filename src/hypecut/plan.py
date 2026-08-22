"""Read a cut list back in, so a render can be driven by an edited plan.

The sidecar HypeCut writes is not just a receipt — it is a complete,
re-renderable description of a cut: the source, the segments with their
framing decisions, and the exact config that produced them. Being able to
load one back closes the loop that ``analyze`` / ``render_plan`` opened:
propose a cut, let a human or an agent change it, render what they approved.

Everything in a loaded plan is treated as untrusted input. It may have been
hand-edited, produced by a model, or arrived over HTTP; the numbers in it end
up as ffmpeg seek arguments either way, so they are clamped and checked here
rather than anywhere further in.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .ffmpeg import probe
from .types import Candidate, HighlightPlan

__all__ = ["load_plan", "plan_from_dict"]


def load_plan(
    path: str | Path, *, source: str | Path | None = None
) -> tuple[HighlightPlan, Config]:
    """Load a ``.hypecut.json`` sidecar (or any edited copy of one)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return plan_from_dict(payload, source=source)


def plan_from_dict(
    payload: dict[str, Any], *, source: str | Path | None = None
) -> tuple[HighlightPlan, Config]:
    """Rebuild a plan and its config from the sidecar's dict form.

    ``source`` overrides the recorded path, which matters more often than it
    sounds: a cut list is portable, and the video it describes routinely lives
    somewhere else by the time anyone re-renders it.
    """
    src = str(source or payload.get("source") or "")
    if not src:
        raise ValueError("Plan has no source video; pass source= explicitly.")
    if not Path(src).exists():
        raise FileNotFoundError(f"Source video not found: {src}")

    info = probe(src)
    cfg = Config().merged(payload["config"]) if payload.get("config") else Config()

    raw = payload.get("segments")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Plan contains no segments.")

    segments: list[Candidate] = []
    for position, item in enumerate(raw, start=1):
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Segment {position} needs numeric start and end.") from exc
        if not (np.isfinite(start) and np.isfinite(end)):
            raise ValueError(f"Segment {position} has a non-finite time.")

        start = max(0.0, min(start, info.duration))
        end = max(0.0, min(end, info.duration))
        if end - start < 0.05:
            raise ValueError(
                f"Segment {position} is {end - start:.3f}s long after clamping to the "
                f"source ({info.duration:.1f}s) — check the times."
            )

        segments.append(
            Candidate(
                start=start,
                end=end,
                score=float(item.get("score", 0.0)),
                reasons={str(k): float(v) for k, v in (item.get("reasons") or {}).items()},
                meta=dict(item.get("meta") or {}),
            )
        )

    segments.sort(key=lambda c: c.start)
    plan = HighlightPlan(
        info=info,
        segments=segments,
        # The curve is analysis output, not part of the cut. Rendering never
        # reads it, and a sidecar carrying 36,000 floats per hour would be a
        # bad trade for a file people are meant to open and edit by hand.
        curve=np.zeros(0),
        times=np.zeros(0),
    )
    return plan, cfg
