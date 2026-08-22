"""Measuring whether a cut is any good.

Until this existed, every default in HypeCut was a reasoned guess checked
against footage built to have the property being checked for. That proves the
code does what was intended; it says nothing about whether the intention
matches real video. A profile PR could not be reviewed on evidence, and a
change to ``percentile`` could not be shown to help.

Three decisions shape this module, and each of them is a judgement call worth
disagreeing with:

**A hit means the clip contains the moment, not that the edges line up.**
Overlap-based scores (IoU and friends) conflate two different questions —
*did you find it* and *did you frame it well* — and the second one is
snapping and trimming's job, measured separately below as coverage. So a clip
hits a labelled highlight when it contains that highlight's midpoint.

**Labels ship without video.** A benchmark that needs a corpus of gameplay
and broadcast footage cannot be distributed, so a labels file references a
video by path and carries only timestamps. Anyone can build one from footage
they already have; nobody has to host anything.

**One annotator, and say so.** Two people mark different highlights in the
same match, and pretending otherwise would give the numbers a false
authority. A labels file records who made it. Comparing two profiles against
one annotator is still a valid experiment; comparing scores across annotators
is not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["Highlight", "Labels", "Score", "load_labels", "write_labels", "score_plan"]


@dataclass(frozen=True)
class Highlight:
    """One moment a human said was worth keeping."""

    start: float
    end: float
    label: str = ""

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class Labels:
    """A hand-marked answer key for one video."""

    video: str
    highlights: list[Highlight]
    annotator: str = ""
    notes: str = ""
    profile: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "annotator": self.annotator,
            "notes": self.notes,
            "profile": self.profile,
            "highlights": [
                {
                    "start": round(h.start, 3),
                    "end": round(h.end, 3),
                    **({"label": h.label} if h.label else {}),
                }
                for h in self.highlights
            ],
        }


@dataclass
class Score:
    """How one cut list did against one answer key."""

    video: str
    found: int = 0
    missed: list[Highlight] = field(default_factory=list)
    spurious: list[tuple[float, float]] = field(default_factory=list)
    clips: int = 0
    coverage: float = 0.0

    @property
    def total(self) -> int:
        return self.found + len(self.missed)

    @property
    def recall(self) -> float:
        return self.found / self.total if self.total else 0.0

    @property
    def precision(self) -> float:
        return (self.clips - len(self.spurious)) / self.clips if self.clips else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "video": self.video,
            "clips": self.clips,
            "labelled": self.total,
            "found": self.found,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "coverage": round(self.coverage, 4),
            "missed": [
                {"start": round(h.start, 3), "end": round(h.end, 3), "label": h.label}
                for h in self.missed
            ],
            "spurious": [{"start": round(a, 3), "end": round(b, 3)} for a, b in self.spurious],
        }


def load_labels(path: str | Path) -> Labels:
    """Read a labels file (YAML or JSON), rejecting anything unusable."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        payload = yaml.safe_load(text) or {}
    else:
        payload = json.loads(text)

    video = str(payload.get("video") or "")
    if not video:
        raise ValueError(f"{path}: no `video` field.")

    raw = payload.get("highlights") or []
    highlights: list[Highlight] = []
    for position, item in enumerate(raw, start=1):
        # A draft written by `hypecut label` marks every proposal `keep: null`.
        # Anything still undecided is not an answer, so it is skipped rather
        # than silently counted as either a highlight or a rejection.
        keep = item.get("keep", True)
        if keep is None or keep is False:
            continue
        try:
            start, end = float(item["start"]), float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{path}: highlight {position} needs numeric start and end.") from exc
        if end <= start:
            raise ValueError(f"{path}: highlight {position} ends before it starts.")
        highlights.append(Highlight(start, end, str(item.get("label") or "")))

    if not highlights:
        raise ValueError(
            f"{path}: no highlights marked. A draft needs `keep: true` on the ones "
            "that are real, and entries added for anything the detector missed."
        )

    return Labels(
        video=video,
        highlights=sorted(highlights, key=lambda h: h.start),
        annotator=str(payload.get("annotator") or ""),
        notes=str(payload.get("notes") or ""),
        profile=str(payload.get("profile") or ""),
    )


def write_labels(
    labels: Labels, dest: str | Path, *, draft: list[dict[str, Any]] | None = None
) -> Path:
    """Write a labels file. ``draft`` entries keep their ``keep`` markers."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = labels.to_dict()
    if draft is not None:
        payload["highlights"] = draft

    if dest.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        body = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        dest.write_text(_LABELS_HEADER + body, encoding="utf-8")
    else:
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return dest


def score_plan(labels: Labels, clips: list[tuple[float, float]]) -> Score:
    """Compare a cut list against an answer key.

    ``coverage`` is reported separately from recall on purpose: it answers
    "given that the moment was found, how much of it made the clip", which is
    a question about edge placement rather than detection. A profile can score
    perfect recall and poor coverage — that means the detector is right and
    the rolls are too tight, which is a completely different fix from a
    detector that misses things.
    """
    result = Score(video=labels.video, clips=len(clips))
    hit_clips: set[int] = set()
    covered: list[float] = []

    for highlight in labels.highlights:
        matches = [
            index for index, (start, end) in enumerate(clips) if start <= highlight.midpoint <= end
        ]
        if matches:
            result.found += 1
            hit_clips.update(matches)
            covered.append(_covered_fraction(highlight, [clips[i] for i in matches]))
        else:
            result.missed.append(highlight)

    result.spurious = [clip for index, clip in enumerate(clips) if index not in hit_clips]
    result.coverage = sum(covered) / len(covered) if covered else 0.0
    return result


def _covered_fraction(highlight: Highlight, clips: list[tuple[float, float]]) -> float:
    """How much of a labelled span the clips actually contain, 0-1."""
    if highlight.duration <= 0:
        return 1.0
    overlap = sum(
        max(0.0, min(end, highlight.end) - max(start, highlight.start)) for start, end in clips
    )
    return min(1.0, overlap / highlight.duration)


_LABELS_HEADER = """\
# HypeCut labels — a hand-marked answer key for one video.
#
# `hypecut label` writes this as a draft: every proposal starts with
# `keep: null`, which counts as neither a highlight nor a rejection. Go
# through them (the contact sheet next to this file has one tile per entry),
# set `keep: true` on the real ones and `keep: false` on the rest, and add
# entries by hand for anything the detector missed entirely — those matter
# most, because they are the failures the score would otherwise never see.
#
# Then: hypecut eval this-file.yaml --profile configs/whatever.yaml
"""
