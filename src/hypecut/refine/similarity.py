"""De-duplication by what the frames actually look like.

The ``diversity`` refiner spreads clips out by *time*, which is a proxy for
sameness and a poor one. It penalises a spectacular save thirty seconds after
a goal, and lets through the fifth identical spawn-camp kill because they
happened to be minutes apart. Time is not the question. The question is
whether the viewer is about to watch the same thing twice.

**A replay is not a duplicate.** This is the design decision that shapes the
whole module. A broadcast shows the goal, then shows it again in slow motion,
then again from behind the net, and a highlight reel that keeps all three is
not making a mistake — that is how the edit is supposed to look. Cutting them
would be removing editing, not duplication.

So similarity alone decides nothing. Similarity plus *time* decides:

* **similar, close together** — the same event being shown again. Kept, and
  tagged with a shared ``moment`` id so later stages can see the group.
* **similar, far apart** — a different occurrence that happens to look
  identical: the same corridor, the same camera angle, the same menu, the
  same celebration cam. That is the repetition worth penalising.

Cost is close to nothing. The descriptor is built from the 96x54 grayscale
frames already decoded on the context, so no model, no extra decode, and no
dependency beyond the numpy that is already required.
"""

from __future__ import annotations

import numpy as np

from ..types import AnalysisContext, Candidate, VideoInfo
from .base import Refiner, register

__all__ = ["Similarity", "descriptor", "cosine"]

#: Descriptor grid. Coarse on purpose — the question is "same kind of shot",
#: not "same frame", and a fine grid answers the second one.
_ROWS, _COLS = 6, 8

#: Mean per-pixel movement, in luma levels, below which a clip is treated as
#: too still to have a shape worth comparing. Measured, not guessed: a static
#: scene under x264 sits at 0.013-0.027, a locked camera watching one small
#: moving object reaches 0.19, and ordinary footage runs 0.35 upward. 0.1 sits
#: an order of magnitude above the noise and below anything real.
_MIN_MOTION = 0.1


def descriptor(ctx: AnalysisContext, seg: Candidate) -> np.ndarray | None:
    """A signature of *where a clip moves*, pooled to a coarse grid.

    The obvious descriptor — what the clip looks like — does not work, and
    the way it fails is worth recording. Averaged frames of a locked-camera
    football match are the same green rectangle every time, so every pair of
    clips scores above 0.99 and the whole video reads as one repeated
    moment. Appearance describes the venue, not the play.

    Motion does describe the play. Averaging the frame-to-frame differences
    cancels the static background exactly and leaves a map of where things
    happened: an attack down the left wing and one down the right produce
    different maps in the same stadium, while a goal and its slow-motion
    replay produce similar ones. That is precisely the distinction this
    module needs to make.

    Returns ``None`` when a clip barely moves at all. Such a map is mostly
    sensor noise, and a normalised noise vector correlates with whatever it
    is compared against — which would invent duplicates out of stillness.
    """
    if ctx.gray is None or ctx.gray.shape[0] < 2:
        return None

    lo, hi = seg.protected()
    i0 = int(np.clip(round(lo * ctx.grid_fps), 0, ctx.gray.shape[0] - 2))
    i1 = int(np.clip(round(hi * ctx.grid_fps) + 1, i0 + 2, ctx.gray.shape[0]))
    block = ctx.gray[i0:i1].astype(np.float32)
    if block.shape[0] < 2:
        return None

    moved = np.abs(np.diff(block, axis=0)).mean(axis=0)
    h, w = moved.shape
    # Trim to a multiple of the grid so the reshape-based pooling is exact;
    # cropping a few edge pixels is harmless and avoids an interpolation
    # dependency.
    rows, cols = _ROWS * (h // _ROWS), _COLS * (w // _COLS)
    if rows == 0 or cols == 0:
        return None
    pooled = moved[:rows, :cols].reshape(_ROWS, rows // _ROWS, _COLS, cols // _COLS)
    vector = pooled.mean(axis=(1, 3)).ravel()

    # Same absolute floor as `scene_change` uses, and for the same reason: a
    # ratio has no opinion about scale, so without this, codec flicker
    # normalises into a confident-looking pattern.
    if float(vector.mean()) < _MIN_MOTION:
        return None

    vector = vector - vector.mean()
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        # Motion spread perfectly evenly over the frame — a whip pan, a
        # fade. It matches everything and nothing; refuse to compare it.
        return None
    return vector / norm


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Similarity of two descriptors, -1 to 1. Both are already unit length."""
    return float(np.dot(a, b))


@register("similarity")
class Similarity(Refiner):
    """Penalise clips that look like an earlier clip from a different moment.

    Params
    ------
    threshold: cosine similarity above which two clips count as alike
        (default 0.85). Descriptors are zero-meaned and normalised, so this
        is a shape-of-the-shot match, not a pixel match.
    replay_window: seconds within which two similar clips are treated as the
        same event being shown again — a replay or another angle — and are
        kept (default 90). Broadcast replay packages run well under a minute;
        90 gives room for a long one without reaching the next attack.
    strength: 0-1, how hard a genuine repeat is penalised (default 0.5).
    """

    description = "Penalise repeated-looking moments, while keeping replays and angles."

    def refine(self, info: VideoInfo, candidates: list[Candidate]) -> list[Candidate]:
        ctx = self.ctx
        if ctx is None or ctx.gray is None or len(candidates) < 2:
            return candidates

        threshold = float(self.params.get("threshold", 0.85))
        window = float(self.params.get("replay_window", 90.0))
        strength = float(np.clip(float(self.params.get("strength", 0.5)), 0.0, 1.0))

        ordered = sorted(candidates, key=lambda c: c.start)
        vectors = [descriptor(ctx, seg) for seg in ordered]

        # Union-find over "same moment" links, so a goal, its slow-motion
        # replay and a third angle all end up in one group even when the
        # first and last are further apart than the window.
        parent = list(range(len(ordered)))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> None:
            ri, rj = find(i), find(j)
            if ri != rj:
                parent[max(ri, rj)] = min(ri, rj)

        alike: list[tuple[int, int, float]] = []
        for i in range(len(ordered)):
            if vectors[i] is None:
                continue
            for j in range(i + 1, len(ordered)):
                if vectors[j] is None:
                    continue
                score = cosine(vectors[i], vectors[j])
                if score < threshold:
                    continue
                alike.append((i, j, score))
                if ordered[j].start - ordered[i].end <= window:
                    union(i, j)

        # Grouping has to finish before anything is penalised. A goal, its
        # replay and a third angle can chain past the window end to end — the
        # first and the last are further apart than any single hop — and
        # penalising as we went would demote the third angle for matching the
        # goal it is a replay of.
        for i, j, score in alike:
            if find(i) == find(j):
                continue
            # Different moments that still look identical: the same thing
            # happening again. Demote the weaker of the two — the better take
            # is the one worth the viewer's attention.
            weaker = ordered[i] if ordered[i].score <= ordered[j].score else ordered[j]
            excess = (score - threshold) / max(1e-6, 1.0 - threshold)
            factor = 1.0 - strength * min(1.0, excess)
            weaker.score *= max(0.0, factor)
            weaker.meta["repeat_penalty"] = round(
                max(weaker.meta.get("repeat_penalty", 0.0), 1.0 - factor), 4
            )

        moments: dict[int, int] = {}
        for index, seg in enumerate(ordered):
            root = find(index)
            seg.meta["moment"] = moments.setdefault(root, len(moments) + 1)

        return candidates
