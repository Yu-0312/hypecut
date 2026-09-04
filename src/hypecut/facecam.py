"""Find the facecam box from the footage itself.

``reframe.facecam`` is the one setting a user has to supply by hand, and it
is the one thing a layout-dependent feature (`stack`, `react_to_facecam`)
cannot work without. This module removes that setting: it locates the
webcam overlay from the same tiny grayscale frames every other stage reads,
so `facecam: auto` costs no decode, no model and no dependency.

The tell is *persistence*, and it is worth spelling out why. A webcam is a
small rectangle that is always slightly alive — sensor noise, compression
shimmer, a person who never holds still. Almost nothing else on screen
shares that profile:

* a kill feed or a scoreboard changes **hard** but **rarely** — high
  per-event energy, low duty cycle;
* gameplay changes **often** but over a **large, shifting** area — no fixed
  compact box;
* a static HUD changes **never**.

So the frame is divided into cells, each cell is scored by how often it
moved (duty cycle) times how much it moved, the best rectangle of live
cells is grown from the strongest seed, and the liveness has to hold across
both halves of the timeline — a region busy during one fight does not
qualify. A corner prior finishes it, because streamers put webcams in
corners and almost nowhere else.

This is deliberately not a face detector. It finds the *overlay box*, which
is what the crop actually needs — a face detector would find the face,
which the box only has to contain.
"""

from __future__ import annotations

import numpy as np

from .types import AnalysisContext

__all__ = ["locate_facecam"]

#: Side of a cell in analysis-frame pixels. Six pixels at the 96x54 analysis
#: resolution is small enough to outline a corner webcam and large enough
#: that single-pixel noise cannot light a cell up on its own.
CELL = 6

#: A cell counts as "moved" in a frame when its mean change clears this many
#: luma levels. Per-*cell* means, not per-pixel: averaging first is what
#: makes compression flicker and sensor noise fall below the bar while a
#: genuinely busy little overlay clears it every frame.
DUTY_BAR = 0.6

#: Cells below this duty cycle (fraction of frames with visible change) are
#: never webcam candidates — a kill feed cannot survive it, a webcam can.
MIN_DUTY = 0.35

#: Sanity bounds on the box a webcam can occupy, as a fraction of the frame.
MIN_SPAN = 0.06
MAX_SPAN = 0.55

#: Below this confidence the answer is "not found" rather than a guess.
MIN_CONFIDENCE = 0.25


def locate_facecam(ctx: AnalysisContext) -> dict[str, object] | None:
    """Locate the webcam overlay, or return ``None`` if nothing qualifies.

    Returns ``{"box": [x0, y0, x1, y1], "confidence": float}`` with the box
    in 0-1 coordinates, ready for ``reframe.facecam``.
    """
    if ctx.gray is None or ctx.gray.shape[0] < 30:
        return None

    f = ctx.gray.astype(np.float32, copy=False)
    delta = np.abs(np.diff(f, axis=0))
    h, w = delta.shape[1], delta.shape[2]

    cols, rows = max(1, w // CELL), max(1, h // CELL)
    trimmed = delta[:, : rows * CELL, : cols * CELL].reshape(delta.shape[0], rows, CELL, cols, CELL)
    # Mean change per cell per frame — (T-1, rows, cols). Averaging the cell
    # before thresholding is the noise immunity; see DUTY_BAR.
    activity = trimmed.mean(axis=(2, 4))

    energy = activity.mean(axis=0)  # how much, overall
    duty = (activity > DUTY_BAR).mean(axis=0)  # how often
    if not (duty >= MIN_DUTY).any():
        return None

    half = activity.shape[0] // 2
    duty_first = (activity[:half] > DUTY_BAR).mean(axis=0)
    duty_second = (activity[half:] > DUTY_BAR).mean(axis=0)

    best = _best_box(energy, duty, duty_first, duty_second, rows, cols)
    if best is None:
        return None
    score, (r0, c0), (r1, c1) = best
    confidence = float(np.clip(score, 0.0, 1.0))
    if confidence < MIN_CONFIDENCE:
        return None

    box = [
        round(c0 * CELL / w, 4),
        round(r0 * CELL / h, 4),
        round(min(cols, c1) * CELL / w, 4),
        round(min(rows, r1) * CELL / h, 4),
    ]
    return {"box": box, "confidence": round(confidence, 3)}


# --------------------------------------------------------------------- private


def _best_box(
    energy: np.ndarray,
    duty: np.ndarray,
    duty_first: np.ndarray,
    duty_second: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[float, tuple[int, int], tuple[int, int]] | None:
    """Score rectangles of live cells; return the best one found.

    Rather than search every rectangle (quadratic in cells), grow candidates
    from the strongest seed: expand while the ring being added is at least
    half as live as the core, then score the result. Webcams are rectangles;
    expansion that stops at a falloff traces one.
    """
    live = duty >= MIN_DUTY
    strength = duty * energy
    order = np.argsort(strength, axis=None)[::-1]

    best: tuple[float, tuple[int, int], tuple[int, int]] | None = None
    for seed in order[: 2 * rows * cols // 5 + 4]:
        r, c = divmod(int(seed), cols)
        if not live[r, c]:
            break  # the rest are weaker seeds on dead cells
        r0, r1, c0, c1 = r, r + 1, c, c + 1
        core = float(duty[r, c])
        changed = True
        while changed and (r1 - r0) < rows and (c1 - c0) < cols:
            changed = False
            ring: list[tuple[int, int]] = []
            if r0 > 0:
                ring += [(r0 - 1, cc) for cc in range(c0, c1)]
            if r1 < rows:
                ring += [(r1, cc) for cc in range(c0, c1)]
            if c0 > 0:
                ring += [(rr, c0 - 1) for rr in range(r0, r1)]
            if c1 < cols:
                ring += [(rr, c1) for rr in range(r0, r1)]
            mean_duty = float(np.mean([duty[rr, cc] for rr, cc in ring])) if ring else 0.0
            if mean_duty >= MIN_DUTY and mean_duty >= 0.5 * core:
                r0 = min(r0, min(rr for rr, _ in ring))
                r1 = max(r1, max(rr for rr, _ in ring) + 1)
                c0 = min(c0, min(cc for _, cc in ring))
                c1 = max(c1, max(cc for _, cc in ring) + 1)
                changed = True

        score = _box_score(energy, duty, duty_first, duty_second, r0, r1, c0, c1, rows, cols)
        if best is None or score > best[0]:
            best = (score, (r0, c0), (r1, c1))
    return best


def _box_score(
    energy: np.ndarray,
    duty: np.ndarray,
    duty_first: np.ndarray,
    duty_second: np.ndarray,
    r0: int,
    r1: int,
    c0: int,
    c1: int,
    rows: int,
    cols: int,
) -> float:
    """How webcam-like one candidate box is, on a 0-1 scale.

    Five equally weighted judgements, each doing real work:

    * *liveness* — the box is actually busy (duty);
    * *solidity* — the box is full of live cells, not a bounding rectangle
      around scattered ones (a webcam is solid; two unrelated HUD elements
      joined by their bbox are not);
    * *persistence* — the liveness holds in both halves of the timeline, so
      a region that was busy only during one fight does not qualify;
    * *size prior* — webcams occupy a sane fraction of the frame;
    * *corner prior* — webcams sit in corners.
    """
    box_duty = duty[r0:r1, c0:c1]
    box_energy = energy[r0:r1, c0:c1]

    liveness = float(np.average(np.clip(box_duty, 0, 1), weights=box_energy + 1e-6))
    solidity = float((box_duty >= MIN_DUTY).mean())
    first = float(duty_first[r0:r1, c0:c1].mean())
    second = float(duty_second[r0:r1, c0:c1].mean())
    persistence = min(first, second) / max(first, second, 1e-6)

    wf = (c1 - c0) / cols
    hf = (r1 - r0) / rows
    span_ok = MIN_SPAN <= wf <= MAX_SPAN and MIN_SPAN <= hf <= MAX_SPAN
    size = 1.0 if span_ok else 0.3

    corner = _corner_prior(r0, r1, c0, c1, rows, cols)

    return 0.28 * liveness + 0.22 * solidity + 0.20 * persistence + 0.15 * size + 0.15 * corner


def _corner_prior(r0: int, r1: int, c0: int, c1: int, rows: int, cols: int) -> float:
    """Webcams sit in corners; a full 0-1 score needs the box to touch one."""
    touches_vertical = r0 == 0 or r1 == rows
    touches_horizontal = c0 == 0 or c1 == cols
    if touches_vertical and touches_horizontal:
        return 1.0
    if touches_vertical or touches_horizontal:
        return 0.6
    return 0.2
