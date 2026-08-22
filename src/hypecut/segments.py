"""Curve -> clip list.

Three things have to be true of a good highlight reel and none of them
fall out of thresholding alone:

* clips start *before* the spike (you need the wind-up to read the payoff),
* clips do not overlap or stutter into each other,
* the reel fits a budget — a 3-hour VOD still has to become ~2 minutes.

This module handles all three, in that order.
"""

from __future__ import annotations

import numpy as np

from .config import SegmentConfig
from .fusion import explain
from .types import Candidate, SignalTrack

__all__ = ["find_regions", "build_candidates", "merge", "select", "out_point_floor"]


def out_point_floor(seg: Candidate, guard: float) -> float:
    """Earliest place a clip's out-point may be moved to.

    Normally that is the end of the event: an edge must not cut the exchange
    short. The exception is a clip the length budget already truncated
    mid-event — there the event is being lost regardless of what any edge does,
    so refusing to move would only mean ending on an arbitrary frame instead of
    on a cut. In that case the only remaining requirement is to keep ``guard``
    seconds of the opening.
    """
    lo, hi = seg.protected()
    if hi <= seg.end:
        return max(hi, lo + guard)
    return lo + guard


def find_regions(curve: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Contiguous index ranges where ``curve >= threshold`` (end-exclusive)."""
    if curve.size == 0:
        return []
    mask = curve >= threshold
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist(), strict=False))


def build_candidates(
    curve: np.ndarray,
    times: np.ndarray,
    cfg: SegmentConfig,
    *,
    grid_fps: float,
    duration: float,
    tracks: list[SignalTrack] | None = None,
) -> list[Candidate]:
    """Expand above-threshold regions into padded, scored candidates."""
    if curve.size == 0:
        return []

    threshold = float(np.percentile(curve, np.clip(cfg.percentile, 50.0, 99.9)))
    regions = find_regions(curve, threshold)
    if not regions:
        return []

    tracks = tracks or []
    lag = max(0.0, cfg.reaction_lag)
    out: list[Candidate] = []
    for i0, i1 in regions:
        peak_idx = int(i0 + np.argmax(curve[i0:i1]))

        # `reaction_lag` shifts the in-point only, and it is asymmetric on
        # purpose. In sports the detectable event is the crowd's reaction,
        # which arrives after the play — so the clip has to start earlier to
        # contain the play at all. The out-point is left alone: the roar and
        # the celebration *are* worth keeping, and pulling the end back by the
        # same amount would cut them off.
        start = float(times[i0]) - cfg.pre_roll - lag
        end = float(times[min(i1, times.size - 1)]) + cfg.post_roll
        start = max(0.0, start)
        end = min(duration, max(end, start + 0.1))

        # The moment itself, as opposed to the reaction that revealed it. Every
        # downstream guard (snapping, trimming) protects this, so it has to be
        # the play rather than the roar.
        moment = max(0.0, float(times[peak_idx]) - lag)

        # The span that must survive later edge moves. Its start is shifted by
        # the lag (that is where the play was) and its end is not (the reaction
        # is worth protecting too), so together they cover both.
        event_start = max(0.0, float(times[i0]) - lag)
        event_end = min(duration, float(times[min(i1, times.size - 1)]))

        # Grow short clips symmetrically around the event instead of from the
        # left edge, so the interesting part stays centred.
        if end - start < cfg.min_duration:
            centre = (event_start + max(event_start, event_end)) / 2.0
            half = cfg.min_duration / 2.0
            start = max(0.0, centre - half)
            end = min(duration, start + cfg.min_duration)
            start = max(0.0, end - cfg.min_duration)

        score = float(curve[i0:i1].mean() * 0.5 + curve[peak_idx] * 0.5)
        meta: dict[str, object] = {
            "peak_time": round(moment, 3),
            "event_start": round(event_start, 3),
            "event_end": round(max(event_start, event_end), 3),
        }
        if lag:
            meta["reaction_time"] = round(float(times[peak_idx]), 3)
        out.append(
            Candidate(
                start=start,
                end=end,
                score=score,
                reasons=explain(tracks, i0, i1) if tracks else {},
                meta=meta,
            )
        )
    return out


def merge(candidates: list[Candidate], cfg: SegmentConfig) -> list[Candidate]:
    """Join clips separated by less than ``merge_gap``; cap at ``max_duration``."""
    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda c: c.start)
    merged: list[Candidate] = [ordered[0]]
    for cand in ordered[1:]:
        last = merged[-1]
        if cand.start - last.end <= cfg.merge_gap:
            new_end = max(last.end, cand.end)
            weight_a, weight_b = last.duration, cand.duration
            total = (weight_a + weight_b) or 1.0
            last.end = new_end
            last.score = (last.score * weight_a + cand.score * weight_b) / total
            for key, value in cand.reasons.items():
                last.reasons[key] = max(last.reasons.get(key, value), value)

            # The merged clip covers both events, so its protected span does too.
            a_lo, a_hi = last.protected()
            b_lo, b_hi = cand.protected()
            last.meta["event_start"] = round(min(a_lo, b_lo), 3)
            last.meta["event_end"] = round(max(a_hi, b_hi), 3)

            if last.duration > cfg.max_duration:
                # Too long to keep whole. Hold on to the event and drop padding
                # rather than centring on a peak that may sit anywhere inside.
                lo, hi = last.protected()
                start = max(last.start, min(lo, hi - cfg.max_duration))
                if hi - lo > cfg.max_duration:
                    start = max(last.start, lo)  # the event alone overflows
                last.start = start
                last.end = min(last.end, start + cfg.max_duration)
        else:
            merged.append(cand)
    return merged


def select(candidates: list[Candidate], cfg: SegmentConfig) -> list[Candidate]:
    """Keep the best clips within the clip-count and total-duration budget.

    Selection is by score, output is by timeline order — a reel should still
    tell the match's story front to back.
    """
    pool = [c for c in candidates if c.score >= cfg.min_score and c.duration > 0.2]
    pool.sort(key=lambda c: c.score, reverse=True)

    chosen: list[Candidate] = []
    total = 0.0
    for cand in pool:
        if cfg.max_clips and len(chosen) >= cfg.max_clips:
            break
        # Allow the first clip through even if it alone busts the budget,
        # otherwise a short target yields an empty reel.
        if chosen and cfg.target_duration and total + cand.duration > cfg.target_duration:
            continue
        chosen.append(cand)
        total += cand.duration

    chosen.sort(key=lambda c: c.start)
    for rank, cand in enumerate(chosen, start=1):
        cand.meta["rank"] = rank
    return chosen
