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

__all__ = ["find_regions", "build_candidates", "merge", "select"]


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

        # Grow short clips symmetrically around the moment instead of from the
        # left edge, so the interesting part stays centred.
        if end - start < cfg.min_duration:
            half = cfg.min_duration / 2.0
            start = max(0.0, moment - half)
            end = min(duration, start + cfg.min_duration)
            start = max(0.0, end - cfg.min_duration)

        score = float(curve[i0:i1].mean() * 0.5 + curve[peak_idx] * 0.5)
        meta: dict[str, object] = {"peak_time": round(moment, 3)}
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
            if last.duration > cfg.max_duration:
                # Keep the window around the strongest peak in the merge.
                peak = last.meta.get("peak_time", (last.start + last.end) / 2)
                half = cfg.max_duration / 2.0
                start = max(last.start, peak - half)
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
