#!/usr/bin/env python3
"""Example: a custom signal, registered at runtime and used immediately.

Run it against any video:

    python examples/custom_signal.py my_vod.mp4

The signal here rewards *sustained* high motion rather than instantaneous
change — useful for footage where the interesting thing is a long chase or
teamfight rather than a single hit.
"""

from __future__ import annotations

import sys

import numpy as np

from hypecut import Config, analyze
from hypecut.signals import Signal, register


@register("sustained_motion")
class SustainedMotion(Signal):
    """Rolling mean of frame difference over a multi-second window.

    Params
    ------
    window_seconds: length of the rolling window (default 5).
    """

    description = "Sustained motion over several seconds, not single-frame spikes."
    requires_video = True

    def compute(self, ctx):
        window = int(self.params.get("window_seconds", 5) * ctx.grid_fps)
        f = ctx.gray.astype(np.float32)
        if f.shape[0] < 2:
            return np.zeros(ctx.n)
        d = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
        d = np.concatenate([d[:1], d])
        kernel = np.ones(max(1, window)) / max(1, window)
        return np.convolve(
            np.pad(d, (window // 2, window // 2), mode="edge"), kernel, mode="valid"
        )[: ctx.n]


def main(path: str) -> int:
    cfg = Config().merged(
        {
            "signals": {
                "enabled": ["audio_rms", "audio_transient", "sustained_motion"],
                "weights": {"audio_rms": 1.0, "audio_transient": 1.0, "sustained_motion": 1.5},
                "params": {"sustained_motion": {"window_seconds": 6}},
            },
            "segments": {"percentile": 90, "target_duration": 90.0},
        }
    )

    plan = analyze(path, cfg, progress=lambda p, m: print(f"  {p:5.0%} {m}"))
    print(f"\n{len(plan.segments)} clips:")
    for seg in plan.segments:
        print(f"  {seg.start:8.1f}s → {seg.end:8.1f}s  score {seg.score:.3f}  {seg.reasons}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    raise SystemExit(main(sys.argv[1]))
