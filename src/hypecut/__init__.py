"""HypeCut — automatic highlight reels for gameplay, esports and sports.

Upload a long video, get back a short one containing only the moments
worth watching, plus a machine-readable cut list explaining every choice.

    from hypecut import run, load_config

    result = run("vod.mp4", "reel.mp4", load_config("configs/valorant.yaml"))
    print(result.plan.to_dict())
"""

from __future__ import annotations

from .config import Config, ReframeConfig, RenderConfig, SegmentConfig, SignalConfig, load_config
from .pipeline import PipelineResult, analyze, render_plan, run
from .types import Candidate, HighlightPlan, SignalTrack, VideoInfo

__version__ = "0.7.0"

__all__ = [
    "__version__",
    "Config",
    "SignalConfig",
    "SegmentConfig",
    "RenderConfig",
    "ReframeConfig",
    "load_config",
    "analyze",
    "render_plan",
    "run",
    "PipelineResult",
    "VideoInfo",
    "Candidate",
    "SignalTrack",
    "HighlightPlan",
]
