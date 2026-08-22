"""Configuration objects and YAML/JSON loading.

A profile is just a nested dict merged over the defaults, so a community
preset for a new game is a ~20 line YAML file and needs no Python.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

__all__ = [
    "SignalConfig",
    "SegmentConfig",
    "ReframeConfig",
    "RenderConfig",
    "Config",
    "load_config",
]


@dataclass
class SignalConfig:
    """Which detectors run, and how much each one is trusted.

    Weights are relative; every track is z-normalised before weighting, so
    ``audio_rms: 1.0`` and ``scene_change: 0.5`` means "loudness matters
    twice as much as cuts", regardless of each signal's native units.
    """

    enabled: list[str] = field(
        default_factory=lambda: [
            "audio_rms",
            "audio_transient",
            "scene_change",
            "motion",
            "roi_activity",
        ]
    )
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "audio_rms": 1.0,
            "audio_transient": 1.2,
            "scene_change": 0.6,
            "motion": 0.8,
            "roi_activity": 1.0,
        }
    )
    params: dict[str, dict[str, Any]] = field(default_factory=dict)
    grid_fps: float = 10.0
    frame_width: int = 96
    frame_height: int = 54
    audio_sr: int = 16_000
    smooth_seconds: float = 1.5


@dataclass
class SegmentConfig:
    """How the fused curve becomes a list of clips."""

    min_duration: float = 4.0
    max_duration: float = 20.0
    pre_roll: float = 3.0
    post_roll: float = 2.0
    merge_gap: float = 2.0
    # How long *after* the moment its detectable reaction arrives. Zero for
    # gameplay, where the kill and its sound are simultaneous; a couple of
    # seconds for sports, where the goal is silent and the roar is not.
    reaction_lag: float = 0.0
    percentile: float = 92.0
    max_clips: int = 20
    target_duration: float | None = 120.0
    min_score: float = 0.0

    # Shot-boundary snapping. A clip edge that lands mid-shot reads as a slice;
    # the same edge moved half a second onto a real cut reads as an edit.
    snap_to_shots: bool = True
    snap_window: float = 2.0  # how far an edge may travel to reach a boundary
    snap_fine: bool = True  # re-check at native frame rate for frame accuracy
    snap_guard: float = 0.75  # minimum footage kept after the peak when snapping the end
    snap_to_dissolves: bool = True  # also land on crossfades and fades, not just hard cuts

    # Silence-aware trimming — the audio half of the same problem. Only edges
    # that found no shot boundary are considered, so a real cut always wins.
    trim_to_silence: bool = True
    silence_window: float = 1.5  # how far an edge may travel to reach a pause
    min_silence: float = 0.30  # a pause has to last this long to count
    silence_drop_db: float = 14.0  # how far below the clip's own level is "quiet"
    silence_pad: float = 0.12  # breathing room kept on the speech side of the pause


@dataclass
class ReframeConfig:
    """Turning a landscape capture into a vertical (or square) crop.

    ``mode``
        ``off``       leave the framing alone (default)
        ``crop``      motion-centred crop — the usual choice for gameplay
        ``stack``     facecam on top, gameplay below (the classic Shorts look)
        ``blur_pad``  whole frame letterboxed over a blurred enlargement
    """

    mode: str = "off"
    width: int = 1080
    height: int = 1920
    track: bool = False  # let the crop pan to follow the action
    smooth_seconds: float = 2.5
    max_pan: float = 0.10  # fraction of frame width the crop may travel per second
    keyframes: int = 6  # pan resolution; more means a longer filter expression
    facecam: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.28, 0.28])
    gameplay: list[float] = field(default_factory=lambda: [0.0, 0.0, 1.0, 1.0])
    facecam_share: float = 0.32  # portion of output height for the facecam pane
    blur_sigma: float = 24.0

    # Reaction-aware crop: when the facecam is busy (the streamer is reacting),
    # pull the crop toward it; otherwise stay on the action. Needs `facecam` to
    # actually match the layout, which is why it is off by default.
    react_to_facecam: bool = False
    react_weight: float = 0.6  # 0 = ignore the facecam, 1 = frame it exclusively
    react_threshold: float = 1.6  # multiple of the clip's median facecam activity

    def __post_init__(self) -> None:
        # YAML 1.1 reads a bare `off` as the boolean False, so the most natural
        # way to write "no reframing" in a profile silently produces the wrong
        # type. Accept it rather than failing three stages later with a
        # confusing message about mode `False`.
        if self.mode is False:
            self.mode = "off"
        if self.mode is True:
            raise ValueError(
                "render.reframe.mode is `true` — YAML read a bare `on` as a boolean. "
                'Quote the mode name, e.g. mode: "crop".'
            )


@dataclass
class RenderConfig:
    """Output encoding and transitions."""

    width: int | None = None
    height: int | None = None
    fps: float | None = None
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    crf: int = 20
    preset: str = "veryfast"
    audio_bitrate: str = "192k"
    fade: float = 0.25
    normalize_audio: bool = True  # even out dynamics *inside* each clip
    # ...and match loudness *between* clips, which dynaudnorm cannot do. Two
    # passes: measure each clip's integrated loudness, then apply a static gain.
    loudness_target: float = -16.0  # LUFS; the usual target for web delivery
    loudness_match: float = 0.9  # 0 disables; 1 flattens every clip to the target
    loudness_max_gain: float = 12.0  # dB, so a near-silent clip cannot explode
    write_chapters: bool = True
    reframe: ReframeConfig = field(default_factory=ReframeConfig)


@dataclass
class Config:
    """Top-level configuration; one of these fully determines a run."""

    profile: str = "default"
    signals: SignalConfig = field(default_factory=SignalConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    refiners: list[str] = field(default_factory=list)
    refiner_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    #: Extra renders of the *same* analysis, keyed by name. Each value is a
    #: partial ``render`` override, so a landscape reel and a vertical cutdown
    #: come out of one decode and one set of decisions instead of two runs.
    variants: dict[str, dict[str, Any]] = field(default_factory=dict)

    def render_for(self, variant: str | None = None) -> RenderConfig:
        """The render config for the base output, or for a named variant."""
        if not variant:
            return self.render
        if variant not in self.variants:
            raise KeyError(
                f"Unknown variant {variant!r}. Defined: {', '.join(sorted(self.variants)) or '-'}"
            )
        merged = _deep_merge(asdict(self.render), self.variants[variant])
        return _from_dict(RenderConfig, merged)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def merged(self, overrides: dict[str, Any]) -> Config:
        return _from_dict(Config, _deep_merge(self.to_dict(), overrides))


def load_config(path: str | Path | None = None, **overrides: Any) -> Config:
    """Load a profile from YAML/JSON, then apply keyword overrides.

    ``load_config("configs/valorant.yaml", segments={"max_clips": 8})``
    """
    base = Config()
    if path is not None:
        data = _read_structured(Path(path))
        base = base.merged(data)
    if overrides:
        base = base.merged(overrides)
    return base


def _read_structured(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:  # pragma: no cover
            raise RuntimeError(
                "PyYAML is required to read .yaml profiles: pip install pyyaml"
            ) from exc
        return yaml.safe_load(text) or {}
    return json.loads(text)


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Rebuild a (possibly nested) config dataclass from plain dicts.

    Annotations in this module are strings, because of ``from __future__
    import annotations``, so the real types have to come from
    ``get_type_hints``. That is what lets an arbitrarily nested section such
    as ``render.reframe`` be reconstructed without this function knowing any
    field names.
    """
    if not is_dataclass(cls):
        return data
    hints = get_type_hints(cls)
    known = {f.name for f in fields(cls)}

    unknown = set(data) - known
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {', '.join(sorted(unknown))}")

    kwargs: dict[str, Any] = {}
    for name in known & set(data):
        value = data[name]
        hint = hints.get(name)
        if isinstance(value, dict) and is_dataclass(hint):
            kwargs[name] = _from_dict(hint, value)  # type: ignore[arg-type]
        else:
            kwargs[name] = value
    return cls(**kwargs)
