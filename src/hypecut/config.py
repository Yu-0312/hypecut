"""Configuration objects and YAML/JSON loading.

A profile is just a nested dict merged over the defaults, so a community
preset for a new game is a ~20 line YAML file and needs no Python.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

__all__ = ["SignalConfig", "SegmentConfig", "RenderConfig", "Config", "load_config"]


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
    percentile: float = 92.0
    max_clips: int = 20
    target_duration: float | None = 120.0
    min_score: float = 0.0


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
    normalize_audio: bool = True
    write_chapters: bool = True


@dataclass
class Config:
    """Top-level configuration; one of these fully determines a run."""

    profile: str = "default"
    signals: SignalConfig = field(default_factory=SignalConfig)
    segments: SegmentConfig = field(default_factory=SegmentConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    refiners: list[str] = field(default_factory=list)
    refiner_params: dict[str, dict[str, Any]] = field(default_factory=dict)

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
    if not is_dataclass(cls):
        return data
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        if is_dataclass(f.type) and isinstance(value, dict):
            kwargs[name] = _from_dict(f.type, value)  # type: ignore[arg-type]
        elif isinstance(value, dict) and name in {"signals", "segments", "render"}:
            kwargs[name] = _from_dict(
                {"signals": SignalConfig, "segments": SegmentConfig, "render": RenderConfig}[name],
                value,
            )
        else:
            kwargs[name] = value
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"Unknown config key(s) for {cls.__name__}: {', '.join(sorted(unknown))}")
    return cls(**kwargs)
