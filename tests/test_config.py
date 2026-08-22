"""Configuration loading and profile merging."""

from __future__ import annotations

import json

import pytest

from hypecut.config import Config, load_config


def test_defaults_are_sane():
    cfg = Config()
    assert cfg.segments.min_duration < cfg.segments.max_duration
    assert set(cfg.signals.weights) >= set(cfg.signals.enabled)


def test_merge_is_deep_not_replacing():
    cfg = Config().merged({"segments": {"max_clips": 3}})
    assert cfg.segments.max_clips == 3
    assert cfg.segments.min_duration == Config().segments.min_duration


def test_load_json_profile(tmp_path):
    path = tmp_path / "p.json"
    path.write_text(json.dumps({"profile": "x", "segments": {"percentile": 80}}))
    cfg = load_config(path)
    assert cfg.profile == "x"
    assert cfg.segments.percentile == 80


def test_load_yaml_profile(tmp_path):
    yaml = pytest.importorskip("yaml")
    path = tmp_path / "p.yaml"
    path.write_text(yaml.safe_dump({"refiners": ["diversity"]}))
    assert load_config(path).refiners == ["diversity"]


def test_unknown_key_is_rejected_loudly():
    with pytest.raises(ValueError, match="Unknown config key"):
        Config().merged({"segments": {"nope": 1}})


def test_shipped_profiles_all_parse():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "configs"
    profiles = sorted(root.glob("*.yaml"))
    assert profiles, "no profiles shipped"
    for path in profiles:
        cfg = load_config(path)
        assert cfg.signals.enabled
        for name in cfg.signals.enabled:
            assert name in cfg.signals.weights, f"{path.name}: {name} has no weight"
