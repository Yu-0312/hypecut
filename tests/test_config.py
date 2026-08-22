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


def test_reframe_config_is_nested_and_merges_deeply():
    cfg = Config().merged({"render": {"reframe": {"mode": "crop", "track": True}}})
    assert cfg.render.reframe.mode == "crop"
    assert cfg.render.reframe.track is True
    assert cfg.render.reframe.width == 1080  # untouched defaults survive
    assert cfg.render.crf == Config().render.crf


def test_yaml_bare_off_is_accepted_as_the_off_mode():
    """`mode: off` in YAML is the boolean False; it must not blow up later."""
    from hypecut.config import ReframeConfig

    assert ReframeConfig(mode=False).mode == "off"


def test_yaml_bare_on_is_rejected_with_a_useful_message():
    from hypecut.config import ReframeConfig

    with pytest.raises(ValueError, match="Quote the mode name"):
        ReframeConfig(mode=True)


def test_unknown_nested_key_is_rejected():
    with pytest.raises(ValueError, match="Unknown config key"):
        Config().merged({"render": {"reframe": {"nope": 1}}})
