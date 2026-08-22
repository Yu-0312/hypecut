"""CLI argument handling, variant presets and batch discovery."""

from __future__ import annotations

import pytest

from hypecut.cli import VARIANT_PRESETS, _collect, _config_from_args, _parse_box, build_parser


def _args(argv: list[str]):
    return build_parser().parse_args(argv)


def test_vertical_shorthand_sets_a_crop_reframe():
    cfg = _config_from_args(_args(["cut", "v.mp4", "--vertical"]))
    assert cfg.render.reframe.mode == "crop"
    assert (cfg.render.reframe.width, cfg.render.reframe.height) == (1080, 1920)


def test_explicit_reframe_beats_the_shorthand():
    cfg = _config_from_args(_args(["cut", "v.mp4", "--vertical", "--reframe", "stack"]))
    assert cfg.render.reframe.mode == "stack"


def test_size_flags_land_on_the_reframe_when_reframing():
    """Reframing owns the geometry, so --width must not go to the scale/pad path."""
    cfg = _config_from_args(
        _args(["cut", "v.mp4", "--vertical", "--width", "720", "--height", "1280"])
    )
    assert (cfg.render.reframe.width, cfg.render.reframe.height) == (720, 1280)
    assert cfg.render.width is None and cfg.render.height is None


def test_size_flags_stay_on_render_without_reframing():
    cfg = _config_from_args(_args(["cut", "v.mp4", "--width", "1280", "--height", "720"]))
    assert (cfg.render.width, cfg.render.height) == (1280, 720)


def test_react_implies_crop_and_takes_a_facecam_box():
    cfg = _config_from_args(_args(["cut", "v.mp4", "--react", "--facecam", "0,0,0.25,0.3"]))
    assert cfg.render.reframe.mode == "crop"
    assert cfg.render.reframe.react_to_facecam is True
    assert cfg.render.reframe.facecam == [0.0, 0.0, 0.25, 0.3]


@pytest.mark.parametrize("text", ["1,2,3", "0,0,2,1", "a,b,c,d", ""])
def test_bad_facecam_boxes_are_rejected(text):
    with pytest.raises(ValueError, match="--facecam"):
        _parse_box(text)


def test_edge_stages_can_be_disabled():
    cfg = _config_from_args(_args(["cut", "v.mp4", "--no-snap", "--no-trim"]))
    assert cfg.segments.snap_to_shots is False
    assert cfg.segments.trim_to_silence is False


def test_also_adds_variants_without_touching_the_base_render():
    cfg = _config_from_args(_args(["cut", "v.mp4", "--also", "vertical", "--also", "square"]))
    assert set(cfg.variants) == {"vertical", "square"}
    assert cfg.render.reframe.mode == "off", "the base output stays landscape"


def test_repeated_also_is_deduplicated():
    cfg = _config_from_args(_args(["cut", "v.mp4", "--also", "vertical", "--also", "vertical"]))
    assert list(cfg.variants) == ["vertical"]


def test_render_for_merges_a_variant_over_the_base():
    cfg = _config_from_args(_args(["cut", "v.mp4", "--crf", "27", "--also", "vertical"]))
    variant = cfg.render_for("vertical")
    assert variant.reframe.mode == "crop"
    assert variant.crf == 27, "variants inherit everything they do not override"
    assert cfg.render_for(None).reframe.mode == "off"


def test_render_for_rejects_an_unknown_variant():
    cfg = _config_from_args(_args(["cut", "v.mp4"]))
    with pytest.raises(KeyError, match="Unknown variant"):
        cfg.render_for("nope")


def test_every_variant_preset_builds_a_valid_render_config():
    for name in VARIANT_PRESETS:
        cfg = _config_from_args(_args(["cut", "v.mp4", "--also", name]))
        render = cfg.render_for(name)
        assert render.reframe.mode != "off"
        assert render.reframe.width > 0 and render.reframe.height > 0


def test_collect_finds_videos_and_ignores_everything_else(tmp_path):
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.mkv").touch()
    (tmp_path / "notes.txt").touch()
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.mov").touch()

    flat = _collect(tmp_path, None, recursive=False)
    assert [p.name for p in flat] == ["a.mp4", "b.mkv"]

    deep = _collect(tmp_path, None, recursive=True)
    assert [p.name for p in deep] == ["a.mp4", "b.mkv", "c.mov"]


def test_collect_honours_explicit_patterns(tmp_path):
    (tmp_path / "a.mp4").touch()
    (tmp_path / "b.mkv").touch()
    assert [p.name for p in _collect(tmp_path, ["*.mkv"], recursive=False)] == ["b.mkv"]
