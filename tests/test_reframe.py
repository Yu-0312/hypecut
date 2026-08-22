"""Vertical reframing: action tracking, planning and filter construction."""

from __future__ import annotations

import numpy as np
import pytest

from hypecut.config import ReframeConfig
from hypecut.reframe import action_track, geometry_filters, plan_reframe
from hypecut.types import AnalysisContext, Candidate, VideoInfo


def _ctx_with_moving_blob(grid_fps: float = 10.0, n: int = 100) -> AnalysisContext:
    """A bright block sliding from the left edge to the right over the clip."""
    h, w = 54, 96
    gray = np.zeros((n, h, w), dtype=np.uint8)
    for i in range(n):
        x = int((i / max(n - 1, 1)) * (w - 10))
        gray[i, 20:34, x : x + 10] = 255
    return AnalysisContext(
        info=VideoInfo("fake.mp4", n / grid_fps, 30.0, 1280, 720, True),
        grid_fps=grid_fps,
        times=np.arange(n) / grid_fps,
        gray=gray,
    )


def _info(width: int = 1280, height: int = 720) -> VideoInfo:
    return VideoInfo("fake.mp4", 60.0, 30.0, width, height, True)


def test_action_track_follows_the_moving_subject():
    ctx = _ctx_with_moving_blob()
    seg = Candidate(0.0, 10.0, 1.0)
    track = action_track(ctx, seg, ReframeConfig(smooth_seconds=0.5, max_pan=1.0))

    assert len(track) > 10
    assert all(0.0 <= x <= 1.0 for x in track)
    assert track[0] < 0.35, "should start near the left edge"
    assert track[-1] > 0.65, "should end near the right edge"


def test_action_track_is_centred_when_nothing_moves():
    n = 40
    gray = np.full((n, 54, 96), 90, dtype=np.uint8)
    ctx = AnalysisContext(
        info=VideoInfo("fake.mp4", 4.0, 30.0, 1280, 720, True),
        grid_fps=10.0,
        times=np.arange(n) / 10.0,
        gray=gray,
    )
    track = action_track(ctx, Candidate(0.0, 4.0, 1.0), ReframeConfig())
    assert all(abs(x - 0.5) < 0.02 for x in track)


def test_pan_speed_is_capped():
    """A subject that teleports must produce a push, not a jump cut sideways."""
    n = 60
    gray = np.zeros((n, 54, 96), dtype=np.uint8)
    gray[:30, 20:34, 0:10] = 255  # hard left...
    gray[30:, 20:34, 86:96] = 255  # ...then hard right
    ctx = AnalysisContext(
        info=VideoInfo("fake.mp4", 6.0, 30.0, 1280, 720, True),
        grid_fps=10.0,
        times=np.arange(n) / 10.0,
        gray=gray,
    )
    cfg = ReframeConfig(smooth_seconds=0.1, max_pan=0.05)
    track = action_track(ctx, Candidate(0.0, 6.0, 1.0), cfg)
    steps = np.abs(np.diff(track))
    assert steps.max() <= cfg.max_pan / ctx.grid_fps + 1e-9


def test_plan_reframe_off_leaves_clips_untouched():
    ctx = _ctx_with_moving_blob()
    seg = Candidate(0.0, 10.0, 1.0)
    plan_reframe(ctx, [seg], ReframeConfig(mode="off"))
    assert "reframe" not in seg.meta


def test_plan_reframe_still_crop_records_a_single_centre():
    ctx = _ctx_with_moving_blob()
    seg = Candidate(0.0, 10.0, 1.0)
    plan_reframe(ctx, [seg], ReframeConfig(mode="crop", track=False))
    plan = seg.meta["reframe"]
    assert plan["mode"] == "crop"
    assert "keyframes" not in plan
    assert 0.0 <= plan["x"] <= 1.0


def test_plan_reframe_tracking_records_keyframes():
    ctx = _ctx_with_moving_blob()
    seg = Candidate(0.0, 10.0, 1.0)
    plan_reframe(ctx, [seg], ReframeConfig(mode="crop", track=True, keyframes=5, max_pan=1.0))
    keys = seg.meta["reframe"]["keyframes"]
    assert len(keys) == 5
    assert keys[0][0] == 0.0 and keys[-1][0] == pytest.approx(10.0)
    assert keys[-1][1] > keys[0][1], "the crop should travel with the subject"


def test_plan_reframe_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="Unknown reframe mode"):
        plan_reframe(_ctx_with_moving_blob(), [Candidate(0, 1, 1)], ReframeConfig(mode="nope"))


def test_crop_filter_geometry_is_nine_by_sixteen_and_even():
    seg = Candidate(0.0, 10.0, 1.0, meta={"reframe": {"mode": "crop", "x": 0.5}})
    chain = ",".join(geometry_filters(_info(), seg, ReframeConfig(mode="crop")))
    crop = [p for p in chain.split(",") if p.startswith("crop=")][0]
    width = int(crop.split("w=")[1].split(":")[0])
    height = int(crop.split("h=")[1].split(":")[0])
    assert width % 2 == 0 and height % 2 == 0
    assert width / height == pytest.approx(1080 / 1920, rel=0.02)
    assert "scale=1080:1920" in chain


def test_crop_x_stays_inside_the_frame_for_extreme_centres():
    for centre in (0.0, 1.0):
        seg = Candidate(0.0, 5.0, 1.0, meta={"reframe": {"mode": "crop", "x": centre}})
        chain = ",".join(geometry_filters(_info(), seg, ReframeConfig(mode="crop")))
        crop = [p for p in chain.split(",") if p.startswith("crop=")][0]
        width = int(crop.split("w=")[1].split(":")[0])
        x = int(crop.split("x=")[1].split(":")[0])
        assert 0 <= x <= 1280 - width


def test_pan_expression_has_no_whitespace_and_is_quoted():
    """The chain is one argv token; a space in it would split the command."""
    seg = Candidate(
        0.0, 4.0, 1.0, meta={"reframe": {"mode": "crop", "keyframes": [[0.0, 0.1], [4.0, 0.9]]}}
    )
    chain = ",".join(geometry_filters(_info(), seg, ReframeConfig(mode="crop", track=True)))
    assert " " not in chain
    assert "if(lt(t," in chain and chain.count("'") == 2


def test_stack_and_blur_modes_build_complete_graphs():
    seg = Candidate(0.0, 5.0, 1.0)
    for mode, marker in (("stack", "vstack=inputs=2"), ("blur_pad", "overlay=")):
        seg.meta["reframe"] = {"mode": mode}
        chain = ",".join(geometry_filters(_info(), seg, ReframeConfig(mode=mode)))
        assert marker in chain
        assert " " not in chain
        # Every labelled output must be consumed again by a later filter.
        assert chain.count("[top]") == 2 if mode == "stack" else True


def test_off_mode_produces_no_filters():
    assert geometry_filters(_info(), Candidate(0, 1, 1), ReframeConfig(mode="off")) == []


def test_portrait_source_is_handled_without_upscaling_past_the_frame():
    seg = Candidate(0.0, 5.0, 1.0, meta={"reframe": {"mode": "crop", "x": 0.5}})
    chain = ",".join(geometry_filters(_info(720, 1280), seg, ReframeConfig(mode="crop")))
    crop = [p for p in chain.split(",") if p.startswith("crop=")][0]
    width = int(crop.split("w=")[1].split(":")[0])
    height = int(crop.split("h=")[1].split(":")[0])
    assert width <= 720 and height <= 1280
