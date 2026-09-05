"""End-to-end tests. These need ffmpeg and are skipped without it."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hypecut import analyze, load_config, run
from hypecut.config import Config
from tests.conftest import EXCITING_WINDOWS, requires_ffmpeg


@requires_ffmpeg
def test_probe_reads_duration_and_audio(sample_vod):
    from hypecut.ffmpeg import probe

    info = probe(sample_vod)
    assert info.has_audio is True
    assert 40 < info.duration < 50
    assert info.width == 320 and info.height == 180


@requires_ffmpeg
def test_analyze_lands_clips_on_the_loud_stretches(sample_vod):
    cfg = Config().merged(
        {"segments": {"percentile": 88, "min_duration": 3.0, "target_duration": 30.0}}
    )
    plan = analyze(sample_vod, cfg)
    assert plan.segments, "expected at least one highlight"

    for lo, hi in EXCITING_WINDOWS:
        centre = (lo + hi) / 2
        assert any(s.start <= centre <= s.end for s in plan.segments), (
            f"no clip covers the loud stretch at {centre}s"
        )


@requires_ffmpeg
def test_analyze_never_exceeds_the_clip_budget(sample_vod):
    cfg = Config().merged({"segments": {"max_clips": 1, "percentile": 85}})
    plan = analyze(sample_vod, cfg)
    assert len(plan.segments) <= 1


@requires_ffmpeg
def test_run_produces_a_playable_reel(sample_vod, tmp_path):
    from hypecut.ffmpeg import probe

    out = tmp_path / "reel.mp4"
    cfg = Config().merged(
        {
            "segments": {"percentile": 88, "target_duration": 20.0},
            "render": {"fade": 0.1, "write_chapters": True},
        }
    )
    result = run(sample_vod, out, cfg)

    assert out.exists() and out.stat().st_size > 1000
    reel = probe(out)
    assert reel.has_audio
    assert reel.duration == pytest.approx(result.plan.total_duration, abs=1.5)

    sidecar = out.with_suffix(".hypecut.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text())
    assert payload["segments"] and "config" in payload
    assert out.with_suffix(".edl").exists()


@requires_ffmpeg
def test_silent_video_still_works(silent_vod, tmp_path):
    """A video with no audio must fall back to visual signals, not crash.

    ``min_prominence`` is off here on purpose. The fixture is a test pattern:
    it moves constantly and uniformly, so nothing in it stands out and the
    emptiness check is right to refuse it. That is a different question from
    the one this test asks, which is whether the visual signals carry the run
    when there is no audio track to lean on.
    """
    cfg = Config().merged(
        {"segments": {"percentile": 80, "target_duration": 8.0, "min_prominence": 0}}
    )
    plan = analyze(silent_vod, cfg)
    assert plan.segments
    assert all(t.name in {"scene_change", "motion", "roi_activity", "flash"} for t in plan.tracks)

    out = tmp_path / "silent_reel.mp4"
    run(silent_vod, out, cfg)
    assert out.exists()


@requires_ffmpeg
def test_refiners_change_the_ordering_but_keep_the_contract(sample_vod):
    base = analyze(sample_vod, Config().merged({"segments": {"percentile": 85}}))
    refined = analyze(
        sample_vod,
        Config().merged({"segments": {"percentile": 85}, "refiners": ["diversity", "pacing"]}),
    )
    assert refined.segments
    assert all(s.duration > 0 for s in refined.segments)
    assert len(refined.segments) <= len(base.segments) + 2


@requires_ffmpeg
def test_shipped_profiles_run_end_to_end(sample_vod):
    root = Path(__file__).resolve().parents[1] / "configs"
    for path in sorted(root.glob("*.yaml")):
        cfg = load_config(path).merged({"segments": {"percentile": 80}, "refiners": []})
        analyze(sample_vod, cfg)  # must not raise


@requires_ffmpeg
def test_clips_land_on_the_synthetic_shot_boundaries(sample_vod):
    """The fixture is concatenated at 15/20/35/40 s — real cuts to snap to."""
    cfg = Config().merged(
        {"segments": {"percentile": 85, "min_duration": 3.0, "target_duration": 40.0}}
    )
    plan = analyze(sample_vod, cfg)
    assert plan.segments

    cuts = [15.0, 20.0, 35.0, 40.0]
    snapped = [s for s in plan.segments if s.meta.get("snapped")]
    assert snapped, "expected at least one edge to reach a boundary"
    for seg in snapped:
        for edge in ("start", "end"):
            if edge in seg.meta["snapped"]:
                value = getattr(seg, edge)
                assert min(abs(value - c) for c in cuts) < 0.2, (
                    f"{edge} at {value:.2f}s is not on a cut"
                )


@requires_ffmpeg
def test_snapping_can_be_switched_off(sample_vod):
    cfg = Config().merged({"segments": {"percentile": 85, "snap_to_shots": False}})
    plan = analyze(sample_vod, cfg)
    assert all("snapped" not in s.meta for s in plan.segments)


@requires_ffmpeg
@pytest.mark.parametrize("mode", ["crop", "stack", "blur_pad"])
def test_vertical_reel_renders_at_the_requested_size(sample_vod, tmp_path, mode):
    from hypecut.ffmpeg import probe

    out = tmp_path / f"{mode}.mp4"
    cfg = Config().merged(
        {
            "segments": {"percentile": 88, "target_duration": 12.0},
            "render": {
                "fade": 0.1,
                "reframe": {"mode": mode, "width": 540, "height": 960, "track": mode == "crop"},
            },
        }
    )
    result = run(sample_vod, out, cfg)

    # `result.output`, not `out`: the deliberately tiny 12 s budget here can
    # spill the cut across several parts, and part 1 is not named `out`.
    reel = probe(result.output)
    assert (reel.width, reel.height) == (540, 960)
    assert all(s.meta.get("reframe", {}).get("mode") == mode for s in result.plan.segments)


@requires_ffmpeg
def test_edges_move_into_the_pauses_on_footage_with_no_cuts(talk_vod):
    """The case snapping cannot help with: one locked-off shot, all audio."""
    from tests.conftest import TALK_LOUD, TALK_PAUSES

    cfg = Config().merged({"segments": {"percentile": 88, "min_duration": 4.0}})
    plan = analyze(talk_vod, cfg)
    assert plan.segments

    seg = max(plan.segments, key=lambda s: s.score)
    assert seg.meta.get("trimmed"), "expected an edge to reach a pause"
    assert not seg.meta.get("snapped"), "there are no shot boundaries in this fixture"

    # Both edges should now sit inside a pause, close to the loud stretch.
    for edge in (seg.start, seg.end):
        assert any(lo - 0.4 <= edge <= hi + 0.4 for lo, hi in TALK_PAUSES), (
            f"edge at {edge:.2f}s is not in a pause"
        )
    assert seg.start <= TALK_LOUD[0] + 0.3
    assert seg.end >= TALK_LOUD[1] - 0.3


@requires_ffmpeg
def test_trimming_can_be_switched_off(talk_vod):
    cfg = Config().merged({"segments": {"percentile": 88, "trim_to_silence": False}})
    plan = analyze(talk_vod, cfg)
    assert all("trimmed" not in s.meta for s in plan.segments)


@requires_ffmpeg
def test_a_snapped_edge_is_never_overwritten_by_trimming(sample_vod):
    cfg = Config().merged({"segments": {"percentile": 85, "min_duration": 3.0}})
    plan = analyze(sample_vod, cfg)
    for seg in plan.segments:
        overlap = set(seg.meta.get("snapped", {})) & set(seg.meta.get("trimmed", {}))
        assert not overlap, f"{overlap} was decided twice"


@requires_ffmpeg
def test_one_analysis_renders_several_aspect_ratios(sample_vod, tmp_path):
    """The point of variants: decode and decide once, encode several times."""
    from hypecut.ffmpeg import probe
    from hypecut.pipeline import render_variants

    cfg = Config().merged(
        {
            "segments": {"percentile": 88, "target_duration": 12.0},
            "render": {"fade": 0.1},
            "variants": {
                "vertical": {"reframe": {"mode": "crop", "width": 540, "height": 960}},
                "square": {"reframe": {"mode": "crop", "width": 480, "height": 480}},
            },
        }
    )
    plan = analyze(sample_vod, cfg)
    assert plan.segments

    outputs = render_variants(plan, tmp_path / "reel.mp4", cfg)

    assert set(outputs) == {"base", "vertical", "square"}
    assert outputs["vertical"].name == "reel_vertical.mp4"
    base = probe(outputs["base"])
    assert (base.width, base.height) == (320, 180), "the base output keeps the source shape"
    assert (probe(outputs["vertical"]).width, probe(outputs["vertical"]).height) == (540, 960)
    assert (probe(outputs["square"]).width, probe(outputs["square"]).height) == (480, 480)

    # Each variant's framing is planned separately and recorded separately.
    seg = plan.segments[0]
    assert "reframe:vertical" in seg.meta and "reframe:square" in seg.meta


@requires_ffmpeg
def test_run_reports_variant_outputs(sample_vod, tmp_path):
    cfg = Config().merged(
        {
            "segments": {"percentile": 88, "target_duration": 10.0},
            "render": {"fade": 0.1},
            "variants": {"vertical": {"reframe": {"mode": "crop", "width": 540, "height": 960}}},
        }
    )
    result = run(sample_vod, tmp_path / "reel.mp4", cfg)
    assert set(result.variants) == {"vertical"}
    assert result.variants["vertical"].exists()
    assert result.sidecar is not None and result.sidecar.exists()


@requires_ffmpeg
def test_sports_profile_keeps_the_play_not_just_the_reaction(sports_vod):
    """The gameplay defaults start after the goal; the sports profile must not."""
    from tests.conftest import SPORT_DECOY, SPORT_GOAL, SPORT_ROAR

    root = Path(__file__).resolve().parents[1] / "configs"
    cfg = load_config(root / "sports-broadcast.yaml").merged(
        {
            "segments": {"percentile": 92, "min_duration": 6.0, "target_duration": 40.0},
            "signals": {"params": {"roi_change": {"box": [0.0, 0.0, 0.22, 0.16]}}},
        }
    )
    plan = analyze(sports_vod, cfg)
    assert plan.segments

    best = max(plan.segments, key=lambda s: s.score)
    assert best.start < SPORT_GOAL, "the clip has to contain the play, not only the roar"
    assert best.end > SPORT_ROAR[0], "and the reaction that follows it"
    assert best.meta.get("reaction_time", 0) > best.meta["peak_time"], "lag is recorded"
    assert not (best.start <= SPORT_DECOY <= best.end), "the brief shout is not a highlight"


@requires_ffmpeg
def test_crowd_roar_beats_raw_loudness_on_stadium_audio(sports_vod):
    """The decoy shout is louder than the roar; only one of them is a moment."""
    from hypecut.pipeline import _build_context
    from hypecut.signals import build_signals
    from tests.conftest import SPORT_DECOY, SPORT_ROAR

    cfg = Config().merged({"signals": {"enabled": ["crowd_roar", "audio_rms"]}})
    ctx = _build_context(
        probe_info := __import__("hypecut.ffmpeg", fromlist=["probe"]).probe(sports_vod), cfg
    )
    assert probe_info.has_audio

    tracks = {s.name: s.track(ctx).values for s in build_signals(cfg.signals.enabled)}
    roar_peak = float(np.argmax(tracks["crowd_roar"])) / ctx.grid_fps
    loud_peak = float(np.argmax(tracks["audio_rms"])) / ctx.grid_fps

    assert SPORT_ROAR[0] <= roar_peak <= SPORT_ROAR[1]
    assert not (SPORT_ROAR[0] <= loud_peak <= SPORT_ROAR[1]), (
        "raw loudness is expected to pick something else — that is the point"
    )
    assert abs(loud_peak - SPORT_DECOY) < 3.0 or loud_peak < SPORT_ROAR[0]


@requires_ffmpeg
def test_shipped_sports_profiles_run_end_to_end(sports_vod):
    root = Path(__file__).resolve().parents[1] / "configs"
    for name in ("sports-broadcast.yaml", "sports-field.yaml"):
        cfg = load_config(root / name).merged(
            {"segments": {"percentile": 88, "min_duration": 5.0}, "refiners": []}
        )
        plan = analyze(sports_vod, cfg)
        assert plan.segments, f"{name} found nothing"


@requires_ffmpeg
def test_evaluation_separates_the_sports_profile_from_the_default(sports_vod, tmp_path):
    """The whole point of the harness: turn a claim about quality into numbers.

    On this fixture both profiles *find* the goal, so recall, precision and F1
    all tie at 1.0 — and that tie is exactly why coverage is reported
    separately. The gameplay default fires on the goal frame itself and rolls
    out before the crowd has finished, keeping about two thirds of the moment;
    the sports profile knows the evidence arrives late and keeps all of it.
    Same detection, different framing, and only one number says so.
    """
    from hypecut.evaluation import Highlight, Labels, score_plan
    from tests.conftest import SPORT_GOAL, SPORT_ROAR

    # The fixture's answer key is known by construction: one moment, the goal
    # and the roar that follows it. The whistle and the decoy shout are not.
    labels = Labels(
        video=str(sports_vod),
        highlights=[Highlight(SPORT_GOAL - 2.0, SPORT_ROAR[1], "the goal")],
        annotator="fixture",
    )

    root = Path(__file__).resolve().parents[1] / "configs"
    scores = {}
    for name in ("default.yaml", "sports-broadcast.yaml"):
        cfg = load_config(root / name).merged({"segments": {"percentile": 90}})
        plan = analyze(sports_vod, cfg)
        scores[name] = score_plan(labels, [(s.start, s.end) for s in plan.segments])

    sport = scores["sports-broadcast.yaml"]
    generic = scores["default.yaml"]

    assert sport.recall == 1.0, "the sports profile has to find the goal"
    assert generic.recall == 1.0, "so does the default — the difference is not detection"
    assert sport.f1 >= generic.f1, "the sports profile must never score worse"
    assert sport.coverage > generic.coverage + 0.2, (
        f"the sports profile should keep more of the moment "
        f"({sport.coverage:.2f} vs {generic.coverage:.2f})"
    )


@requires_ffmpeg
def test_an_empty_video_yields_no_reel_and_says_so(boring_vod, tmp_path):
    """The point of `min_prominence`: "nothing here" has to be an answer.

    Every other threshold is relative to the video it is given — `fuse` even
    min-max rescales the curve to 0-1 — so without this check a percentile
    always finds something and three hours of an idle lobby comes back as a
    confident reel of its least-boring moments.
    """
    result = run(boring_vod, tmp_path / "nothing.mp4", Config())

    assert result.reels == []
    assert result.output is None
    assert not (tmp_path / "nothing.mp4").exists(), "no file should be written"
    assert "stands out" in result.plan.empty_reason
    assert result.plan.prominence < result.plan.min_prominence


def _tone_vod(path: Path, bitrate: str) -> Path:
    """One constant tone over one uniform test pattern, at a given bitrate."""
    from tests.conftest import _run

    _run(
        "ffmpeg -v error -y -f lavfi -i testsrc2=s=320x180:r=15:d=12 "
        "-f lavfi -i sine=frequency=400:r=48000:d=12 "
        "-filter_complex [1:a]volume=6.0[a] -map 0:v -map [a] "
        "-c:v libx264 -preset ultrafast -crf 32 -pix_fmt yuv420p "
        "-c:a aac -b:a {br} {out}",
        br=bitrate,
        out=str(path),
    )
    return path


@requires_ffmpeg
@pytest.mark.parametrize("bitrate", ["64k", "32k", "16k"])
def test_a_uniform_tone_is_empty_whatever_the_encoder_left_behind(tmp_path, bitrate):
    """Emptiness has to be a property of the footage, not of the codec.

    A constant tone over a constant test pattern contains no event, so the
    answer must be "nothing here" whichever encoder wrote the file. It was
    not. `audio_transient` declared no noise floor, so on footage with no
    onsets its median and its MAD both collapsed towards zero and the ratio
    `prominence` computes became noise divided by smaller noise. At 64k the
    AAC encoder's own artefacts measured a rise of 0.06 against a MAD of
    1e-4 and reported a prominence of 405 — the emptiness check answering
    "definitely something here" about a video containing nothing. The same
    tone at 32k scored 0.9 and came back empty.

    Which answer you got therefore depended on the ffmpeg build, which is how
    this reached CI as a macOS-only failure with every Linux job green.
    Parameterising the bitrate is the whole point: all three must agree.
    """
    result = run(_tone_vod(tmp_path / f"tone_{bitrate}.mp4", bitrate), tmp_path / "out.mp4")

    assert result.reels == []
    assert "stands out" in result.plan.empty_reason
    assert result.plan.prominence < result.plan.min_prominence


@requires_ffmpeg
def test_the_same_empty_video_cuts_fine_once_the_check_is_off(boring_vod, tmp_path):
    """The gate is the only thing stopping it — proof the footage is workable."""
    cfg = Config().merged({"segments": {"min_prominence": 0, "percentile": 85}})
    result = run(boring_vod, tmp_path / "anyway.mp4", cfg)
    assert result.reels, "with the check disabled the old behaviour returns"


@requires_ffmpeg
def test_a_long_cut_is_split_into_watchable_parts(repeat_vod, tmp_path):
    cfg = Config().merged(
        {
            "segments": {
                "percentile": 85,
                "min_prominence": 0,
                "clips_per_reel": 3,
                "target_duration": 500.0,
            }
        }
    )
    result = run(repeat_vod, tmp_path / "match.mp4", cfg)

    assert len(result.reels) > 1, "more clips than fit in one reel"
    assert all(reel.clips <= 3 for reel in result.reels)
    assert [reel.path.name for reel in result.reels[:2]] == ["match.part1.mp4", "match.part2.mp4"]
    assert all(reel.path.exists() for reel in result.reels)

    # Each part carries its own cut list, and it has to describe that part.
    for reel in result.reels:
        assert reel.sidecar and reel.sidecar.exists()
        written = json.loads(reel.sidecar.read_text())
        assert len(written["segments"]) == reel.clips

    # Parts run front to back, so the reel still tells the match's story.
    ends = [max(s.end for s in group) for group in result.plan.reels()]
    starts = [min(s.start for s in group) for group in result.plan.reels()]
    assert all(ends[i] <= starts[i + 1] for i in range(len(ends) - 1))


@requires_ffmpeg
def test_a_plan_is_always_chronological_and_non_overlapping(sample_vod, talk_vod):
    """The invariant the whole cut rests on, asserted end to end.

    `merge` establishes it and `select` preserves it, but snapping and trimming
    both move edges afterwards by more than the gap `merge` guarantees. Those
    two stages are now bounded by their neighbours; this checks the property
    that matters through the real pipeline rather than at the seam.
    """
    for source in (sample_vod, talk_vod):
        segments = analyze(source, Config()).segments
        for earlier, later in zip(segments, segments[1:], strict=False):
            assert later.start >= earlier.end, (
                f"{Path(source).name}: [{earlier.start:.2f}, {earlier.end:.2f}] and "
                f"[{later.start:.2f}, {later.end:.2f}] overlap"
            )
        for seg in segments:
            assert seg.start < seg.end
            assert seg.start >= 0.0
