"""Cross-clip loudness matching."""

from __future__ import annotations

import numpy as np
import pytest

from hypecut.config import RenderConfig
from hypecut.render import audio_filters, plan_loudness_gains
from hypecut.types import Candidate, VideoInfo
from tests.conftest import requires_ffmpeg


@pytest.fixture(scope="session")
def uneven_vod(tmp_path_factory):
    """Three clips of identical content at wildly different levels."""
    from tests.conftest import HAS_FFMPEG, _run

    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not installed")

    d = tmp_path_factory.mktemp("loud")
    sr = 48_000

    def tone(amp: float, seconds: float) -> np.ndarray:
        t = np.arange(int(seconds * sr)) / sr
        return (amp * np.sin(2 * np.pi * 300 * t)).astype(np.float32)

    raw = d / "audio.f32"
    raw.write_bytes(np.concatenate([tone(0.02, 10), tone(0.5, 10), tone(0.02, 10)]).tobytes())

    out = d / "uneven.mp4"
    _run(
        "ffmpeg -v error -y -f f32le -ar 48000 -ac 1 -i {raw} "
        "-f lavfi -i color=c=#203040:s=320x180:r=15 -shortest "
        "-c:v libx264 -preset ultrafast -crf 32 -pix_fmt yuv420p -c:a aac -b:a 96k {out}",
        raw=str(raw),
        out=str(out),
    )
    return out


def _segments() -> list[Candidate]:
    return [Candidate(2.0, 8.0, 0.9), Candidate(12.0, 18.0, 0.9), Candidate(22.0, 28.0, 0.9)]


@requires_ffmpeg
def test_matching_collapses_the_spread_between_clips(uneven_vod):
    from hypecut.ffmpeg import probe
    from hypecut.render import measure_loudness

    info = probe(uneven_vod)
    cfg = RenderConfig()
    segments = _segments()

    before = [measure_loudness(info, s, cfg) for s in segments]
    assert None not in before
    spread_before = max(before) - min(before)
    assert spread_before > 8.0, "fixture should actually be uneven"

    gains = plan_loudness_gains(info, segments, cfg)
    after = [b + g for b, g in zip(before, gains, strict=True)]
    spread_after = max(after) - min(after)

    # match=0.9 leaves a tenth of the original spread: inaudible as a jump,
    # still audible as character.
    assert spread_after == pytest.approx(spread_before * 0.1, rel=0.4)


@requires_ffmpeg
def test_full_match_flattens_everything(uneven_vod):
    from hypecut.ffmpeg import probe
    from hypecut.render import measure_loudness

    info = probe(uneven_vod)
    cfg = RenderConfig(loudness_match=1.0)
    segments = _segments()

    before = [measure_loudness(info, s, cfg) for s in segments]
    gains = plan_loudness_gains(info, segments, cfg)
    after = [b + g for b, g in zip(before, gains, strict=True)]
    assert max(after) - min(after) < 1.0


@requires_ffmpeg
def test_matching_can_be_switched_off(uneven_vod):
    from hypecut.ffmpeg import probe

    gains = plan_loudness_gains(probe(uneven_vod), _segments(), RenderConfig(loudness_match=0.0))
    assert gains == [0.0, 0.0, 0.0]


@requires_ffmpeg
def test_gain_is_clamped_so_near_silence_cannot_explode(uneven_vod):
    from hypecut.ffmpeg import probe

    cfg = RenderConfig(loudness_target=0.0, loudness_max_gain=3.0)
    gains = plan_loudness_gains(probe(uneven_vod), _segments(), cfg)
    assert all(abs(g) <= 3.0 + 1e-9 for g in gains)


def test_a_silent_source_needs_no_measurement():
    info = VideoInfo("silent.mp4", 30.0, 30.0, 320, 180, has_audio=False)
    assert plan_loudness_gains(info, _segments(), RenderConfig()) == [0.0, 0.0, 0.0]


def test_gain_leads_the_chain_and_is_omitted_when_zero():
    """Order matters: the gain has to precede the compressor, not follow it."""
    seg = Candidate(0.0, 10.0, 1.0)
    cfg = RenderConfig()

    with_gain = audio_filters(seg, cfg, gain_db=-4.0)
    assert with_gain[0] == "volume=-4.00dB"
    assert any(f.startswith("dynaudnorm") for f in with_gain)

    assert not any(f.startswith("volume=") for f in audio_filters(seg, cfg, gain_db=0.0))
