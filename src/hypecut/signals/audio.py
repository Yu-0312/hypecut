"""Audio-derived excitement signals.

Loudness is the single most reliable cheap proxy for "something happened"
in gameplay footage: the streamer reacts, the game plays a kill sting, the
crowd in the VOD chatter spikes. Two complementary views are computed —
sustained energy, and sudden onsets — because a long shout and a single
gunshot look very different on the same waveform.
"""

from __future__ import annotations

import numpy as np

from ..types import AnalysisContext
from .base import Signal, register

__all__ = ["AudioRms", "AudioTransient", "SpeechBand"]


@register("audio_rms")
class AudioRms(Signal):
    """Short-term loudness in dBFS, the workhorse signal.

    Params
    ------
    floor_db: values below this are clamped (default -60).
    """

    description = "Short-term RMS loudness (dBFS) — sustained excitement."
    requires_audio = True
    # dB. A rise of under 3 dB over the whole video is room tone, not an event.
    noise_floor = 3.0

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        floor_db = float(self.params.get("floor_db", -60.0))
        frames = ctx.audio_frames()
        rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1) + 1e-12)
        db = 20.0 * np.log10(np.maximum(rms, 1e-6))
        return np.maximum(db, floor_db)


@register("audio_transient")
class AudioTransient(Signal):
    """Positive spectral-flux-like onset strength.

    Rewards *changes* in loudness rather than loudness itself, so a
    consistently loud stream does not read as one 40-minute highlight.

    Params
    ------
    lag: grid steps to look back when differencing (default 2).
    """

    description = "Onset strength — sudden jumps in energy (kills, hits, shouts)."
    requires_audio = True
    # Log-energy units. Onset strength is a difference of log energies, so it
    # has a scale of its own and needs a floor like every other signal here —
    # and it needs one *more* than the others, because on footage with no
    # onsets at all its median and its MAD both collapse towards zero. The
    # ratio `prominence` computes is then noise over smaller noise: a constant
    # tone through the AAC encoder measured a rise of 0.06 against a MAD of
    # 1e-4 and reported a prominence of 405, which is the emptiness check
    # answering "definitely something here" about a video containing nothing.
    # Re-encoding the same tone at a different bitrate moved that 405 to 0.9.
    # A real onset — silence into a shout, a quiet bed into a crowd — rises by
    # 0.5 or more, so 0.15 clears codec noise by a comfortable margin without
    # coming near a genuine event.
    noise_floor = 0.15

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        lag = max(1, int(self.params.get("lag", 2)))
        frames = ctx.audio_frames()
        energy = np.log1p(np.mean(np.abs(frames, dtype=np.float64), axis=1) * 100.0)
        prev = np.concatenate([np.repeat(energy[:1], lag), energy[:-lag]])
        return np.maximum(energy - prev, 0.0)


@register("speech_band")
class SpeechBand(Signal):
    """Energy concentrated in the 300-3400 Hz voice band.

    Useful for commentary-heavy footage: it separates "the caster is
    yelling" from "the game music got loud".

    Params
    ------
    low_hz / high_hz: band edges (defaults 300 / 3400).
    """

    description = "Voice-band (300-3400 Hz) energy — commentary and reactions."
    requires_audio = True

    def compute(self, ctx: AnalysisContext) -> np.ndarray:
        low = float(self.params.get("low_hz", 300.0))
        high = float(self.params.get("high_hz", 3400.0))
        frames = ctx.audio_frames()
        n_fft = frames.shape[1]
        if n_fft < 8:
            return np.zeros(ctx.n)
        window = np.hanning(n_fft)
        spec = np.abs(np.fft.rfft(frames * window, axis=1))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / ctx.audio_sr)
        band = (freqs >= low) & (freqs <= high)
        if not band.any():
            return np.zeros(ctx.n)
        total = spec.sum(axis=1) + 1e-9
        return spec[:, band].sum(axis=1) / total * np.log1p(total)
