"""Built-in refiners.

``diversity`` and ``pacing`` need no extra dependencies and are on by
default; ``clip_rerank`` and ``speech_keywords`` are opt-in and pull in
torch / faster-whisper only when actually used.
"""

from __future__ import annotations

import math
import subprocess
import tempfile
from pathlib import Path

from ..ffmpeg import cmd
from ..types import Candidate, VideoInfo
from .base import Refiner, register

__all__ = ["Diversity", "Pacing", "ClipRerank", "SpeechKeywords"]


@register("diversity")
class Diversity(Refiner):
    """Penalise clips that cluster in one stretch of the video.

    Without this a single chaotic teamfight eats the whole reel. The
    penalty is a soft exponential in the gap to the nearest already-kept
    clip, so genuinely outstanding neighbours still survive.

    Params
    ------
    min_gap: seconds below which the penalty applies (default 45).
    strength: 0-1, how hard to penalise (default 0.35).
    """

    description = "Spread clips across the timeline instead of clustering."

    def refine(self, info: VideoInfo, candidates: list[Candidate]) -> list[Candidate]:
        min_gap = float(self.params.get("min_gap", 45.0))
        strength = float(self.params.get("strength", 0.35))
        kept: list[Candidate] = []
        for cand in sorted(candidates, key=lambda c: c.score, reverse=True):
            if kept:
                gap = min(abs(cand.start - k.start) for k in kept)
                if gap < min_gap:
                    factor = 1.0 - strength * (1.0 - gap / min_gap)
                    cand.score *= max(0.0, factor)
                    cand.meta["diversity_penalty"] = round(1.0 - factor, 4)
            kept.append(cand)
        return candidates


@register("pacing")
class Pacing(Refiner):
    """Nudge clip scores toward a comfortable viewing length.

    Very short clips read as jump-cuts; very long ones lose the room. The
    multiplier is a gentle bell around ``ideal``.

    Params
    ------
    ideal: preferred clip length in seconds (default 9).
    tolerance: width of the bell (default 6).
    """

    description = "Favour clips near a target length; damp very short/long ones."

    def refine(self, info: VideoInfo, candidates: list[Candidate]) -> list[Candidate]:
        ideal = float(self.params.get("ideal", 9.0))
        tol = max(1e-3, float(self.params.get("tolerance", 6.0)))
        for cand in candidates:
            factor = math.exp(-((cand.duration - ideal) ** 2) / (2 * tol**2))
            cand.score *= 0.6 + 0.4 * factor
            cand.meta["pacing_factor"] = round(factor, 4)
        return candidates


@register("clip_rerank")
class ClipRerank(Refiner):
    """Rescore candidates with CLIP image-text similarity.

    One frame is sampled at each candidate's peak and scored against a set
    of prompts describing what a highlight looks like for the profile
    ("a player getting a kill", "an explosion"), minus a set of negative
    prompts ("an empty menu screen", "a loading screen"). This is the step
    that separates "loud" from "loud *and* actually a fight".

    Requires ``pip install hypecut[ml]`` (torch + open_clip_torch).

    Params
    ------
    model: open_clip model name (default ``ViT-B-32``).
    pretrained: weights tag (default ``laion2b_s34b_b79k``).
    positive / negative: prompt lists.
    weight: 0-1 blend against the heuristic score (default 0.5).
    """

    description = "CLIP image-text rescoring of candidate peaks (needs [ml] extra)."

    DEFAULT_POSITIVE = [
        "an intense video game firefight",
        "a player scoring a kill in a first person shooter",
        "an explosion and visual effects in a video game",
        "a celebration after winning a round",
    ]
    DEFAULT_NEGATIVE = [
        "a video game main menu screen",
        "a loading screen",
        "a player standing still doing nothing",
        "an inventory or settings screen",
    ]

    def available(self) -> tuple[bool, str]:
        try:
            import open_clip  # noqa: F401
            import torch  # noqa: F401
        except ModuleNotFoundError as exc:
            return False, f"clip_rerank needs `pip install hypecut[ml]` ({exc.name})"
        return True, ""

    def refine(self, info: VideoInfo, candidates: list[Candidate]) -> list[Candidate]:
        ok, reason = self.available()
        if not ok:
            for cand in candidates:
                cand.meta["clip_rerank"] = "skipped"
            import warnings

            warnings.warn(reason, stacklevel=2)
            return candidates

        import open_clip
        import torch
        from PIL import Image

        model_name = self.params.get("model", "ViT-B-32")
        pretrained = self.params.get("pretrained", "laion2b_s34b_b79k")
        blend = float(self.params.get("weight", 0.5))
        positive = list(self.params.get("positive", self.DEFAULT_POSITIVE))
        negative = list(self.params.get("negative", self.DEFAULT_NEGATIVE))

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        model = model.to(device).eval()
        tokenizer = open_clip.get_tokenizer(model_name)

        with tempfile.TemporaryDirectory(prefix="hypecut-clip-") as tmp:
            images = []
            for idx, cand in enumerate(candidates):
                at = cand.meta.get("peak_time", (cand.start + cand.end) / 2)
                frame = Path(tmp) / f"{idx:04d}.jpg"
                subprocess.run(
                    cmd(
                        "ffmpeg -v error -nostdin -y -ss {at} -i {src} -frames:v 1 {dest}",
                        at=f"{at:.3f}",
                        src=info.path,
                        dest=str(frame),
                    ),
                    check=False,
                )
                if frame.exists():
                    images.append((idx, preprocess(Image.open(frame).convert("RGB"))))

            if not images:
                return candidates

            with torch.no_grad():
                batch = torch.stack([im for _, im in images]).to(device)
                image_features = model.encode_image(batch)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                text = tokenizer(positive + negative).to(device)
                text_features = model.encode_text(text)
                text_features /= text_features.norm(dim=-1, keepdim=True)

                sim = (image_features @ text_features.T).cpu().numpy()

            n_pos = len(positive)
            for row, (idx, _) in enumerate(images):
                pos = float(sim[row, :n_pos].max())
                neg = float(sim[row, n_pos:].max())
                semantic = max(0.0, min(1.0, (pos - neg) * 5.0 + 0.5))
                cand = candidates[idx]
                cand.score = (1 - blend) * cand.score + blend * semantic
                cand.reasons["clip_semantic"] = round(semantic, 4)
        return candidates


@register("speech_keywords")
class SpeechKeywords(Refiner):
    """Boost clips whose transcript contains reaction words.

    "Oh my god", "let's go", "no way" are near-perfect highlight labels and
    the streamer supplies them for free. Only candidate windows are
    transcribed, so this stays cheap even on long VODs.

    Requires ``pip install hypecut[asr]`` (faster-whisper).

    Params
    ------
    keywords: list of lowercase substrings to reward.
    boost: additive score per matched window (default 0.15).
    model: whisper size (default ``base``).
    """

    description = "Transcribe candidates and boost reaction keywords (needs [asr])."

    DEFAULT_KEYWORDS = [
        "oh my god",
        "let's go",
        "lets go",
        "no way",
        "what the",
        "insane",
        "clutch",
        "got him",
        "holy",
        "ace",
        "triple",
        "quad",
        "不會吧",
        "太扯了",
        "誇張",
        "神啊",
        "衝啊",
    ]

    def available(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except ModuleNotFoundError as exc:
            return False, f"speech_keywords needs `pip install hypecut[asr]` ({exc.name})"
        return True, ""

    def refine(self, info: VideoInfo, candidates: list[Candidate]) -> list[Candidate]:
        ok, reason = self.available()
        if not ok:
            import warnings

            warnings.warn(reason, stacklevel=2)
            return candidates
        if not info.has_audio:
            return candidates

        from faster_whisper import WhisperModel

        keywords = [k.lower() for k in self.params.get("keywords", self.DEFAULT_KEYWORDS)]
        boost = float(self.params.get("boost", 0.15))
        model = WhisperModel(self.params.get("model", "base"), compute_type="int8")

        with tempfile.TemporaryDirectory(prefix="hypecut-asr-") as tmp:
            for idx, cand in enumerate(candidates):
                wav = Path(tmp) / f"{idx:04d}.wav"
                subprocess.run(
                    cmd(
                        "ffmpeg -v error -nostdin -y -ss {start} -t {dur} -i {src} "
                        "-vn -ac 1 -ar 16000 {dest}",
                        start=f"{cand.start:.3f}",
                        dur=f"{cand.duration:.3f}",
                        src=info.path,
                        dest=str(wav),
                    ),
                    check=False,
                )
                if not wav.exists():
                    continue
                segments, _ = model.transcribe(str(wav), beam_size=1)
                text = " ".join(s.text for s in segments).lower()
                hits = [k for k in keywords if k in text]
                if hits:
                    cand.score += boost * min(3, len(hits))
                    cand.reasons["speech_keywords"] = float(len(hits))
                    cand.meta["transcript"] = text.strip()[:280]
        return candidates
