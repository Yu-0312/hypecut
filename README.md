# HypeCut

**Drop in a VOD, get back the good parts.**

HypeCut watches a long gameplay recording, finds the moments that matter, and
stitches them into a single highlight reel — with a machine-readable cut list
explaining why every clip was chosen.

```bash
hypecut cut vod.mp4 -o reel.mp4 --target 120
```

```
3 clips, 118.4s reel from 5412.0s source (2.2% kept)
  00:14:22–00:14:41  score 0.912  top signal: roi_activity
  00:41:07–00:41:29  score 0.884  top signal: audio_transient
  01:22:55–01:23:31  score 0.871  top signal: audio_rms
```

Or run the web UI and drag the file onto the page:

```bash
pip install "hypecut[web]"
hypecut serve            # → http://127.0.0.1:8000
```

---

## Why another highlight tool

Most auto-highlight tools are either a black box behind an API, or a single
heuristic ("loud = good") wearing a UI. HypeCut is built around two ideas:

**1. Hybrid detection, in that order.** Cheap signals run over the entire
video and propose generous candidates. Expensive models — CLIP, Whisper — run
only on those few dozen windows. You get model-grade judgement at heuristic
cost, and the expensive stage is always optional. A three-hour VOD is analysed
on a laptop CPU in about a minute; the optional CLIP pass adds seconds, not
hours, because it never touches the boring 98%.

**2. Every decision is inspectable.** Each clip carries the per-signal scores
that produced it. The JSON sidecar and the EDL export mean HypeCut can be your
first pass rather than your only pass — open the cut in Resolve or Premiere and
finish by hand.

## Install

Needs Python 3.10+ and **ffmpeg** on your `PATH`.

```bash
pip install hypecut          # CLI + library
pip install "hypecut[web]"   # + the upload UI
pip install "hypecut[ml]"    # + CLIP semantic reranking
pip install "hypecut[asr]"   # + Whisper reaction-keyword detection
```

Docker, if you'd rather not think about it:

```bash
docker compose up -d         # → http://localhost:8000
```

## Usage

### Web UI

`hypecut serve` starts a single-page app: drop a file, set the reel length and
sensitivity, watch the progress bar, download the result. Uploads and outputs
stay on your machine — there is no external service anywhere in the pipeline.

### CLI

```bash
# The basics
hypecut cut vod.mp4 -o reel.mp4

# A tighter, more selective reel
hypecut cut vod.mp4 --target 90 --max-clips 8 --percentile 95

# Use a game profile and turn on the CLIP reranker
hypecut cut vod.mp4 --profile configs/fps-shooter.yaml --refiner clip_rerank

# See the cut list without spending an encode
hypecut analyze vod.mp4 --json plan.json

# What detectors are available?
hypecut signals
```

### Python

```python
from hypecut import analyze, render_plan, load_config

cfg = load_config("configs/moba.yaml")
plan = analyze("vod.mp4", cfg)

# Inspect and edit the cut before rendering
for clip in plan.segments:
    print(clip.start, clip.end, clip.score, clip.reasons)
plan.segments = [c for c in plan.segments if c.score > 0.8]

render_plan(plan, "reel.mp4", cfg)
```

## How it works

```
       ┌──────────┐
video ─┤ decode×1 ├─► 10 Hz grid: tiny grayscale frames + mono audio
       └──────────┘
            │
            ├─► audio_rms ───────┐
            ├─► audio_transient ─┤
            ├─► scene_change ────┼─► normalise → weight → sum → smooth
            ├─► motion ──────────┤        (the excitement curve)
            └─► roi_activity ────┘
                                          │
                        top-N% regions ───┴──► candidates (+pre/post roll)
                                                   │
                     stage 2 · refiners ───────────┤   diversity, pacing,
                     (candidates only)             │   clip_rerank, speech_keywords
                                                   │
                          merge → budget select ───┴──► ffmpeg cut + concat
                                                          │
                                        reel.mp4 · .hypecut.json · .edl
```

The video is decoded exactly once, into a 96×54 grayscale plane at 10 Hz and
16 kHz mono audio — about 180 MB of RAM per hour of footage. Every signal reads
from that shared buffer, so adding a detector costs milliseconds rather than
another decode pass.

Signals are normalised with median/MAD rather than mean/σ, so a single
explosion can't flatten the rest of the curve into noise. Clips are grown
around their peak, not their leading edge, so the wind-up survives — a kill
without the approach reads as a jump cut.

Full detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Profiles

A profile is a small YAML file. Shipped ones:

| Profile | For | What's different |
|---|---|---|
| `default.yaml` | anything | balanced weights, 2-minute reel |
| `fps-shooter.yaml` | VALORANT, CS2, R6 | kill-feed ROI weighted heavily, short tight clips |
| `moba.yaml` | LoL, Dota 2 | long pre-roll for the engage, caster voice band |
| `just-chatting.yaml` | talk streams, podcasts | visual signals off, long clips, keyword boost |

Copy one, change numbers, pass `--profile my.yaml`. No Python required.

## Writing a detector

A signal answers one question about every moment: *how interesting is now, by
my measure?* Roughly twenty lines:

```python
from hypecut.signals import Signal, register

@register("chat_spike")
class ChatSpike(Signal):
    """Messages per second from a Twitch chat log."""
    description = "Chat message rate — the audience already did the labelling."

    def compute(self, ctx):
        import numpy as np
        rate = np.zeros(ctx.n)
        for ts in load_chat(self.params["log"]):
            idx = int(ts * ctx.grid_fps)
            if 0 <= idx < ctx.n:
                rate[idx] += 1
        return rate
```

Register it under the `hypecut.signals` entry-point group and it appears in
`hypecut signals` for everyone who installs your package. Same story for
refiners. See [docs/EXTENDING.md](docs/EXTENDING.md).

## Roadmap

Near-term: shot-boundary-aware cut points, per-clip vertical reframing for
Shorts/TikTok, a proper queue backend for multi-user deployments, and community
profiles for more games. Details and open design questions in
[docs/ROADMAP.md](docs/ROADMAP.md) — that file is the best place to find
something to work on.

## Contributing

Issues and PRs welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). The most useful
contribution right now is a **game profile** — you don't need to write Python,
just tune a YAML file against footage you know well and open a PR.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

HypeCut only ever reads files you give it. It ships no models, downloads
nothing at runtime unless you enable an optional refiner, and sends nothing
anywhere.
