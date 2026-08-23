# HypeCut

**Drop in a VOD, get back the good parts.**

HypeCut watches a long recording — a gameplay VOD, a football match, a lecture
— finds the moments that matter, and stitches them into a single highlight
reel, with a machine-readable cut list explaining why every clip was chosen.

```bash
hypecut cut vod.mp4 -o reel.mp4 --target 120
```

```
3 clips, 118.4s reel from 5412.0s source (2.2% kept)
2/3 clips moved onto a shot boundary
2/3 clips had an edge moved into a pause
  00:14:22–00:14:41  score 0.912  top signal: roi_activity     [snap start-0.84s]
  00:41:07–00:41:29  score 0.884  top signal: audio_transient  [snap start-1.20s; pause end+0.43s]
  01:22:55–01:23:31  score 0.871  top signal: audio_rms        [pause start+0.31s end+0.67s]
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

`hypecut serve` starts a single-page app. It asks two questions in plain
language — what kind of video is this, what shape do you want — and neither
is required; drop a file and press the button. Thresholds, detectors and edge
placement are all there under "Advanced" for anyone who wants them, and
folded away for everyone who does not. Uploads and outputs stay on your
machine — there is no external service anywhere in the pipeline.

### CLI

```bash
# The basics
hypecut cut vod.mp4 -o reel.mp4

# A tighter, more selective reel
hypecut cut vod.mp4 --target 90 --max-clips 8 --percentile 95

# Use a game profile and turn on the CLIP reranker
hypecut cut vod.mp4 --profile configs/fps-shooter.yaml --refiner clip_rerank

# Vertical for Shorts/Reels/TikTok — crop follows the action
hypecut cut vod.mp4 --vertical --reframe-track
hypecut cut vod.mp4 --profile configs/shorts.yaml
hypecut cut vod.mp4 --reframe stack      # facecam on top, gameplay below

# Vertical that commits to the facecam while the streamer is reacting
hypecut cut vod.mp4 --vertical --react --facecam 0,0,0.26,0.3

# One analysis, three aspect ratios
hypecut cut vod.mp4 --also vertical --also square

# A folder of recordings, one reel each
hypecut batch ~/Recordings -o ~/Reels --recursive

# Leave the edges exactly where the rolls put them
hypecut cut vod.mp4 --no-snap --no-trim

# See the cut list without spending an encode
hypecut analyze vod.mp4 --json plan.json

# Edit plan.json by hand, then render exactly that
hypecut render plan.json -o reel.mp4

# One labelled image showing what the footage is (or what got picked)
hypecut contact-sheet vod.mp4 -o sheet.png
hypecut contact-sheet vod.mp4 --plan plan.json -o picked.png

# What profiles exist, and what each is for
hypecut profiles

# What detectors are available?
hypecut signals

# Is a profile actually better? Mark a video, then score profiles against it
hypecut label vod.mp4                       # writes vod.labels.yaml + a sheet
hypecut eval vod.labels.yaml -p configs/sports-broadcast.yaml -p configs/default.yaml
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
            ├─► motion ──────────┤       = the excitement curve
            └─► roi_activity ────┘
                                 │
                                 ▼
                   top-N% regions → candidates (+ pre/post roll)
                                 │
        stage 2 · refiners ──────┤  diversity · pacing · similarity
        (candidates only)        │  clip_rerank · speech_keywords
                                 ▼
                        merge → budget select
                                 │
         snap edges to real cuts ┤  hard cuts and dissolves
        trim the rest to pauses ─┤  only edges no cut claimed
        plan each framing ───────┤  one per aspect ratio wanted
                                 ▼
                        ffmpeg cut + concat
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              reel.mp4   reel_vertical   reel_square
                    └──── .hypecut.json · .edl ────┘
             (or reel.part1.mp4, reel.part2.mp4 … when
              there is more here than fits in one reel)
```

The video is decoded exactly once, into a 96×54 grayscale plane at 10 Hz and
16 kHz mono audio — about 180 MB of RAM per hour of footage. Every signal reads
from that shared buffer, so adding a detector costs milliseconds rather than
another decode pass.

Signals are normalised with median/MAD rather than mean/σ, so a single
explosion can't flatten the rest of the curve into noise. Clips are grown
around their peak, not their leading edge, so the wind-up survives — a kill
without the approach reads as a jump cut.

**Edges land on real cuts.** Before rendering, every clip edge is allowed to
travel up to a couple of seconds to reach an actual shot boundary — a round
transition, a killcam, a scene switch. A clip that starts three frames into a
continuous shot looks *sliced*; the same clip started on the cut looks
*edited*. Boundaries are found on the 10 Hz frames already in memory, then
each accepted edge is re-checked at the source frame rate. Nothing may cross
into the clip's *event* — the above-threshold span it was built around, not
merely its loudest frame — and a snap that would break the length budget is
refused. That distinction is not pedantry: in a twenty-second rally the
loudest frame can be the third shot, and a guard placed there would let the
first quarter be trimmed away.

Crossfades and fades count too, and they get treated as the intervals they
are: an in-point lands on the *far* side of a dissolve so the clip opens on
the incoming shot, an out-point on the *near* side so it leaves before the
picture starts mixing away.

**What the cuts miss, the pauses catch.** A locked-off talking-head stream has
no shot boundaries at all, so snapping has nothing to work with and edges land
three words into a sentence. Any edge that found no boundary is then moved into
the nearest pause instead — measured against that clip's own speech level, so a
whispered aside and a shouted play are both handled. A real cut always wins:
trimming never touches an edge snapping already decided.

**The reel sounds like one piece.** `dynaudnorm` evens out dynamics inside a
clip and says nothing about how two clips compare, which is exactly the
artefact people notice when a reel jumps in volume halfway through. So there
is a measurement pass: every clip's integrated loudness (EBU R128), then a
static gain toward a target. Matching is 0.9 rather than 1.0 on purpose —
flattening every clip to the same number makes a quiet moment and a stadium
roar equally loud, which is technically correct and editorially wrong.

**Vertical is a crop, not a letterbox.** `--vertical` takes a 9:16 slice
centred on where the motion actually is, computed per clip from the same
decoded frames. Add `--reframe-track` and the crop pans to follow the action,
velocity-limited so it reads as a camera push rather than a twitch. The
alternatives are there too: `--reframe stack` for facecam-over-gameplay, and
`--reframe blur_pad` when the whole frame matters. With `--react` and a
`--facecam` box, the crop commits to the streamer while the webcam is busy and
returns to the action when it isn't — the reaction is half the highlight, and a
crop that averages the two frames neither.

**One analysis, several aspect ratios.** `--also vertical --also square`
renders extra cutdowns from the same decode and the same cut decisions — each
framing is planned separately while the frames are still in memory, so the
vertical crop is centred on its own action track rather than being a
letterboxed copy. Only the encode is repeated.

Full detail in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Letting an AI drive it

Everything above is also reachable by an agent, and the pieces that make that
work are deliberate rather than incidental: `analyze` and `render` are
separate steps, the cut list is JSON that carries the reasoning *and* the
config, and `contact-sheet` hands a model one labelled image so it can
actually see the footage instead of guessing from a filename.

Two ways to use it:

- **Give an AI the repo.** [AGENTS.md](AGENTS.md) is the contract — the loop,
  the commands, the judgement calls. No special agent mode exists; the same
  commands do the same things for a person.
- **Install the skill.** `skill/hypecut/` (packaged as `hypecut.skill`) is a
  thin conversational wrapper: drop a video into a chat, get a reel back. It
  deliberately duplicates nothing — profile descriptions come from
  `hypecut profiles`, detector descriptions from `hypecut signals --json`, so
  the skill cannot drift out of sync with the tool.

Both need a shell, a filesystem and ffmpeg.

## Profiles

A profile is a small YAML file. Shipped ones:

| Profile | For | What's different |
|---|---|---|
| `default.yaml` | anything | balanced weights, 2-minute reel |
| `fps-shooter.yaml` | VALORANT, CS2, R6 | kill-feed ROI weighted heavily, short tight clips |
| `moba.yaml` | LoL, Dota 2 | long pre-roll for the engage, caster voice band |
| `just-chatting.yaml` | talk streams, podcasts | visual signals off, long clips, keyword boost |
| `shorts.yaml` | Shorts / Reels / TikTok | 9:16 tracking crop, picky, short punchy clips |
| `sports-broadcast.yaml` | televised football, basketball, hockey | crowd roar, whistle, scoreboard, reaction lag |
| `sports-field.yaml` | phone on the sideline, club matches | crowd roar + motion only, no cuts to snap to |

Copy one, change numbers, pass `--profile my.yaml`. No Python required.

### Sport is not gameplay

Three assumptions break when you point this at a match, and the sports
profiles exist because of them.

**The moment is silent; the reaction is not.** A goal makes no sound. The roar
arrives a second or two later and then *holds*. Detectors built for gameplay —
where the kill and its sound are simultaneous — fire on the roar and produce a
clip that starts *after* the goal. `segments.reaction_lag` shifts the in-point
back by that offset, and only the in-point: the celebration is worth keeping.

**A roar is a plateau, not a spike.** `crowd_roar` takes a rolling *minimum*,
so a door slam or a clipped microphone cannot survive it while eight seconds
of crowd can. On the test footage, plain loudness picks a brief shout over the
goal; `crowd_roar` does not.

**The scoreboard is ground truth, not a proxy.** `roi_change` measures a small
box *against the rest of the frame*, so a digit flipping registers and a camera
cut — which moves the box and everything else equally — cancels to zero.

The same three questions are how you'd adapt HypeCut to anything else: when
does the evidence arrive relative to the event, is the evidence an edge or a
plateau, and is there a region of the frame that already knows the answer.
[docs/EXTENDING.md](docs/EXTENDING.md) walks through it.

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

## When there is nothing to cut, you get nothing

Every threshold in HypeCut is relative to the video it is given — a quiet VOD
and a loud one should both yield reels — which means a percentile always
selects *something*. Feed in three hours of an idle lobby and older versions
returned a confident reel of its least-boring moments.

`segments.min_prominence` is the one cross-video check: how far the best
moment stands above that video's own background, measured per signal in its
own units. Below the bar, you get no reel and a sentence saying why.

```
$ hypecut cut afk-stream.mp4
Nothing to cut: nothing in this video stands out from its own background
(prominence 2.2, needs 4.0).
```

`hypecut batch` counts those separately from failures, because a folder of
recordings normally contains a few with nothing in them. If you disagree with
the verdict, lower `min_prominence` — or set it to 0 to skip the check.

## A long recording becomes several reels

A three-hour match has more than one reel's worth of highlights in it, and
truncating to the best twenty clips throws away the second half. Past
`clips_per_reel` (10) clips or `target_duration` seconds, the cut spills into
the next part:

```
3 reels — the cut was too long for one:
Part 1:  match.part1.mp4  (10 clips)
Part 2:  match.part2.mp4  (10 clips)
Part 3:  match.part3.mp4  (4 clips)
```

Parts are chronological, so the reel still tells the match's story front to
back, and each carries its own cut list and EDL. There is no cap on how many
there can be. `max_clips` is now the only setting that discards a highlight
rather than moving it; it defaults to no cap.

## Replays are not duplicates

The `diversity` refiner spreads clips out by *time*, which is a proxy for
sameness and a poor one: it penalises a save thirty seconds after a goal and
lets through the fifth identical spawn-camp kill because they were minutes
apart.

`similarity` asks the better question — do these two clips *move* the same
way — and then uses time only to interpret the answer:

| | close together | far apart |
|---|---|---|
| **look alike** | the same event again: a replay, another angle. **Kept**, and tagged with a shared `moment` id | the same thing happening twice. The weaker take is demoted |
| **look different** | untouched | untouched |

That first cell is the whole design. A broadcast shows the goal, the slow
motion, and the angle from behind the net; a reel that keeps all three is not
repeating itself, it is edited. Cutting them would be removing the edit.

It compares frame *differences*, not frames. Averaged frames of a
locked-camera football match are the same green rectangle every time — every
pair scores above 0.99 and the whole video reads as one repeated moment.
Motion cancels the background and describes the play. Cost is one pass over
frames already in memory: no model, no extra decode, no new dependency.

## Knowing whether it worked

Every threshold in this project started as a reasoned guess. `hypecut eval`
is how you find out whether a guess was right — and how a profile PR gets
reviewed on evidence instead of on the author's confidence.

```bash
hypecut label match.mp4 --annotator max        # writes match.labels.yaml + a sheet
# open the sheet, then edit the file: keep: true / keep: false,
# and add entries by hand for anything it missed
hypecut eval match.labels.yaml -p configs/default.yaml -p configs/sports-broadcast.yaml
```

```
profile               clips        found   prec  recall     F1  cover
default                   3     1/1        0.33    1.00   0.50   0.54
sports-broadcast          1     1/1        1.00    1.00   1.00   1.00
```

Three things about that table are deliberate:

**A clip hits when it *contains* the moment**, not when the edges line up.
"Did you find it" and "did you frame it well" are different questions with
different fixes, so they get different columns.

**`cover` is that second question** — how much of the labelled moment
survived into the clip. Perfect recall with low coverage means the detector
is right and the rolls are too tight. A single blended score would hide that.

**Labels carry no video.** A labels file is a path plus timestamps, so you
can publish an answer key for footage you cannot redistribute. It also names
one annotator: two people mark different highlights in the same match, so
comparing profiles against one key is an experiment and comparing scores
across keys is not.

`hypecut label` over-proposes on purpose. Throwing away a bad proposal takes
a second; noticing a moment nobody proposed takes watching the video, which
is why the file asks you to add those by hand — they are the failures a score
would otherwise never see.

## Roadmap

Near-term: auto-locating the facecam, wipe detection, VOD URLs as input, a
chat-log signal, a proper queue backend for multi-user deployments, and
community profiles for more games. Details and open design questions in
[docs/ROADMAP.md](docs/ROADMAP.md) — that file is the best place to find
something to work on.

## Publishing your own copy

```bash
bash scripts/push-to-github.sh <your-github-username> hypecut public
```

Ordinary git and `gh` — read it before running it. It commits, creates the
repository, pushes `main` and tags the current version. Without `gh` it
prints the manual steps instead.

## Contributing

Issues and PRs welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). The most useful
contribution right now is a **game profile** — you don't need to write Python,
just tune a YAML file against footage you know well and open a PR.

## Licence

Apache-2.0. See [LICENSE](LICENSE).

HypeCut only ever reads files you give it. It ships no models, downloads
nothing at runtime unless you enable an optional refiner, and sends nothing
anywhere.
