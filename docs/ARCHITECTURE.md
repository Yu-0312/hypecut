# Architecture

## The shape of the problem

A highlight detector has to answer two questions that pull in opposite
directions:

1. **Where is something happening?** — cheap to approximate, needs to run over
   every second of a possibly multi-hour file.
2. **Is that thing actually interesting?** — expensive to judge well, and the
   good judges (vision-language models, ASR) cost orders of magnitude more per
   second than the video is worth.

Running a model over the whole VOD answers (2) well and (1) wastefully. Running
only heuristics answers (1) cheaply and (2) badly — loudness alone happily
selects the streamer sneezing.

HypeCut splits the two. **Stage 1** is cheap and generous: it runs over
everything and over-proposes. **Stage 2** is expensive and selective: it only
ever sees the ~30 windows stage 1 nominated. This is the single structural
decision the rest of the codebase follows from.

## Data flow

```
VideoInfo ──► AnalysisContext ──► [SignalTrack] ──► curve ──► [Candidate]
  ffprobe        decode ×1           signals        fusion      segments
                                                                   │
                                                        refiners ──┤ stage 2
                                                                   │
                                              merge + select ──► HighlightPlan
                                                                   │
                                                         render ──►  mp4 + json + edl
```

Each arrow is a module; each box is a type in `types.py`. Nothing in the chain
reaches backwards, so any stage can be tested with hand-built inputs.

## Stage 0 — decode once

`pipeline._build_context` decodes the video exactly one time into two buffers:

* **Video** — 96×54 grayscale, sampled at 10 Hz. ~5 KB/frame, so an hour is
  ~180 MB. Small enough to hold, large enough for scene changes, motion
  coverage, and a HUD region of interest.
* **Audio** — mono float32 at 16 kHz, reshaped on demand into one row per grid
  step.

Both live on an `AnalysisContext`. Signals are forbidden from touching the file
themselves; this is what keeps a ten-signal profile as fast as a two-signal
one.

**Why 10 Hz?** It is a hundred-millisecond resolution — finer than any cut
point a human would place, coarse enough that an hour of video is 36,000
samples, which numpy treats as a rounding error.

**Why grayscale?** Colour buys almost nothing for change detection and triples
the memory. The one place colour would help — team-coloured kill feeds — is
better served by a purpose-built plugin that decodes its own crops.

## Stage 1 — signals

A signal is a pure function from context to a `(T,)` float array. It does not
normalise, does not threshold, and does not know its own weight. Built-ins:

| Signal | Reads | Catches |
|---|---|---|
| `audio_rms` | audio | sustained loudness — shouting, sustained fights |
| `audio_transient` | audio | onsets — gunshots, kill stings, sudden reactions |
| `speech_band` | audio | 300–3400 Hz energy — commentary vs. game music |
| `scene_change` | frames | hard cuts, killcams, respawns, menu transitions |
| `motion` | frames | fraction of the frame moving — action density |
| `flash` | frames | global luminance jumps — explosions, ults, flashbangs |
| `roi_activity` | frames | change inside a normalised box — kill feed, scoreboard |

`roi_activity` is the interesting one. It encodes game-specific knowledge — *the
kill feed lives in the top-right* — without OCR, without a model, and without a
line of Python: the box is four numbers in a YAML profile. Most per-game tuning
turns out to be finding the right rectangle.

## Fusion

```
for each track:  z = clip((v - median(v)) / (1.4826 · MAD(v)), ±4)
curve = smooth(Σ wᵢ·zᵢ / Σ|wᵢ|, 1.5 s)  →  rescaled to [0, 1]
```

**Median/MAD, not mean/σ.** A single 12-sigma explosion under mean/σ
normalisation compresses everything else toward zero, and the rest of the video
becomes indistinguishable flatline. MAD is unmoved by a handful of extremes, so
ordinary excitement stays legible. The ±4 clip then caps what any one outlier
can contribute.

**Smoothing before thresholding, not after.** A 1.5 s moving average means no
single frame can create a clip, which kills the compression-artefact false
positives that plague frame-difference approaches.

Every step is arithmetic you can do in your head, which matters: when a user
asks why a clip was picked, `fusion.explain()` answers with the same numbers
the selector used, not a post-hoc rationalisation.

## Candidates

Regions above the *n*th percentile (default 92) become candidates. Three
non-obvious rules:

* **Pre-roll and post-roll are asymmetric** (3 s / 2 s by default). The wind-up
  to a moment is longer than the reaction to it, and a clip that starts on the
  kill reads as a jump cut.
* **Short regions grow around their peak, not their left edge.** Otherwise
  every minimum-length clip ends on its own climax.
* **Percentile, not absolute threshold.** A quiet stream and a loud one should
  both yield a reel; the top 8% of *this* video is meaningful, "above -20 dBFS"
  is not.

## Stage 2 — refiners

A refiner sees the candidate list and may rescore, retime, or reject. Because
it never touches the full video, an expensive model is affordable here.

* `diversity` — penalises clips clustered in one stretch, so a single chaotic
  teamfight can't eat the whole reel.
* `pacing` — a soft bell around a target clip length.
* `clip_rerank` — CLIP similarity of each candidate's peak frame against
  positive prompts ("an intense firefight") minus negatives ("a menu screen").
  This is the step that separates *loud* from *loud and actually a fight*.
* `speech_keywords` — Whisper on candidate windows only, boosting reaction
  phrases. The streamer labels their own highlights for free.

Refiners must degrade gracefully: `available()` reports a missing dependency
and the run continues without it. An optional feature that can fail the whole
job is not optional.

## Selection

Score-ordered greedy fill against two budgets (clip count and total duration),
then re-sorted into timeline order. Reels should still tell the match's story
front to back — chronology is free narrative structure and viewers notice when
it's missing.

One deliberate exception: if the first clip alone busts the duration budget it
is admitted anyway. A too-tight budget should produce a short reel, never an
empty one.

## Rendering

Two-pass: every segment is re-encoded to an identical intermediate, then joined
with the concat demuxer.

The tempting alternative — `-c copy` straight from the source — is much faster
and wrong here. Stream copy can only cut on keyframes, and gameplay captures
routinely run 2–10 s GOPs. A clip that starts 1.8 s late has already missed the
shot. Re-encoding costs time; landing on the wrong frame costs the highlight.

Alongside the mp4, every run writes:

* `.hypecut.json` — the full cut list with per-signal reasons and the exact
  config used, so a result is reproducible.
* `.edl` — a CMX3600 edit list, openable in Resolve/Premiere. HypeCut finds
  moments; the user may well want to finish the edit themselves, and a tool
  that refuses to hand over its work is a tool people stop trusting.
* MP4 chapters — one marker per clip, with its source timecode.

## The web layer

One FastAPI app, one background worker thread, an in-memory job store bounded
at 200 jobs with FIFO eviction that deletes files as it goes.

Deliberately **not** Celery + Redis + a broker. The realistic deployment for an
open-source tool like this is one person on one box, and the cost of that stack
is paid at install time by every one of them. The `JobStore` interface is four
methods wide, so swapping in RQ for a multi-user deployment is a contained
change rather than a rewrite.

Server paths never leave the process — `Job.public()` strips them before
serialisation.

## What is deliberately absent

* **No database.** Nothing here needs to survive a restart. Jobs are ephemeral
  by nature and a schema is a maintenance burden a v0.1 should not carry.
* **No model in the default path.** The tool must work fully offline on a CPU
  with nothing downloaded. Models are opt-in extras, never a hard dependency.
* **No per-game hardcoding in Python.** Game knowledge lives in YAML profiles.
  A contributor who knows Rocket League should not have to learn this codebase
  to encode what they know.
* **No cloud anything.** No telemetry, no remote calls, no accounts.
