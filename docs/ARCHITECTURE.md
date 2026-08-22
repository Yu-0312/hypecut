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
                                              merge + select ──────┤
                                                                   │
                                     snapping + reframe planning ──┤ post
                                                                   │
                                                        HighlightPlan
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
| `crowd_roar` | audio | sustained stadium noise, measured as a plateau not a spike |
| `whistle` | audio | narrowband tonal burst — referee whistles |
| `roi_change` | frames | change *isolated* to a box — scoreboards, clocks, counters |

The last three exist because sport breaks assumptions the first seven were
built on; `docs/EXTENDING.md` covers the reasoning. The pair worth contrasting
is `roi_activity` and `roi_change`: the first measures how busy a region is,
the second measures how much busier it is *than the rest of the frame*. A
camera cut maxes out the first and cancels to zero in the second, which is why
a scoreboard needs the second one.

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

* **`reaction_lag` shifts only the in-point.** When the evidence for a moment
  arrives after the moment — a crowd roar following a goal — the clip has to
  start earlier to contain the play at all, but the reaction is worth keeping,
  so the out-point stays. The moment (`peak_time`) and the evidence
  (`reaction_time`) are both recorded, and every later guard uses the former.
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

## Post-selection: snapping and reframing

Two steps run after selection and before rendering. Both need the decoded
frames, and both are decisions about *the cut* rather than about the encode —
so they live on the analysis side and travel to the renderer inside each
clip's metadata. That keeps `analyze()` / `render_plan()` split intact: a
caller can inspect or edit the plan, and the sidecar JSON records exactly what
was decided.

### Shot-boundary snapping (`snapping.py`)

Where an edge lands is the most visible difference between an auto-cut and a
hand-cut reel. Three seconds of wind-up is a guess; a hard cut two seconds
before the peak is *evidence*. So every edge may travel to the nearest real
boundary.

Detection is coarse-then-fine. Frame difference over a running median gives
"how unusual is this difference for this stretch of video", which is what a cut
actually is — raw difference is useless, because a chaotic teamfight differs
more frame-to-frame than a hard cut in a menu does. Peaks above 2.5× baseline
become candidate boundaries; each accepted edge is then re-decoded at the
source frame rate over a one-second window, turning a ±50 ms answer into a
frame-exact one for the cost of a few dozen tiny frames.

Three rules keep it honest:

* **The peak is a hard stop.** No edge may cross it — that moment is the
  entire reason the clip exists.
* **The in-point and out-point are not symmetric.** The in-point may move
  right up to the peak: if a hard cut sits between the old start and the peak,
  the wind-up it would have kept belonged to a *different scene*, and opening
  on unrelated footage is worse than opening tight. The out-point keeps a
  guard, because there is no equivalent argument for cutting the payoff short.
* **A snap that breaks the length budget is refused, not clamped.** A clip
  suddenly a second under `min_duration` is a worse outcome than an unsnapped
  edge.

The travel allowance is `max(snap_window, pre_roll)`. A fixed 2 s window can
never reach the cut that started the scene when the pre-roll placed the edge
3 s earlier, and that is precisely the case snapping exists for.

Rounding is deliberately late rather than early: the returned time is the
frame *after* the largest difference. One frame late is invisible; one frame
early shows a flash of the outgoing shot — exactly the artefact this removes.

Two kinds of boundary are detected. A **hard cut** is one enormous frame
difference. A **dissolve** spreads the same total change over a second or
two, so no single frame stands out and a peak detector walks straight past
it; the answer is to look at difference *accumulated* over a run rather than
the maximum within it. That alone would also flag a camera pan — likewise
sustained, likewise spike-free — so the discriminator is spatial contrast: a
blend of two images is flatter than either, and a pan's contrast is not.

Cuts detected *inside* a dissolve are dropped. A one-second crossfade is not
spike-free at 10 Hz and some of its frames clear the cut detector's bar on
their own, but they are the middle of one transition, not several — landing
an edge there would put the clip in the mix between two shots.

There is also an absolute floor on frame difference (1.5 luma levels), and it
earns its place. On footage that holds perfectly still — a menu, a paused
game — the local baseline collapses toward zero and a single frame of
compression flicker measures as *hundreds of times* the baseline. Relative
evidence alone calls that a shot change. The floor stays low because a cut
between two dark scenes can be worth only two or three levels.

### Silence-aware trimming (`trimming.py`)

The audio half of the same problem, and the only half available on footage
with no cuts in it — a locked-off webcam stream has exactly one shot.

An edge belongs in a gap, not in the middle of a sound. The in-point lands
where sound resumes; the out-point lands where it stops. Both keep a small pad
so the clip does not open or close on a hard transient.

"Quiet" is measured against the clip's own speech level (75th percentile minus
14 dB), never a global threshold: a whispered aside and a shouted play share no
absolute dBFS range but both have the same *gap* between talking and not. Two
guards stop it inventing pauses — if almost nothing is below the threshold
(continuous sound) or almost everything is (continuous quiet), the clip is left
alone rather than moved on noise.

**Precedence is the important part.** Trimming only ever considers edges that
found no shot boundary. A hard cut is unambiguous evidence about where a moment
ends; a pause is a good guess. Running both and letting the second overwrite
the first would mean the weaker signal decides.

One by-product feeds the renderer: whether the clip's last moment is quiet.
The video fade stays constant so the reel keeps one visual rhythm, but the
audio fade lengthens for clips cut off mid-sound, where a hard stop would read
as a dropout. That flag is computed even when there are no usable pauses —
a clip buried inside continuous sound is precisely the one that needs it.

### Vertical reframing (`reframe.py`)

A 16:9 clip in a 9:16 slot loses two thirds of its height. Three ways out:

* **`crop`** — a 9:16 slice, positioned from motion energy. Column-wise motion
  mass summed over rows gives a distribution across the frame; its centroid is
  where things are happening. A crude estimator is fine here: the crop is 56%
  of the frame wide, so it only has to be right to within a few percent.
* **`stack`** — facecam pane over gameplay pane, both from normalised boxes in
  the profile. No analysis needed.
* **`blur_pad`** — the whole frame over a blurred blow-up of itself. Loses
  nothing, wastes half the screen; the right default when the important thing
  might be anywhere (minimaps, scoreboards).

With `react_to_facecam`, the crop centre is pulled toward the facecam box
during the moments that box is busy. The reaction is as often the highlight as
the play is, and a motion centroid — being an average — frames the midpoint
between the two, which shows neither. "Busy" is measured against the *25th
percentile* of facecam activity in the clip, not the median: a reaction filling
most of the clip would drag the median up into itself and then measure as
normal, the same mistake as normalising a signal against a window containing
the thing you are detecting. No detector is involved; the box comes from the
profile, which keeps this dependency-free and correct for whatever layout the
streamer actually uses.

Panning is off by default. A crop that chases every centroid wobble reads as
camera drift, and a still frame placed at the *median* of the action is what
most hand-made vertical edits do. With `track: true` the centre is smoothed
over ~2.5 s and then velocity-limited, which turns a subject teleporting across
the screen into a slow push instead of a jump — what a camera operator would
do, and what a viewer can follow.

The pan becomes a piecewise-linear ffmpeg expression over a handful of
keyframes rather than a per-frame command stream. It is bounded in size, it
survives being written to and read back from JSON, and it needs no extra
tooling — at the cost of straight-line interpolation between keyframes, which
at six keyframes over a ten-second clip is imperceptible.

One constraint shapes the code: the whole filter chain is a single argv token,
so no filter string may contain whitespace, and expressions carrying commas are
single-quoted so ffmpeg does not read them as filter separators.

## The protected span

`build_candidates` derives a clip from an above-threshold region and then
records that region's bounds on the clip as `event_start` / `event_end`.
Everything downstream that moves an edge — snapping, trimming, the length
clamp in `merge` — asks `Candidate.protected()` which part must survive.

This used to be a single number, `peak_time`, and that was a real bug rather
than a simplification. For a goal the loudest frame *is* the event, so a point
describes it fine. For a twenty-second rally the maximum can be the third
shot of twelve, and a guard placed there permits an edge to be dragged to it —
legally deleting the first quarter of the exchange. Carrying the span costs
two floats and removes the whole class of error.

Degenerate spans fall back to the peak, so hand-built clips and anything
without event bounds behave exactly as before.

## Loudness, between clips and not just inside them

`dynaudnorm` is a within-clip tool: it evens out dynamics in one piece of
audio and knows nothing about the piece before it. The artefact everyone
actually notices — a reel that jumps in volume halfway — lives *between*
clips, so it needs a measurement that spans them.

Two passes, because integrated loudness is a property of a whole clip and
there is no way to know the right gain until you have heard it all: measure
each segment through the same filter chain the encode will use, then prepend a
static `volume` gain toward the target. Measuring the raw segment and then
applying a compressor would compute a gain for audio that no longer exists,
so the chain is built once and shared by both passes.

`loudness_match` defaults to 0.9, not 1.0. Full matching is the obvious
choice and the wrong one: it makes a whispered aside and a stadium roar
equally loud. At 0.9 the spread collapses to a tenth — inaudible as a jump,
still audible as character.

## Variants — one analysis, several aspect ratios

`Config.variants` maps a name to a partial `render` override. Everything
before the encode is shared: one decode, one set of signals, one curve, one
selection, one set of snapped edges. Only the encode runs per variant.

The subtle part is framing. A vertical cutdown is not a letterboxed copy of
the landscape reel — it needs its own crop centre, computed from the motion
in each clip. That computation needs the decoded frames, which exist only
during analysis, so *every* framing the run will produce is planned during
`analyze()` and stored under its own metadata key (`reframe:vertical`). The
renderer then stays a pure function of the plan, and the sidecar records how
every variant was framed.

The alternative — re-deriving the crop at render time — would either mean
keeping the frames alive far past their usefulness or falling back to a naive
centre crop. The first costs memory for the whole render; the second throws
away the feature.

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

* **No parallel encoding of variants.** They are rendered one after another.
  ffmpeg already saturates the cores it is given, and running three encodes
  at once on a laptop makes all three slower while making progress reporting
  meaningless.
* **No agent mode.** `contact-sheet`, `render --plan` and the JSON
  catalogues exist because they are useful to people too. A separate code
  path for models would be a second path to keep working, and the first sign
  it had rotted would be a model quietly doing the wrong thing.
* **No database.** Nothing here needs to survive a restart. Jobs are ephemeral
  by nature and a schema is a maintenance burden a v0.1 should not carry.
* **No model in the default path.** The tool must work fully offline on a CPU
  with nothing downloaded. Models are opt-in extras, never a hard dependency.
* **No optical flow or object tracking for reframing.** Motion centroid on
  already-decoded frames is ~free and good enough for a crop that wide. A
  tracker would add a dependency, a model, and failure modes, to move the crop
  by a few percent.
* **No per-game hardcoding in Python.** Game knowledge lives in YAML profiles.
  A contributor who knows Rocket League should not have to learn this codebase
  to encode what they know.
* **No cloud anything.** No telemetry, no remote calls, no accounts.
