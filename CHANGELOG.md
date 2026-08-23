# Changelog

All notable changes are documented here. This project follows
[Semantic Versioning](https://semver.org).

## [Unreleased]

## [0.8.0] — nothing, too much, and twice

Three changes that all answer the same complaint: the cut you get back should
match what is actually in the video.

### Added
- **"Nothing here" is now an answer.** Every threshold in HypeCut is relative
  to the video it is given — `fuse` even min-max rescales the curve to 0-1 —
  so a percentile always selected *something*, and three hours of an idle
  lobby came back as a confident reel of its least-boring moments.
  `segments.min_prominence` measures how far the best moment stands above
  that video's own background, in MAD-scale units per signal, and returns no
  segments when the answer is "not far". Being a ratio it needs no
  calibration corpus. `hypecut batch` counts such files separately from
  failures; the web UI says so in words rather than showing an error.
- **A cut too long for one reel becomes several.** Past
  `segments.clips_per_reel` (10) clips or `segments.target_duration` seconds,
  the cut spills into `reel.part2.mp4`, `reel.part3.mp4` and so on, split
  chronologically so each part still tells the story in order. Each carries
  its own cut list and EDL. There is no cap on the number of parts.
- **`similarity`** — a refiner that de-duplicates by what the frames do
  rather than by how far apart they are, at no extra decode or dependency
  cost. Enabled by default where footage varies shot to shot.

### Changed
- **`segments.max_clips` now defaults to 0 (no cap)** and the shipped
  profiles follow, except `shorts.yaml`. It is now the only setting that
  *discards* highlights; length is `target_duration` and `clips_per_reel`'s
  job. A three-hour recording no longer silently loses its second half.
- **`segments.target_duration` is per reel**, not per run.
- **The web UI asks two plain questions** — what kind of video, what shape —
  and hides every threshold behind an "Advanced" fold that nothing requires.
  It was eight technical controls before, four of which needed you to know
  what a percentile is.
- Refiners receive the decoded frames as `self.ctx`. Existing refiners are
  unaffected; it is an attribute rather than a new argument for that reason.
- `merge` carries `repeat_penalty`, `diversity_penalty` and `moment` across a
  join, so a merged clip can still explain the score it was given.
- `_describe_profiles` moved to `hypecut.config.describe_profiles` (public).

### Notes on why replays are not duplicates

The obvious de-duplication rule — drop clips that look like an earlier clip —
is wrong for the footage this project targets. A broadcast shows the goal,
then the slow-motion, then the angle from behind the net, and a reel that
keeps all three is not repeating itself; that is what the edit is *for*.

So similarity alone decides nothing here. Similar **and close together** is
one event being shown again: kept, and tagged with a shared `moment` id.
Similar **and far apart** is a different occurrence that happens to look
identical — the same corridor, the same camera angle, the same celebration
cam — and only that is penalised.

Two measurements shaped the implementation, both of which contradicted the
first design:

- **Appearance does not work.** Averaged frames of a locked-camera football
  match are the same green rectangle every time: every pair of clips scored
  above 0.99 and the whole video read as one repeated moment. The descriptor
  is built from frame *differences* instead, which cancels the static
  background exactly and describes where the play happened.
- **A ratio needs an absolute floor.** The same failure appeared in the
  emptiness check: with a near-zero baseline, codec flicker of five
  hundredths of a luma level divides out to "nine times the usual". Signals
  now declare a `noise_floor` in their own units, and a clip that barely
  moves declines to be compared at all rather than matching everything.


## [0.7.0] — evidence

Until this release every default in HypeCut was a reasoned guess, checked
against footage built to have the property being checked for. That proves the
code does what was intended; it says nothing about whether the intention
matches real video, and it left profile PRs unreviewable on evidence. This
release makes the question answerable.

### Added
- **`hypecut label VIDEO`** — writes an answer key. It deliberately
  over-proposes (percentile 80) and marks every proposal `keep: null`, which
  counts as neither a highlight nor a rejection; a contact sheet is written
  alongside with one tile per entry. The human sets `keep: true`/`false` and
  adds entries for anything the detector missed — those matter most, because
  they are the failures a score would otherwise never see. A draft nobody has
  been through is rejected rather than scored.
- **`hypecut eval LABELS… [--profile P]…`** — scores any number of profiles
  against any number of labels files, as a table or `--json`. This is the
  command a profile PR should include the output of.
- **`hypecut.evaluation`** in the API: `Highlight`, `Labels`, `Score`,
  `load_labels`, `write_labels`, `score_plan`.

### Notes on the metric

Three decisions, each worth disagreeing with:

- **A hit means the clip contains the moment**, not that the edges line up.
  Overlap scores (IoU and friends) conflate *did you find it* with *did you
  frame it well*; the second is snapping and trimming's job and is reported
  separately as **coverage**. Perfect recall with poor coverage means the
  detector is right and the rolls are too tight — a completely different fix
  from a detector that misses things.
- **Labels ship without video.** A benchmark that needs a corpus of gameplay
  and broadcast footage cannot be distributed, so a labels file references a
  video by path and carries only timestamps.
- **One annotator, named in the file.** Comparing two profiles against one
  annotator is a valid experiment; comparing scores across annotators is not.

On the synthetic sports fixture the harness reproduces exactly the split it
was built to expose: the gameplay default and `sports-broadcast` both find
the goal — recall, precision and F1 all tie at 1.0 — and only coverage
separates them (0.64 vs 1.00), because the default rolls out before the crowd
does. Had the harness reported a single blended number, that difference would
have been invisible.

## [0.6.0] — sound, spans, and agents

### Added
- **Cross-clip loudness matching.** A measurement pass reads each clip's
  integrated loudness (EBU R128) through the same filter chain the encode
  will use, then applies a static gain toward `render.loudness_target`.
  `loudness_match` defaults to 0.9 rather than 1.0: full matching makes a
  whispered aside and a stadium roar equally loud.
- **`hypecut contact-sheet`** — one labelled grid of frames, either sampled
  across the video or one per proposed clip. The caption sits at the bottom
  so it cannot cover the scoreboard corner.
- **`hypecut render plan.json`** — render an edited cut list. Times are
  clamped to the source and validated; an edited plan is untrusted input.
- **`hypecut profiles`** and **`hypecut signals --json`** — machine-readable
  catalogues. Profile summaries come from each profile's own first comment,
  so they cannot drift from the file.
- **`AGENTS.md`** and a **skill package** (`skill/hypecut/`, `hypecut.skill`)
  for driving HypeCut from an AI assistant. No separate agent code path —
  the same commands serve people.
- `hypecut.plan.load_plan()` / `plan_from_dict()` in the API.

### Fixed
- **Clip edges could be moved into the middle of a long event.** Snapping and
  trimming guarded a single point (`peak_time`); the loudest frame of a
  twenty-second rally can be its third shot, so the guard permitted the rest
  to be trimmed away. Clips now carry `event_start` / `event_end` and the
  guards protect the whole span. Degenerate spans fall back to the old
  behaviour, so instant events are unchanged.

## [0.5.0] — sport

HypeCut is no longer only for gameplay. Sport breaks three assumptions the
gameplay signals were built on, and this release addresses each one.

### Added
- **`crowd_roar`** — sustained stadium noise, found with a rolling *minimum*
  so a plateau survives and a spike does not. A goal's roar wins over a louder
  but briefer shout, which raw loudness gets backwards.
- **`whistle`** — referee whistles, detected as narrowband tonal bursts
  (concentration × band share, so neither loudness nor frequency alone
  qualifies).
- **`roi_change`** — change *isolated* to a small box, by subtracting the
  whole-frame difference. A scoreboard digit flipping registers; a camera cut,
  which moves the box and everything else equally, cancels to zero.
- **`segments.reaction_lag`** — how long after a moment its detectable
  reaction arrives. Shifts the in-point earlier only (the celebration is worth
  keeping) and records both `peak_time` (the moment) and `reaction_time` (the
  evidence) so every downstream guard protects the play.
- **`sports-broadcast.yaml`** and **`sports-field.yaml`** profiles, the second
  for a phone on the sideline with no scoreboard, no director and no cuts.
- A written method for adapting HypeCut to any new domain, in
  `docs/EXTENDING.md`.

### Changed
- Positioning broadened from "gameplay and esports" to "gameplay, esports and
  sports" across the README, package metadata and CLI help.

## [0.4.0] — dissolves, variants, batch

### Added
- **Dissolve and fade detection.** Snapping now finds gradual transitions as
  well as hard cuts, using accumulated difference with a spatial-contrast dip
  to tell a crossfade from a camera pan. A dissolve is treated as an interval:
  in-points land on its far side, out-points on its near side. Each snapped
  edge records which kind it landed on (`meta.snap_kind`).
  `segments.snap_to_dissolves` turns it off.
- **Variants — one analysis, several aspect ratios.** `--also vertical --also
  square` (or `Config.variants` in a profile) renders extra cutdowns from the
  same decode and the same cut decisions. Every framing is planned during
  analysis, so a vertical variant is centred on its own action track rather
  than being a letterboxed copy. Available in the web UI too, with a download
  link per variant.
- **Batch mode.** `hypecut batch FOLDER -o OUTDIR [--recursive] [--pattern]`
  cuts every video in a directory, skips ones already cut, carries on past
  individual failures and reports what happened.
- `Config.render_for(variant)` and `pipeline.render_variants()` in the API.

### Fixed
- Shot detection no longer invents boundaries in perfectly static footage. A
  near-zero local baseline made a single frame of compression flicker measure
  as hundreds of times "normal"; an absolute floor of 1.5 luma levels stops
  that while staying below the smallest real cut (dark scene to dark scene).

## [0.3.0] — pauses and reactions

### Added
- **Silence-aware trimming** (on by default). Clip edges that found no shot
  boundary are moved into the nearest pause instead: the in-point lands where
  sound resumes, the out-point where it stops. "Quiet" is relative to the
  clip's own speech level, and a clip with no usable contrast is left alone.
  `--no-trim` on the CLI, `segments.trim_to_silence` / `silence_*` in profiles.
- **Adaptive audio fades.** Clips that end mid-sound get a longer audio ramp so
  the stop does not read as a dropout; the video fade is unchanged.
- **Reaction-aware reframing.** `render.reframe.react_to_facecam` pulls the
  9:16 crop toward the facecam box while that box is busy, and back to the
  action when it is not. `--react` and `--facecam X0,Y0,X1,Y1` on the CLI.
- Web UI toggles for trimming and facecam reaction; per-clip badges for edges
  moved into a pause.

### Changed
- Precedence between the two edge stages is explicit: snapping runs first and
  trimming only considers edges it did not claim.

## [0.2.0] — cut points and vertical

### Added
- **Shot-boundary snapping** (on by default). Clip edges move onto real cuts
  instead of landing mid-shot: coarse detection on the analysis frames, then a
  frame-exact pass at the source frame rate. Guards keep the peak inside the
  clip and refuse snaps that would break the length budget.
  `--no-snap` / `--snap-window` on the CLI, `segments.snap_*` in profiles.
- **Vertical reframing** for Shorts/Reels/TikTok: `crop` (9:16 slice centred on
  the motion, optionally panning to follow it), `stack` (facecam over gameplay)
  and `blur_pad`. `--vertical`, `--reframe MODE`, `--reframe-track`, or
  `render.reframe` in a profile. Crop decisions are recorded in the sidecar.
- `configs/shorts.yaml` — a vertical-first profile.
- Web UI controls for framing mode, pan-to-follow and snapping; per-clip badges
  showing which edges were snapped and how each clip was framed.
- `hypecut.ffmpeg.decode_gray_frames` accepts `start` / `duration`, so a window
  can be re-decoded at native frame rate without touching the rest of the file.

### Changed
- Config sections nest arbitrarily now (`render.reframe.*`); `_from_dict`
  resolves real types instead of a hardcoded section list.
- A bare `off` in YAML (which parses as boolean `false`) is accepted as the
  `off` reframe mode rather than failing later with a confusing message.

## [0.1.0] — first release

### Added
- Hybrid two-stage highlight detection: cheap signals over the whole video,
  optional models over candidates only.
- Seven built-in signals: `audio_rms`, `audio_transient`, `speech_band`,
  `scene_change`, `motion`, `flash`, `roi_activity`.
- Four refiners: `diversity`, `pacing`, `clip_rerank` (CLIP),
  `speech_keywords` (Whisper).
- CLI: `cut`, `analyze`, `signals`, `serve`.
- Web UI with drag-and-drop upload, live progress and an inspectable cut list.
- Python API: `analyze()` / `render_plan()` / `run()`.
- Outputs: mp4 with chapters, `.hypecut.json` cut list, CMX3600 `.edl`.
- Profiles for FPS, MOBA and talk-stream footage.
- Plugin discovery via the `hypecut.signals` / `hypecut.refiners` entry points.
- Docker image and compose file.
