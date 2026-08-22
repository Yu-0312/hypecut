# Changelog

All notable changes are documented here. This project follows
[Semantic Versioning](https://semver.org).

## [Unreleased]

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
