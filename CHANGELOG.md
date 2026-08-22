# Changelog

All notable changes are documented here. This project follows
[Semantic Versioning](https://semver.org).

## [Unreleased]

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
