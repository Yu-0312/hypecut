# Changelog

All notable changes are documented here. This project follows
[Semantic Versioning](https://semver.org).

## [Unreleased]

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
