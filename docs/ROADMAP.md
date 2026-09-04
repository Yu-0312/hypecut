# Roadmap

Ordered by *how much it improves the output per unit of work*, not by how
interesting it is to build. Items marked **help wanted** are good entry points.

## Shipped in v0.2

- [x] **Shot-boundary snapping.** Clip edges travel up to `max(snap_window,
      pre_roll)` seconds to land on a real cut, coarse-detected on the analysis
      frames and then refined at the source frame rate.
- [x] **Vertical reframing.** `crop` (motion-centred, optionally panning),
      `stack` (facecam over gameplay) and `blur_pad`, planned during analysis
      and recorded in the sidecar.

## Shipped in v0.3

- [x] **Silence-aware trimming.** Edges that found no shot boundary move into
      the nearest pause instead, measured against the clip's own speech level.
- [x] **Adaptive audio fades.** The video fade stays constant; the audio fade
      lengthens for clips cut off mid-sound.
- [x] **Reaction-aware reframing.** The `crop` centre commits to the facecam
      box while that box is busy — no detector, no extra dependency. The
      original plan was a face model behind an optional extra; a configured
      box turned out to give the same behaviour, verifiably, for nothing.

## Shipped in v0.4

- [x] **Dissolve detection.** Crossfades and fades are found by accumulated
      difference plus a contrast dip, and treated as intervals: in-points land
      on the far side, out-points on the near side.
- [x] **One analysis, several aspect ratios.** `--also vertical --also square`
      shares the decode, the scoring and the cut decisions; each framing is
      planned separately while the frames are still in memory.
- [x] **Batch mode.** `hypecut batch FOLDER` cuts every video in a directory
      and carries on past individual failures.

## Shipped in v0.5

- [x] **Sport.** Three new signals (`crowd_roar`, `whistle`, `roi_change`),
      `segments.reaction_lag` for evidence that arrives after the event, and
      `sports-broadcast` / `sports-field` profiles.
- [x] **A written method for adapting to a new domain**, in EXTENDING.md.

## Shipped in v0.6

- [x] **Cross-clip loudness matching.** Two-pass EBU R128; the reel no longer
      jumps in volume between clips.
- [x] **The protected span is an interval, not a point.** Clips carry the
      event bounds they were built from, so a guard cannot be placed on an
      arbitrary frame inside a long exchange.
- [x] **An agent-facing surface**: `contact-sheet`, `render --plan`,
      `profiles`, `signals --json`, plus `AGENTS.md` and a skill package.

## Shipped in v0.7

- [x] **Evaluation.** `hypecut label` proposes generously and a human keeps or
      drops; `hypecut eval` scores any number of profiles against those
      labels. Labels reference a video by path and carry only timestamps, so
      an answer key can be shared where the footage cannot. A hit means the
      clip *contains* the moment; how much of it survived is reported
      separately as coverage, because "missed it" and "framed it badly" need
      different fixes.

## Shipped in v0.8

- [x] **An empty video gets an empty answer.** `segments.min_prominence`
      measures how far the best moment stands above the video's own
      background, so an idle stream is no longer cut into a reel of its
      least-boring parts.
- [x] **Reels split instead of truncating.** Past `clips_per_reel` clips a
      long cut becomes part 1, part 2, part 3, in chronological order, each
      with its own cut list. `max_clips` now defaults to no cap.
- [x] **De-duplication by content, not by clock.** The `similarity` refiner
      compares where clips *move*, and treats "similar and close together" as
      a replay to keep rather than a duplicate to drop.
- [x] **A web UI an ordinary person can use.** Two plain questions, one
      button, everything else folded away.

## Shipped in v0.9

- [x] **Auto-locate the facecam.** The box `react_to_facecam` and `stack`
      always needed by hand is now found from the frames already decoded:
      a webcam is a small rectangle that is persistently, mildly alive,
      unlike an event-driven kill feed or sprawling gameplay. Opt in with
      `render.reframe.facecam: auto` or `--facecam auto`; low-confidence
      verdicts fall back to the default box. No model, no extra decode.
- [x] **Wipe and slide detection.** A wipe keeps its contrast, so the
      dissolve test never saw one. Wipes are found by their moving front —
      narrow, directional, frame-covering — and treated as intervals like
      dissolves: in-points land where the sweep completes, out-points where
      it begins.
- [x] **Word-boundary trimming.** With `segments.use_asr_words` and the
      `[asr]` extra, pauses come from transcribed word timings instead of
      loudness, so a slow speaker with no level gaps no longer gets cut
      mid-word. Falls back to the loudness path with a warning when the
      extra is missing.

## Shipped in v1.0 (in part)

- [x] **Twitch/YouTube VOD URLs as input**, via yt-dlp as the new
      `[ytdlp]` extra. URLs download once into a local cache keyed by video
      id; `cut`, `analyze`, `label`, `contact-sheet` and `render --source`
      accept them.
- [x] **Chat-log signal.** `chat_rate` reads JSONL, TwitchDownloader JSON
      or plain timestamped logs — a sibling `<video>.chat.jsonl` by name,
      or `--chat <log>`. Message rate fuses into the curve like any other
      signal, and shows up in clip `reasons`.
- [x] **Parallel batch workers.** `hypecut batch --workers N` cuts a folder
      on a process pool; the default of 1 keeps the fine-grained progress
      bar. Failure semantics and the exit code are unchanged.

## v1.0 — what is left

- [ ] **One reel across a whole folder** — batch mode makes one reel per file
      today; a season recap wants the opposite. Needs the renderer to take
      per-clip sources, which touches the sidecar, EDL and chapter formats;
      still waiting on a design that does not complicate the single-source
      path that everything else uses.

## v1.1 — deployments with more than one user

- [ ] Pluggable queue backend (RQ or arq) behind the existing `JobStore`
      interface; the in-process worker stays the default.
- [ ] Optional auth for the web UI (a shared token is probably enough).
- [ ] Resumable/chunked uploads for multi-GB VODs.
- [ ] Live progress over SSE instead of polling.

## Ongoing — profiles

More games, better tuned. This does not need Python and does not need a
release. See [EXTENDING.md](EXTENDING.md#contributing-a-game-profile).

Wanted, gaming: Rocket League, Apex Legends, Overwatch 2, Fortnite, Street
Fighter 6, Minecraft, racing sims, speedruns, chess.

Wanted, sport: tennis and volleyball (rally-based, so the "moment" is a whole
point rather than an instant), motorsport (engine noise swamps the crowd
band), combat sports (the roar and the strike are nearly simultaneous, so
`reaction_lag` should be near zero), and cricket or baseball, where the
scoreboard changes far more often than something interesting happens.

## Open design questions

Opinions welcome in the issues — these are genuinely undecided.

**How should the reel handle audio ducking?** Game audio across a hard cut is
jarring. Options: a short crossfade (current), duck-to-silence between clips,
or a music bed under the whole reel. The last one is the most "produced" and
also the most opinionated — is that HypeCut's job?

**Should scores be comparable across videos?** Half-answered in v0.8.
`min_prominence` is genuinely cross-video — it is a ratio, so it needs no
calibration — but it only answers the binary question, *is there anything
here*. Clip scores are still percentile-relative, so 0.9 in a quiet VOD is
not 0.9 in a loud one, and "only give me clips above 0.8, forever" still does
not work. A real absolute scale needs calibration data, and `hypecut label`
is now the mechanism that could collect it: enough answer keys would let a
mapping be fitted from raw signal values to "a human marked this". Nobody has
gathered them yet, and it is not clear how many it would take.

**Where does the line sit with LLM/VLM APIs?** A hosted VLM would likely beat
CLIP at judging candidates, but it breaks the "works fully offline, sends
nothing anywhere" guarantee. Current thinking: acceptable as an explicitly
opt-in refiner with a loud disclosure, never as a default. Not yet built.

**How should rally sports be modelled?** Football has instants — a goal is a
frame. Tennis has rallies: the interesting unit is twenty seconds long and its
"peak" is arbitrary. Everything here assumes a moment with rolls around it,
and a rally probably wants the region itself kept whole instead. Nobody has
designed that yet.

**What should the metric do about disagreement between annotators?** The
harness shipped in v0.7 answers "is this profile better than that one *for
this person*", which is the question a profile PR actually needs. It does not
answer "is this profile good", because two people mark different highlights
in the same match and there is no principled way to merge them. A labels file
records one annotator by name and scores are never pooled across annotators.
Whether a multi-annotator agreement measure is worth the complexity is
undecided; nobody has yet collected two keys for the same video to find out.

**Should trimming be allowed to override a snap?** Right now a shot boundary
always wins, on the grounds that a cut is evidence and a pause is a guess. But
a clip that snaps to a cut and then opens on half a word is a real failure mode
nobody has measured yet.

**How far should a crop be allowed to pan?** The velocity cap
(`reframe.max_pan`, 10% of frame width per second) is a guess calibrated on
shooters. Racing and sports footage probably want more; a locked-camera MOBA
wants none. Per-profile values exist but nobody has tuned them against real
footage yet.

**Should HypeCut ever re-order clips by score instead of time?** Chronological
order tells the match's story; score order front-loads the payoff, which is how
social-first edits are cut. Probably a flag, but which is the default?

## Non-goals

* Becoming a video editor. HypeCut finds moments and hands off an EDL.
* Shipping model weights in the package.
* Any hosted service, account system, or telemetry.
