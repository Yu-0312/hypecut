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

## v0.5 — cut points, continued

- [ ] **Auto-locate the facecam.** `react_to_facecam` needs the box to be
      right, which is the one thing a user has to supply by hand. A one-off
      detection pass (optional extra) could find it and remove the setting.
      **help wanted**
- [ ] **Wipe and slide detection.** Dissolves are covered; a wipe keeps its
      contrast and so is invisible to the current test. **help wanted**
- [ ] **Word-boundary trimming.** Pauses are found from loudness alone, so a
      slow speaker with no gaps still gets cut mid-word. ASR word timings
      (behind the existing `[asr]` extra) would fix it. **help wanted**

## v0.6 — reach

- [ ] **Twitch/YouTube VOD URLs as input**, via yt-dlp as an optional extra.
- [ ] **Chat-log signal.** Twitch chat message rate is close to a free
      human-labelled highlight track. Needs a log format adapter. **help wanted**
- [ ] **One reel across a whole folder** — batch mode makes one reel per file
      today; a season recap wants the opposite.
- [ ] **Parallel batch workers.** One file at a time is right on a laptop and
      wasteful on a workstation.

## v0.7 — deployments with more than one user

- [ ] Pluggable queue backend (RQ or arq) behind the existing `JobStore`
      interface; the in-process worker stays the default.
- [ ] Optional auth for the web UI (a shared token is probably enough).
- [ ] Resumable/chunked uploads for multi-GB VODs.
- [ ] Live progress over SSE instead of polling.

## Ongoing — profiles

More games, better tuned. This does not need Python and does not need a
release. See [EXTENDING.md](EXTENDING.md#contributing-a-game-profile).

Wanted: Rocket League, Apex Legends, Overwatch 2, Fortnite, Street Fighter 6,
Minecraft, racing sims, speedruns, chess.

## Open design questions

Opinions welcome in the issues — these are genuinely undecided.

**How should the reel handle audio ducking?** Game audio across a hard cut is
jarring. Options: a short crossfade (current), duck-to-silence between clips,
or a music bed under the whole reel. The last one is the most "produced" and
also the most opinionated — is that HypeCut's job?

**Should scores be comparable across videos?** Today scores are
percentile-relative, so 0.9 in a quiet VOD is not 0.9 in a loud one. An
absolute scale would let users set one threshold forever, but it would need
calibration data we don't have.

**Where does the line sit with LLM/VLM APIs?** A hosted VLM would likely beat
CLIP at judging candidates, but it breaks the "works fully offline, sends
nothing anywhere" guarantee. Current thinking: acceptable as an explicitly
opt-in refiner with a loud disclosure, never as a default. Not yet built.

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
