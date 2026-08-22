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

## v0.3 — better cut points, continued

- [ ] **Silence-aware trimming.** Pull the out-point back to the first pause
      after the reaction instead of a fixed post-roll. Snapping fixed the
      *visual* edges; this is the audio half of the same problem. **help wanted**
- [ ] **Beat/loudness-aware transitions.** Crossfade lengths that follow the
      audio instead of a constant 0.25 s.
- [ ] **Face-aware reframing.** When a facecam is present, bias the `crop`
      centre toward it during reaction beats and toward the action otherwise.
      Needs an optional face detector, so it belongs behind an extra. **help wanted**
- [ ] **Dissolve detection.** Snapping only finds hard cuts today; fades and
      wipes are missed. **help wanted**

## v0.4 — reach

- [ ] **Twitch/YouTube VOD URLs as input**, via yt-dlp as an optional extra.
- [ ] **Chat-log signal.** Twitch chat message rate is close to a free
      human-labelled highlight track. Needs a log format adapter. **help wanted**
- [ ] **Batch mode**: point at a folder, get one reel per file, or one reel
      across all of them.
- [ ] **One source, both aspect ratios** in a single pass — landscape reel plus
      vertical cutdowns, sharing the analysis.

## v0.5 — deployments with more than one user

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
