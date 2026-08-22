# Roadmap

Ordered by *how much it improves the output per unit of work*, not by how
interesting it is to build. Items marked **help wanted** are good entry points.

## v0.2 — better cut points

The single biggest quality gap today is that clips start and end at arbitrary
frames rather than at natural boundaries.

- [ ] **Shot-boundary snapping.** Detect cuts in a ±2 s neighbourhood of each
      clip edge and snap to them. Free, and it makes reels look edited rather
      than sliced. **help wanted**
- [ ] **Silence-aware trimming.** Pull the out-point back to the first pause
      after the reaction instead of a fixed post-roll.
- [ ] **Beat/loudness-aware transitions.** Crossfade lengths that follow the
      audio instead of a constant 0.25 s.

## v0.3 — reach

- [ ] **Vertical reframing** for Shorts/TikTok/Reels: per-clip 9:16 crop that
      follows the action (or a fixed facecam + gameplay stack). Frequently the
      first thing anyone asks for. **help wanted**
- [ ] **Twitch/YouTube VOD URLs as input**, via yt-dlp as an optional extra.
- [ ] **Chat-log signal.** Twitch chat message rate is close to a free
      human-labelled highlight track. Needs a log format adapter. **help wanted**
- [ ] **Batch mode**: point at a folder, get one reel per file, or one reel
      across all of them.

## v0.4 — deployments with more than one user

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

**Should HypeCut ever re-order clips by score instead of time?** Chronological
order tells the match's story; score order front-loads the payoff, which is how
social-first edits are cut. Probably a flag, but which is the default?

## Non-goals

* Becoming a video editor. HypeCut finds moments and hands off an EDL.
* Shipping model weights in the package.
* Any hosted service, account system, or telemetry.
