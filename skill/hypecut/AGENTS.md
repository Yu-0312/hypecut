# Driving HypeCut as an agent

You are an AI agent with a shell. Someone has given you a video and asked for
a highlight reel. This file is the contract: it tells you the loop, the
commands, and the judgement calls you are expected to make.

Everything here is also true for a human. HypeCut has no agent mode — the
same commands do the same things either way, and that is deliberate. A tool
with a special path for models is a tool with two paths to keep working.

## Before anything else

```bash
hypecut --version || pip install "hypecut[web]"
ffmpeg -version | head -1     # required; HypeCut cannot work without it
```

If ffmpeg is missing, stop and say so. Do not try to work around it — every
stage shells out to it.

## The loop

### 1. Look at the footage

```bash
hypecut contact-sheet input.mp4 -o sheet.png --count 12
```

One image, twelve labelled frames. **Open it.** You cannot choose a profile
from a filename, and the cut list you get later will say things like
`roi_activity 2.1` that mean nothing unless you have seen the screen.

What you are looking for:

- What kind of footage is this — first-person shooter, MOBA, broadcast sport,
  someone talking to camera?
- Is there a HUD element that already knows when something happened? A kill
  feed, a scoreboard, a lap counter. **Note which corner**, in 0-1
  coordinates; you will need the box.
- Is there a facecam, and where?
- Does the camera cut between angles, or is it one locked-off shot?

### 2. Choose a profile

```bash
hypecut profiles                  # names + what each is for
hypecut profiles --json           # same, machine-readable
hypecut signals --json            # every detector and what it measures
```

Pick the closest one. If nothing fits, start from `default` and override on
the command line. `docs/EXTENDING.md` has the three questions that decide
whether a domain needs anything new — read it before inventing a signal.

### 3. Propose a cut, without spending an encode

```bash
hypecut analyze input.mp4 --profile configs/fps-shooter.yaml --json plan.json
```

`plan.json` is the whole decision: every clip with its start, end, score, the
per-signal `reasons` that produced it, and the exact config used. Read it.

Then look at what it picked:

```bash
hypecut contact-sheet input.mp4 --plan plan.json -o picked.png
```

**Open that too.** This is the step that catches the mistake that matters:
clips landing on menus, replays, or downtime. The `reasons` tell you which
detector is responsible, so a fix is usually one weight or one box.

### 4. Adjust, and say why

Common corrections, in the order they usually apply:

| Symptom | Fix |
|---|---|
| Clips on menus or dead time | raise `--percentile`, or lower the weight of whatever `reasons` shows dominating |
| Scoreboard ignored | set the `roi_change` / `roi_activity` box to the corner you found in step 1 |
| Clips start after the moment | raise `segments.reaction_lag` (sport) or `pre_roll` |
| Everything from one stretch | raise `diversity.min_gap` |
| Too many, too short | raise `--min-duration`, lower `--max-clips` |

Write the result as a profile so the run is reproducible and the user can
re-use it:

```bash
cp configs/default.yaml my-profile.yaml   # then edit
hypecut analyze input.mp4 --profile my-profile.yaml --json plan.json
```

### 5. Show the user the cut list before rendering

In their words, not JSON. "Six clips, 1m54s: a triple kill at 14:22, a
clutch defuse at 41:07…" Ask whether to drop or move anything. Rendering is
the expensive step and the only irreversible-feeling one.

### 6. Render what they approved

If they changed nothing:

```bash
hypecut cut input.mp4 -o reel.mp4 --profile my-profile.yaml
```

If they did, edit `plan.json` — change `start`/`end`, delete objects from
`segments` — and render that exact cut:

```bash
hypecut render plan.json -o reel.mp4
```

Vertical cutdowns cost one extra encode and no extra analysis:

```bash
hypecut cut input.mp4 -o reel.mp4 --also vertical --also square
```

A whole folder:

```bash
hypecut batch ~/Recordings -o ~/Reels --recursive
```

## What you get back

- `reel.mp4` — the reel, with a chapter marker per clip
- `reel.hypecut.json` — the cut list *and* the full config. Re-renderable with
  `hypecut render`; this is the file to keep.
- `reel.edl` — a CMX3600 edit list, openable in Resolve or Premiere

## Rules

**Look at the pictures.** Both contact sheets. Choosing a profile without
seeing a frame is guessing, and the user can tell.

**Analyse before you render.** `analyze` is seconds; `cut` is minutes. Never
burn an encode to find out the percentile was wrong.

**Never invent timestamps.** If you want a clip the detector missed, add it
to `plan.json` with real times you found on the contact sheet, and say that
you added it by hand.

**Explain in signals, not vibes.** "It picked 14:22 because `roi_activity`
spiked — that is the kill feed" is checkable. "It looked exciting" is not.
The `reasons` field exists so you can do the first one.

**Say when you are unsure.** Amateur footage with no HUD, no crowd and a
static camera is genuinely hard. Saying "this profile is a guess, check the
cut list carefully" is more useful than confident silence.

**Keep it offline.** The core needs no model and no network. The optional
`clip_rerank` and `speech_keywords` refiners download weights on first use —
mention that before enabling either.

## When something breaks

| Error | Meaning |
|---|---|
| `Missing required binaries` | ffmpeg is not on PATH — stop |
| `No highlights found` | the percentile is too high for this footage; try 85 |
| `No video stream found` | not a video, or the file is truncated |
| `Unknown config key(s)` | a typo in a profile; the message names the key |
| `Segment N is 0.0s long after clamping` | an edited plan has times outside the source |

`hypecut batch` reports per-file failures and carries on; the exit code is
non-zero if any file failed.
