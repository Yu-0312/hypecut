---
name: hypecut
description: Turn a long video into a highlight reel. Use when someone shares a gameplay recording, a VOD, a match, a stream archive, a lecture or any long video and wants the good parts, a highlights cut, a montage, a short version, clips for Shorts/Reels/TikTok, or a vertical cutdown. Also use for "剪成集錦", "精華", "重點片段", "做成短影音".
---

# HypeCut — automatic highlight reels

HypeCut finds the moments worth watching in a long video and stitches them
into a short one, with a machine-readable cut list explaining every choice.
You drive it; the user decides.

## Check the environment first

```bash
hypecut --version && ffmpeg -version | head -1
```

Not installed:

```bash
pip install "hypecut[web]"
```

**ffmpeg is required and cannot be worked around.** If it is missing, say so
and stop — on macOS `brew install ffmpeg`, on Debian/Ubuntu
`sudo apt install ffmpeg`.

This skill needs a shell and the video on disk. In a chat with no filesystem,
tell the user that and stop.

## The loop

Follow `AGENTS.md` in the HypeCut repo for the full contract. The short form:

1. **Look.** `hypecut contact-sheet input.mp4 -o sheet.png --count 12`, then
   open `sheet.png`. Identify the footage, and note where any scoreboard or
   kill feed sits in 0-1 coordinates.
2. **Choose.** `hypecut profiles` — pick the closest. Nothing fits? Start
   from `default`.
3. **Propose.** `hypecut analyze input.mp4 --profile P --json plan.json`,
   then `hypecut contact-sheet input.mp4 --plan plan.json -o picked.png` and
   open that too.
4. **Adjust.** Every clip lists the signals responsible in `reasons`; a bad
   pick is usually one weight or one ROI box. Save the result as a profile so
   the run is reproducible.
5. **Confirm.** Tell the user what was picked, in plain language with
   timestamps, and ask before rendering.
6. **Render.** `hypecut cut …`, or `hypecut render plan.json` if they edited
   the cut list.

7. **Measure, if there is a choice to make.** Two plausible profiles and no
   way to pick? `hypecut label input.mp4` writes a draft answer key plus a
   sheet; the **user** marks `keep: true/false` and adds what was missed;
   `hypecut eval labels.yaml -p a.yaml -p b.yaml` scores both. `recall` is
   *did we find it*, `cover` is *how much of it survived* — they need
   different fixes. Never fill in the labels yourself: scoring your own
   detector against your own opinion measures nothing.

## Non-negotiables

- **Open both contact sheets.** Choosing a profile without seeing a frame is
  guessing.
- **`analyze` before `cut`.** Analysis is seconds, rendering is minutes.
- **Never invent timestamps.** Add a missed moment by editing `plan.json`
  with times you actually found, and say you did.
- **Explain with signals.** "`roi_activity` spiked — that is the kill feed"
  is checkable; "it looked exciting" is not.
- **Pass on "nothing here".** If the video has nothing that stands out,
  HypeCut says so and cuts nothing. Report that instead of forcing a reel out
  of it by setting `segments.min_prominence: 0`.
- **Count the files.** A long recording comes back as `reel.part1.mp4`,
  `reel.part2.mp4` … in time order. Hand over all of them.

## Things people ask for

| Ask | Command |
|---|---|
| Shorter / more selective | `--target 60 --max-clips 6 --percentile 95` |
| Vertical for Shorts/Reels | `--profile configs/shorts.yaml`, or `--vertical --reframe-track` |
| Both landscape and vertical | `--also vertical` (one analysis, one extra encode) |
| Facecam on top, game below | `--reframe stack --facecam 0,0,0.26,0.3` |
| A whole folder | `hypecut batch FOLDER -o OUT --recursive` |
| Football / basketball match | `--profile configs/sports-broadcast.yaml` |
| Phone on the sideline | `--profile configs/sports-field.yaml` |
| "Is this profile actually better?" | `hypecut label VIDEO`, they mark it, `hypecut eval LABELS -p A -p B` |
| Nothing came back at all | the video may genuinely be empty — the message gives the measured prominence; `--percentile 85` or a profile first, `segments.min_prominence: 0` last |
| Shorter parts / longer parts | `segments.clips_per_reel` (default 10), `--target` is per part |

## Deliver

Give the user `reel.mp4` and tell them `reel.hypecut.json` is the re-editable
cut list — `hypecut render reel.hypecut.json` re-renders it after any edit.
`reel.edl` opens in Resolve or Premiere if they want to finish by hand.
