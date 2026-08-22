# Extending HypeCut

Two extension points: **signals** (stage 1) and **refiners** (stage 2). Both
are plain Python classes with one required method, and both can ship in a
separate package.

## Writing a signal

A signal answers one question about every moment of the video: *how interesting
is now, by my particular measure?*

Rules:

* Return a float array aligned to `ctx.times`. Length mismatches are padded or
  trimmed for you, but getting it right is cheaper.
* **Do not normalise.** Return your natural units — dBFS, pixel counts,
  messages per second. Fusion z-scores everything.
* **Do not threshold.** A signal that returns 0/1 throws away the gradient the
  selector needs.
* **Do not decode the video.** Everything is on the context already. If you
  genuinely need something else (colour, higher resolution), decode a *crop*
  and cache it in `ctx.extras`.

```python
import numpy as np
from hypecut.signals import Signal, register


@register("hud_number_churn")
class HudNumberChurn(Signal):
    """How fast a HUD counter (ammo, score, HP) is changing.

    Params
    ------
    box: [x0, y0, x1, y1] in normalised 0-1 coordinates.
    """

    description = "Rate of change in a HUD number region."
    requires_video = True

    def compute(self, ctx):
        box = self.params.get("box", [0.85, 0.88, 0.98, 0.97])
        f = ctx.gray.astype(np.float32)
        h, w = f.shape[1:]
        x0, y0 = int(box[0] * w), int(box[1] * h)
        x1, y1 = int(box[2] * w), int(box[3] * h)
        roi = f[:, y0:y1, x0:x1]
        d = np.abs(np.diff(roi, axis=0)).mean(axis=(1, 2))
        return np.concatenate([d[:1], d])
```

Enable it in a profile:

```yaml
signals:
  enabled: [audio_rms, audio_transient, hud_number_churn]
  weights:
    hud_number_churn: 1.2
  params:
    hud_number_churn:
      box: [0.85, 0.88, 0.98, 0.97]
```

### Gating on what's available

Set `requires_audio` / `requires_video` and the signal is skipped rather than
crashing when the input lacks that stream. Override `applicable(ctx)` for
anything more specific.

## Writing a refiner

Refiners see the candidate list — typically a few dozen windows — and may
rescore, retime, or drop them. This is where an expensive model belongs.

```python
from hypecut.refine import Refiner, register


@register("face_reaction")
class FaceReaction(Refiner):
    """Boost clips where the facecam shows a strong reaction."""

    description = "Detect facecam expression changes (needs mediapipe)."

    def available(self):
        try:
            import mediapipe  # noqa: F401
        except ModuleNotFoundError as exc:
            return False, f"face_reaction needs mediapipe ({exc.name})"
        return True, ""

    def refine(self, info, candidates):
        ok, why = self.available()
        if not ok:
            import warnings
            warnings.warn(why)
            return candidates          # degrade, never fail the run
        for cand in candidates:
            score = analyse_facecam(info.path, cand.start, cand.end)
            cand.score = 0.7 * cand.score + 0.3 * score
            cand.reasons["face_reaction"] = score
        return candidates
```

**`available()` is not optional.** A refiner that raises `ImportError` in the
middle of a two-hour job has destroyed the run for a feature the user did not
ask to be mandatory. Report and continue.

## What is *not* a plugin

Shot-boundary snapping and vertical reframing are core pipeline stages, not
signals or refiners. Both need the decoded frames, which refiners deliberately
do not get — a refiner sees candidates, so that an expensive model never has to
touch the whole video. Rather than widen that contract, the two stages run
after selection inside `pipeline.analyze()` and are configured through
`segments.snap_*` and `render.reframe.*`.

If you want to change how they behave, tune the profile first
(`snap_window`, `snap_guard`, `reframe.max_pan`, `reframe.keyframes`). If you
need genuinely different behaviour, `snapping.find_boundaries` and
`reframe.action_track` are plain functions over numpy arrays and are the right
places to start.

## Shipping as a package

Declare entry points and HypeCut discovers your plugin on install — it will
show up in `hypecut signals` for anyone who has both packages.

```toml
# your-package/pyproject.toml
[project.entry-points."hypecut.signals"]
hud_number_churn = "hypecut_hud:HudNumberChurn"

[project.entry-points."hypecut.refiners"]
face_reaction = "hypecut_face:FaceReaction"
```

## Adapting to a new kind of footage

Adding a game is tuning. Adding a *domain* — sport, lectures, wildlife,
dashcam — sometimes needs new detectors, and the sports work is the worked
example. Three questions decide what you need.

**1. When does the evidence arrive relative to the event?**

In gameplay the kill and its sound are simultaneous, so the detected peak is
the moment. In sport the goal is silent and the roar arrives a second or two
later, so a clip anchored on the peak starts *after* the thing worth watching.
That is what `segments.reaction_lag` is for. It shifts the in-point earlier
and records both times on the clip (`peak_time` is the moment,
`reaction_time` is the evidence), so every downstream guard protects the play
rather than the celebration.

Note it does *not* shift the out-point. The reaction is usually worth keeping;
only the start was in the wrong place.

**2. Is the evidence an edge or a plateau?**

`audio_transient` rewards change, which is right for a gunshot and wrong for a
crowd: it fires as the roar begins and loses interest exactly when the stadium
is loudest. `crowd_roar` takes a rolling *minimum* instead, so only sustained
noise survives. If your domain's evidence is "loud for a while" rather than
"suddenly loud", copy that shape rather than reaching for a model.

A caveat worth internalising: frequency will not separate a crowd from a
commentator — a voice fundamental sits at 85-255 Hz, inside any sensible crowd
band. Duration does, because a person breathes and a stadium does not. When a
band-based idea does not work, ask what *else* is different.

**3. Does some region of the frame already know the answer?**

Kill feeds, scoreboards, lap counters, lower-thirds. `roi_activity` asks "is
this box busy", which a camera cut answers as loudly as a goal does.
`roi_change` asks the better question — is this box changing *more than the
rest of the frame* — by subtracting the global difference. A digit flip
survives; a cut cancels to zero. Prefer it whenever the region is small and
the rest of the frame moves on its own.

Once you know the answers, most of the work is a profile. Only reach for a new
signal when none of the existing ones measures the thing you care about.

## Contributing a game profile

The highest-value contribution, and it needs no Python.

1. Record or find 10–20 minutes of footage you know well.
2. `hypecut analyze footage.mp4 --json plan.json` with the default profile.
3. Look at what it picked and what it missed. Adjust:
   * clips landing on menus or downtime → raise `percentile`, lower the weight
     of whichever signal `reasons` shows dominating;
   * clips starting late → raise `pre_roll`;
   * everything clustered in one fight → raise `diversity.min_gap`;
   * the kill feed being ignored → fix the `roi_activity` box for that HUD.
4. Iterate until the cut list matches what you'd have picked by hand.
5. Open a PR adding `configs/<game>.yaml` with a comment block explaining *why*
   each weight is what it is. The reasoning is the valuable part — it's what
   lets the next person adapt it.

The shipped `sports-broadcast.yaml` is the fullest worked example: every
number in it has a one-line justification, including the ones that turn
features *off* (`trim_to_silence: false`, because a stadium is never quiet
and there are no pauses to find).

## Testing your extension

```python
import numpy as np
from hypecut.types import AnalysisContext, VideoInfo

def make_ctx(n=300, fps=10.0):
    return AnalysisContext(
        info=VideoInfo("x.mp4", n / fps, 30.0, 320, 180, True),
        grid_fps=fps,
        times=np.arange(n) / fps,
        gray=np.random.randint(0, 255, (n, 54, 96), dtype=np.uint8),
        audio=np.random.normal(0, 0.1, int(n / fps * 16000)).astype(np.float32),
    )

def test_my_signal_shape():
    out = MySignal().compute(make_ctx())
    assert out.shape == (300,)
    assert np.isfinite(out).all()
```

Signals should be deterministic given a context — no wall-clock, no network, no
global state. That is what makes a cut list reproducible from its sidecar.
