# Contributing to HypeCut

Thanks for looking. The three most useful things you can do, in order:

1. **Tune a game profile.** No Python needed. See
   [docs/EXTENDING.md](docs/EXTENDING.md#contributing-a-game-profile).
2. **Tell us where it got the cut wrong.** A timestamp, the profile you used,
   and what you'd have picked instead is a genuinely useful bug report. Attach
   the `.hypecut.json` sidecar if you can — it contains the exact config and
   the per-signal scores.
3. **Pick something off [the roadmap](docs/ROADMAP.md)**, especially anything
   marked *help wanted*.

## Setup

```bash
git clone https://github.com/hypecut/hypecut && cd hypecut
python -m venv .venv && source .venv/bin/activate
make dev            # pip install -e ".[web,dev]"
make test
```

You need **ffmpeg** on your `PATH`. Tests that need it skip themselves if it's
missing, so a passing run does not mean much on its own — check the skip count.

Run the suite as `pytest` (which is what `make test` and CI do), not as
`python -m pytest`. The two are not equivalent: `python -m` prepends the
working directory to `sys.path`, so it can import things a real installation
cannot. A suite that only passes under `python -m pytest` is a suite that
will fail in CI.

## Before you open a PR

```bash
make fmt      # ruff --fix + format
make lint
make test
```

CI runs the same three on Linux (3.10/3.11/3.12) and macOS, plus a Docker
build-and-boot smoke test.

## House style

* **Comments explain *why*, never *what*.** `# add 1 to i` is noise; `# the
  wind-up matters more than the reaction, hence the asymmetric roll` is the
  reason someone won't "simplify" it away next year.
* **New behaviour comes with a test.** Pure logic (fusion, segments, config)
  tests without ffmpeg — prefer that where possible.
* **Optional dependencies stay optional.** Anything importing torch, whisper,
  or similar must be behind a refiner's `available()` check and must degrade to
  a warning, never an exception.
* **No new required dependency** without a paragraph in the PR on why the
  problem can't be solved with numpy and ffmpeg. The core install being tiny is
  a feature.
* Type hints on public functions. `mypy` is advisory for now.

## Adding a signal or refiner

See [docs/EXTENDING.md](docs/EXTENDING.md). If it's generally useful, a PR
adding it to `src/hypecut/signals/` is welcome — include a one-line
`description` (it shows up in `hypecut signals`) and say in the docstring what
kind of footage it's for.

If it needs a heavy dependency, ship it as a separate package using the
`hypecut.signals` entry-point group instead.

## Commit and PR conventions

Conventional-commit prefixes (`feat:`, `fix:`, `docs:`, `perf:`, `refactor:`,
`test:`, `chore:`) are appreciated but not enforced. Keep PRs focused; a
profile addition and a rendering change should be two PRs.

## Reporting a bug

Include:

* the command or UI settings you used,
* `hypecut --version`, your OS, and `ffmpeg -version | head -1`,
* the `.hypecut.json` sidecar if a run completed,
* what you expected the cut list to look like.

## Security

Please don't open a public issue for a security problem — see
[SECURITY.md](SECURITY.md).

## Licence

Contributions are accepted under Apache-2.0, the project's licence. There is no
CLA.
