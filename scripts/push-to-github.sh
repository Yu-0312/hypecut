#!/usr/bin/env bash
#
# Create the GitHub repository and push this project to it.
#
#   bash scripts/push-to-github.sh <your-github-username> [repo-name] [public|private]
#
# Everything here is ordinary git and gh — nothing HypeCut-specific. Read it
# before running it; it creates a public repository by default.

set -euo pipefail

USER="${1:?usage: push-to-github.sh <github-username> [repo-name] [public|private]}"
REPO="${2:-hypecut}"
VISIBILITY="${3:-public}"

command -v git >/dev/null || { echo "git is not installed"; exit 1; }

echo "==> Repository: $USER/$REPO ($VISIBILITY)"

# --- 1. make sure this directory is a git repo with everything committed
if [ ! -d .git ]; then
  git init -b main
fi
git add -A
git diff --cached --quiet || git commit -m "HypeCut — automatic highlight reels"

# main, not master: GitHub's default since 2020, and the branch protection
# and Actions defaults all assume it.
git branch -M main

# --- 2. create the remote repo and push
if command -v gh >/dev/null && gh auth status >/dev/null 2>&1; then
  echo "==> Creating via gh"
  gh repo create "$USER/$REPO" \
    --"$VISIBILITY" \
    --source=. \
    --remote=origin \
    --description "Automatic highlight reels for gameplay, esports and sports" \
    --push
else
  cat <<MANUAL
==> gh is unavailable or not logged in. Do this instead:

  1. Create an empty repo at https://github.com/new
       name:        $REPO
       visibility:  $VISIBILITY
       DO NOT add a README, .gitignore or licence — this repo has them,
       and adding them there makes the first push a conflict.

  2. Then run:

       git remote add origin git@github.com:$USER/$REPO.git
       git push -u origin main

     (HTTPS instead of SSH: https://github.com/$USER/$REPO.git)
MANUAL
  exit 0
fi

# --- 3. tag the release so the release workflow has something to build
VERSION="$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)"
if [ -n "$VERSION" ] && ! git rev-parse "v$VERSION" >/dev/null 2>&1; then
  echo "==> Tagging v$VERSION"
  git tag -a "v$VERSION" -m "HypeCut v$VERSION"
  git push origin "v$VERSION"
fi

cat <<DONE

==> Done: https://github.com/$USER/$REPO

Worth doing next, in the repo's Settings:

  * Actions → General → allow GitHub Actions (CI runs on push and PR)
  * Topics → add: video, highlights, ffmpeg, esports, sports, python
  * About → set the description and tick "Releases"

The release workflow (.github/workflows/release.yml) publishes to PyPI and
GHCR when you push a v* tag. It needs, in Settings → Secrets and variables:

  * an environment named "pypi" with PyPI trusted publishing configured
    (https://docs.pypi.org/trusted-publishers/) — no token needed
  * nothing for GHCR; it uses the built-in GITHUB_TOKEN

Until you set those up, the tag push will fail that job and nothing else.
DONE
