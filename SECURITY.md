# Security Policy

## Supported versions

HypeCut is pre-1.0; only the latest release receives fixes.

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting** (Security → Report a
vulnerability) rather than a public issue. If that is unavailable, open an
issue titled "security contact request" with no details and a maintainer will
follow up.

Please include reproduction steps and the affected version. Expect an
acknowledgement within a week.

## Threat model, honestly stated

HypeCut is designed for **single-user, self-hosted** use. If you expose the web
UI to a network, be aware:

* **There is no authentication.** Anyone who can reach the port can upload
  files, consume CPU, and download any reel produced on that instance. Put it
  behind a reverse proxy with auth, or bind it to localhost (the default).
* **Uploads are attacker-controlled input to ffmpeg.** ffmpeg has a large parse
  surface. Keep it patched, and prefer running HypeCut in the provided
  container, which runs as a non-root user with only `/data` writable.
* **Job IDs are unguessable but not secret.** Anyone holding a job's ID can
  fetch that reel. Treat IDs as bearer tokens.
* **Disk is not bounded by default.** The job store evicts after 200 jobs and
  deletes the files it evicts, but a burst of large uploads can still fill the
  volume before that. Set `HYPECUT_MAX_UPLOAD_MB` and give `/data` its own
  volume with a quota.

## What HypeCut does not do

* No network calls at runtime. The only exception is the first use of an
  optional ML refiner, which downloads its model from Hugging Face.
* No telemetry, analytics, or crash reporting.
* No execution of anything from an uploaded file beyond handing it to ffmpeg.
