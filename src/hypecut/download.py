"""VOD URLs as input, via yt-dlp.

Twitch and YouTube keep the recordings; HypeCut just wants a file. This
module closes the gap: a URL passed to `cut`, `analyze`, `label` or
`contact-sheet` is downloaded once into a local cache and then treated like
any other input. The cache is keyed by the platform's own video id, so
re-running a command does not re-download.

yt-dlp is an optional extra — `pip install "hypecut[ytdlp]"` — because the
core promise is that HypeCut needs nothing but ffmpeg. Nothing here phones
home beyond the URL the user asked for.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["looks_like_url", "download_dir", "ensure_local", "DownloadError"]

URL_SCHEMES = ("http://", "https://")

#: Where downloads land. ``HYPECUT_DOWNLOAD_DIR`` wins, then the standard
#: cache location; the same video id always maps to the same file, so the
#: cache can be shared across runs and wiped carelessly without harm.
_DOWNLOAD_SUBDIR = Path("hypecut") / "downloads"


class DownloadError(RuntimeError):
    """Raised when a URL could not be turned into a local file."""


def looks_like_url(text: str) -> bool:
    """Is this a URL rather than a path?"""
    return text.strip().lower().startswith(URL_SCHEMES)


def download_dir() -> Path:
    override = os.environ.get("HYPECUT_DOWNLOAD_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / _DOWNLOAD_SUBDIR


def ensure_local(source: str, *, progress=None) -> Path:
    """A local path for ``source`` — downloaded first if it is a URL.

    Non-URLs pass through untouched; the pipeline's own error messages
    remain the authority on what to do with a missing file.
    """
    if not looks_like_url(source):
        return Path(source)

    try:
        import yt_dlp
    except ModuleNotFoundError as exc:
        raise DownloadError(
            f'{source!r} looks like a URL and needs yt-dlp: pip install "hypecut[ytdlp]"'
        ) from exc

    target_dir = download_dir()
    target_dir.mkdir(parents=True, exist_ok=True)

    # ``requested_downloads[0].filepath`` is set after the post-processing
    # merge, so it names the file that actually exists even when the
    # container changed during download.
    options = {
        "outtmpl": str(target_dir / "%(id)s.%(ext)s"),
        "format": "bv*[height<=1080]+ba/b",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
    }
    if progress is not None:
        options["progress_hooks"] = [_hook(progress)]

    try:
        with yt_dlp.YoutubeDL(options) as ydl:
            info = ydl.extract_info(source, download=True)
            requested = (info or {}).get("requested_downloads") or []
            filename = requested[0].get("filepath") if requested else None
            if not filename:
                filename = ydl.prepare_filename(info or {})
    except DownloadError:
        raise
    except Exception as exc:
        raise DownloadError(f"could not download {source}: {exc}") from exc

    path = Path(filename)
    if not path.exists():
        raise DownloadError(f"download reported success but {path} does not exist")
    if progress is not None:
        progress(1.0, f"downloaded {path.name}")
    return path


def _hook(progress) -> dict[str, object]:
    """Map a yt-dlp progress hook onto HypeCut's progress callback."""

    def report(payload: dict) -> None:
        status = payload.get("status")
        if status == "downloading":
            total = payload.get("total_bytes") or payload.get("total_bytes_estimate") or 0
            done = payload.get("downloaded_bytes") or 0
            fraction = done / total if total else 0.0
            progress(fraction * 0.95, f"downloading {payload.get('info_dict', {}).get('id', '')}")
        elif status == "finished":
            progress(0.98, "merging downloaded streams")

    return report
