"""URL inputs, downloaded through yt-dlp into a local cache."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from hypecut.download import DownloadError, download_dir, ensure_local, looks_like_url


@pytest.fixture
def fake_ytdlp(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """A yt_dlp stand-in that 'downloads' by writing a tiny file."""

    class FakeYoutubeDL:
        def __init__(self, options: dict) -> None:
            self.options = options
            self.calls: list[str] = []

        def extract_info(self, url: str, download: bool) -> dict:
            assert download is True
            self.calls.append(url)
            out = self.options["outtmpl"]
            path = out.replace("%(id)s.%(ext)s", "abc123.mp4")
            Path(path).write_bytes(b"fake video")
            self.info = {"id": "abc123", "requested_downloads": [{"filepath": path}]}
            return self.info

        def prepare_filename(self, info: dict) -> str:
            return self.options["outtmpl"].replace("%(id)s.%(ext)s", f"{info['id']}.mp4")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    module = types.ModuleType("yt_dlp")
    module.YoutubeDL = FakeYoutubeDL
    monkeypatch.setitem(sys.modules, "yt_dlp", module)
    monkeypatch.setenv("HYPECUT_DOWNLOAD_DIR", str(tmp_path / "cache"))
    return module


def test_looks_like_url():
    assert looks_like_url("https://www.twitch.tv/videos/123")
    assert looks_like_url("HTTP://example.com/v.mp4")
    assert not looks_like_url("vod.mp4")
    assert not looks_like_url("/mnt/recordings/vod.mp4")
    assert not looks_like_url("ftp://host/vod.mp4")


def test_plain_paths_pass_through_untouched(tmp_path):
    src = tmp_path / "vod.mp4"
    assert ensure_local(str(src)) == src


def test_url_is_downloaded_into_the_cache(fake_ytdlp):
    path = ensure_local("https://www.twitch.tv/videos/123")
    assert path == download_dir() / "abc123.mp4"
    assert path.exists()


def test_a_missing_extra_is_a_friendly_error(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    monkeypatch.setenv("HYPECUT_DOWNLOAD_DIR", str(tmp_path))
    with pytest.raises(DownloadError, match='pip install "hypecut\\[ytdlp\\]"'):
        ensure_local("https://example.com/vod")


def test_a_failed_download_reports_why(fake_ytdlp):
    class Failing(fake_ytdlp.YoutubeDL):
        def extract_info(self, url, download):
            raise RuntimeError("HTTP Error 404")

    fake_ytdlp.YoutubeDL = Failing
    with pytest.raises(DownloadError, match="404"):
        ensure_local("https://example.com/gone")


def test_progress_is_reported(fake_ytdlp):
    seen: list[tuple[float, str]] = []
    ensure_local("https://www.twitch.tv/videos/123", progress=lambda p, m: seen.append((p, m)))
    assert seen and seen[-1][0] == 1.0
