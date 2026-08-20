from __future__ import annotations

import shutil
import sqlite3
import subprocess

import pytest

from discql import audio_download as ad
from discql import db, youtube_matching as ym
from discql.discogs_api import ArtistData, ReleaseData, TrackData, VideoData
from discql.sync import upsert_release

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_av_file(path):
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=1",
            "-f", "lavfi", "-i", "sine=duration=1",
            "-c:v", "libx264", "-c:a", "aac", str(path),
        ],
        check=True,
    )


def _make_audio_only_file(path):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "sine=duration=1", "-c:a", "aac", str(path)],
        check=True,
    )


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    db.migrate(connection)
    yield connection
    connection.close()


def seed_matched_release(conn, release_id=1):
    data = ReleaseData(
        id=release_id,
        title="Test Release",
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri=f"https://discogs.com/release/{release_id}",
        cover_image_url=None,
        artists=[ArtistData(id=10, name="Test Artist")],
        labels=[],
        tracks=[TrackData(position="A1", title="Rave-O-Lution", duration="4:30", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=270, description=None)],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")

    def fake_extractor(url):
        return {"title": "Rave-O-Lution", "duration": 270, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=lambda p: (_ for _ in ()).throw(AssertionError))


# --- track_audio_stem / find_existing_download -------------------------------


def test_track_audio_stem_includes_position_and_sanitized_title(tmp_path):
    stem = ad.track_audio_stem(tmp_path, 1, "A1", 'Weird / Title: "Quoted"')
    assert stem.parent == tmp_path / "1"
    assert "/" not in stem.name
    assert ":" not in stem.name
    assert '"' not in stem.name
    assert "A1" in stem.name


def test_track_audio_stem_without_position_uses_title_only(tmp_path):
    stem = ad.track_audio_stem(tmp_path, 1, None, "Some Title")
    assert stem.name == "Some Title"


def test_find_existing_download_returns_none_when_absent(tmp_path):
    assert ad.find_existing_download(tmp_path / "1" / "A1 - Track") is None


def test_find_existing_download_finds_file_regardless_of_extension(tmp_path):
    stem = tmp_path / "1" / "A1 - Track"
    stem.parent.mkdir(parents=True)
    real_file = stem.with_suffix(".m4a")
    real_file.write_bytes(b"fake audio")

    found = ad.find_existing_download(stem)

    assert found == real_file


# --- download_release_tracks -------------------------------------------------


def test_download_release_tracks_downloads_matched_track(conn, tmp_path):
    seed_matched_release(conn)
    calls = []

    def fake_downloader(url, dest_stem):
        calls.append((url, dest_stem))
        dest_stem.parent.mkdir(parents=True, exist_ok=True)
        path = dest_stem.with_suffix(".m4a")
        path.write_bytes(b"fake audio")
        return path

    results = ad.download_release_tracks(conn, 1, tmp_path, downloader=fake_downloader)

    assert len(calls) == 1
    assert calls[0][0] == "https://youtube.com/watch?v=abc"
    assert len(results) == 1
    assert results[0]["status"] == "downloaded"
    assert results[0]["position"] == "A1"
    assert results[0]["path"].exists()


def test_download_release_tracks_skips_already_downloaded(conn, tmp_path):
    seed_matched_release(conn)

    def fake_downloader(url, dest_stem):
        dest_stem.parent.mkdir(parents=True, exist_ok=True)
        path = dest_stem.with_suffix(".m4a")
        path.write_bytes(b"fake audio")
        return path

    ad.download_release_tracks(conn, 1, tmp_path, downloader=fake_downloader)
    calls = []

    def unreachable_downloader(url, dest_stem):
        calls.append((url, dest_stem))
        raise AssertionError("should not re-download")

    results = ad.download_release_tracks(conn, 1, tmp_path, downloader=unreachable_downloader)

    assert len(calls) == 0
    assert results[0]["status"] == "already_downloaded"


def test_download_release_tracks_records_failure_without_raising(conn, tmp_path):
    seed_matched_release(conn)

    def failing_downloader(url, dest_stem):
        raise RuntimeError("network error")

    errors = []
    results = ad.download_release_tracks(
        conn,
        1,
        tmp_path,
        downloader=failing_downloader,
        on_error=lambda pos, exc: errors.append((pos, str(exc))),
        retry_delay=0,
    )

    assert results[0]["status"] == "failed"
    assert results[0]["path"] is None
    assert len(errors) == 1
    assert errors[0][0] == "A1"


def test_download_release_tracks_retries_transient_failures_then_succeeds(conn, tmp_path):
    seed_matched_release(conn)
    attempts = []

    def flaky_downloader(url, dest_stem):
        attempts.append(url)
        if len(attempts) < 2:
            raise RuntimeError("transient 403")
        dest_stem.parent.mkdir(parents=True, exist_ok=True)
        path = dest_stem.with_suffix(".m4a")
        path.write_bytes(b"fake audio")
        return path

    errors = []
    results = ad.download_release_tracks(
        conn,
        1,
        tmp_path,
        downloader=flaky_downloader,
        on_error=lambda pos, exc: errors.append((pos, str(exc))),
        retry_delay=0,
    )

    assert len(attempts) == 2
    assert results[0]["status"] == "downloaded"
    assert errors == []


def test_download_release_tracks_gives_up_after_max_attempts(conn, tmp_path):
    seed_matched_release(conn)
    attempts = []

    def always_failing_downloader(url, dest_stem):
        attempts.append(url)
        raise RuntimeError("persistent failure")

    results = ad.download_release_tracks(
        conn, 1, tmp_path, downloader=always_failing_downloader, retry_delay=0, max_attempts=3
    )

    assert len(attempts) == 3
    assert results[0]["status"] == "failed"


def test_download_release_tracks_ignores_tracks_without_a_selected_match(conn, tmp_path):
    # A track that was never matched (no track_video_matches row at all)
    # simply isn't downloadable — no error, just excluded.
    data = ReleaseData(
        id=1,
        title="Test Release",
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri="https://discogs.com/release/1",
        cover_image_url=None,
        artists=[ArtistData(id=10, name="Test Artist")],
        labels=[],
        tracks=[TrackData(position="A1", title="Unmatched Track", duration="4:30", artist=None)],
        videos=[],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")

    results = ad.download_release_tracks(conn, 1, tmp_path, downloader=lambda u, s: (_ for _ in ()).throw(AssertionError))

    assert results == []


def test_download_release_tracks_reports_progress(conn, tmp_path):
    seed_matched_release(conn)

    def fake_downloader(url, dest_stem):
        dest_stem.parent.mkdir(parents=True, exist_ok=True)
        path = dest_stem.with_suffix(".m4a")
        path.write_bytes(b"fake audio")
        return path

    calls = []
    ad.download_release_tracks(
        conn, 1, tmp_path, downloader=fake_downloader, on_progress=lambda *args: calls.append(args)
    )

    assert calls == [(1, 1, "A1")]


# --- _has_video_stream / _strip_video_track ----------------------------------


@requires_ffmpeg
def test_has_video_stream_true_for_combined_file(tmp_path):
    path = tmp_path / "combined.mp4"
    _make_av_file(path)

    assert ad._has_video_stream(path) is True


@requires_ffmpeg
def test_has_video_stream_false_for_audio_only_file(tmp_path):
    path = tmp_path / "audio.m4a"
    _make_audio_only_file(path)

    assert ad._has_video_stream(path) is False


@requires_ffmpeg
def test_strip_video_track_produces_audio_only_file_via_stream_copy(tmp_path):
    # Real-world case: yt-dlp's "/best" fallback can hand back a combined
    # video+audio file when no pure audio-only stream is available. Stripping
    # must be a lossless stream copy (-acodec copy), not a re-encode.
    path = tmp_path / "combined.mp4"
    _make_av_file(path)

    result = ad._strip_video_track(path)

    assert result.suffix == ".m4a"
    assert result.exists()
    assert not path.exists()
    assert ad._has_video_stream(result) is False


@requires_ffmpeg
def test_strip_video_track_falls_back_to_original_on_ffmpeg_failure(tmp_path, monkeypatch):
    path = tmp_path / "combined.mp4"
    _make_av_file(path)

    def failing_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg")

    monkeypatch.setattr(ad.subprocess, "run", failing_run)

    result = ad._strip_video_track(path)

    assert result == path
    assert path.exists()


# --- build_downloader: optional cookies-from-browser auth --------------------


class _FakeYoutubeDL:
    captured_opts = None

    def __init__(self, opts):
        _FakeYoutubeDL.captured_opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def extract_info(self, url, download=True):
        return {"ext": "m4a"}


def test_build_downloader_without_cookies_browser_omits_cookiesfrombrowser(monkeypatch, tmp_path):
    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeDL)
    monkeypatch.setattr(ad, "_has_video_stream", lambda path: False)

    downloader = ad.build_downloader()
    downloader("https://youtube.com/watch?v=abc", tmp_path / "track")

    assert "cookiesfrombrowser" not in _FakeYoutubeDL.captured_opts


def test_build_downloader_with_cookies_browser_sets_cookiesfrombrowser(monkeypatch, tmp_path):
    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeDL)
    monkeypatch.setattr(ad, "_has_video_stream", lambda path: False)

    downloader = ad.build_downloader("chrome")
    downloader("https://youtube.com/watch?v=abc", tmp_path / "track")

    assert _FakeYoutubeDL.captured_opts["cookiesfrombrowser"] == ("chrome",)


# --- build_downloader: optional max-bitrate cap -------------------------------


def test_build_downloader_without_max_bitrate_uses_unrestricted_bestaudio(monkeypatch, tmp_path):
    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeDL)
    monkeypatch.setattr(ad, "_has_video_stream", lambda path: False)

    downloader = ad.build_downloader()
    downloader("https://youtube.com/watch?v=abc", tmp_path / "track")

    assert _FakeYoutubeDL.captured_opts["format"] == "bestaudio/best"


def test_build_downloader_with_max_bitrate_caps_format_selector(monkeypatch, tmp_path):
    import yt_dlp

    monkeypatch.setattr(yt_dlp, "YoutubeDL", _FakeYoutubeDL)
    monkeypatch.setattr(ad, "_has_video_stream", lambda path: False)

    downloader = ad.build_downloader(max_bitrate_kbps=128)
    downloader("https://youtube.com/watch?v=abc", tmp_path / "track")

    assert _FakeYoutubeDL.captured_opts["format"] == "bestaudio[abr<=128]/bestaudio/best"
