from __future__ import annotations

import re
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Callable

# (url, dest_stem) -> final downloaded path (dest_stem + real extension)
VideoDownloader = Callable[[str, Path], Path]

# Called after each track is downloaded/skipped: (index, total, position)
ProgressCallback = Callable[[int, int, str | None], None]
# Called when a single track's download fails: (position, exception)
ErrorCallback = Callable[[str | None, Exception], None]

# Downloads can fail transiently (rate limiting, momentary 403s - observed
# concretely on real releases this session), so each track gets a few
# attempts before being reported as failed.
DOWNLOAD_MAX_ATTEMPTS = 3
DOWNLOAD_RETRY_DELAY_SECONDS = 2.0

_UNSAFE_FILENAME_RE = re.compile(r'[\\/:*?"<>|]')


def sanitize_filename(text: str) -> str:
    text = _UNSAFE_FILENAME_RE.sub("_", text)
    return re.sub(r"\s+", " ", text).strip()


def track_audio_stem(audio_dir: Path, release_id: int, position: str | None, title: str) -> Path:
    label = f"{position} - {title}" if position else title
    return audio_dir / str(release_id) / sanitize_filename(label)


def find_existing_download(dest_stem: Path) -> Path | None:
    """Return the already-downloaded file for dest_stem (any extension), if any."""
    parent = dest_stem.parent
    if not parent.is_dir():
        return None
    for path in parent.iterdir():
        if path.stem == dest_stem.name:
            return path
    return None


def count_downloaded_tracks(audio_dir: Path, release_id: int) -> int:
    """Number of files in release_id's persisted-download directory
    (audio_dir/<id>/, see track_audio_stem) - one file per downloaded
    track, so this is a release-list-page-cheap stand-in for "how many
    tracks are downloaded" without matching filenames back to specific
    tracks.
    """
    release_dir = audio_dir / str(release_id)
    if not release_dir.is_dir():
        return 0
    return sum(1 for _ in release_dir.iterdir())


def _has_video_stream(path: Path) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
    )
    return bool(result.stdout.strip())


def _strip_video_track(path: Path) -> Path:
    """`bestaudio` isn't always available (observed concretely: YouTube's
    SABR streaming restriction can hide audio-only formats for a specific
    video on some clients), so yt-dlp's "/best" fallback can hand back a
    combined video+audio file. Losslessly extract just the audio stream via
    ffmpeg -acodec copy (a stream copy, not a re-encode — bit-identical audio
    data, just remuxed without the video track) rather than keeping the
    unwanted video around.
    """
    audio_only = path.with_suffix(".m4a")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vn", "-acodec", "copy", str(audio_only)],
            check=True,
            capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return path
    path.unlink()
    return audio_only


class _YoutubeDownloader:
    """Download the best-quality audio-only stream for a track URL, optionally
    authenticated via --cookies-from-browser (yt-dlp's cookiesfrombrowser
    option) so age-restricted videos — which YouTube otherwise refuses to
    serve at all, regardless of player_client — can be downloaded too.

    A class (holding cookies_browser/max_bitrate_kbps as plain attributes)
    rather than build_downloader() returning a closure, so instances are
    picklable — analyze_library() passes its downloader across a
    ProcessPoolExecutor boundary, and a closure over a local function can't
    be pickled (a plain function reference or a picklable-state callable
    can).

    max_bitrate_kbps, if set, caps which audio-only stream yt-dlp picks
    (format selector "bestaudio[abr<=N]") rather than always the highest
    available (typically ~160kbps Opus) - still one of YouTube's own native
    streams, still no re-encoding, just a smaller one. Falls back to
    unrestricted bestaudio/best if no stream meets the cap.
    """

    def __init__(self, cookies_browser: str | None = None, max_bitrate_kbps: int | None = None):
        self.cookies_browser = cookies_browser
        self.max_bitrate_kbps = max_bitrate_kbps

    def __call__(self, url: str, dest_stem: Path) -> Path:
        """Download with no re-encoding (no postprocessors — the file is
        saved exactly as YouTube serves it, whatever container/codec that
        happens to be), as dest_stem.<real extension>. Falls back to a
        combined video+audio format only if no audio-only stream is
        available at all, in which case the video track is then stripped
        losslessly — see _strip_video_track.
        """
        from yt_dlp import YoutubeDL

        format_selector = "bestaudio/best"
        if self.max_bitrate_kbps is not None:
            format_selector = f"bestaudio[abr<={self.max_bitrate_kbps}]/{format_selector}"

        opts = {
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,
            "nocheckcertificate": True,
            "format": format_selector,
            "outtmpl": f"{dest_stem}.%(ext)s",
            # Deliberately NOT the same client list as metadata fetching
            # (youtube_matching.build_extractor uses android_vr too, for
            # good reason there - see its comment). For the actual media
            # download, android_vr reproducibly causes a hard "HTTP Error
            # 403: Forbidden" on the real GET (confirmed concretely against
            # a real video: fails identically whether android_vr is first,
            # last, or the only client in the list - yt-dlp merges formats
            # across all listed clients, so its presence poisons format
            # selection regardless of position). Likely fallout from
            # YouTube's "SABR-only streaming" experiment breaking
            # android/android_vr audio-only formats - see
            # https://github.com/yt-dlp/yt-dlp/issues/12482. Dropping it
            # here still leaves "android" (bestaudio may be unavailable, but
            # then falls through to a combined format - see
            # _strip_video_track) plus "web".
            "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        }
        if self.cookies_browser:
            opts["cookiesfrombrowser"] = (self.cookies_browser,)
            # Cookies make yt-dlp skip "android" (no cookie-auth support),
            # falling back to "web" alone, which needs its signature/n-
            # challenge solved to resolve any real formats - see the same
            # note in youtube_matching.build_extractor.
            opts["remote_components"] = ["ejs:github"]
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
        path = dest_stem.with_suffix(f".{info.get('ext', 'webm')}")

        if _has_video_stream(path):
            path = _strip_video_track(path)

        return path


def build_downloader(cookies_browser: str | None = None, max_bitrate_kbps: int | None = None) -> VideoDownloader:
    return _YoutubeDownloader(cookies_browser, max_bitrate_kbps)


default_video_downloader = build_downloader()


def _downloadable_tracks(conn: sqlite3.Connection, release_id: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT t.position, t.title, rv.url
        FROM tracks t
        JOIN track_video_matches tvm ON tvm.track_id = t.id AND tvm.is_selected = 1
        JOIN release_videos rv ON rv.id = tvm.video_id
        WHERE t.release_id = ?
        ORDER BY t.id
        """,
        (release_id,),
    ).fetchall()


def download_with_retries(
    downloader: VideoDownloader,
    url: str,
    dest_stem: Path,
    max_attempts: int,
    retry_delay: float,
) -> Path:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return downloader(url, dest_stem)
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                time.sleep(retry_delay)
    assert last_exc is not None
    raise last_exc


def download_release_tracks(
    conn: sqlite3.Connection,
    release_id: int,
    audio_dir: Path,
    downloader: VideoDownloader = default_video_downloader,
    on_progress: ProgressCallback | None = None,
    on_error: ErrorCallback | None = None,
    max_attempts: int = DOWNLOAD_MAX_ATTEMPTS,
    retry_delay: float = DOWNLOAD_RETRY_DELAY_SECONDS,
) -> list[dict]:
    """Download best-quality audio for every track in the release that has a
    primary matched video (track_video_matches.is_selected=1), skipping
    tracks that already have a downloaded file. Returns one dict per
    downloadable track: {"position", "title", "path", "status"} where status
    is "downloaded", "already_downloaded", or "failed". Transient failures
    (rate limiting, momentary 403s) are retried up to max_attempts times per
    track before being reported as failed.
    """
    tracks = _downloadable_tracks(conn, release_id)
    total = len(tracks)
    results = []

    for index, track in enumerate(tracks, start=1):
        stem = track_audio_stem(audio_dir, release_id, track["position"], track["title"])
        existing = find_existing_download(stem)
        if existing is not None:
            results.append(
                {"position": track["position"], "title": track["title"], "path": existing, "status": "already_downloaded"}
            )
        else:
            try:
                stem.parent.mkdir(parents=True, exist_ok=True)
                path = download_with_retries(downloader, track["url"], stem, max_attempts, retry_delay)
                results.append(
                    {"position": track["position"], "title": track["title"], "path": path, "status": "downloaded"}
                )
            except Exception as exc:
                if on_error is not None:
                    on_error(track["position"], exc)
                results.append(
                    {"position": track["position"], "title": track["title"], "path": None, "status": "failed"}
                )
        if on_progress is not None:
            on_progress(index, total, track["position"])

    return results
