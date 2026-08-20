from __future__ import annotations

import sqlite3
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable

from discql import audio_analysis, audio_download, db, local_audio, ollama_client, youtube_matching
from discql.audio_download import VideoDownloader
from discql.youtube_matching import OllamaCaller, VideoExtractor

# Called after each release finishes (or fails): (completed, total, release_id, phase)
# phase is "match" during the match+download pass, "analyze" during the analysis pass.
ProgressCallback = Callable[[int, int, int, str], None]
# Called when a release's pipeline run raises: (release_id, exception)
ErrorCallback = Callable[[int, Exception], None]

# A release still has something left to do (for a non-force run) if any of
# its videos hasn't been fetched yet, or any of its matched tracks hasn't
# been analyzed yet - same condition used to skip already-finished releases,
# reused here so the UI can show a pending count. "Matched" (for the
# analysis half) means either a selected YouTube match or a confident local
# file match (local_audio.py) - either is a usable audio source, see
# audio_analysis.analyze_release_tracks.
_PENDING_CONDITION = """
    AND (
        EXISTS (
            SELECT 1 FROM release_videos rv2
            WHERE rv2.release_id = r.id AND rv2.yt_fetched_at IS NULL
        )
        OR EXISTS (
            SELECT 1 FROM tracks t
            LEFT JOIN track_video_matches tvm ON tvm.track_id = t.id AND tvm.is_selected = 1
            WHERE t.release_id = r.id AND t.analyzed_at IS NULL
              AND (tvm.video_id IS NOT NULL OR t.local_audio_path IS NOT NULL)
        )
    )
"""


def count_pending_releases(conn: sqlite3.Connection) -> int:
    """How many releases a non-force "Run All" would still touch. Not
    joined against release_videos - a release with zero Discogs-listed
    videos but a confident local match still needs analysis (see
    _PENDING_CONDITION's second branch), so it must still be counted."""
    row = conn.execute(
        f"""
        SELECT COUNT(DISTINCT r.id) AS n
        FROM releases r
        WHERE r.removed_from_discogs_at IS NULL
        {_PENDING_CONDITION}
        """
    ).fetchone()
    return row["n"]


def _needs_match(conn: sqlite3.Connection, release_id: int) -> bool:
    return (
        conn.execute(
            "SELECT 1 FROM release_videos WHERE release_id = ? AND video_type IS NULL LIMIT 1",
            (release_id,),
        ).fetchone()
        is not None
    )


def _needs_analysis(conn: sqlite3.Connection, release_id: int) -> bool:
    return (
        conn.execute(
            """
            SELECT 1 FROM tracks t
            LEFT JOIN track_video_matches tvm ON tvm.track_id = t.id AND tvm.is_selected = 1
            WHERE t.release_id = ? AND t.analyzed_at IS NULL
              AND (tvm.video_id IS NOT NULL OR t.local_audio_path IS NOT NULL)
            LIMIT 1
            """,
            (release_id,),
        ).fetchone()
        is not None
    )


def _locally_covered(conn: sqlite3.Connection, release_id: int) -> bool:
    """True if every track in the release's tracklist already has a
    confident local file match (local_audio.py) - nothing left for YouTube
    matching/downloading to contribute for this release. A release with no
    tracks at all is trivially not "covered" (nothing to skip for)."""
    row = conn.execute(
        "SELECT COUNT(*) AS total, COUNT(local_audio_path) AS covered FROM tracks WHERE release_id = ?",
        (release_id,),
    ).fetchone()
    return row["total"] > 0 and row["total"] == row["covered"]


def _match_and_download_one_release(
    db_path: Path,
    release_id: int,
    release_title: str,
    audio_dir: Path,
    downloader: VideoDownloader,
    extractor: VideoExtractor,
    call_ollama: OllamaCaller,
    force: bool,
    confident_threshold: float,
    tie_margin: float,
    denoise_terms: list[str],
) -> None:
    # A dedicated connection per worker thread - sqlite3.Connection isn't
    # safe for concurrent use from multiple threads at once - so each
    # release's match+download pair runs independently of whatever other
    # releases are doing at the same time.
    conn = db.connect(db_path)
    try:
        # A release whose entire tracklist already has a confident local
        # file match (local_audio.py, run as a pre-step by
        # match_and_download_library before this is ever submitted) has
        # nothing left for YouTube matching/downloading to contribute -
        # skip both entirely rather than spending yt-dlp/Ollama calls on a
        # release that's already fully covered from the local library.
        if not force and _locally_covered(conn, release_id):
            return

        if force or _needs_match(conn, release_id):
            youtube_matching.match_release(
                conn,
                release_id,
                release_title,
                force,
                youtube_matching.DEFAULT_WORKERS,
                extractor,
                call_ollama,
                confident_threshold=confident_threshold,
                tie_margin=tie_margin,
                denoise_terms=denoise_terms,
            )
        # Only bother downloading audio a release doesn't need analyzed yet
        # (already fully analyzed) - re-checked *after* matching, since
        # matching above may have just produced the selected match this
        # depends on. download_release_tracks itself already skips any
        # track that already has a persisted file, so this is safe to call
        # even for a release that's only partially done.
        if force or _needs_analysis(conn, release_id):
            audio_download.download_release_tracks(conn, release_id, audio_dir, downloader=downloader)
    finally:
        conn.close()


def match_and_download_library(
    conn: sqlite3.Connection,
    db_path: Path,
    audio_dir: Path,
    downloader: VideoDownloader,
    extractor: VideoExtractor,
    call_ollama: OllamaCaller,
    force: bool = False,
    max_workers: int = 1,
    confident_threshold: float = youtube_matching.CONFIDENT_THRESHOLD,
    tie_margin: float = youtube_matching.TIE_MARGIN,
    denoise_terms: list[str] = youtube_matching.DEFAULT_DENOISE_TERMS,
    flac_dir: Path | None = None,
    local_confident_threshold: float = local_audio.DEFAULT_CONFIDENT_THRESHOLD,
    on_progress: ProgressCallback | None = None,
    on_error: ErrorCallback | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Match + download every release with at least one YouTube video
    candidate, up to max_workers releases at a time in parallel. This is
    the standalone "Match YT" button's implementation, and also
    run_full_pipeline()'s ("Run All") phase 1 - see there for how the two
    relate.

    Network/LLM-bound (yt-dlp metadata fetches, Ollama disambiguation,
    yt-dlp audio downloads), so concurrency across releases actually buys
    real wall-clock speedup, unlike audio analysis. A release's match step
    is skipped if it's already fully classified (unless force=True); its
    download step is skipped if the release is already fully analyzed
    (nothing left that would need the audio). Ollama is started once for
    the whole run (not once per release/thread) and stopped again before
    returning.

    Runs local_audio.match_library() as a cheap pre-step first (main
    thread, before the worker pool starts - a plain filesystem walk plus
    in-memory fuzzy scoring, not worth parallelizing or repeating per
    worker). If flac_dir is None (feature disabled) or doesn't exist (e.g.
    an unmounted NFS share), this is a no-op - see local_audio.match_library.
    Any release match_library confidently covers end-to-end gets skipped
    below (_locally_covered, checked per-release in
    _match_and_download_one_release) - YouTube becomes a fallback only for
    tracks without a local copy.

    If `should_stop` starts returning True, releases still in flight are
    allowed to finish but no new ones are submitted to the pool.
    """
    if flac_dir is not None:
        local_audio.match_library(conn, flac_dir, confident_threshold=local_confident_threshold, force=force)

    query = """
        SELECT DISTINCT r.id, r.title
        FROM releases r
        JOIN release_videos rv ON rv.release_id = r.id
        WHERE r.removed_from_discogs_at IS NULL
    """
    if not force:
        query += _PENDING_CONDITION
    release_rows = conn.execute(query).fetchall()
    releases = [(row["id"], row["title"]) for row in release_rows]
    total = len(releases)
    completed = 0
    workers = max(1, max_workers)

    def _stopped() -> bool:
        return should_stop is not None and should_stop()

    with ollama_client.ensure_ollama_running():
        with ThreadPoolExecutor(max_workers=workers) as pool:
            pending = list(releases)
            in_flight: dict = {}

            def submit_more() -> None:
                while pending and len(in_flight) < workers and not _stopped():
                    release_id, release_title = pending.pop(0)
                    future = pool.submit(
                        _match_and_download_one_release,
                        db_path,
                        release_id,
                        release_title,
                        audio_dir,
                        downloader,
                        extractor,
                        call_ollama,
                        force,
                        confident_threshold,
                        tie_margin,
                        denoise_terms,
                    )
                    in_flight[future] = release_id

            submit_more()
            while in_flight:
                done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
                for future in done:
                    release_id = in_flight.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        if on_error is not None:
                            on_error(release_id, exc)
                    completed += 1
                    if on_progress is not None:
                        on_progress(completed, total, release_id, "match")
                submit_more()


def run_full_pipeline(
    conn: sqlite3.Connection,
    db_path: Path,
    audio_dir: Path,
    waveform_dir: Path,
    models_dir: Path,
    downloader: VideoDownloader,
    extractor: VideoExtractor,
    call_ollama: OllamaCaller,
    force: bool = False,
    max_workers: int = 1,
    confident_threshold: float = youtube_matching.CONFIDENT_THRESHOLD,
    tie_margin: float = youtube_matching.TIE_MARGIN,
    denoise_terms: list[str] = youtube_matching.DEFAULT_DENOISE_TERMS,
    flac_dir: Path | None = None,
    local_confident_threshold: float = local_audio.DEFAULT_CONFIDENT_THRESHOLD,
    on_progress: ProgressCallback | None = None,
    on_error: ErrorCallback | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Two passes over the whole library, run to completion one after the
    other rather than interleaved per release:

    1. match_and_download_library() - see there. CPU/GPU-bound analysis
       hasn't started yet, so Ollama (started inside that call) is already
       stopped again by the time this returns - no reason to keep it (and
       whatever GPU/RAM it holds) around while phase 2 grinds through
       Essentia.
    2. Analyze every release with something left to analyze, up to
       max_workers releases at a time in parallel (delegates to
       audio_analysis.analyze_library - CPU/GPU-bound, so unlike phase 1's
       network/LLM-bound work, parallelism here has less headroom to gain
       from, but still helps when cores are available). Reuses whatever
       phase 1 already downloaded; a release matched but never downloaded
       (shouldn't happen, but not guaranteed - e.g. a stopped phase 1) is
       still handled correctly, since analyze_release_tracks falls back to
       downloading on demand for any track without a persisted file.

    Phase 2 only starts if phase 1 wasn't stopped early - see `should_stop`.
    """
    match_and_download_library(
        conn,
        db_path,
        audio_dir,
        downloader,
        extractor,
        call_ollama,
        force=force,
        max_workers=max_workers,
        confident_threshold=confident_threshold,
        tie_margin=tie_margin,
        denoise_terms=denoise_terms,
        flac_dir=flac_dir,
        local_confident_threshold=local_confident_threshold,
        on_progress=on_progress,
        on_error=on_error,
        should_stop=should_stop,
    )

    if should_stop is not None and should_stop():
        return

    def analyze_progress(completed: int, total: int, release_id: int) -> None:
        if on_progress is not None:
            on_progress(completed, total, release_id, "analyze")

    audio_analysis.analyze_library(
        conn,
        db_path,
        audio_dir,
        waveform_dir,
        models_dir,
        downloader=downloader,
        force=force,
        max_workers=max_workers,
        on_progress=analyze_progress,
        on_error=on_error,
        should_stop=should_stop,
    )
