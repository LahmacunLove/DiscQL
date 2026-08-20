"""Matches releases against a local audio library (e.g. Bandcamp FLAC
purchases) so YouTube matching/download becomes unnecessary wherever a
digital copy already exists locally - see README.md's "Local FLAC library
matching" section for the full picture (scan -> folder match -> file match
-> pipeline/analysis integration).

Verified against the real collection (~740 releases, ~170 local folders):
confidently matched 66 releases with token_sort_ratio (see
folder_match_score), all spot-checked correct bar one residual edge case -
a "Various" (compilation) artist release whose title is only a catalog
number (e.g. "SUN001") scored just above DEFAULT_CONFIDENT_THRESHOLD
against an unrelated folder sharing only the word "Various" and a
similar-shaped-but-different catalog number. No fuzzy text matcher can
reliably tell two different catalog codes apart from character similarity
alone, and this class of release (generic artist, no real title text) is
rare - accepted as a known residual false-positive risk for this first
pass rather than solved now. If you spot a wrong match in practice, raise
LOCAL_MATCH_CONFIDENT_THRESHOLD.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from rapidfuzz import fuzz
from rapidfuzz.utils import default_process as fuzz_process

from discql.youtube_matching import TrackInput

# Fuzzy score (0-1) above which a local folder is trusted as a release's
# match, and a local file is trusted as a track's match. Overridable per
# call so it can be configured via discql.config
# (LOCAL_MATCH_CONFIDENT_THRESHOLD) instead of hardcoded.
DEFAULT_CONFIDENT_THRESHOLD = 0.75

AUDIO_EXTENSIONS = {".flac", ".mp3", ".wav", ".m4a", ".aiff", ".ogg"}

# Called after each release is processed: (index, total, release_id)
ProgressCallback = Callable[[int, int, int], None]
# Called when a single release's local matching fails: (release_id, exception)
ErrorCallback = Callable[[int, Exception], None]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LocalFolder:
    path: Path
    files: list[Path]  # audio files directly inside, sorted


def scan_local_folders(root: Path) -> list[LocalFolder]:
    """Recursively finds every directory under root that directly contains
    at least one audio file, treating each as one release's folder
    candidate. Naturally covers both the dominant "one folder per release"
    layout and a label folder one level up with no audio files of its own
    (os.walk still descends into and yields its per-release subfolders,
    each of which does contain files directly). A folder of loose files
    with no per-release subfolder at all (a flat "misc singles" dump) is
    treated as a single candidate whose "release" text is just that
    folder's name - fuzzy matching against it will generally fail to match
    any real release, which is harmless (no match = no effect), but such
    loose files won't be individually discovered - not handled in this
    first pass.

    Returns [] if root doesn't exist (e.g. an unmounted NFS share) rather
    than raising - callers treat that the same as "nothing to match".
    """
    if not root.is_dir():
        return []

    folders = []
    for dirpath, _dirnames, filenames in os.walk(root):
        audio_files = sorted(
            Path(dirpath) / name for name in filenames if Path(name).suffix.lower() in AUDIO_EXTENSIONS
        )
        if audio_files:
            folders.append(LocalFolder(path=Path(dirpath), files=audio_files))
    return folders


def folder_match_score(folder_name: str, release_title: str, artist_names: list[str]) -> float:
    """Combined artist+title vs. folder name fuzzy score, normalized to
    [0, 1]. token_sort_ratio, not WRatio (unlike youtube_matching.fuzzy_score's
    title comparison) - confirmed concretely against the real collection
    that WRatio produces false positives here: its partial-ratio component
    dominates whenever one string is much shorter than the other (a short
    "artist + title" against a long, noisy folder name - catalog numbers,
    repeated artist/title text - is exactly that shape), scoring completely
    unrelated release/folder pairs at a suspiciously uniform ~0.85 as long
    as the folder name is long enough for pieces of the short string to
    coincidentally line up somewhere in it (observed multiple *different*
    releases all scoring identically against the same wrong folder).
    token_sort_ratio has no such escape hatch (word-order-independent exact
    comparison, not a partial/substring one) - measured cleanly separating
    real matches (~95-100) from that same false-positive set (~30-48) on
    real collection data, at the cost of also scoring some genuinely-correct
    but heavily-noisy folder names (e.g. one with the artist/title text
    repeated twice plus a catalog number) too low to confidently match.
    That's the right tradeoff for this feature: a missed match just leaves
    YouTube matching in place as it is today, unchanged; a false positive
    would silently attach the wrong audio to a track.
    """
    combined = f"{' '.join(artist_names)} {release_title}".strip()
    return fuzz.token_sort_ratio(combined, folder_name, processor=fuzz_process) / 100


def match_release_to_folder(
    release_title: str, artist_names: list[str], folders: list[LocalFolder]
) -> tuple[LocalFolder | None, float]:
    """Best-scoring folder for this release, plus its score (0.0, None if
    folders is empty)."""
    best_folder: LocalFolder | None = None
    best_score = 0.0
    for folder in folders:
        score = folder_match_score(folder.path.name, release_title, artist_names)
        if score > best_score:
            best_score = score
            best_folder = folder
    return best_folder, best_score


_TRACK_NUM_PREFIX_RE = re.compile(r"^\d{1,3}[\s._-]+")


def _clean_filename(path: Path) -> str:
    """Filename stem with a leading track-number prefix (e.g. "01 - ", "03.",
    "12_") stripped - ripped/converted filenames commonly lead with the
    track number, which is pure noise for fuzzy title matching against the
    tracklist (same reasoning as youtube_matching.denoise_video_title
    stripping boilerplate before scoring).
    """
    return _TRACK_NUM_PREFIX_RE.sub("", path.stem)


def match_tracks_to_files(tracks: list[TrackInput], files: list[Path]) -> dict[int, tuple[Path, float]]:
    """Best track<->file assignment within one matched folder. Unlike
    youtube_matching (which lets multiple videos map to the same track and
    breaks ties with an LLM), a local file should never be assigned to more
    than one track, and there's no LLM step here - so this computes the
    full track x file score matrix, then greedily assigns the
    highest-scoring pair first, removing both the track and file from
    further consideration, until no pairs are left. Returns every track
    that got a file assigned (however low its score) - callers filter by
    confident_threshold before treating a pair as trustworthy.
    """
    if not tracks or not files:
        return {}

    pairs: list[tuple[float, TrackInput, Path]] = []
    for track in tracks:
        track_text = f"{track.position or ''} {track.title}".strip()
        for file in files:
            file_text = _clean_filename(file)
            score = fuzz.WRatio(track_text, file_text, processor=fuzz_process)
            if track.artist:
                score = max(score, fuzz.WRatio(f"{track.title} {track.artist}", file_text, processor=fuzz_process))
            pairs.append((score / 100, track, file))

    pairs.sort(key=lambda p: p[0], reverse=True)

    assigned: dict[int, tuple[Path, float]] = {}
    used_files: set[Path] = set()
    for score, track, file in pairs:
        if track.id in assigned or file in used_files:
            continue
        assigned[track.id] = (file, score)
        used_files.add(file)

    return assigned


def _release_tracks(conn: sqlite3.Connection, release_id: int) -> list[TrackInput]:
    rows = conn.execute(
        "SELECT id, position, title, artist FROM tracks WHERE release_id = ?",
        (release_id,),
    ).fetchall()
    return [
        TrackInput(id=row["id"], position=row["position"], title=row["title"], artist=row["artist"], duration_seconds=None)
        for row in rows
    ]


def _release_artist_names(conn: sqlite3.Connection, release_id: int) -> list[str]:
    rows = conn.execute(
        """
        SELECT a.name FROM release_artists ra
        JOIN artists a ON a.id = ra.artist_id
        WHERE ra.release_id = ?
        """,
        (release_id,),
    ).fetchall()
    return [row["name"] for row in rows]


def match_library(
    conn: sqlite3.Connection,
    flac_dir: Path,
    confident_threshold: float = DEFAULT_CONFIDENT_THRESHOLD,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
    on_error: ErrorCallback | None = None,
) -> None:
    """Scan flac_dir once, then try to match every release that hasn't been
    checked yet (unless force=True re-checks everything) against the
    scanned folders, and each matched folder's files against that release's
    tracklist. A no-op (no DB writes at all) if flac_dir doesn't exist -
    see scan_local_folders - so a temporarily-unmounted share never gets
    releases incorrectly stamped as "checked, no local copy".

    A release whose best folder score clears confident_threshold gets
    releases.local_folder_path/local_folder_match_score/local_matched_at
    written; its tracks get local_audio_path/local_match_score written for
    every track whose file score also clears confident_threshold (NULL for
    the rest - explicitly, even on a force re-check where a track
    previously had a confident match that no longer holds). A release whose
    best folder score doesn't clear the threshold (including "no folders at
    all") still gets local_matched_at stamped (so a non-force run doesn't
    keep re-scanning it every time) but local_folder_path stays NULL, and
    any of its tracks' local_audio_path/local_match_score are cleared too.
    """
    if not flac_dir.is_dir():
        return

    folders = scan_local_folders(flac_dir)

    query = "SELECT id, title FROM releases WHERE removed_from_discogs_at IS NULL"
    if not force:
        query += " AND local_matched_at IS NULL"
    release_rows = conn.execute(query).fetchall()

    total = len(release_rows)
    completed = 0

    for row in release_rows:
        release_id = row["id"]
        try:
            _match_one_release(conn, release_id, row["title"], folders, confident_threshold)
        except Exception as exc:
            if on_error is not None:
                on_error(release_id, exc)
            continue
        completed += 1
        if on_progress is not None:
            on_progress(completed, total, release_id)


def _match_one_release(
    conn: sqlite3.Connection,
    release_id: int,
    release_title: str,
    folders: list[LocalFolder],
    confident_threshold: float,
) -> None:
    artist_names = _release_artist_names(conn, release_id)
    tracks = _release_tracks(conn, release_id)
    best_folder, folder_score = match_release_to_folder(release_title, artist_names, folders)
    now = _utcnow_iso()

    is_confident = best_folder is not None and folder_score >= confident_threshold
    file_matches = match_tracks_to_files(tracks, best_folder.files) if is_confident else {}

    with conn:
        conn.execute(
            """
            UPDATE releases
            SET local_folder_path = ?, local_folder_match_score = ?, local_matched_at = ?
            WHERE id = ?
            """,
            (str(best_folder.path) if is_confident else None, folder_score, now, release_id),
        )
        for track in tracks:
            match = file_matches.get(track.id)
            is_track_confident = match is not None and match[1] >= confident_threshold
            conn.execute(
                "UPDATE tracks SET local_audio_path = ?, local_match_score = ? WHERE id = ?",
                (str(match[0]) if is_track_confident else None, match[1] if match else None, track.id),
            )


def clear_local_match(conn: sqlite3.Connection, release_id: int) -> None:
    """Wipe local-matching results for a release back to a pristine "never
    checked" state, same "reset without re-scanning" pattern as
    youtube_matching.clear_matches / audio_analysis.clear_analysis.
    """
    with conn:
        conn.execute(
            """
            UPDATE releases
            SET local_folder_path = NULL, local_folder_match_score = NULL, local_matched_at = NULL
            WHERE id = ?
            """,
            (release_id,),
        )
        conn.execute(
            "UPDATE tracks SET local_audio_path = NULL, local_match_score = NULL WHERE release_id = ?",
            (release_id,),
        )
