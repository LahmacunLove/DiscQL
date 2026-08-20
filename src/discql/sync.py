from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from discql.db import utcnow_iso
from discql.discogs_api import DiscogsApi, ReleaseData

# Called after each release is processed: (index, total, release_id, title)
ProgressCallback = Callable[[int, int, int, str], None]
# Called when a single release's fetch fails: (release_id, exception)
ErrorCallback = Callable[[int, Exception], None]

DEFAULT_WORKERS = 5


def sync_collection(
    conn: sqlite3.Connection,
    api: DiscogsApi,
    on_progress: ProgressCallback | None = None,
    on_error: ErrorCallback | None = None,
    max_workers: int = DEFAULT_WORKERS,
    limit: int | None = None,
    force_refetch: bool = False,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """Mirror the Discogs collection into the local DB (Discogs -> local, one-directional).

    Release details are fetched concurrently (bounded pool) since each one is an
    independent network round-trip; the underlying discogs_client already backs off
    on 429s, so a small worker count stays safely under Discogs' rate limit while
    avoiding the dead time of doing every request strictly one after another.
    DB writes happen only on the main thread, so sqlite access stays single-threaded.

    If `limit` is set, only the first `limit` releases are synced (for fast dev
    iteration) and the soft-delete reconciliation is skipped, since a partial sync
    says nothing about whether releases outside the limit are still in the collection.

    By default, releases already present locally and not soft-deleted are skipped
    (not re-fetched from Discogs) — repeated syncs are then cheap and only pick up
    genuinely new or reinstated releases. Soft-deleted releases are always re-fetched
    so they can be reinstated if they've reappeared in the collection. Pass
    `force_refetch=True` to refresh every matched release's metadata regardless.

    If `should_stop` starts returning True partway through, already-queued
    fetches are cancelled (unstarted ones only - see cancel_futures below)
    and soft-delete reconciliation is skipped, same as the `limit` case,
    since a partial sync says nothing about what's still in the collection.
    """
    now = utcnow_iso()
    items = list(api.fetch_collection_items())
    if limit is not None:
        items = items[:limit]
    date_added_by_id = dict(items)
    seen_ids = set(date_added_by_id)

    if force_refetch:
        to_fetch = seen_ids
    else:
        active_ids = {
            row["id"]
            for row in conn.execute(
                "SELECT id FROM releases WHERE removed_from_discogs_at IS NULL"
            )
        }
        to_fetch = seen_ids - active_ids

    completed = 0
    stopped = False
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        futures = {
            executor.submit(api.fetch_release_detail, release_id): release_id
            for release_id in to_fetch
        }
        for future in as_completed(futures):
            if should_stop is not None and should_stop():
                stopped = True
                break
            release_id = futures[future]
            try:
                data = future.result()
                upsert_release(conn, data, date_added=date_added_by_id[release_id], now=now)
            except Exception as exc:
                if on_error is not None:
                    on_error(release_id, exc)
                continue
            completed += 1
            if on_progress is not None:
                on_progress(completed, len(to_fetch), release_id, data.title)
    finally:
        # cancel_futures drops anything not yet started; futures already
        # running finish in the background but their results are simply
        # never consumed above once stopped=True.
        executor.shutdown(wait=not stopped, cancel_futures=stopped)

    if limit is None and not stopped:
        soft_delete_missing(conn, seen_ids, now=now)


def upsert_release(
    conn: sqlite3.Connection,
    data: ReleaseData,
    date_added: str | None,
    now: str,
) -> None:
    with conn:
        existing = conn.execute(
            "SELECT added_at FROM releases WHERE id = ?", (data.id,)
        ).fetchone()
        added_at = existing["added_at"] if existing else now

        conn.execute(
            """
            INSERT INTO releases (
                id, title, year, formats_json, genres_json, styles_json,
                discogs_uri, date_added, cover_image_url, country, images_json,
                added_at, last_synced_at, removed_from_discogs_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                year = excluded.year,
                formats_json = excluded.formats_json,
                genres_json = excluded.genres_json,
                styles_json = excluded.styles_json,
                discogs_uri = excluded.discogs_uri,
                date_added = excluded.date_added,
                cover_image_url = excluded.cover_image_url,
                country = excluded.country,
                images_json = excluded.images_json,
                last_synced_at = excluded.last_synced_at,
                removed_from_discogs_at = NULL
            """,
            (
                data.id,
                data.title,
                data.year,
                json.dumps(data.formats),
                json.dumps(data.genres),
                json.dumps(data.styles),
                data.discogs_uri,
                date_added,
                data.cover_image_url,
                data.country,
                json.dumps(data.images),
                added_at,
                now,
            ),
        )

        for artist in data.artists:
            conn.execute(
                """
                INSERT INTO artists (id, name) VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET name = excluded.name
                """,
                (artist.id, artist.name),
            )
        conn.execute("DELETE FROM release_artists WHERE release_id = ?", (data.id,))
        for artist in data.artists:
            conn.execute(
                """
                INSERT INTO release_artists (release_id, artist_id) VALUES (?, ?)
                ON CONFLICT DO NOTHING
                """,
                (data.id, artist.id),
            )

        for label in data.labels:
            conn.execute(
                """
                INSERT INTO labels (id, name) VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET name = excluded.name
                """,
                (label.id, label.name),
            )
        conn.execute("DELETE FROM release_labels WHERE release_id = ?", (data.id,))
        for label in data.labels:
            conn.execute(
                """
                INSERT INTO release_labels (release_id, label_id, catalog_number)
                VALUES (?, ?, ?)
                ON CONFLICT DO NOTHING
                """,
                (data.id, label.id, label.catalog_number),
            )

        conn.execute("DELETE FROM tracks WHERE release_id = ?", (data.id,))
        for track in data.tracks:
            conn.execute(
                """
                INSERT INTO tracks (release_id, position, title, duration, artist)
                VALUES (?, ?, ?, ?, ?)
                """,
                (data.id, track.position, track.title, track.duration, track.artist),
            )

        conn.execute("DELETE FROM release_videos WHERE release_id = ?", (data.id,))
        for video in data.videos:
            conn.execute(
                """
                INSERT INTO release_videos (release_id, title, url, duration, description)
                VALUES (?, ?, ?, ?, ?)
                """,
                (data.id, video.title, video.url, video.duration, video.description),
            )

        for credit in data.credits:
            conn.execute(
                """
                INSERT INTO artists (id, name) VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET name = excluded.name
                """,
                (credit.artist_id, credit.artist_name),
            )
        conn.execute("DELETE FROM release_credits WHERE release_id = ?", (data.id,))
        for credit in data.credits:
            conn.execute(
                """
                INSERT INTO release_credits (release_id, artist_id, role, tracks, name_variation)
                VALUES (?, ?, ?, ?, ?)
                """,
                (data.id, credit.artist_id, credit.role, credit.tracks, credit.name_variation),
            )


def soft_delete_missing(conn: sqlite3.Connection, seen_ids: set[int], now: str) -> None:
    with conn:
        rows = conn.execute(
            "SELECT id FROM releases WHERE removed_from_discogs_at IS NULL"
        ).fetchall()
        missing_ids = [row["id"] for row in rows if row["id"] not in seen_ids]
        conn.executemany(
            "UPDATE releases SET removed_from_discogs_at = ? WHERE id = ?",
            [(now, release_id) for release_id in missing_ids],
        )
