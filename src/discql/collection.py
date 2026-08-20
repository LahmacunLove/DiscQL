from __future__ import annotations

import sqlite3

from discql.db import utcnow_iso
from discql.discogs_api import DiscogsApi
from discql.sync import upsert_release


def add_release(conn: sqlite3.Connection, api: DiscogsApi, release_id: int) -> None:
    """Add a release to the Discogs collection, then mirror it into the local DB."""
    api.add_release_to_collection(release_id)
    now = utcnow_iso()
    data = api.fetch_release_detail(release_id)
    upsert_release(conn, data, date_added=now, now=now)


def remove_release(conn: sqlite3.Connection, api: DiscogsApi, release_id: int) -> None:
    """Remove a release from the Discogs collection, then soft-delete it locally."""
    api.remove_release_from_collection(release_id)
    now = utcnow_iso()
    with conn:
        conn.execute(
            "UPDATE releases SET removed_from_discogs_at = ? WHERE id = ?",
            (now, release_id),
        )
