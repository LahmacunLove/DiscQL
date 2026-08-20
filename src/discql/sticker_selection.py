from __future__ import annotations

import sqlite3

from discql.db import utcnow_iso


def add_release(conn: sqlite3.Connection, release_id: int) -> None:
    """Add a release to the sticker selection - idempotent, a silent no-op
    if it's already in there."""
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO sticker_selection (release_id, added_at) VALUES (?, ?)",
            (release_id, utcnow_iso()),
        )


def add_many(conn: sqlite3.Connection, release_ids: list[int]) -> None:
    """Bulk version of add_release, for "select all filtered releases"."""
    now = utcnow_iso()
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO sticker_selection (release_id, added_at) VALUES (?, ?)",
            [(release_id, now) for release_id in release_ids],
        )


def remove_release(conn: sqlite3.Connection, release_id: int) -> None:
    """Remove a release from the sticker selection - a no-op if it wasn't in there."""
    with conn:
        conn.execute("DELETE FROM sticker_selection WHERE release_id = ?", (release_id,))


def clear(conn: sqlite3.Connection) -> None:
    with conn:
        conn.execute("DELETE FROM sticker_selection")


def selected_ids(conn: sqlite3.Connection) -> set[int]:
    rows = conn.execute("SELECT release_id FROM sticker_selection").fetchall()
    return {row["release_id"] for row in rows}


def is_selected(conn: sqlite3.Connection, release_id: int) -> bool:
    row = conn.execute("SELECT 1 FROM sticker_selection WHERE release_id = ?", (release_id,)).fetchone()
    return row is not None
