from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from discql.db import utcnow_iso


@dataclass
class Crate:
    id: int
    name: str
    created_at: str


def create_crate(conn: sqlite3.Connection, name: str) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO crates (name, created_at) VALUES (?, ?)", (name, utcnow_iso())
        )
    return cur.lastrowid


def rename_crate(conn: sqlite3.Connection, crate_id: int, name: str) -> None:
    with conn:
        conn.execute("UPDATE crates SET name = ? WHERE id = ?", (name, crate_id))


def delete_crate(conn: sqlite3.Connection, crate_id: int) -> None:
    with conn:
        conn.execute("DELETE FROM crates WHERE id = ?", (crate_id,))


def get_crate(conn: sqlite3.Connection, crate_id: int) -> Crate | None:
    row = conn.execute(
        "SELECT id, name, created_at FROM crates WHERE id = ?", (crate_id,)
    ).fetchone()
    return Crate(id=row["id"], name=row["name"], created_at=row["created_at"]) if row else None


def add_release(conn: sqlite3.Connection, crate_id: int, release_id: int) -> None:
    """Add a release to a crate - never modifies/removes anything in the
    library itself, only membership. Idempotent: adding an already-present
    release is a silent no-op, not an error."""
    with conn:
        conn.execute(
            "INSERT OR IGNORE INTO crate_releases (crate_id, release_id, added_at) VALUES (?, ?, ?)",
            (crate_id, release_id, utcnow_iso()),
        )


def remove_release(conn: sqlite3.Connection, crate_id: int, release_id: int) -> None:
    """Remove a release from a crate - a no-op if it wasn't in there."""
    with conn:
        conn.execute(
            "DELETE FROM crate_releases WHERE crate_id = ? AND release_id = ?",
            (crate_id, release_id),
        )
