from __future__ import annotations

import sqlite3

import pytest

from discql import crates, db
from discql.sync import upsert_release
from tests.test_sync import make_release


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    db.migrate(connection)
    yield connection
    connection.close()


def seed_release(conn, release_id):
    upsert_release(conn, make_release(release_id), date_added="2024-01-01T00:00:00", now="2024-01-01T00:00:00")


def test_create_crate_returns_new_id_and_is_retrievable(conn):
    crate_id = crates.create_crate(conn, "Warehouse Gig")

    crate = crates.get_crate(conn, crate_id)
    assert crate is not None
    assert crate.name == "Warehouse Gig"
    assert crate.created_at


def test_get_crate_returns_none_for_unknown_id(conn):
    assert crates.get_crate(conn, 999) is None


def test_rename_crate(conn):
    crate_id = crates.create_crate(conn, "Old Name")

    crates.rename_crate(conn, crate_id, "New Name")

    assert crates.get_crate(conn, crate_id).name == "New Name"


def test_delete_crate(conn):
    crate_id = crates.create_crate(conn, "Temp")

    crates.delete_crate(conn, crate_id)

    assert crates.get_crate(conn, crate_id) is None


def test_delete_crate_cascades_to_membership_rows(conn):
    seed_release(conn, 1)
    crate_id = crates.create_crate(conn, "Temp")
    crates.add_release(conn, crate_id, 1)

    crates.delete_crate(conn, crate_id)

    rows = conn.execute("SELECT * FROM crate_releases WHERE crate_id = ?", (crate_id,)).fetchall()
    assert rows == []


def test_add_release_adds_membership_without_touching_the_release_itself(conn):
    seed_release(conn, 1)
    crate_id = crates.create_crate(conn, "Warehouse Gig")

    crates.add_release(conn, crate_id, 1)

    row = conn.execute(
        "SELECT * FROM crate_releases WHERE crate_id = ? AND release_id = ?", (crate_id, 1)
    ).fetchone()
    assert row is not None
    assert row["added_at"]
    # Adding to a crate never modifies/removes anything in the library itself.
    release = conn.execute("SELECT removed_from_discogs_at FROM releases WHERE id = 1").fetchone()
    assert release["removed_from_discogs_at"] is None


def test_add_release_is_idempotent(conn):
    seed_release(conn, 1)
    crate_id = crates.create_crate(conn, "Warehouse Gig")

    crates.add_release(conn, crate_id, 1)
    crates.add_release(conn, crate_id, 1)  # not an error, no duplicate row

    rows = conn.execute("SELECT * FROM crate_releases WHERE crate_id = ?", (crate_id,)).fetchall()
    assert len(rows) == 1


def test_remove_release(conn):
    seed_release(conn, 1)
    crate_id = crates.create_crate(conn, "Warehouse Gig")
    crates.add_release(conn, crate_id, 1)

    crates.remove_release(conn, crate_id, 1)

    rows = conn.execute("SELECT * FROM crate_releases WHERE crate_id = ?", (crate_id,)).fetchall()
    assert rows == []


def test_remove_release_not_present_is_a_noop(conn):
    seed_release(conn, 1)
    crate_id = crates.create_crate(conn, "Warehouse Gig")

    crates.remove_release(conn, crate_id, 1)  # never added - should not raise
