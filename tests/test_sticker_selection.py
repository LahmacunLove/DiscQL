from __future__ import annotations

import sqlite3

import pytest

from discql import db, sticker_selection
from tests.test_sync import make_release
from discql.sync import upsert_release


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


def test_add_release_adds_membership_without_touching_the_release_itself(conn):
    seed_release(conn, 1)

    sticker_selection.add_release(conn, 1)

    assert sticker_selection.is_selected(conn, 1) is True
    release = conn.execute("SELECT removed_from_discogs_at FROM releases WHERE id = 1").fetchone()
    assert release["removed_from_discogs_at"] is None


def test_add_release_is_idempotent(conn):
    seed_release(conn, 1)

    sticker_selection.add_release(conn, 1)
    sticker_selection.add_release(conn, 1)

    rows = conn.execute("SELECT * FROM sticker_selection WHERE release_id = 1").fetchall()
    assert len(rows) == 1


def test_add_many_adds_all_given_releases(conn):
    seed_release(conn, 1)
    seed_release(conn, 2)
    seed_release(conn, 3)

    sticker_selection.add_many(conn, [1, 2, 3])

    assert sticker_selection.selected_ids(conn) == {1, 2, 3}


def test_remove_release(conn):
    seed_release(conn, 1)
    sticker_selection.add_release(conn, 1)

    sticker_selection.remove_release(conn, 1)

    assert sticker_selection.is_selected(conn, 1) is False


def test_remove_release_not_present_is_a_noop(conn):
    seed_release(conn, 1)

    sticker_selection.remove_release(conn, 1)  # never added - should not raise


def test_clear_empties_the_whole_selection(conn):
    seed_release(conn, 1)
    seed_release(conn, 2)
    sticker_selection.add_many(conn, [1, 2])

    sticker_selection.clear(conn)

    assert sticker_selection.selected_ids(conn) == set()


def test_is_selected_false_for_unselected_release(conn):
    seed_release(conn, 1)

    assert sticker_selection.is_selected(conn, 1) is False
