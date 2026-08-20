from __future__ import annotations

import sqlite3

import pytest

from discql import collection, db
from tests.test_sync import make_release


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    db.migrate(connection)
    yield connection
    connection.close()


class FakeApi:
    def __init__(self):
        self.added_ids: list[int] = []
        self.removed_ids: list[int] = []
        self.release = make_release(200, title="Pushed Release")

    def add_release_to_collection(self, release_id: int) -> None:
        self.added_ids.append(release_id)

    def remove_release_from_collection(self, release_id: int) -> None:
        self.removed_ids.append(release_id)

    def fetch_release_detail(self, release_id: int):
        return self.release


def test_add_release_pushes_to_discogs_then_local_db(conn):
    api = FakeApi()

    collection.add_release(conn, api, 200)

    assert api.added_ids == [200]
    row = conn.execute("SELECT * FROM releases WHERE id = 200").fetchone()
    assert row["title"] == "Pushed Release"


def test_remove_release_pushes_to_discogs_then_soft_deletes_local(conn):
    api = FakeApi()
    collection.add_release(conn, api, 200)

    collection.remove_release(conn, api, 200)

    assert api.removed_ids == [200]
    row = conn.execute("SELECT * FROM releases WHERE id = 200").fetchone()
    assert row["removed_from_discogs_at"] is not None
