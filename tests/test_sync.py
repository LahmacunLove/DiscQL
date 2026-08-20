from __future__ import annotations

import json
import sqlite3

import pytest

from discql import db, sync
from discql.discogs_api import (
    ArtistData,
    CreditData,
    LabelData,
    ReleaseData,
    TrackData,
    VideoData,
)


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    db.migrate(connection)
    yield connection
    connection.close()


class FakeApi:
    def __init__(self, releases: dict[int, ReleaseData], date_added: dict[int, str]):
        self.releases = releases
        self.date_added = date_added

    def fetch_collection_items(self):
        for release_id in self.releases:
            yield release_id, self.date_added.get(release_id)

    def fetch_release_detail(self, release_id: int) -> ReleaseData:
        return self.releases[release_id]


def make_release(release_id: int, title: str = "Test Release") -> ReleaseData:
    return ReleaseData(
        id=release_id,
        title=title,
        year=2020,
        genres=["Electronic"],
        styles=["Techno"],
        formats=[{"name": "Vinyl", "qty": "1"}],
        discogs_uri=f"https://discogs.com/release/{release_id}",
        cover_image_url="https://example.com/cover.jpg",
        country="Testland",
        images=[
            {"type": "primary", "uri": "https://example.com/cover.jpg", "uri150": None, "width": 600, "height": 600},
            {"type": "secondary", "uri": "https://example.com/back.jpg", "uri150": None, "width": 600, "height": 600},
        ],
        artists=[ArtistData(id=1, name="Some Artist")],
        labels=[LabelData(id=1, name="Some Label", catalog_number="CAT001")],
        tracks=[
            TrackData(position="A1", title="Track One", duration="3:45", artist=None),
            TrackData(position="A2", title="Track Two", duration="4:12", artist=None),
        ],
        videos=[
            VideoData(title="Track One (video)", url="https://youtube.com/watch?v=abc", duration=225, description="desc"),
        ],
        credits=[
            CreditData(artist_id=2, artist_name="Mix Engineer", role="Mixed By", tracks="", name_variation=""),
        ],
    )


def test_sync_inserts_new_release(conn):
    release = make_release(100)
    api = FakeApi({100: release}, {100: "2024-01-01T00:00:00"})

    sync.sync_collection(conn, api)

    row = conn.execute("SELECT * FROM releases WHERE id = 100").fetchone()
    assert row["title"] == "Test Release"
    assert row["removed_from_discogs_at"] is None
    assert row["country"] == "Testland"

    images = json.loads(row["images_json"])
    assert len(images) == 2
    assert images[0]["type"] == "primary"

    tracks = conn.execute("SELECT * FROM tracks WHERE release_id = 100").fetchall()
    assert len(tracks) == 2

    videos = conn.execute("SELECT * FROM release_videos WHERE release_id = 100").fetchall()
    assert len(videos) == 1
    assert videos[0]["title"] == "Track One (video)"
    assert videos[0]["url"] == "https://youtube.com/watch?v=abc"

    credits = conn.execute("SELECT * FROM release_credits WHERE release_id = 100").fetchall()
    assert len(credits) == 1
    assert credits[0]["role"] == "Mixed By"
    assert credits[0]["artist_id"] == 2

    credited_artist = conn.execute("SELECT name FROM artists WHERE id = 2").fetchone()
    assert credited_artist["name"] == "Mix Engineer"

    artists = conn.execute("SELECT * FROM release_artists WHERE release_id = 100").fetchall()
    assert len(artists) == 1


def test_sync_force_refetch_replaces_videos_and_credits(conn):
    release_v1 = make_release(100)
    api1 = FakeApi({100: release_v1}, {100: "2024-01-01T00:00:00"})
    sync.sync_collection(conn, api1)

    release_v2 = make_release(100)
    release_v2.videos = [
        VideoData(title="New video", url="https://youtube.com/watch?v=zzz", duration=300, description=None)
    ]
    release_v2.credits = [
        CreditData(artist_id=3, artist_name="New Credit", role="Remix", tracks="A1", name_variation=""),
    ]
    api2 = FakeApi({100: release_v2}, {100: "2024-01-01T00:00:00"})
    sync.sync_collection(conn, api2, force_refetch=True)

    videos = conn.execute("SELECT * FROM release_videos WHERE release_id = 100").fetchall()
    assert [v["title"] for v in videos] == ["New video"]

    credits = conn.execute("SELECT * FROM release_credits WHERE release_id = 100").fetchall()
    assert len(credits) == 1
    assert credits[0]["artist_id"] == 3
    assert credits[0]["role"] == "Remix"


def test_sync_reports_progress(conn):
    api = FakeApi(
        {100: make_release(100), 101: make_release(101, title="Second")},
        {100: "2024-01-01T00:00:00", 101: "2024-01-02T00:00:00"},
    )
    calls = []

    sync.sync_collection(conn, api, on_progress=lambda *args: calls.append(args))

    # Release details are fetched concurrently, so completion order isn't guaranteed.
    indices = sorted(call[0] for call in calls)
    totals = {call[1] for call in calls}
    release_ids = sorted(call[2] for call in calls)
    assert indices == [1, 2]
    assert totals == {2}
    assert release_ids == [100, 101]


def test_sync_continues_after_single_release_failure(conn):
    class FlakyApi(FakeApi):
        def fetch_release_detail(self, release_id: int) -> ReleaseData:
            if release_id == 100:
                raise RuntimeError("boom")
            return super().fetch_release_detail(release_id)

    api = FlakyApi(
        {100: make_release(100), 101: make_release(101, title="Second")},
        {100: "2024-01-01T00:00:00", 101: "2024-01-02T00:00:00"},
    )
    errors = []

    sync.sync_collection(conn, api, on_error=lambda release_id, exc: errors.append(release_id))

    assert errors == [100]
    assert conn.execute("SELECT id FROM releases WHERE id = 100").fetchone() is None
    assert conn.execute("SELECT id FROM releases WHERE id = 101").fetchone() is not None


def test_sync_continues_when_upsert_fails_for_one_release(conn, monkeypatch):
    api = FakeApi(
        {100: make_release(100), 101: make_release(101, title="Second")},
        {100: "2024-01-01T00:00:00", 101: "2024-01-02T00:00:00"},
    )
    real_upsert = sync.upsert_release

    def flaky_upsert(conn, data, date_added, now):
        if data.id == 100:
            raise sqlite3.IntegrityError("simulated constraint failure")
        return real_upsert(conn, data, date_added, now)

    monkeypatch.setattr(sync, "upsert_release", flaky_upsert)
    errors = []

    sync.sync_collection(conn, api, on_error=lambda release_id, exc: errors.append(release_id))

    assert errors == [100]
    assert conn.execute("SELECT id FROM releases WHERE id = 100").fetchone() is None
    assert conn.execute("SELECT id FROM releases WHERE id = 101").fetchone() is not None


def test_upsert_release_tolerates_duplicate_artist_ids(conn):
    release = make_release(100)
    release.artists = [ArtistData(id=1, name="Some Artist"), ArtistData(id=1, name="Some Artist")]

    sync.upsert_release(conn, release, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")

    rows = conn.execute("SELECT * FROM release_artists WHERE release_id = 100").fetchall()
    assert len(rows) == 1


def test_sync_limit_syncs_only_first_n_and_skips_soft_delete(conn):
    releases = {i: make_release(i, title=f"Release {i}") for i in range(1, 6)}
    date_added = {i: f"2024-01-{i:02d}T00:00:00" for i in range(1, 6)}
    api = FakeApi(releases, date_added)

    # Full sync first, so all 5 releases exist locally.
    sync.sync_collection(conn, api)

    # Now a limited re-sync of the same (unchanged) 5-release collection: nothing
    # should be soft-deleted just because it fell outside the limit.
    sync.sync_collection(conn, api, limit=1)

    rows = conn.execute("SELECT id, removed_from_discogs_at FROM releases").fetchall()
    assert len(rows) == 5
    assert all(row["removed_from_discogs_at"] is None for row in rows)


def test_sync_should_stop_prevents_any_fetch(conn):
    releases = {i: make_release(i, title=f"Release {i}") for i in range(1, 4)}
    date_added = {i: f"2024-01-{i:02d}T00:00:00" for i in range(1, 4)}
    api = FakeApi(releases, date_added)

    sync.sync_collection(conn, api, should_stop=lambda: True)

    rows = conn.execute("SELECT id FROM releases").fetchall()
    assert rows == []


def test_sync_should_stop_skips_soft_delete_reconciliation(conn):
    releases = {i: make_release(i, title=f"Release {i}") for i in range(1, 4)}
    date_added = {i: f"2024-01-{i:02d}T00:00:00" for i in range(1, 4)}
    api = FakeApi(releases, date_added)

    # All 3 releases exist locally first.
    sync.sync_collection(conn, api)

    # Now the collection shrinks to just release 1, but the sync is stopped
    # immediately - releases 2/3 must NOT be soft-deleted from a run that
    # never got to look at the full picture.
    api.releases = {1: releases[1]}
    api.date_added = {1: date_added[1]}
    sync.sync_collection(conn, api, force_refetch=True, should_stop=lambda: True)

    rows = conn.execute("SELECT id, removed_from_discogs_at FROM releases").fetchall()
    assert len(rows) == 3
    assert all(row["removed_from_discogs_at"] is None for row in rows)


def test_sync_default_skips_already_known_active_releases(conn):
    class CountingApi(FakeApi):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.fetch_calls: list[int] = []

        def fetch_release_detail(self, release_id: int) -> ReleaseData:
            self.fetch_calls.append(release_id)
            return super().fetch_release_detail(release_id)

    api = CountingApi(
        {100: make_release(100, title="Original"), 101: make_release(101, title="New")},
        {100: "2024-01-01T00:00:00", 101: "2024-01-02T00:00:00"},
    )
    sync.sync_collection(conn, api, limit=1)  # only release 100 exists locally so far
    assert api.fetch_calls == [100]

    # Mutate what Discogs would return for 100, to prove the default behavior leaves it alone.
    api.releases[100] = make_release(100, title="Changed on Discogs")
    api.fetch_calls.clear()

    sync.sync_collection(conn, api)

    assert api.fetch_calls == [101]
    row = conn.execute("SELECT title FROM releases WHERE id = 100").fetchone()
    assert row["title"] == "Original"
    row = conn.execute("SELECT title FROM releases WHERE id = 101").fetchone()
    assert row["title"] == "New"


def test_sync_force_refetch_updates_existing_release(conn):
    api1 = FakeApi({100: make_release(100, title="Old Title")}, {100: "2024-01-01T00:00:00"})
    sync.sync_collection(conn, api1)

    api2 = FakeApi({100: make_release(100, title="New Title")}, {100: "2024-01-01T00:00:00"})
    sync.sync_collection(conn, api2, force_refetch=True)

    row = conn.execute("SELECT * FROM releases WHERE id = 100").fetchone()
    assert row["title"] == "New Title"
    # added_at should be preserved across re-syncs
    tracks = conn.execute("SELECT * FROM tracks WHERE release_id = 100").fetchall()
    assert len(tracks) == 2


def test_sync_soft_deletes_missing_release(conn):
    api1 = FakeApi({100: make_release(100)}, {100: "2024-01-01T00:00:00"})
    sync.sync_collection(conn, api1)

    api2 = FakeApi({}, {})
    sync.sync_collection(conn, api2)

    row = conn.execute("SELECT * FROM releases WHERE id = 100").fetchone()
    assert row["removed_from_discogs_at"] is not None


def test_sync_reinstates_previously_removed_release(conn):
    api1 = FakeApi({100: make_release(100)}, {100: "2024-01-01T00:00:00"})
    sync.sync_collection(conn, api1)

    sync.sync_collection(conn, FakeApi({}, {}))
    row = conn.execute("SELECT * FROM releases WHERE id = 100").fetchone()
    assert row["removed_from_discogs_at"] is not None

    sync.sync_collection(conn, api1)
    row = conn.execute("SELECT * FROM releases WHERE id = 100").fetchone()
    assert row["removed_from_discogs_at"] is None
