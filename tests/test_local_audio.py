from __future__ import annotations

from pathlib import Path

import pytest

from discql import db, local_audio
from discql.discogs_api import ArtistData, ReleaseData, TrackData
from discql.sync import upsert_release
from discql.youtube_matching import TrackInput


@pytest.fixture()
def conn(tmp_path):
    path = tmp_path / "test.db"
    c = db.connect(path)
    db.migrate(c)
    return c


def seed_release(conn, release_id, title, artist, tracks):
    data = ReleaseData(
        id=release_id,
        title=title,
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri=f"https://discogs.com/release/{release_id}",
        cover_image_url=None,
        artists=[ArtistData(id=release_id * 10, name=artist)],
        labels=[],
        tracks=[TrackData(position=pos, title=t, duration=None, artist=None) for pos, t in tracks],
        videos=[],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")


def touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


# --- scan_local_folders ------------------------------------------------------


def test_scan_local_folders_finds_flat_folder_with_files(tmp_path):
    touch(tmp_path / "Artist - Title" / "01 Track.flac")

    folders = local_audio.scan_local_folders(tmp_path)

    assert len(folders) == 1
    assert folders[0].path == tmp_path / "Artist - Title"
    assert folders[0].files == [tmp_path / "Artist - Title" / "01 Track.flac"]


def test_scan_local_folders_finds_nested_label_folder_not_the_label_itself(tmp_path):
    touch(tmp_path / "Label" / "Artist - Title" / "01 Track.flac")

    folders = local_audio.scan_local_folders(tmp_path)

    assert len(folders) == 1
    assert folders[0].path == tmp_path / "Label" / "Artist - Title"


def test_scan_local_folders_ignores_non_audio_files(tmp_path):
    touch(tmp_path / "Artist - Title" / "cover.jpg")

    assert local_audio.scan_local_folders(tmp_path) == []


def test_scan_local_folders_missing_root_returns_empty(tmp_path):
    assert local_audio.scan_local_folders(tmp_path / "does-not-exist") == []


# --- folder_match_score / match_release_to_folder ----------------------------


def test_match_release_to_folder_picks_best_scoring_folder(tmp_path):
    good = local_audio.LocalFolder(path=tmp_path / "1200 Micrograms - 1200 Micrograms", files=[])
    bad = local_audio.LocalFolder(path=tmp_path / "Totally Unrelated Folder", files=[])

    best, score = local_audio.match_release_to_folder("1200 Micrograms", ["1200 Micrograms"], [good, bad])

    assert best is good
    assert score > 0.7


def test_match_release_to_folder_tolerates_mild_punctuation_noise(tmp_path):
    folder = local_audio.LocalFolder(path=tmp_path / "Magnetrixx- Ticon - Monolith 45 series I", files=[])

    best, score = local_audio.match_release_to_folder(
        "Monolith 45 series I", ["Magnetrixx", "Ticon"], [folder]
    )

    assert best is folder
    assert score >= local_audio.DEFAULT_CONFIDENT_THRESHOLD


def test_match_release_to_folder_rejects_heavily_duplicated_folder_name(tmp_path):
    """token_sort_ratio (not WRatio - see folder_match_score's docstring for
    why) doesn't confidently match a real release against a folder name
    that repeats the artist/title text twice plus a label prefix and
    catalog number - a real, if unfortunate, example from the collection.
    That's an accepted false negative (YouTube matching simply stays in
    place for it, unchanged from today), not a bug: the alternative
    (WRatio) was confirmed to produce false *positives* instead - see
    test_match_release_to_folder_never_confidently_matches_unrelated_release
    below for the concrete case that caused the switch.
    """
    folder = local_audio.LocalFolder(
        path=tmp_path / "Bound by Endogamy - Raw Ambassador - AAR028 - Bound by Endogamy - Raw Ambassador - Acid Avengers 028",
        files=[],
    )

    best, score = local_audio.match_release_to_folder("Raw Ambassador", ["Bound by Endogamy"], [folder])

    assert score < local_audio.DEFAULT_CONFIDENT_THRESHOLD


def test_match_release_to_folder_never_confidently_matches_unrelated_release(tmp_path):
    """Regression test for a concrete false-positive found on the real
    collection: WRatio scored *three different, unrelated* releases at an
    identical ~0.855 against this one folder, because its partial-ratio
    component dominates whenever one string is much shorter than the other
    - see folder_match_score's docstring. token_sort_ratio must not repeat
    this."""
    folder = local_audio.LocalFolder(
        path=tmp_path
        / "Cabasa- Innershades- Trancesetters Of Westphalia - NACHT01- Cabasa- Innershades- Trancesetters Of Westphalia",
        files=[],
    )

    for title, artists in [
        ("Tree Of Life EP", ["Nurse Erica"]),
        ("Hard 2 The Power Of Core", ["Clouds"]),
        ("Return Of The Sexmachine", ["Various"]),
    ]:
        _, score = local_audio.match_release_to_folder(title, artists, [folder])
        assert score < local_audio.DEFAULT_CONFIDENT_THRESHOLD


def test_match_release_to_folder_no_folders_returns_none_and_zero():
    best, score = local_audio.match_release_to_folder("Title", ["Artist"], [])
    assert best is None
    assert score == 0.0


# --- match_tracks_to_files ----------------------------------------------------


def test_match_tracks_to_files_strips_track_number_prefix_and_assigns_correctly(tmp_path):
    tracks = [
        TrackInput(id=1, position="A1", title="Ayahuasca", artist=None, duration_seconds=None),
        TrackInput(id=2, position="A2", title="Hashish", artist=None, duration_seconds=None),
    ]
    files = [
        tmp_path / "1200 Micrograms - 1200 Micrograms - 01 Ayahuasca.flac",
        tmp_path / "1200 Micrograms - 1200 Micrograms - 02 Hashish.flac",
    ]

    result = local_audio.match_tracks_to_files(tracks, files)

    assert result[1][0] == files[0]
    assert result[2][0] == files[1]
    assert result[1][1] > 0.7
    assert result[2][1] > 0.7


def test_match_tracks_to_files_never_assigns_same_file_twice(tmp_path):
    tracks = [
        TrackInput(id=1, position="A1", title="Song", artist=None, duration_seconds=None),
        TrackInput(id=2, position="A2", title="Song", artist=None, duration_seconds=None),
    ]
    files = [tmp_path / "01 Song.flac"]

    result = local_audio.match_tracks_to_files(tracks, files)

    assert len(result) == 1
    assert list(result.values())[0][0] == files[0]


def test_match_tracks_to_files_empty_inputs():
    assert local_audio.match_tracks_to_files([], [Path("x")]) == {}
    assert local_audio.match_tracks_to_files(
        [TrackInput(id=1, position="A1", title="X", artist=None, duration_seconds=None)], []
    ) == {}


# --- match_library -------------------------------------------------------------


def test_match_library_writes_confident_matches_and_stamps_matched_at(conn, tmp_path):
    seed_release(conn, 1, "1200 Micrograms", "1200 Micrograms", [("A1", "Ayahuasca"), ("A2", "Hashish")])
    flac_dir = tmp_path / "flac"
    touch(flac_dir / "1200 Micrograms - 1200 Micrograms" / "1200 Micrograms - 01 Ayahuasca.flac")
    touch(flac_dir / "1200 Micrograms - 1200 Micrograms" / "1200 Micrograms - 02 Hashish.flac")

    local_audio.match_library(conn, flac_dir)

    release = conn.execute("SELECT * FROM releases WHERE id = 1").fetchone()
    assert release["local_folder_path"] is not None
    assert release["local_folder_match_score"] >= local_audio.DEFAULT_CONFIDENT_THRESHOLD
    assert release["local_matched_at"] is not None

    tracks = conn.execute("SELECT * FROM tracks WHERE release_id = 1 ORDER BY position").fetchall()
    assert all(t["local_audio_path"] is not None for t in tracks)
    assert all(t["local_match_score"] >= local_audio.DEFAULT_CONFIDENT_THRESHOLD for t in tracks)


def test_match_library_stamps_matched_at_even_without_a_match(conn, tmp_path):
    seed_release(conn, 1, "Completely Unrelated Release Name", "Nobody", [("A1", "Track")])
    flac_dir = tmp_path / "flac"
    touch(flac_dir / "Some Other Artist - Some Other Album" / "01 Whatever.flac")

    local_audio.match_library(conn, flac_dir)

    release = conn.execute("SELECT * FROM releases WHERE id = 1").fetchone()
    assert release["local_folder_path"] is None
    assert release["local_matched_at"] is not None


def test_match_library_noop_when_flac_dir_missing(conn, tmp_path):
    seed_release(conn, 1, "Some Release", "Some Artist", [("A1", "Track")])

    local_audio.match_library(conn, tmp_path / "does-not-exist")

    release = conn.execute("SELECT * FROM releases WHERE id = 1").fetchone()
    assert release["local_matched_at"] is None


def test_match_library_skips_already_checked_releases_unless_forced(conn, tmp_path):
    seed_release(conn, 1, "1200 Micrograms", "1200 Micrograms", [("A1", "Ayahuasca")])
    flac_dir = tmp_path / "flac"
    # A folder present from the start so the very first scan can actually
    # run (and stamp "checked, no match") - an empty/missing flac_dir is a
    # no-op that stamps nothing at all (see the noop test above), so it
    # wouldn't exercise the "already checked" skip path being tested here.
    touch(flac_dir / "Unrelated Folder" / "01 Whatever.flac")

    local_audio.match_library(conn, flac_dir)
    first_stamp = conn.execute("SELECT local_matched_at FROM releases WHERE id = 1").fetchone()["local_matched_at"]
    assert first_stamp is not None

    touch(flac_dir / "1200 Micrograms - 1200 Micrograms" / "1200 Micrograms - 01 Ayahuasca.flac")

    local_audio.match_library(conn, flac_dir)  # non-force: release already checked, skipped
    release = conn.execute("SELECT * FROM releases WHERE id = 1").fetchone()
    assert release["local_folder_path"] is None
    assert release["local_matched_at"] == first_stamp

    local_audio.match_library(conn, flac_dir, force=True)  # force: re-checked, now matches
    release = conn.execute("SELECT * FROM releases WHERE id = 1").fetchone()
    assert release["local_folder_path"] is not None


def test_clear_local_match_resets_fields(conn, tmp_path):
    seed_release(conn, 1, "1200 Micrograms", "1200 Micrograms", [("A1", "Ayahuasca")])
    flac_dir = tmp_path / "flac"
    touch(flac_dir / "1200 Micrograms - 1200 Micrograms" / "1200 Micrograms - 01 Ayahuasca.flac")
    local_audio.match_library(conn, flac_dir)

    local_audio.clear_local_match(conn, 1)

    release = conn.execute("SELECT * FROM releases WHERE id = 1").fetchone()
    assert release["local_folder_path"] is None
    assert release["local_folder_match_score"] is None
    assert release["local_matched_at"] is None
    track = conn.execute("SELECT * FROM tracks WHERE release_id = 1").fetchone()
    assert track["local_audio_path"] is None
    assert track["local_match_score"] is None
