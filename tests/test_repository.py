from __future__ import annotations

import sqlite3

import pytest

from discql import crates, db
from discql.discogs_api import ArtistData, LabelData, ReleaseData, TrackData
from discql.sync import upsert_release
from discql.web import repository


@pytest.fixture()
def conn():
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.create_function("strip_accents", 1, db.strip_accents, deterministic=True)
    db.migrate(connection)
    yield connection
    connection.close()


def seed(conn, release_id, title, artist_name, genres, year=2020, styles=None, label_name="Some Label"):
    data = ReleaseData(
        id=release_id,
        title=title,
        year=year,
        genres=genres,
        styles=styles or [],
        formats=[{"name": "Vinyl", "qty": "1"}],
        discogs_uri=f"https://discogs.com/release/{release_id}",
        cover_image_url="https://example.com/cover.jpg",
        artists=[ArtistData(id=release_id * 10, name=artist_name)],
        labels=[LabelData(id=release_id * 100, name=label_name, catalog_number="CAT001")],
        tracks=[
            TrackData(position="A1", title="Track One", duration="3:45", artist=None),
            TrackData(position="A2", title="Track Two", duration="4:12", artist=None),
        ],
    )
    upsert_release(conn, data, date_added=f"2024-01-{release_id:02d}T00:00:00", now="2024-02-01T00:00:00")


def test_list_releases_returns_all_by_default(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])

    results = repository.list_releases(conn)

    assert {r.title for r in results} == {"Alpha", "Beta"}


def test_list_releases_reports_track_match_and_analysis_counts(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])

    result = repository.list_releases(conn)[0]
    assert result.track_count == 2
    assert result.matched_count == 0
    assert result.analyzed_count == 0
    assert result.fully_matched is False
    assert result.fully_analyzed is False

    track_ids = [row["id"] for row in conn.execute("SELECT id FROM tracks WHERE release_id = 1 ORDER BY id")]
    video_id = conn.execute(
        "INSERT INTO release_videos (release_id, url) VALUES (1, 'https://youtube.com/watch?v=abc') RETURNING id"
    ).fetchone()["id"]
    conn.execute(
        "INSERT INTO track_video_matches (track_id, video_id, fuzzy_score, is_selected, created_at) VALUES (?, ?, 0.9, 1, '2024-01-01')",
        (track_ids[0], video_id),
    )
    conn.execute("UPDATE tracks SET analyzed_at = '2024-01-01' WHERE id = ?", (track_ids[0],))
    conn.commit()

    partial = repository.list_releases(conn)[0]
    assert partial.matched_count == 1
    assert partial.analyzed_count == 1
    assert partial.fully_matched is False
    assert partial.fully_analyzed is False

    conn.execute(
        "INSERT INTO track_video_matches (track_id, video_id, fuzzy_score, is_selected, created_at) VALUES (?, ?, 0.9, 1, '2024-01-01')",
        (track_ids[1], video_id),
    )
    conn.execute("UPDATE tracks SET analyzed_at = '2024-01-01' WHERE id = ?", (track_ids[1],))
    conn.commit()

    full = repository.list_releases(conn)[0]
    assert full.fully_matched is True
    assert full.fully_analyzed is True


def test_list_releases_reports_matching_processed(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])

    fresh = repository.list_releases(conn)[0]
    assert fresh.matching_processed is False

    video_id = conn.execute(
        "INSERT INTO release_videos (release_id, url) VALUES (1, 'https://youtube.com/watch?v=abc') RETURNING id"
    ).fetchone()["id"]
    conn.commit()
    unprocessed_video = repository.list_releases(conn)[0]
    assert unprocessed_video.matching_processed is False  # fetched but not yet classified

    conn.execute("UPDATE release_videos SET video_type = 'unrelated' WHERE id = ?", (video_id,))
    conn.commit()
    processed = repository.list_releases(conn)[0]
    assert processed.matching_processed is True


def test_list_releases_filters_by_title(conn):
    seed(conn, 1, "Acid Test", "Artist A", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])

    results = repository.list_releases(conn, q="acid")

    assert [r.title for r in results] == ["Acid Test"]


def test_list_releases_q_matches_artist_name(conn):
    seed(conn, 1, "Alpha", "DJ Sparkle", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])

    results = repository.list_releases(conn, q="sparkle")

    assert [r.title for r in results] == ["Alpha"]


def test_list_releases_escapes_like_wildcards(conn):
    seed(conn, 1, "100% Vinyl", "Artist A", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])

    results = repository.list_releases(conn, q="100%")

    assert [r.title for r in results] == ["100% Vinyl"]


def test_list_releases_filters_by_genre(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic", "Techno"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])

    results = repository.list_releases(conn, genres=["Techno"])

    assert [r.title for r in results] == ["Alpha"]


def test_list_releases_filters_by_genre_matches_any_selected(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])
    seed(conn, 3, "Gamma", "Artist C", ["Jazz"])

    results = repository.list_releases(conn, genres=["Electronic", "Rock"])

    assert {r.title for r in results} == {"Alpha", "Beta"}


def test_list_releases_filters_by_style(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"], styles=["Acid", "Techno"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"], styles=["Punk"])

    results = repository.list_releases(conn, styles=["Acid"])

    assert [r.title for r in results] == ["Alpha"]


def test_list_releases_filters_by_year(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"], year=1999)
    seed(conn, 2, "Beta", "Artist B", ["Rock"], year=2020)

    results = repository.list_releases(conn, year=1999)

    assert [r.title for r in results] == ["Alpha"]


def test_list_releases_filters_by_label_name(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"], label_name="Acid Records")
    seed(conn, 2, "Beta", "Artist B", ["Rock"], label_name="Rock Records")

    results = repository.list_releases(conn, label_name="Acid Records")

    assert [r.title for r in results] == ["Alpha"]


def test_list_releases_filters_by_artist_name(conn):
    seed(conn, 1, "Alpha", "DJ Sparkle", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])

    results = repository.list_releases(conn, artist_name="DJ Sparkle")

    assert [r.title for r in results] == ["Alpha"]


def test_list_releases_filters_by_artist_name_substring(conn):
    seed(conn, 1, "Alpha", "Klex (3)", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])

    results = repository.list_releases(conn, artist_name="Klex")

    assert [r.title for r in results] == ["Alpha"]


def test_list_releases_filters_by_artist_name_ignoring_accents(conn):
    seed(conn, 1, "Alpha", "Kulor Sound", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])

    results = repository.list_releases(conn, artist_name="Kulør")

    assert [r.title for r in results] == ["Alpha"]


def test_list_releases_filters_by_label_name_ignoring_accents(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"], label_name="Kulør")
    seed(conn, 2, "Beta", "Artist B", ["Rock"], label_name="Other Label")

    results = repository.list_releases(conn, label_name="Kulor")

    assert [r.title for r in results] == ["Alpha"]


@pytest.mark.parametrize("query", ["Müller", "Mueller", "Muller"])
def test_list_releases_filters_by_artist_name_umlaut_spellings_are_equivalent(conn, query):
    seed(conn, 1, "Alpha", "Müller", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])

    results = repository.list_releases(conn, artist_name=query)

    assert [r.title for r in results] == ["Alpha"]


def test_list_releases_combines_filters(conn):
    seed(conn, 1, "Alpha", "DJ Sparkle", ["Electronic"], year=1999, label_name="Acid Records")
    seed(conn, 2, "Beta", "DJ Sparkle", ["Electronic"], year=2020, label_name="Acid Records")

    results = repository.list_releases(conn, artist_name="DJ Sparkle", year=1999)

    assert [r.title for r in results] == ["Alpha"]


def set_bpm(conn, release_id, position, bpm):
    conn.execute(
        "UPDATE tracks SET bpm = ? WHERE release_id = ? AND position = ?", (bpm, release_id, position)
    )
    conn.commit()


def test_list_releases_filters_by_bpm_exact_matches_rounded_value(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])
    set_bpm(conn, 1, "A1", 128.2)  # rounds to 128
    set_bpm(conn, 2, "A1", 130.0)

    results = repository.list_releases(conn, bpm=128, bpm_tolerance_pct=0)

    assert [r.title for r in results] == ["Alpha"]


def test_list_releases_filters_by_bpm_exact_excludes_outside_half_bpm_band(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    set_bpm(conn, 1, "A1", 129.0)

    results = repository.list_releases(conn, bpm=128, bpm_tolerance_pct=0)

    assert results == []


def test_list_releases_filters_by_bpm_tolerance_band(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])
    set_bpm(conn, 1, "A1", 138)  # 128 * 1.08 = 138.24 - just inside +8%
    set_bpm(conn, 2, "A1", 145)  # outside +8%, but within +16% (128 * 1.16 = 148.48)

    results_8pct = repository.list_releases(conn, bpm=128, bpm_tolerance_pct=0.08)
    results_16pct = repository.list_releases(conn, bpm=128, bpm_tolerance_pct=0.16)

    assert [r.title for r in results_8pct] == ["Alpha"]
    assert {r.title for r in results_16pct} == {"Alpha", "Beta"}


def test_list_releases_filters_by_bpm_matches_release_with_at_least_one_matching_track(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    set_bpm(conn, 1, "A1", 128)
    set_bpm(conn, 1, "A2", 90)  # unrelated second track - shouldn't disqualify the release

    results = repository.list_releases(conn, bpm=128, bpm_tolerance_pct=0)

    assert [r.title for r in results] == ["Alpha"]


def test_list_releases_bpm_filter_ignored_without_tolerance(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    set_bpm(conn, 1, "A1", 128)

    # bpm alone, no tolerance selected - filter isn't applied at all (mirrors
    # the web layer's "no checkbox checked" case, see releases() in app.py).
    results = repository.list_releases(conn, bpm=999)

    assert [r.title for r in results] == ["Alpha"]


def test_list_releases_filters_by_crate_id(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])
    crate_id = crates.create_crate(conn, "Warehouse Gig")
    crates.add_release(conn, crate_id, 1)

    results = repository.list_releases(conn, crate_id=crate_id)

    assert [r.title for r in results] == ["Alpha"]


def test_list_crates_with_counts_reports_item_count(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])
    crate_id = crates.create_crate(conn, "Warehouse Gig")

    empty = repository.list_crates_with_counts(conn)
    assert [c.item_count for c in empty] == [0]

    crates.add_release(conn, crate_id, 1)
    crates.add_release(conn, crate_id, 2)

    full = repository.list_crates_with_counts(conn)
    assert full[0].name == "Warehouse Gig"
    assert full[0].item_count == 2


def test_list_crates_with_counts_orders_newest_first(conn):
    first_id = crates.create_crate(conn, "First")
    second_id = crates.create_crate(conn, "Second")
    # created_at has second resolution in practice, but two crates created
    # back-to-back in a test can land on the exact same timestamp - force a
    # visible ordering rather than relying on wall-clock timing.
    conn.execute("UPDATE crates SET created_at = '2024-01-01T00:00:00' WHERE id = ?", (first_id,))
    conn.execute("UPDATE crates SET created_at = '2024-01-02T00:00:00' WHERE id = ?", (second_id,))
    conn.commit()

    results = repository.list_crates_with_counts(conn)

    assert [c.name for c in results] == ["Second", "First"]


def test_crate_ids_for_release(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    crate_a = crates.create_crate(conn, "A")
    crate_b = crates.create_crate(conn, "B")
    crates.add_release(conn, crate_a, 1)

    assert repository.crate_ids_for_release(conn, 1) == {crate_a}

    crates.add_release(conn, crate_b, 1)
    assert repository.crate_ids_for_release(conn, 1) == {crate_a, crate_b}


def test_crate_ids_for_release_empty_when_not_in_any_crate(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    assert repository.crate_ids_for_release(conn, 1) == set()


def test_list_releases_limit_none_returns_everything(conn):
    for i in range(1, 6):
        seed(conn, i, f"Release {i}", "Artist", ["Electronic"])

    results = repository.list_releases(conn, sort="title_asc", limit=None)

    assert [r.title for r in results] == [f"Release {i}" for i in range(1, 6)]


def test_sort_by_artist_and_label(conn):
    seed(conn, 1, "Alpha", "Zebra Artist", ["Electronic"], label_name="Zeta Records")
    seed(conn, 2, "Beta", "Alpha Artist", ["Electronic"], label_name="Alpha Records")

    by_artist = repository.list_releases(conn, sort="artist_asc")
    by_label = repository.list_releases(conn, sort="label_asc")

    assert [r.title for r in by_artist] == ["Beta", "Alpha"]
    assert [r.title for r in by_label] == ["Beta", "Alpha"]


def test_list_years_returns_distinct_sorted_desc(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"], year=1999)
    seed(conn, 2, "Beta", "Artist B", ["Rock"], year=2020)

    assert repository.list_years(conn) == [2020, 1999]


def test_list_styles_returns_distinct_sorted(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"], styles=["Techno", "Acid"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"], styles=["Acid"])

    assert repository.list_styles(conn) == ["Acid", "Techno"]


def test_list_labels_and_artists_return_distinct_sorted(conn):
    seed(conn, 1, "Alpha", "Zebra Artist", ["Electronic"], label_name="Zeta Records")
    seed(conn, 2, "Beta", "Alpha Artist", ["Rock"], label_name="Alpha Records")

    assert repository.list_labels(conn) == ["Alpha Records", "Zeta Records"]
    assert repository.list_artists(conn) == ["Alpha Artist", "Zebra Artist"]


def test_list_artists_strips_disambiguation_suffix_and_dedupes(conn):
    seed(conn, 1, "Alpha", "Klex (3)", ["Electronic"])
    seed(conn, 2, "Beta", "Klex (7)", ["Rock"])  # a different Discogs artist, same display name

    assert repository.list_artists(conn) == ["Klex"]


def test_list_releases_strips_artist_disambiguation_suffix(conn):
    seed(conn, 1, "Alpha", "Klex (3)", ["Electronic"])

    [summary] = repository.list_releases(conn)
    assert summary.artist_names == "Klex"


def test_list_releases_excludes_soft_deleted(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    conn.execute(
        "UPDATE releases SET removed_from_discogs_at = '2024-03-01T00:00:00' WHERE id = 1"
    )
    conn.commit()

    results = repository.list_releases(conn)

    assert results == []


def test_list_releases_pagination(conn):
    for i in range(1, 6):
        seed(conn, i, f"Release {i}", "Artist", ["Electronic"])

    page1 = repository.list_releases(conn, sort="title_asc", limit=2, offset=0)
    page2 = repository.list_releases(conn, sort="title_asc", limit=2, offset=2)

    assert [r.title for r in page1] == ["Release 1", "Release 2"]
    assert [r.title for r in page2] == ["Release 3", "Release 4"]


def test_count_releases_matches_filtered_results(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Rock"])

    assert repository.count_releases(conn) == 2
    assert repository.count_releases(conn, genres=["Rock"]) == 1


def test_list_genres_returns_distinct_sorted(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Techno", "Electronic"])
    seed(conn, 2, "Beta", "Artist B", ["Electronic"])

    assert repository.list_genres(conn) == ["Electronic", "Techno"]


def test_get_release_detail_surfaces_youtube_match_info(conn):
    from discql import youtube_matching as ym
    from discql.discogs_api import VideoData

    data = ReleaseData(
        id=1,
        title="Alpha",
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri="https://discogs.com/release/1",
        cover_image_url=None,
        artists=[ArtistData(id=10, name="Artist A")],
        labels=[],
        tracks=[TrackData(position="A1", title="Rave-O-Lution", duration="4:30", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=270, description=None)],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")

    def fake_extractor(url):
        return {"title": "Rave-O-Lution", "duration": 270, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=lambda p: (_ for _ in ()).throw(AssertionError))

    detail = repository.get_release_detail(conn, 1)

    assert detail.videos[0].video_type == "full_track"
    assert detail.videos[0].matched_track_position == "A1"
    assert detail.videos[0].fuzzy_score > 0.9
    assert detail.videos[0].is_selected is True
    assert detail.tracks[0].matched_video_url == "https://youtube.com/watch?v=abc"


def test_get_release_detail_marks_non_primary_full_track_candidate_as_not_selected(conn):
    # Real-world case: two videos both legitimately fuzzy-match the same
    # track (e.g. two uploads of the same song); only one is "the" primary
    # pick, but the other is still a valid full_track match, not a rejected
    # low-score guess - the repository/template layer must be able to tell
    # them apart (this used to surface as "below match threshold" on a video
    # whose score was actually well above the confidence threshold).
    from discql import youtube_matching as ym
    from discql.discogs_api import VideoData

    data = ReleaseData(
        id=1,
        title="Alpha",
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri="https://discogs.com/release/1",
        cover_image_url=None,
        artists=[ArtistData(id=10, name="Artist A")],
        labels=[],
        tracks=[TrackData(position="A1", title="Rave-O-Lution", duration="4:30", artist=None)],
        videos=[
            VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=270, description=None),
            VideoData(title="ignored", url="https://youtube.com/watch?v=xyz", duration=268, description=None),
        ],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")

    def fake_extractor(url):
        duration = 270 if "abc" in url else 268
        return {"title": "Rave-O-Lution", "duration": duration, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=lambda p: (_ for _ in ()).throw(AssertionError))

    detail = repository.get_release_detail(conn, 1)

    assert len(detail.videos) == 2
    assert all(v.video_type == "full_track" for v in detail.videos)
    assert all(v.matched_track_position == "A1" for v in detail.videos)
    selected = [v for v in detail.videos if v.is_selected]
    not_selected = [v for v in detail.videos if not v.is_selected]
    assert len(selected) == 1
    assert len(not_selected) == 1
    # The non-primary one still has its own valid track match reported, not
    # treated as if it were a rejected/unmatched guess.
    assert not_selected[0].matched_track_position == "A1"
    assert not_selected[0].fuzzy_score is not None


def test_get_release_detail_returns_full_data(conn):
    seed(conn, 1, "Alpha", "Artist A", ["Electronic"], styles=["Acid"])

    detail = repository.get_release_detail(conn, 1)

    assert detail is not None
    assert detail.title == "Alpha"
    assert detail.artist_names == ["Artist A"]
    assert detail.labels[0].name == "Some Label"
    assert detail.labels[0].catalog_number == "CAT001"
    assert detail.genres == ["Electronic"]
    assert detail.styles == ["Acid"]
    assert [t.title for t in detail.tracks] == ["Track One", "Track Two"]


def test_get_release_detail_flags_distinct_track_artists(conn):
    data = ReleaseData(
        id=1,
        title="Various Artists Comp",
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri="https://discogs.com/release/1",
        cover_image_url=None,
        artists=[ArtistData(id=1, name="Various")],
        labels=[],
        tracks=[
            TrackData(position="A1", title="Track One", duration="3:00", artist="Artist A"),
            TrackData(position="A2", title="Track Two", duration="3:30", artist="Artist B"),
        ],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")

    detail = repository.get_release_detail(conn, 1)

    assert detail.tracks_have_distinct_artists is True
    assert detail.tracks[0].artist == "Artist A"
    assert detail.tracks[1].artist == "Artist B"


def test_get_release_detail_does_not_flag_when_track_artists_match_release(conn):
    data = ReleaseData(
        id=1,
        title="Single Artist Album",
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri="https://discogs.com/release/1",
        cover_image_url=None,
        artists=[ArtistData(id=1, name="Solo Artist")],
        labels=[],
        tracks=[
            TrackData(position="A1", title="Track One", duration="3:00", artist=None),
            TrackData(position="A2", title="Track Two", duration="3:30", artist="Solo Artist"),
        ],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")

    detail = repository.get_release_detail(conn, 1)

    assert detail.tracks_have_distinct_artists is False


def test_get_release_detail_strips_artist_disambiguation_suffix(conn):
    data = ReleaseData(
        id=1,
        title="Various Artists Comp",
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri="https://discogs.com/release/1",
        cover_image_url=None,
        artists=[ArtistData(id=1, name="Klex (3)")],
        labels=[],
        tracks=[
            TrackData(position="A1", title="Track One", duration="3:00", artist="Klex (3)"),
            TrackData(position="A2", title="Track Two", duration="3:30", artist="Other Artist (12)"),
        ],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")

    detail = repository.get_release_detail(conn, 1)

    assert detail.artist_names == ["Klex"]
    assert detail.tracks[0].artist == "Klex"
    assert detail.tracks[1].artist == "Other Artist"
    # The first track's (stripped) artist matches the (stripped) release
    # artist, so only the second track should trigger the distinct-artists flag.
    assert detail.tracks_have_distinct_artists is True


def test_get_release_detail_does_not_flag_distinct_artists_once_disambiguation_is_stripped(conn):
    data = ReleaseData(
        id=1,
        title="Single Artist Album",
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri="https://discogs.com/release/1",
        cover_image_url=None,
        artists=[ArtistData(id=1, name="Klex (3)")],
        labels=[],
        tracks=[
            TrackData(position="A1", title="Track One", duration="3:00", artist="Klex (3)"),
        ],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")

    detail = repository.get_release_detail(conn, 1)

    assert detail.tracks_have_distinct_artists is False


def test_get_release_detail_returns_none_for_missing(conn):
    assert repository.get_release_detail(conn, 999) is None


def test_get_release_detail_includes_country_images_videos_credits(conn):
    from discql.discogs_api import CreditData, VideoData

    data = ReleaseData(
        id=1,
        title="Alpha",
        year=2020,
        genres=["Electronic"],
        styles=[],
        formats=[],
        discogs_uri="https://discogs.com/release/1",
        cover_image_url="https://example.com/cover.jpg",
        country="Sweden",
        images=[
            {"type": "primary", "uri": "https://example.com/cover.jpg", "uri150": None, "width": 600, "height": 600},
            {"type": "secondary", "uri": "https://example.com/back.jpg", "uri150": None, "width": 600, "height": 600},
        ],
        artists=[ArtistData(id=10, name="Artist A")],
        labels=[],
        tracks=[],
        videos=[
            VideoData(title="A video", url="https://youtube.com/watch?v=abc", duration=225, description=None),
        ],
        credits=[
            CreditData(artist_id=20, artist_name="Mix Engineer", role="Mixed By", tracks="", name_variation=""),
        ],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")

    detail = repository.get_release_detail(conn, 1)

    assert detail.country == "Sweden"
    assert len(detail.images) == 2
    assert detail.images[0]["type"] == "primary"
    assert [v.title for v in detail.videos] == ["A video"]
    assert detail.videos[0].url == "https://youtube.com/watch?v=abc"
    assert [c.artist_name for c in detail.credits] == ["Mix Engineer"]
    assert detail.credits[0].role == "Mixed By"
    assert detail.credits[0].artist_id == 20
