from __future__ import annotations

import contextlib
import threading
import time

import pytest
from fastapi.testclient import TestClient

from discql import db
from discql.discogs_api import ArtistData, CreditData, LabelData, ReleaseData, TrackData, VideoData
from discql.sync import upsert_release
from discql.web import tasks
from discql.web.app import _availability_status, _matching_status, app, get_config


@pytest.fixture(autouse=True)
def reset_sync_task():
    tasks._tasks.clear()
    yield
    tasks._tasks.clear()


@pytest.fixture(autouse=True)
def no_real_ollama_process(monkeypatch):
    """Endpoints that trigger YouTube matching wrap it in
    ollama_client.ensure_ollama_running(), which - on a machine with the
    `ollama` CLI installed - would otherwise actually spawn a real `ollama
    serve` subprocess on every such test. Replaced with a no-op context
    manager here so tests stay hermetic regardless of the host's local
    Ollama installation.
    """

    @contextlib.contextmanager
    def fake_ensure_ollama_running(*args, **kwargs):
        yield

    monkeypatch.setattr("discql.web.app.ollama_client.ensure_ollama_running", fake_ensure_ollama_running)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "dummy")
    monkeypatch.setenv("DISCOGS_DB_PATH", str(db_path))
    # Without these, get_config() falls back to the real ~/.cache/discql
    # dirs - confirmed concretely this leaks test-written files into the
    # user's actual cache (e.g. a stray waveforms/secret.txt from
    # test_get_waveform_rejects_path_traversal, or a numeric "1.jpg" from a
    # cover-art test colliding with a later test expecting a fresh fetch).
    monkeypatch.setenv("WAVEFORM_DIR", str(tmp_path / "waveforms"))
    monkeypatch.setenv("ESSENTIA_MODELS_DIR", str(tmp_path / "essentia_models"))
    monkeypatch.setenv("COVER_DIR", str(tmp_path / "covers"))
    # Likewise for CONFIG_DIR - without it, load_config() reads/writes the
    # real ~/.config/discql/config (e.g. MAX_WORKERS, FUZZY_* defaults),
    # coupling tests to whatever's actually in it on this machine.
    from discql import config as config_module

    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path / "config")
    get_config.cache_clear()

    conn = db.connect(db_path)
    db.migrate(conn)

    data = ReleaseData(
        id=1,
        title="Acid Test",
        year=2020,
        genres=["Electronic"],
        styles=["Acid"],
        formats=[],
        discogs_uri="https://discogs.com/release/1",
        cover_image_url="https://example.com/cover.jpg",
        country="Sweden",
        artists=[ArtistData(id=10, name="DJ Sparkle")],
        labels=[LabelData(id=100, name="Some Label", catalog_number="CAT001")],
        tracks=[
            TrackData(position="A1", title="Track One", duration="3:45", artist=None),
        ],
        videos=[
            VideoData(title="Acid Test (video)", url="https://youtube.com/watch?v=abc", duration=225, description=None),
        ],
        credits=[
            CreditData(artist_id=30, artist_name="Mix Engineer", role="Mixed By", tracks="", name_variation=""),
        ],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")

    data2 = ReleaseData(
        id=2,
        title="Rock Anthology",
        year=1999,
        genres=["Rock"],
        styles=[],
        formats=[],
        discogs_uri="https://discogs.com/release/2",
        cover_image_url=None,
        artists=[ArtistData(id=20, name="The Rockers")],
        labels=[],
        tracks=[],
    )
    upsert_release(conn, data2, date_added="2024-01-02T00:00:00", now="2024-02-01T00:00:00")
    conn.close()

    with TestClient(app) as test_client:
        yield test_client

    get_config.cache_clear()


@pytest.fixture()
def client_without_auth(tmp_path, monkeypatch):
    """Same isolation as `client`, but deliberately no Discogs token/OAuth
    credentials configured at all - for testing the setup-redirect
    middleware and the /settings page itself."""
    db_path = tmp_path / "test.db"
    for key in (
        "DISCOGS_USER_TOKEN",
        "DISCOGS_CONSUMER_KEY",
        "DISCOGS_CONSUMER_SECRET",
        "DISCOGS_OAUTH_TOKEN",
        "DISCOGS_OAUTH_TOKEN_SECRET",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("DISCOGS_DB_PATH", str(db_path))
    monkeypatch.setenv("WAVEFORM_DIR", str(tmp_path / "waveforms"))
    monkeypatch.setenv("ESSENTIA_MODELS_DIR", str(tmp_path / "essentia_models"))
    monkeypatch.setenv("COVER_DIR", str(tmp_path / "covers"))

    from discql import config as config_module

    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path / "config")
    get_config.cache_clear()

    conn = db.connect(db_path)
    db.migrate(conn)
    conn.close()

    with TestClient(app) as test_client:
        yield test_client

    get_config.cache_clear()


def test_index_redirects_to_releases(client):
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/releases"


def test_releases_page_lists_seeded_releases(client):
    response = client.get("/releases")
    assert response.status_code == 200
    assert "Acid Test" in response.text
    assert "Rock Anthology" in response.text


def test_releases_search_filters_by_title(client):
    response = client.get("/releases", params={"q": "acid"})
    assert response.status_code == 200
    assert "Acid Test" in response.text
    assert "Rock Anthology" not in response.text


def test_releases_genre_filter(client):
    response = client.get("/releases", params={"genre": "Rock"})
    assert response.status_code == 200
    assert "Rock Anthology" in response.text
    assert "Acid Test" not in response.text


def test_releases_list_view_renders_without_error(client):
    response = client.get("/releases", params={"view": "list"})
    assert response.status_code == 200
    assert "Acid Test" in response.text


def test_releases_list_view_shows_status_indicators(client):
    response = client.get("/releases", params={"view": "list"})
    assert response.status_code == 200
    assert "status-indicators" in response.text
    assert "status-dot" in response.text


def test_releases_grid_view_omits_status_indicators(client):
    response = client.get("/releases", params={"view": "grid"})
    assert response.status_code == 200
    assert "status-indicators" not in response.text


@pytest.mark.parametrize(
    "count, available, total, expected",
    [
        (0, 0, 3, ""),  # nothing matched yet - nothing to show
        (0, 2, 3, ""),  # 2 tracks matched, neither downloaded/analyzed yet
        (1, 2, 3, ""),  # 1 of 2 matched tracks done - still incomplete
        (2, 2, 3, "partial"),  # every matched track done, but 1 track has no match at all
        (3, 3, 3, "done"),  # every track in the release done
    ],
)
def test_availability_status(count, available, total, expected):
    assert _availability_status(count, available, total) == expected


def test_releases_list_view_shows_partial_status_for_releases_with_unmatched_tracks(client, monkeypatch):
    monkeypatch.setattr(
        "discql.web.app.audio_download.count_downloaded_tracks",
        lambda audio_dir, release_id: 1,
    )
    # "Acid Test" (seeded in the client fixture) has 1 track and 1 video, but
    # no match has actually been selected yet - add one, plus a second,
    # never-matched track, so matched_count (1) < track_count (2) while the
    # one matched track is fully analyzed - the "partial" (not "done") case.
    conn = db.connect(get_config().db_path)
    track_id = conn.execute("SELECT id FROM tracks WHERE release_id = 1 AND position = 'A1'").fetchone()[0]
    video_id = conn.execute("SELECT id FROM release_videos WHERE release_id = 1").fetchone()[0]
    conn.execute(
        "INSERT INTO track_video_matches (track_id, video_id, fuzzy_score, is_selected, created_at) VALUES (?, ?, 0.9, 1, '2024-01-01')",
        (track_id, video_id),
    )
    conn.execute("UPDATE tracks SET analyzed_at = '2024-01-01' WHERE id = ?", (track_id,))
    conn.execute("INSERT INTO tracks (release_id, position, title) VALUES (1, 'A2', 'Unmatched Track')")
    conn.commit()
    conn.close()

    response = client.get("/releases", params={"view": "list"})
    assert response.status_code == 200
    assert 'class="status-dot partial"' in response.text


@pytest.mark.parametrize(
    "matched_count, track_count, processed, expected",
    [
        (0, 3, False, ""),  # never run yet
        (0, 3, True, "partial"),  # run, but no candidate video matched anything
        (2, 3, True, "partial"),  # run, some tracks still unmatched
        (3, 3, True, "done"),  # run, every track matched
        (0, 0, True, ""),  # no tracks at all - nothing to report either way
    ],
)
def test_matching_status(matched_count, track_count, processed, expected):
    assert _matching_status(matched_count, track_count, processed) == expected


def test_releases_list_view_shows_partial_matching_status_when_processed_but_incomplete(client):
    # "Acid Test" (seeded in the client fixture) has 1 track and 1 video,
    # never matched. Mark the video as classified (as match_release would)
    # without ever selecting a track match, plus add a second track so
    # matched_count (0) < track_count (2) - matching ran, but nothing
    # matched: "partial", not the same "" a never-processed release shows.
    conn = db.connect(get_config().db_path)
    conn.execute("UPDATE release_videos SET video_type = 'unrelated' WHERE release_id = 1")
    conn.execute("INSERT INTO tracks (release_id, position, title) VALUES (1, 'A2', 'Second Track')")
    conn.commit()
    conn.close()

    response = client.get("/releases", params={"view": "list"})
    assert response.status_code == 200
    assert 'title="YouTube matching: processed, not every track matched (0/2 tracks)"' in response.text


def test_releases_list_view_shows_blank_matching_status_when_never_processed(client):
    response = client.get("/releases", params={"view": "list"})
    assert response.status_code == 200
    assert 'title="YouTube matching: not processed yet (0/1 tracks)"' in response.text


def test_releases_style_filter(client):
    response = client.get("/releases", params={"style": "Acid"})
    assert response.status_code == 200
    assert "Acid Test" in response.text
    assert "Rock Anthology" not in response.text


def test_releases_full_form_submission_with_empty_year_does_not_422(client):
    # Replicates exactly what the real filter form sends: every field present,
    # most left empty (the browser's default "unselected" state for the <select>s).
    response = client.get(
        "/releases",
        params={
            "view": "grid",
            "q": "acid",
            "genre": "",
            "style": "",
            "year": "",
            "label": "",
            "artist": "",
            "sort": "date_added_desc",
        },
    )
    assert response.status_code == 200
    assert "Acid Test" in response.text
    assert "Rock Anthology" not in response.text


def test_releases_year_filter_accepts_string_query_param(client):
    response = client.get("/releases", params={"year": "1999"})
    assert response.status_code == 200
    assert "Rock Anthology" in response.text
    assert "Acid Test" not in response.text


def test_releases_malformed_year_does_not_500(client):
    response = client.get("/releases", params={"year": "not-a-number"})
    assert response.status_code == 200


def test_releases_full_render_shows_result_count_once(client):
    response = client.get("/releases")
    assert response.text.count("result-count") == 1


def test_releases_continuation_request_omits_wrapper_and_count(client):
    response = client.get(
        "/releases", params={"show_all": "true", "continuation": "true"}, headers={"HX-Request": "true"}
    )
    assert response.status_code == 200
    assert "result-count" not in response.text
    assert 'id="results"' not in response.text
    assert "Acid Test" in response.text
    assert "Rock Anthology" in response.text


def test_releases_year_filter(client):
    response = client.get("/releases", params={"year": 1999})
    assert response.status_code == 200
    assert "Rock Anthology" in response.text
    assert "Acid Test" not in response.text


def test_releases_label_filter(client):
    response = client.get("/releases", params={"label": "Some Label"})
    assert response.status_code == 200
    assert "Acid Test" in response.text
    assert "Rock Anthology" not in response.text


def test_releases_artist_filter(client):
    response = client.get("/releases", params={"artist": "The Rockers"})
    assert response.status_code == 200
    assert "Rock Anthology" in response.text
    assert "Acid Test" not in response.text


def test_releases_show_all_returns_everything_without_load_more(client):
    response = client.get("/releases", params={"show_all": "true"})
    assert response.status_code == 200
    assert "Acid Test" in response.text
    assert "Rock Anthology" in response.text
    assert "load-more" not in response.text


def test_releases_sort_by_artist(client):
    response = client.get("/releases", params={"sort": "artist_asc"})
    assert response.status_code == 200
    # "DJ Sparkle" < "The Rockers" alphabetically
    assert response.text.index("Acid Test") < response.text.index("Rock Anthology")


def test_releases_htmx_request_returns_partial_only(client):
    response = client.get("/releases", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert "<html" not in response.text.lower()
    assert "Acid Test" in response.text


def test_release_detail_shows_tracklist(client):
    response = client.get("/releases/1")
    assert response.status_code == 200
    assert "Acid Test" in response.text
    assert "DJ Sparkle" in response.text
    assert "Track One" in response.text


def test_release_detail_shows_country_videos_and_credits(client):
    response = client.get("/releases/1")
    assert response.status_code == 200
    assert "Sweden" in response.text
    assert "Acid Test (video)" in response.text
    assert "https://youtube.com/watch?v=abc" in response.text
    assert "Mixed By" in response.text
    assert "Mix Engineer" in response.text
    assert 'href="https://www.discogs.com/artist/30"' in response.text


def test_release_detail_links_title_artist_and_label_to_discogs(client):
    response = client.get("/releases/1")
    assert response.status_code == 200
    assert 'href="https://discogs.com/release/1"' in response.text
    assert 'href="https://www.discogs.com/artist/10"' in response.text
    assert 'href="https://www.discogs.com/label/100"' in response.text
    assert "View on Discogs" not in response.text


def test_release_detail_404_for_missing_release(client):
    response = client.get("/releases/999")
    assert response.status_code == 404


def test_match_youtube_for_release_success(client, monkeypatch):
    captured = {}

    def fake_match_release(conn, release_id, release_title, force=False, **kwargs):
        captured["release_id"] = release_id
        captured["force"] = force
        captured["confident_threshold"] = kwargs.get("confident_threshold")
        captured["tie_margin"] = kwargs.get("tie_margin")
        captured["denoise_terms"] = kwargs.get("denoise_terms")

    monkeypatch.setattr("discql.web.app.youtube_matching.match_release", fake_match_release)

    response = client.post("/releases/1/match_youtube")

    assert response.status_code == 200
    assert captured["release_id"] == 1
    # Default (and only, from the UI) behavior: only fetch YouTube metadata
    # for videos that don't have it yet, never force a re-fetch.
    assert captured["force"] is False
    assert captured["confident_threshold"] == get_config().fuzzy_confident_threshold
    assert captured["tie_margin"] == get_config().fuzzy_tie_margin
    assert captured["denoise_terms"] == get_config().fuzzy_denoise_terms
    assert "Acid Test" in response.text
    assert "<html" not in response.text.lower()


def test_match_youtube_for_release_force_true_forces_refetch(client, monkeypatch):
    captured = {}

    def fake_match_release(conn, release_id, release_title, force=False, **kwargs):
        captured["force"] = force

    monkeypatch.setattr("discql.web.app.youtube_matching.match_release", fake_match_release)

    response = client.post("/releases/1/match_youtube", params={"force": "true"})

    assert response.status_code == 200
    assert captured["force"] is True


def test_release_detail_page_shows_match_button(client):
    response = client.get("/releases/1")
    assert "match_youtube?force=false" in response.text
    assert "Match YouTube videos for this release" in response.text
    assert "Recompute matches (no re-fetch)" not in response.text


def test_match_youtube_for_release_shows_inline_error_on_failure(client, monkeypatch):
    def failing_match_release(conn, release_id, release_title, force=False, **kwargs):
        raise RuntimeError("yt-dlp exploded")

    monkeypatch.setattr("discql.web.app.youtube_matching.match_release", failing_match_release)

    response = client.post("/releases/1/match_youtube")

    assert response.status_code == 200
    assert "Matching failed" in response.text
    assert "yt-dlp exploded" in response.text


def test_match_youtube_for_release_404_for_missing_release(client):
    response = client.post("/releases/999/match_youtube")
    assert response.status_code == 404


def test_clear_youtube_matches_for_release(client, monkeypatch):
    captured = {}

    def fake_clear_matches(conn, release_id):
        captured["release_id"] = release_id

    monkeypatch.setattr("discql.web.app.youtube_matching.clear_matches", fake_clear_matches)

    response = client.post("/releases/1/clear_youtube_matches")

    assert response.status_code == 200
    assert captured["release_id"] == 1
    assert "Acid Test" in response.text
    assert "<html" not in response.text.lower()


def test_clear_youtube_matches_also_clears_debug_fuzzy_matrix(client, monkeypatch):
    # debug_score_matrix() is always freshly recomputed from yt_title, so it
    # would still show scores for a release just wiped back to "never
    # matched" unless the route explicitly stops passing it through.
    from discql import youtube_matching as ym

    conn = db.connect(get_config().db_path)

    def fake_extractor(url):
        return {"title": "Track One", "duration": 225, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=lambda p: (_ for _ in ()).throw(AssertionError))
    conn.close()

    before = client.get("/releases/1")
    assert "fuzzy-matrix" in before.text

    response = client.post("/releases/1/clear_youtube_matches")

    assert response.status_code == 200
    assert "fuzzy-matrix" not in response.text
    assert "Debug: fuzzy score matrix" not in response.text


def test_clear_youtube_matches_404_for_missing_release(client):
    response = client.post("/releases/999/clear_youtube_matches")
    assert response.status_code == 404


def _seed_a_match(client):
    from discql import youtube_matching as ym

    conn = db.connect(get_config().db_path)

    def fake_extractor(url):
        return {"title": "Track One", "duration": 225, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=lambda p: (_ for _ in ()).throw(AssertionError))
    conn.close()


def test_release_detail_page_hides_download_button_when_nothing_matched(client):
    response = client.get("/releases/1")
    assert "download_tracks" not in response.text


def test_release_detail_page_shows_download_button_once_a_track_is_matched(client):
    _seed_a_match(client)

    response = client.get("/releases/1")

    assert "download_tracks" in response.text
    assert "Download tracks for this release" in response.text


def test_download_tracks_for_release_success(client, monkeypatch):
    _seed_a_match(client)
    captured = {}

    def fake_download_release_tracks(conn, release_id, audio_dir, **kwargs):
        captured["release_id"] = release_id
        captured["audio_dir"] = audio_dir
        return [{"position": "A1", "title": "Track One", "path": audio_dir / "1" / "A1.m4a", "status": "downloaded"}]

    monkeypatch.setattr(
        "discql.web.app.audio_download.download_release_tracks", fake_download_release_tracks
    )

    response = client.post("/releases/1/download_tracks")

    assert response.status_code == 200
    assert captured["release_id"] == 1
    assert captured["audio_dir"] == get_config().audio_dir
    assert "downloaded" in response.text
    assert "<html" not in response.text.lower()


def test_download_tracks_for_release_shows_inline_error_on_failure(client, monkeypatch):
    _seed_a_match(client)

    def failing_download(conn, release_id, audio_dir, **kwargs):
        raise RuntimeError("yt-dlp exploded")

    monkeypatch.setattr("discql.web.app.audio_download.download_release_tracks", failing_download)

    response = client.post("/releases/1/download_tracks")

    assert response.status_code == 200
    assert "Download failed" in response.text
    assert "yt-dlp exploded" in response.text


def test_download_tracks_for_release_404_for_missing_release(client):
    response = client.post("/releases/999/download_tracks")
    assert response.status_code == 404


def test_analyze_tracks_for_release_success(client, monkeypatch):
    _seed_a_match(client)
    captured = {}

    def fake_analyze_release_tracks(conn, release_id, audio_dir, waveform_dir, models_dir, **kwargs):
        captured["release_id"] = release_id
        captured["audio_dir"] = audio_dir
        captured["waveform_dir"] = waveform_dir
        captured["models_dir"] = models_dir
        return [
            {
                "position": "A1",
                "title": "Track One",
                "status": "analyzed",
                "bpm": 128.0,
                "musical_key": "C",
                "mood_summary": "Happy, Danceable",
            }
        ]

    monkeypatch.setattr(
        "discql.web.app.audio_analysis.analyze_release_tracks", fake_analyze_release_tracks
    )

    response = client.post("/releases/1/analyze_tracks")

    assert response.status_code == 200
    assert captured["release_id"] == 1
    assert captured["audio_dir"] == get_config().audio_dir
    assert captured["waveform_dir"] == get_config().waveform_dir
    assert captured["models_dir"] == get_config().essentia_models_dir
    assert "analyzed" in response.text
    assert "Happy, Danceable" in response.text
    assert "<html" not in response.text.lower()


def test_analyze_tracks_for_release_shows_inline_error_on_failure(client, monkeypatch):
    _seed_a_match(client)

    def failing_analyze(conn, release_id, audio_dir, waveform_dir, models_dir, **kwargs):
        raise RuntimeError("essentia exploded")

    monkeypatch.setattr("discql.web.app.audio_analysis.analyze_release_tracks", failing_analyze)

    response = client.post("/releases/1/analyze_tracks")

    assert response.status_code == 200
    assert "Analysis failed" in response.text
    assert "essentia exploded" in response.text


def test_analyze_tracks_for_release_404_for_missing_release(client):
    response = client.post("/releases/999/analyze_tracks")
    assert response.status_code == 404


def test_clear_analysis_for_release_success(client, monkeypatch):
    _seed_a_match(client)
    captured = {}

    def fake_clear_analysis(conn, release_id, waveform_dir):
        captured["release_id"] = release_id
        captured["waveform_dir"] = waveform_dir

    monkeypatch.setattr("discql.web.app.audio_analysis.clear_analysis", fake_clear_analysis)

    response = client.post("/releases/1/clear_analysis")

    assert response.status_code == 200
    assert captured["release_id"] == 1
    assert captured["waveform_dir"] == get_config().waveform_dir
    assert "<html" not in response.text.lower()


def test_clear_analysis_for_release_404_for_missing_release(client):
    response = client.post("/releases/999/clear_analysis")
    assert response.status_code == 404


def test_get_waveform_returns_file(client, tmp_path):
    waveform_dir = get_config().waveform_dir
    release_dir = waveform_dir / "1"
    release_dir.mkdir(parents=True, exist_ok=True)
    (release_dir / "A1.png").write_bytes(b"fake png bytes")

    response = client.get("/waveforms/1/A1.png")

    assert response.status_code == 200
    assert response.content == b"fake png bytes"


def test_get_waveform_404_when_missing(client):
    response = client.get("/waveforms/1/does-not-exist.png")
    assert response.status_code == 404


def test_get_waveform_rejects_path_traversal(client):
    waveform_dir = get_config().waveform_dir
    (waveform_dir).mkdir(parents=True, exist_ok=True)
    secret = waveform_dir / "secret.txt"
    secret.write_text("top secret")

    response = client.get("/waveforms/1/..%2Fsecret.txt")

    assert response.status_code in (404, 400)


def test_get_cover_serves_cached_file_without_fetching(client, monkeypatch):
    cover_dir = get_config().cover_dir
    cover_dir.mkdir(parents=True, exist_ok=True)
    (cover_dir / "1.jpg").write_bytes(b"cached cover bytes")

    def unreachable_get(url, timeout):
        raise AssertionError("should not fetch - already cached")

    monkeypatch.setattr("discql.cover_art.requests.get", unreachable_get)

    response = client.get("/covers/1")

    assert response.status_code == 200
    assert response.content == b"cached cover bytes"


def test_get_cover_fetches_and_caches_when_not_cached(client, monkeypatch):
    class _FakeResponse:
        content = b"fetched cover bytes"

        def raise_for_status(self):
            pass

    captured = {}

    def fake_get(url, timeout, headers):
        captured["url"] = url
        return _FakeResponse()

    monkeypatch.setattr("discql.cover_art.requests.get", fake_get)

    response = client.get("/covers/1")

    assert response.status_code == 200
    assert response.content == b"fetched cover bytes"
    assert captured["url"] == "https://example.com/cover.jpg"
    assert (get_config().cover_dir / "1.jpg").read_bytes() == b"fetched cover bytes"


def test_get_cover_404_for_missing_release(client):
    response = client.get("/covers/999")
    assert response.status_code == 404


def test_get_cover_404_when_release_has_no_cover_url(client):
    response = client.get("/covers/2")  # seeded with cover_image_url=None
    assert response.status_code == 404


def test_release_detail_page_shows_clear_matches_button(client):
    response = client.get("/releases/1")
    assert "clear_youtube_matches" in response.text
    assert "Clear fuzzy matching" in response.text


def test_sync_status_is_idle_by_default(client):
    response = client.get("/sync/status")
    assert response.status_code == 200
    assert "Discogs Sync" in response.text
    assert "sync-widget idle" in response.text


def test_sync_start_runs_in_background_and_reports_done(client, monkeypatch):
    calls = []

    class FakeApi:
        def __init__(self, config):
            pass

    def fake_sync_collection(conn, api, on_progress=None, on_error=None, **kwargs):
        calls.append(1)
        if on_progress:
            on_progress(1, 1, 42, "Test Release")

    monkeypatch.setattr("discql.web.app.build_discogs_api", FakeApi)
    monkeypatch.setattr("discql.web.app.sync.sync_collection", fake_sync_collection)

    response = client.post("/sync/start")
    assert response.status_code == 200

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if tasks.get_task("sync").status == tasks.TaskStatus.DONE:
            break
        time.sleep(0.01)
    else:
        pytest.fail("sync task did not complete in time")

    assert calls == [1]
    state = tasks.get_task("sync")
    assert state.current == 1
    assert state.message == "Test Release"

    status_response = client.get("/sync/status")
    assert "Synced 1 release" in status_response.text


def test_sync_start_passes_force_refetch_through(client, monkeypatch):
    captured = {}

    class FakeApi:
        def __init__(self, config):
            pass

    def fake_sync_collection(conn, api, on_progress=None, on_error=None, **kwargs):
        captured["force_refetch"] = kwargs.get("force_refetch")

    monkeypatch.setattr("discql.web.app.build_discogs_api", FakeApi)
    monkeypatch.setattr("discql.web.app.sync.sync_collection", fake_sync_collection)

    response = client.post("/sync/start", data={"force": "true"})
    assert response.status_code == 200

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if tasks.get_task("sync").status == tasks.TaskStatus.DONE:
            break
        time.sleep(0.01)
    else:
        pytest.fail("sync task did not complete in time")

    assert captured["force_refetch"] is True


def test_sync_start_records_error_on_failure(client, monkeypatch):
    class FakeApi:
        def __init__(self, config):
            pass

    def failing_sync_collection(conn, api, on_progress=None, on_error=None, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("discql.web.app.build_discogs_api", FakeApi)
    monkeypatch.setattr("discql.web.app.sync.sync_collection", failing_sync_collection)

    client.post("/sync/start")

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if tasks.get_task("sync").status == tasks.TaskStatus.ERROR:
            break
        time.sleep(0.01)
    else:
        pytest.fail("sync task did not fail in time")

    response = client.get("/sync/status")
    assert "Sync failed" in response.text
    assert "network down" in response.text


def test_youtube_status_is_idle_by_default(client):
    response = client.get("/youtube/status")
    assert response.status_code == 200
    assert "Match YT" in response.text
    assert "sync-widget idle" in response.text


def test_youtube_start_runs_in_background_and_passes_force_and_configured_workers_through(client, monkeypatch):
    captured = {}

    def fake_match_and_download_library(conn, db_path, audio_dir, on_progress=None, on_error=None, **kwargs):
        captured["force"] = kwargs.get("force")
        captured["max_workers"] = kwargs.get("max_workers")
        if on_progress:
            on_progress(1, 1, 1, "match")

    monkeypatch.setattr(
        "discql.web.app.pipeline.match_and_download_library", fake_match_and_download_library
    )

    response = client.post("/youtube/start", data={"force": "true"})
    assert response.status_code == 200

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if tasks.get_task("youtube").status == tasks.TaskStatus.DONE:
            break
        time.sleep(0.01)
    else:
        pytest.fail("youtube task did not complete in time")

    assert captured["force"] is True
    assert captured["max_workers"] == get_config().max_workers  # not user-configurable in the UI
    status_response = client.get("/youtube/status")
    assert "Matched 1 release" in status_response.text


def test_youtube_start_records_error_on_failure(client, monkeypatch):
    def failing_match_and_download_library(conn, db_path, audio_dir, on_progress=None, on_error=None, **kwargs):
        raise RuntimeError("ollama unreachable")

    monkeypatch.setattr(
        "discql.web.app.pipeline.match_and_download_library", failing_match_and_download_library
    )

    client.post("/youtube/start")

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if tasks.get_task("youtube").status == tasks.TaskStatus.ERROR:
            break
        time.sleep(0.01)
    else:
        pytest.fail("youtube task did not fail in time")

    response = client.get("/youtube/status")
    assert "YouTube matching failed" in response.text


def test_audio_status_is_idle_by_default(client):
    response = client.get("/audio/status")
    assert response.status_code == 200
    assert "Analyze audio" in response.text
    assert "sync-widget idle" in response.text


def test_audio_start_runs_in_background_and_passes_force_through(client, monkeypatch):
    captured = {}

    def fake_analyze_library(conn, db_path, audio_dir, waveform_dir, models_dir, on_progress=None, on_error=None, **kwargs):
        captured["force"] = kwargs.get("force")
        if on_progress:
            on_progress(1, 1, 1)

    monkeypatch.setattr("discql.web.app.audio_analysis.analyze_library", fake_analyze_library)

    response = client.post("/audio/start", data={"force": "true"})
    assert response.status_code == 200

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if tasks.get_task("audio").status == tasks.TaskStatus.DONE:
            break
        time.sleep(0.01)
    else:
        pytest.fail("audio task did not complete in time")

    assert captured["force"] is True
    status_response = client.get("/audio/status")
    assert "Analyzed 1 release" in status_response.text


def test_audio_start_records_error_on_failure(client, monkeypatch):
    def failing_analyze_library(conn, db_path, audio_dir, waveform_dir, models_dir, on_progress=None, on_error=None, **kwargs):
        raise RuntimeError("essentia exploded")

    monkeypatch.setattr("discql.web.app.audio_analysis.analyze_library", failing_analyze_library)

    client.post("/audio/start")

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if tasks.get_task("audio").status == tasks.TaskStatus.ERROR:
            break
        time.sleep(0.01)
    else:
        pytest.fail("audio task did not fail in time")

    response = client.get("/audio/status")
    assert "Audio analysis failed" in response.text
    assert "essentia exploded" in response.text


def test_run_all_status_is_idle_by_default(client):
    response = client.get("/run_all/status")
    assert response.status_code == 200
    assert "Run All" in response.text
    assert "sync-widget idle" in response.text


def test_run_all_start_runs_in_background_and_passes_force_and_configured_workers_through(client, monkeypatch):
    captured = {}

    def fake_run_full_pipeline(conn, db_path, audio_dir, waveform_dir, models_dir, on_progress=None, on_error=None, **kwargs):
        captured["force"] = kwargs.get("force")
        captured["max_workers"] = kwargs.get("max_workers")
        if on_progress:
            on_progress(1, 1, 1, "match")

    monkeypatch.setattr("discql.web.app.pipeline.run_full_pipeline", fake_run_full_pipeline)

    response = client.post("/run_all/start", data={"force": "true"})
    assert response.status_code == 200

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if tasks.get_task("run_all").status == tasks.TaskStatus.DONE:
            break
        time.sleep(0.01)
    else:
        pytest.fail("run_all task did not complete in time")

    assert captured["force"] is True
    assert captured["max_workers"] == get_config().max_workers  # not user-configurable in the UI
    status_response = client.get("/run_all/status")
    assert "Ran all for 1 release" in status_response.text



def test_run_all_start_records_error_on_failure(client, monkeypatch):
    def failing_run_full_pipeline(conn, db_path, audio_dir, waveform_dir, models_dir, on_progress=None, on_error=None, **kwargs):
        raise RuntimeError("pipeline exploded")

    monkeypatch.setattr("discql.web.app.pipeline.run_full_pipeline", failing_run_full_pipeline)

    client.post("/run_all/start")

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if tasks.get_task("run_all").status == tasks.TaskStatus.ERROR:
            break
        time.sleep(0.01)
    else:
        pytest.fail("run_all task did not fail in time")

    response = client.get("/run_all/status")
    assert "Run All failed" in response.text
    assert "pipeline exploded" in response.text


def test_run_all_stop_cancels_the_running_task(client, monkeypatch):
    started = threading.Event()

    def blocking_run_full_pipeline(
        conn, db_path, audio_dir, waveform_dir, models_dir, on_progress=None, on_error=None, should_stop=None, **kwargs
    ):
        started.set()
        while not should_stop():
            time.sleep(0.01)

    monkeypatch.setattr("discql.web.app.pipeline.run_full_pipeline", blocking_run_full_pipeline)

    client.post("/run_all/start")
    assert started.wait(timeout=2)

    stop_response = client.post("/run_all/stop")
    assert stop_response.status_code == 200

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if tasks.get_task("run_all").status == tasks.TaskStatus.CANCELLED:
            break
        time.sleep(0.01)
    else:
        pytest.fail("run_all task did not cancel in time")

    response = client.get("/run_all/status")
    assert "stopped" in response.text
    assert "Run All" in response.text  # can be started again


# --- crates -------------------------------------------------------------


def create_crate(client, name: str) -> int:
    response = client.post("/crates", data={"name": name}, follow_redirects=False)
    assert response.status_code == 303
    return int(response.headers["location"].rsplit("/", 1)[-1])


def test_crates_page_lists_created_crates(client):
    create_crate(client, "Warehouse Gig")

    response = client.get("/crates")

    assert response.status_code == 200
    assert "Warehouse Gig" in response.text


def test_crates_page_empty_state(client):
    response = client.get("/crates")

    assert response.status_code == 200
    assert "No crates yet" in response.text


def test_create_crate_redirects_to_its_detail_page(client):
    response = client.post("/crates", data={"name": "Warehouse Gig"}, follow_redirects=False)

    assert response.status_code == 303
    detail = client.get(response.headers["location"])
    assert "Warehouse Gig" in detail.text


def test_crate_detail_404_for_unknown_crate(client):
    response = client.get("/crates/999")
    assert response.status_code == 404


def test_add_release_to_crate_then_shows_on_detail_page(client):
    crate_id = create_crate(client, "Warehouse Gig")

    response = client.post(
        "/crates/add_release",
        data={"crate_id": crate_id, "release_id": 1, "redirect_to": f"/crates/{crate_id}"},
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/crates/{crate_id}"

    detail = client.get(f"/crates/{crate_id}")
    assert "Acid Test" in detail.text
    assert "Rock Anthology" not in detail.text


def test_adding_a_release_to_a_crate_does_not_change_the_library(client):
    crate_id = create_crate(client, "Warehouse Gig")
    client.post("/crates/add_release", data={"crate_id": crate_id, "release_id": 1})

    # Still shows up in the normal library listing, unaffected.
    response = client.get("/releases")
    assert "Acid Test" in response.text


def test_remove_release_from_crate(client):
    crate_id = create_crate(client, "Warehouse Gig")
    client.post("/crates/add_release", data={"crate_id": crate_id, "release_id": 1})

    response = client.post(f"/crates/{crate_id}/remove_release/1", follow_redirects=False)
    assert response.status_code == 303

    detail = client.get(f"/crates/{crate_id}")
    assert "Acid Test" not in detail.text
    assert "0 releases" in detail.text


def test_rename_crate(client):
    crate_id = create_crate(client, "Old Name")

    client.post(f"/crates/{crate_id}/rename", data={"name": "New Name"})

    detail = client.get(f"/crates/{crate_id}")
    assert "New Name" in detail.text


def test_delete_crate_removes_it_from_the_list(client):
    crate_id = create_crate(client, "Temporary")

    response = client.post(f"/crates/{crate_id}/delete", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/crates"

    crates_page = client.get("/crates")
    assert "Temporary" not in crates_page.text
    assert client.get(f"/crates/{crate_id}").status_code == 404


def test_releases_page_offers_add_to_crate_control_once_a_crate_exists(client):
    create_crate(client, "Warehouse Gig")

    response = client.get("/releases")

    assert "Warehouse Gig" in response.text


def test_release_detail_shows_crate_membership(client):
    crate_id = create_crate(client, "Warehouse Gig")
    client.post("/crates/add_release", data={"crate_id": crate_id, "release_id": 1})

    response = client.get("/releases/1")

    assert "Warehouse Gig" in response.text


# --- settings page / setup-redirect middleware ------------------------------


def test_pages_redirect_to_settings_when_no_discogs_auth_configured(client_without_auth):
    response = client_without_auth.get("/releases", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/settings"


def test_settings_page_itself_does_not_redirect_without_auth(client_without_auth):
    response = client_without_auth.get("/settings")

    assert response.status_code == 200
    assert "Welcome" in response.text


def test_static_assets_are_not_redirected_without_auth(client_without_auth):
    response = client_without_auth.get("/static/styles.css")

    assert response.status_code == 200


def test_htmx_status_polling_requests_are_not_redirected_without_auth(client_without_auth):
    """Regression test: base.html's topbar auto-loads several small status
    widgets via hx-trigger="load" (see require_discogs_auth's docstring).
    Redirecting those to /settings (a full page) made htmx recursively
    re-trigger on the newly-swapped-in content's own copies of the same
    widgets - a runaway loop, observed concretely on the real site. None
    of these need Discogs auth to render their current state."""
    for path in ("/sync/status", "/youtube/status", "/audio/status", "/run_all/status", "/pipeline/pending"):
        response = client_without_auth.get(path, headers={"HX-Request": "true"}, follow_redirects=False)
        assert response.status_code == 200, f"{path} was redirected instead of rendering directly"


def test_settings_page_shows_no_welcome_banner_once_auth_is_configured(client):
    response = client.get("/settings")

    assert response.status_code == 200
    assert "Welcome" not in response.text


def test_saving_settings_with_a_token_unblocks_other_pages(client_without_auth):
    save = client_without_auth.post(
        "/settings",
        data={
            "discogs_user_token": "fresh-token",
            "fuzzy_confident_threshold": "0.8",
            "fuzzy_tie_margin": "0.05",
            "fuzzy_denoise_terms": "Original Mix",
            "youtube_cookies_browser": "",
            "youtube_audio_max_bitrate_kbps": "128",
            "local_flac_dir": "",
            "local_match_confident_threshold": "0.75",
            "max_workers": "4",
        },
        follow_redirects=False,
    )
    assert save.status_code == 303
    assert save.headers["location"] == "/settings?saved=true"

    response = client_without_auth.get("/releases", follow_redirects=False)
    assert response.status_code == 200


def test_saving_settings_with_blank_token_field_keeps_the_existing_token(client):
    # `client` already has DISCOGS_USER_TOKEN=dummy configured via its fixture.
    client.post(
        "/settings",
        data={
            "discogs_user_token": "",
            "fuzzy_confident_threshold": "0.9",
            "fuzzy_tie_margin": "0.05",
            "fuzzy_denoise_terms": "Original Mix",
            "youtube_cookies_browser": "",
            "youtube_audio_max_bitrate_kbps": "128",
            "local_flac_dir": "",
            "local_match_confident_threshold": "0.75",
            "max_workers": "4",
        },
    )

    # Still logged in / no redirect - the token wasn't blanked out, and the
    # other field's edit did take effect.
    response = client.get("/releases", follow_redirects=False)
    assert response.status_code == 200
    assert get_config().fuzzy_confident_threshold == 0.9
    assert get_config().discogs_user_token == "dummy"


def test_saving_settings_persists_tuning_values_to_the_config_file(client, tmp_path):
    client.post(
        "/settings",
        data={
            "discogs_user_token": "",
            "fuzzy_confident_threshold": "0.8",
            "fuzzy_tie_margin": "0.05",
            "fuzzy_denoise_terms": "Original Mix",
            "youtube_cookies_browser": "firefox",
            "youtube_audio_max_bitrate_kbps": "128",
            "local_flac_dir": "",
            "local_match_confident_threshold": "0.75",
            "max_workers": "6",
        },
    )

    assert get_config().max_workers == 6
    assert get_config().youtube_cookies_browser == "firefox"


# --- Discogs OAuth authorize/callback routes ---------------------------------


class _FakeOAuthClient:
    """Stands in for discogs_client.Client for the parts DiscogsApi's OAuth
    flow uses - real HMAC-SHA1 signing/network calls are out of scope for a
    unit test (and untestable without a real registered Discogs
    Application anyway, see the plan)."""

    instances: list["_FakeOAuthClient"] = []

    def __init__(self, user_agent, consumer_key=None, consumer_secret=None, **kwargs):
        self.consumer_key = consumer_key
        self.consumer_secret = consumer_secret
        _FakeOAuthClient.instances.append(self)

    def get_authorize_url(self, callback_url=None):
        self.callback_url = callback_url
        return ("req-token", "req-secret", "https://www.discogs.com/oauth/authorize?oauth_token=req-token")

    def get_access_token(self, verifier):
        self.verifier = verifier
        return ("access-token", "access-secret")


@pytest.fixture(autouse=True)
def reset_fake_oauth_client():
    import discql.web.app as app_module

    _FakeOAuthClient.instances.clear()
    app_module._pending_oauth_client = None
    yield
    _FakeOAuthClient.instances.clear()
    app_module._pending_oauth_client = None


def test_start_discogs_oauth_saves_consumer_credentials_and_redirects_to_discogs(
    client_without_auth, monkeypatch
):
    monkeypatch.setattr("discql.web.app.discogs_client.Client", _FakeOAuthClient)

    response = client_without_auth.post(
        "/settings/discogs/authorize",
        data={"consumer_key": "ck", "consumer_secret": "cs"},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("https://www.discogs.com/oauth/authorize")
    assert get_config().discogs_consumer_key == "ck"
    assert get_config().discogs_consumer_secret == "cs"


def test_discogs_oauth_callback_without_a_pending_flow_shows_an_error(client_without_auth):
    response = client_without_auth.get(
        "/settings/discogs/callback", params={"oauth_verifier": "v"}, follow_redirects=False
    )

    assert response.status_code == 303
    assert response.headers["location"].startswith("/settings?oauth_error=")


def test_full_mocked_oauth_round_trip_configures_discogs_auth(client_without_auth, monkeypatch):
    monkeypatch.setattr("discql.web.app.discogs_client.Client", _FakeOAuthClient)

    authorize_response = client_without_auth.post(
        "/settings/discogs/authorize",
        data={"consumer_key": "ck", "consumer_secret": "cs"},
        follow_redirects=False,
    )
    assert authorize_response.status_code == 303

    callback_response = client_without_auth.get(
        "/settings/discogs/callback", params={"oauth_verifier": "the-verifier"}, follow_redirects=False
    )
    assert callback_response.status_code == 303
    assert callback_response.headers["location"] == "/settings?saved=true"

    config = get_config()
    assert config.has_discogs_auth is True
    assert config.discogs_oauth_token == "access-token"
    assert config.discogs_oauth_token_secret == "access-secret"
    # No longer redirected away from the rest of the app.
    assert client_without_auth.get("/releases", follow_redirects=False).status_code == 200


def test_disconnect_discogs_oauth_clears_oauth_fields(client_without_auth, monkeypatch):
    monkeypatch.setattr("discql.web.app.discogs_client.Client", _FakeOAuthClient)
    client_without_auth.post("/settings/discogs/authorize", data={"consumer_key": "ck", "consumer_secret": "cs"})
    client_without_auth.get("/settings/discogs/callback", params={"oauth_verifier": "v"})
    assert get_config().has_discogs_auth is True

    client_without_auth.post("/settings/discogs/disconnect")

    assert get_config().discogs_oauth_token is None
    assert get_config().has_discogs_auth is False
