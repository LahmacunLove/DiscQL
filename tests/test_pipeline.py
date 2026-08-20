from __future__ import annotations

import contextlib
import threading

import pytest

from discql import db, pipeline
from discql.discogs_api import ArtistData, ReleaseData, TrackData, VideoData
from discql.sync import upsert_release


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "test.db"
    conn = db.connect(path)
    db.migrate(conn)
    conn.close()
    return path


def seed_release(db_path, release_id, has_video=True):
    conn = db.connect(db_path)
    data = ReleaseData(
        id=release_id,
        title=f"Release {release_id}",
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri=f"https://discogs.com/release/{release_id}",
        cover_image_url=None,
        artists=[ArtistData(id=release_id * 10, name="Artist")],
        labels=[],
        tracks=[TrackData(position="A1", title="Track One", duration="3:00", artist=None)],
        videos=(
            [VideoData(title="A video", url=f"https://youtube.com/watch?v={release_id}", duration=180, description=None)]
            if has_video
            else []
        ),
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")
    conn.close()


def mark_matched(conn, release_id, *, analyzed=False):
    """Give a release a selected track/video match (as if match_release had
    really run), optionally also marking its track analyzed."""
    now = "2024-03-01T00:00:00"
    video_id = conn.execute("SELECT id FROM release_videos WHERE release_id = ?", (release_id,)).fetchone()["id"]
    track_id = conn.execute("SELECT id FROM tracks WHERE release_id = ?", (release_id,)).fetchone()["id"]
    conn.execute("UPDATE release_videos SET yt_fetched_at = ?, video_type = 'full_track' WHERE id = ?", (now, video_id))
    if analyzed:
        conn.execute("UPDATE tracks SET analyzed_at = ? WHERE id = ?", (now, track_id))
    conn.execute(
        "INSERT INTO track_video_matches (track_id, video_id, fuzzy_score, is_selected, created_at) "
        "VALUES (?, ?, ?, 1, ?)",
        (track_id, video_id, 1.0, now),
    )
    conn.commit()


@pytest.fixture(autouse=True)
def fake_ollama(monkeypatch):
    calls = {"count": 0, "active": False}

    @contextlib.contextmanager
    def fake_ensure_ollama_running():
        calls["count"] += 1
        calls["active"] = True
        try:
            yield
        finally:
            calls["active"] = False

    monkeypatch.setattr("discql.pipeline.ollama_client.ensure_ollama_running", fake_ensure_ollama_running)
    return calls


def run_pipeline(db_path, conn, **kwargs):
    return pipeline.run_full_pipeline(
        conn, db_path, db_path.parent / "audio", db_path.parent / "waveforms", db_path.parent / "models",
        downloader=None, extractor=None, call_ollama=None,
        **kwargs,
    )


# --- match_and_download_library (standalone entry point, used by the "Match YT" button) ---

def test_match_and_download_library_processes_releases_in_parallel_and_reports_match_phase(
    db_path, monkeypatch, fake_ollama
):
    seed_release(db_path, 1)
    seed_release(db_path, 2)
    conn = db.connect(db_path)

    match_calls = []
    monkeypatch.setattr(
        "discql.pipeline.youtube_matching.match_release",
        lambda conn, release_id, *a, **kw: match_calls.append(release_id),
    )
    monkeypatch.setattr("discql.pipeline.audio_download.download_release_tracks", lambda *a, **kw: None)

    progressed = []
    pipeline.match_and_download_library(
        conn, db_path, db_path.parent / "audio",
        downloader=None, extractor=None, call_ollama=None,
        max_workers=2,
        on_progress=lambda i, t, rid, phase: progressed.append((rid, phase)),
    )
    conn.close()

    assert set(match_calls) == {1, 2}
    assert progressed == [(1, "match"), (2, "match")] or progressed == [(2, "match"), (1, "match")]


def test_match_and_download_library_stops_ollama_before_returning(db_path, monkeypatch, fake_ollama):
    seed_release(db_path, 1)
    conn = db.connect(db_path)
    monkeypatch.setattr("discql.pipeline.youtube_matching.match_release", lambda *a, **kw: None)
    monkeypatch.setattr("discql.pipeline.audio_download.download_release_tracks", lambda *a, **kw: None)

    pipeline.match_and_download_library(
        conn, db_path, db_path.parent / "audio", downloader=None, extractor=None, call_ollama=None,
    )
    conn.close()

    assert fake_ollama["count"] == 1
    assert fake_ollama["active"] is False


# --- local coverage: YouTube becomes obsolete when a track is locally matched ---


def mark_locally_covered(conn, release_id):
    conn.execute(
        "UPDATE tracks SET local_audio_path = ? WHERE release_id = ?",
        (f"/fake/local/{release_id}.flac", release_id),
    )
    conn.commit()


def test_match_and_download_skips_both_steps_when_locally_covered(db_path, monkeypatch):
    seed_release(db_path, 1)
    conn = db.connect(db_path)
    mark_locally_covered(conn, 1)

    match_calls = []
    download_calls = []
    monkeypatch.setattr(
        "discql.pipeline.youtube_matching.match_release",
        lambda conn, release_id, *a, **kw: match_calls.append(release_id),
    )
    monkeypatch.setattr(
        "discql.pipeline.audio_download.download_release_tracks",
        lambda conn, release_id, *a, **kw: download_calls.append(release_id),
    )

    pipeline._match_and_download_one_release(
        db_path, 1, "Release 1", db_path.parent / "audio", None, None, None,
        force=False, confident_threshold=0.8, tie_margin=0.05, denoise_terms=[],
    )
    conn.close()

    assert match_calls == []
    assert download_calls == []


def test_match_and_download_force_ignores_local_coverage(db_path, monkeypatch):
    seed_release(db_path, 1)
    conn = db.connect(db_path)
    mark_locally_covered(conn, 1)

    match_calls = []
    monkeypatch.setattr(
        "discql.pipeline.youtube_matching.match_release",
        lambda conn, release_id, *a, **kw: match_calls.append(release_id),
    )
    monkeypatch.setattr("discql.pipeline.audio_download.download_release_tracks", lambda *a, **kw: None)

    pipeline._match_and_download_one_release(
        db_path, 1, "Release 1", db_path.parent / "audio", None, None, None,
        force=True, confident_threshold=0.8, tie_margin=0.05, denoise_terms=[],
    )
    conn.close()

    assert match_calls == [1]  # force overrides local-coverage skip


def test_match_and_download_library_runs_local_match_as_pre_step(db_path, monkeypatch, fake_ollama):
    seed_release(db_path, 1)
    conn = db.connect(db_path)
    monkeypatch.setattr("discql.pipeline.youtube_matching.match_release", lambda *a, **kw: None)
    monkeypatch.setattr("discql.pipeline.audio_download.download_release_tracks", lambda *a, **kw: None)

    local_match_calls = []
    monkeypatch.setattr(
        "discql.pipeline.local_audio.match_library",
        lambda conn, flac_dir, **kw: local_match_calls.append(flac_dir),
    )

    fake_flac_dir = db_path.parent / "flac"
    pipeline.match_and_download_library(
        conn, db_path, db_path.parent / "audio", downloader=None, extractor=None, call_ollama=None,
        flac_dir=fake_flac_dir,
    )
    conn.close()

    assert local_match_calls == [fake_flac_dir]


def test_match_and_download_library_skips_local_match_when_flac_dir_is_none(db_path, monkeypatch, fake_ollama):
    seed_release(db_path, 1)
    conn = db.connect(db_path)
    monkeypatch.setattr("discql.pipeline.youtube_matching.match_release", lambda *a, **kw: None)
    monkeypatch.setattr("discql.pipeline.audio_download.download_release_tracks", lambda *a, **kw: None)

    local_match_calls = []
    monkeypatch.setattr(
        "discql.pipeline.local_audio.match_library",
        lambda conn, flac_dir, **kw: local_match_calls.append(flac_dir),
    )

    pipeline.match_and_download_library(
        conn, db_path, db_path.parent / "audio", downloader=None, extractor=None, call_ollama=None,
        flac_dir=None,
    )
    conn.close()

    assert local_match_calls == []


# --- _match_and_download_one_release ---------------------------------------

def test_match_and_download_skips_match_when_already_matched(db_path, monkeypatch):
    seed_release(db_path, 1)
    conn = db.connect(db_path)
    mark_matched(conn, 1, analyzed=False)  # matched, not yet analyzed -> still needs download

    match_calls = []
    download_calls = []
    monkeypatch.setattr("discql.pipeline.youtube_matching.match_release", lambda *a, **kw: match_calls.append(1))
    monkeypatch.setattr(
        "discql.pipeline.audio_download.download_release_tracks",
        lambda conn, release_id, *a, **kw: download_calls.append(release_id),
    )

    pipeline._match_and_download_one_release(
        db_path, 1, "Release 1", db_path.parent / "audio", None, None, None,
        force=False, confident_threshold=0.8, tie_margin=0.05, denoise_terms=[],
    )
    conn.close()

    assert match_calls == []
    assert download_calls == [1]


def test_match_and_download_matches_unmatched_release_with_no_match_yet(db_path, monkeypatch):
    seed_release(db_path, 1)  # fresh -> video unmatched, no selected track match at all

    match_calls = []
    download_calls = []
    monkeypatch.setattr(
        "discql.pipeline.youtube_matching.match_release",
        lambda conn, release_id, *a, **kw: match_calls.append(release_id),
    )
    monkeypatch.setattr(
        "discql.pipeline.audio_download.download_release_tracks",
        lambda conn, release_id, *a, **kw: download_calls.append(release_id),
    )

    pipeline._match_and_download_one_release(
        db_path, 1, "Release 1", db_path.parent / "audio", None, None, None,
        force=False, confident_threshold=0.8, tie_margin=0.05, denoise_terms=[],
    )

    assert match_calls == [1]
    assert download_calls == []  # nothing selected to download yet (match is mocked, so no real match got made)


def test_match_and_download_skips_download_when_fully_analyzed(db_path, monkeypatch):
    seed_release(db_path, 1)
    conn = db.connect(db_path)
    mark_matched(conn, 1, analyzed=True)  # fully done: matched and analyzed already

    match_calls = []
    download_calls = []
    monkeypatch.setattr(
        "discql.pipeline.youtube_matching.match_release",
        lambda conn, release_id, *a, **kw: match_calls.append(release_id),
    )
    monkeypatch.setattr(
        "discql.pipeline.audio_download.download_release_tracks",
        lambda conn, release_id, *a, **kw: download_calls.append(release_id),
    )

    pipeline._match_and_download_one_release(
        db_path, 1, "Release 1", db_path.parent / "audio", None, None, None,
        force=False, confident_threshold=0.8, tie_margin=0.05, denoise_terms=[],
    )
    conn.close()

    assert match_calls == []  # already matched, no re-match
    assert download_calls == []  # already analyzed, nothing left to download for


def test_match_and_download_force_always_runs_both_steps(db_path, monkeypatch):
    seed_release(db_path, 1)
    conn = db.connect(db_path)
    mark_matched(conn, 1, analyzed=True)  # fully done otherwise

    match_calls = []
    download_calls = []
    monkeypatch.setattr(
        "discql.pipeline.youtube_matching.match_release",
        lambda conn, release_id, *a, **kw: match_calls.append(release_id),
    )
    monkeypatch.setattr(
        "discql.pipeline.audio_download.download_release_tracks",
        lambda conn, release_id, *a, **kw: download_calls.append(release_id),
    )

    pipeline._match_and_download_one_release(
        db_path, 1, "Release 1", db_path.parent / "audio", None, None, None,
        force=True, confident_threshold=0.8, tie_margin=0.05, denoise_terms=[],
    )
    conn.close()

    assert match_calls == [1]
    assert download_calls == [1]


# --- run_full_pipeline: phase ordering --------------------------------------

def test_run_full_pipeline_finishes_all_matching_before_analysis_starts(db_path, monkeypatch, fake_ollama):
    seed_release(db_path, 1)
    seed_release(db_path, 2)
    conn = db.connect(db_path)
    # Both releases already have a selected (unanalyzed) match, so phase 1
    # both skips re-matching and still downloads for phase 2.
    mark_matched(conn, 1, analyzed=False)
    mark_matched(conn, 2, analyzed=False)

    order = []
    monkeypatch.setattr(
        "discql.pipeline.audio_download.download_release_tracks",
        lambda conn, release_id, *a, **kw: order.append(("download", release_id)),
    )
    monkeypatch.setattr(
        "discql.pipeline.audio_analysis.analyze_library",
        lambda *a, **kw: order.append(("analyze_library", None)),
    )

    run_pipeline(db_path, conn, max_workers=2)
    conn.close()

    assert order == [("download", 1), ("download", 2), ("analyze_library", None)] or order == [
        ("download", 2), ("download", 1), ("analyze_library", None)
    ]


def test_run_full_pipeline_calls_analyze_library_with_expected_args(db_path, monkeypatch, fake_ollama):
    seed_release(db_path, 1, has_video=False)  # nothing for phase 1 to do
    conn = db.connect(db_path)

    captured = {}
    monkeypatch.setattr(
        "discql.pipeline.audio_analysis.analyze_library",
        lambda conn, db_path, audio_dir, waveform_dir, models_dir, **kw: captured.update(kw, audio_dir=audio_dir),
    )

    run_pipeline(db_path, conn, force=True)
    conn.close()

    assert captured["audio_dir"] == db_path.parent / "audio"
    assert captured["force"] is True
    assert callable(captured["on_progress"])
    assert captured["should_stop"] is None  # not passed to run_pipeline() in this test


# --- release selection / skipping -------------------------------------------

def test_run_full_pipeline_skips_releases_without_videos(db_path, monkeypatch, fake_ollama):
    seed_release(db_path, 1, has_video=True)
    seed_release(db_path, 2, has_video=False)

    calls = []
    monkeypatch.setattr(
        "discql.pipeline.youtube_matching.match_release",
        lambda conn, release_id, title, *a, **kw: calls.append(release_id),
    )
    monkeypatch.setattr("discql.pipeline.audio_download.download_release_tracks", lambda *a, **kw: None)
    monkeypatch.setattr("discql.pipeline.audio_analysis.analyze_library", lambda *a, **kw: None)

    conn = db.connect(db_path)
    run_pipeline(db_path, conn)
    conn.close()

    assert calls == [1]


def test_run_full_pipeline_reports_error_and_continues_into_analysis_phase(db_path, monkeypatch, fake_ollama):
    seed_release(db_path, 1)
    seed_release(db_path, 2)

    def failing_match(conn, release_id, title, *a, **kw):
        if release_id == 1:
            raise RuntimeError("boom")

    monkeypatch.setattr("discql.pipeline.youtube_matching.match_release", failing_match)
    monkeypatch.setattr("discql.pipeline.audio_download.download_release_tracks", lambda *a, **kw: None)
    analyze_calls = []
    monkeypatch.setattr(
        "discql.pipeline.audio_analysis.analyze_library", lambda *a, **kw: analyze_calls.append(1)
    )

    conn = db.connect(db_path)
    errors = []
    progressed = []
    run_pipeline(
        db_path, conn,
        on_error=lambda rid, exc: errors.append((rid, str(exc))),
        on_progress=lambda i, t, rid, phase: progressed.append((rid, phase)),
    )
    conn.close()

    assert errors == [(1, "boom")]
    assert {rid for rid, phase in progressed} == {1, 2}
    assert all(phase == "match" for _, phase in progressed)
    # phase 1 failing for one release must not prevent phase 2 from running.
    assert analyze_calls == [1]


def test_run_full_pipeline_skips_already_fully_processed_releases(db_path, monkeypatch, fake_ollama):
    seed_release(db_path, 1)  # left untouched -> video not yet fetched, still pending
    seed_release(db_path, 2)  # marked fully matched + analyzed below -> should be skipped

    conn = db.connect(db_path)
    mark_matched(conn, 2, analyzed=True)

    calls = []
    monkeypatch.setattr(
        "discql.pipeline.youtube_matching.match_release",
        lambda conn, release_id, title, *a, **kw: calls.append(release_id),
    )
    monkeypatch.setattr("discql.pipeline.audio_download.download_release_tracks", lambda *a, **kw: None)
    monkeypatch.setattr("discql.pipeline.audio_analysis.analyze_library", lambda *a, **kw: None)

    run_pipeline(db_path, conn)
    assert calls == [1]

    # force=True must reprocess everything again, including the finished release.
    calls.clear()
    run_pipeline(db_path, conn, force=True)
    conn.close()
    assert set(calls) == {1, 2}


def test_count_pending_releases_matches_what_a_non_force_run_would_touch(db_path):
    seed_release(db_path, 1)  # untouched -> pending
    seed_release(db_path, 2)  # marked fully done below -> not pending

    conn = db.connect(db_path)
    mark_matched(conn, 2, analyzed=True)

    assert pipeline.count_pending_releases(conn) == 1
    conn.close()


# --- should_stop -------------------------------------------------------------

def test_run_full_pipeline_should_stop_prevents_any_submission_or_analysis(db_path, monkeypatch, fake_ollama):
    for release_id in (1, 2, 3):
        seed_release(db_path, release_id)

    calls = []
    analyze_calls = []
    monkeypatch.setattr(
        "discql.pipeline.youtube_matching.match_release",
        lambda conn, release_id, title, *a, **kw: calls.append(release_id),
    )
    monkeypatch.setattr("discql.pipeline.audio_download.download_release_tracks", lambda *a, **kw: None)
    monkeypatch.setattr(
        "discql.pipeline.audio_analysis.analyze_library", lambda *a, **kw: analyze_calls.append(1)
    )

    conn = db.connect(db_path)
    run_pipeline(db_path, conn, max_workers=1, should_stop=lambda: True)
    conn.close()

    assert calls == []
    assert analyze_calls == []


def test_run_full_pipeline_stops_after_current_release_finishes_and_skips_analysis(db_path, monkeypatch, fake_ollama):
    for release_id in (1, 2, 3):
        seed_release(db_path, release_id)

    calls = []
    analyze_calls = []
    monkeypatch.setattr(
        "discql.pipeline.youtube_matching.match_release",
        lambda conn, release_id, title, *a, **kw: calls.append(release_id),
    )
    monkeypatch.setattr("discql.pipeline.audio_download.download_release_tracks", lambda *a, **kw: None)
    monkeypatch.setattr(
        "discql.pipeline.audio_analysis.analyze_library", lambda *a, **kw: analyze_calls.append(1)
    )

    stop_flag = {"stop": False}

    def progress(i, t, rid, phase):
        stop_flag["stop"] = True  # request stop right after the first release completes

    conn = db.connect(db_path)
    run_pipeline(db_path, conn, max_workers=1, should_stop=lambda: stop_flag["stop"], on_progress=progress)
    conn.close()

    assert len(calls) == 1
    assert analyze_calls == []


def test_run_full_pipeline_stops_ollama_before_starting_analysis_phase(db_path, monkeypatch, fake_ollama):
    seed_release(db_path, 1)
    conn = db.connect(db_path)

    ollama_active_during_analyze = {"value": None}
    monkeypatch.setattr("discql.pipeline.youtube_matching.match_release", lambda *a, **kw: None)
    monkeypatch.setattr("discql.pipeline.audio_download.download_release_tracks", lambda *a, **kw: None)
    monkeypatch.setattr(
        "discql.pipeline.audio_analysis.analyze_library",
        lambda *a, **kw: ollama_active_during_analyze.__setitem__("value", fake_ollama["active"]),
    )

    run_pipeline(db_path, conn)
    conn.close()

    assert ollama_active_during_analyze["value"] is False


# --- concurrency --------------------------------------------------------------

def test_run_full_pipeline_can_process_releases_concurrently(db_path, monkeypatch, fake_ollama):
    for release_id in (1, 2, 3):
        seed_release(db_path, release_id)

    barrier = threading.Barrier(3, timeout=5)

    def blocking_match(conn, release_id, title, *a, **kw):
        barrier.wait()  # only passes if all 3 releases are being processed at once

    monkeypatch.setattr("discql.pipeline.youtube_matching.match_release", blocking_match)
    monkeypatch.setattr("discql.pipeline.audio_download.download_release_tracks", lambda *a, **kw: None)
    monkeypatch.setattr("discql.pipeline.audio_analysis.analyze_library", lambda *a, **kw: None)

    conn = db.connect(db_path)
    run_pipeline(db_path, conn, max_workers=3)
    conn.close()
