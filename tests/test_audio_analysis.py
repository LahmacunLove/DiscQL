from __future__ import annotations

import functools
import multiprocessing
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from discql import audio_analysis as aa
from discql import db
from discql.discogs_api import ArtistData, ReleaseData, TrackData, VideoData
from discql.sync import upsert_release
from discql import youtube_matching as ym

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _make_audio_file(path, duration=1):
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", f"sine=duration={duration}", "-c:a", "aac", str(path)],
        check=True,
    )


@pytest.fixture()
def db_path(tmp_path):
    path = tmp_path / "test.db"
    conn = db.connect(path)
    db.migrate(conn)
    conn.close()
    return path


@pytest.fixture()
def conn(db_path):
    # File-backed (not :memory:) since analyze_library() now opens its own
    # connection per worker thread (db.connect(db_path)) to parallelize
    # across releases - :memory: databases aren't shared across connections.
    connection = db.connect(db_path)
    yield connection
    connection.close()


def seed_matched_release(conn, release_id=1):
    data = ReleaseData(
        id=release_id,
        title="Test Release",
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri=f"https://discogs.com/release/{release_id}",
        cover_image_url=None,
        artists=[ArtistData(id=10, name="Test Artist")],
        labels=[],
        tracks=[TrackData(position="A1", title="Rave-O-Lution", duration="4:30", artist=None)],
        videos=[VideoData(title="ignored", url="https://youtube.com/watch?v=abc", duration=270, description=None)],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")

    def fake_extractor(url):
        return {"title": "Rave-O-Lution", "duration": 270, "uploader": "X", "description": "", "tags": [], "chapters": []}

    ym.match_videos(conn, extractor=fake_extractor, call_ollama=lambda p: (_ for _ in ()).throw(AssertionError))


# --- _collapse_predictions ----------------------------------------------------


def test_collapse_predictions_binary_classifier_keeps_positive_class_only():
    result = aa._collapse_predictions("mood_happy", ["happy", "non_happy"], [0.73, 0.27])
    assert result == {"mood_happy": 0.73}


def test_collapse_predictions_binary_classifier_positive_class_can_be_second():
    result = aa._collapse_predictions("mood_sad", ["non_sad", "sad"], [0.6, 0.4])
    assert result == {"mood_sad": 0.4}


def test_collapse_predictions_multiclass_keeps_all_classes():
    result = aa._collapse_predictions("genre_discogs400", ["rock", "jazz"], [0.2, 0.8])
    assert result == {"genre_discogs400:rock": 0.2, "genre_discogs400:jazz": 0.8}


# --- summarize_mood ------------------------------------------------------------


def test_summarize_mood_picks_top_tags_above_threshold():
    scores = {"mood_happy": 0.9, "danceability": 0.7, "mood_sad": 0.1, "genre_discogs400:rock": 0.99}
    summary = aa.summarize_mood(scores)
    assert summary == "Happy, Danceable"


def test_summarize_mood_ignores_multiclass_scores():
    scores = {"genre_discogs400:rock": 0.99, "genre_discogs400:jazz": 0.01}
    assert aa.summarize_mood(scores) == "Neutral"


def test_summarize_mood_neutral_when_nothing_above_threshold():
    scores = {"mood_happy": 0.2, "mood_sad": 0.3}
    assert aa.summarize_mood(scores) == "Neutral"


def test_summarize_mood_respects_max_tags():
    scores = {"mood_happy": 0.9, "danceability": 0.8, "mood_party": 0.7, "mood_relaxed": 0.6}
    summary = aa.summarize_mood(scores, max_tags=2)
    assert summary == "Happy, Danceable"


# --- top_style / top_genre ------------------------------------------------------


def test_top_style_picks_highest_scoring_discogs_style():
    scores = {
        "genre_discogs400:Electronic---Deep House": 0.62,
        "genre_discogs400:Electronic---Techno": 0.81,
        "genre_discogs400:Rock---Krautrock": 0.05,
        "mood_happy": 0.9,
    }
    assert aa.top_style(scores) == "Electronic---Techno"


def test_top_style_none_when_no_genre_scores_present():
    assert aa.top_style({"mood_happy": 0.9}) is None


def test_top_genre_splits_off_broad_genre_from_top_style():
    scores = {
        "genre_discogs400:Electronic---Deep House": 0.81,
        "genre_discogs400:Rock---Krautrock": 0.05,
    }
    assert aa.top_genre(scores) == "Electronic"


def test_top_genre_none_when_no_genre_scores_present():
    assert aa.top_genre({"mood_happy": 0.9}) is None


# --- _ensure_model ---------------------------------------------------------


def test_ensure_model_downloads_and_returns_path(tmp_path, monkeypatch):
    def fake_run(cmd, check, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_bytes(b"model bytes")

    monkeypatch.setattr(aa.subprocess, "run", fake_run)

    path = aa._ensure_model("some-model", "https://example.com/some-model.pb", tmp_path)

    assert path == tmp_path / "some-model.pb"
    assert path.read_bytes() == b"model bytes"
    assert not any(tmp_path.glob("*.tmp"))


def test_ensure_model_skips_download_when_already_present(tmp_path, monkeypatch):
    (tmp_path / "some-model.pb").write_bytes(b"already here")

    def unreachable_run(*a, **kw):
        raise AssertionError("should not download - already cached")

    monkeypatch.setattr(aa.subprocess, "run", unreachable_run)

    path = aa._ensure_model("some-model", "https://example.com/some-model.pb", tmp_path)

    assert path.read_bytes() == b"already here"


def test_ensure_model_concurrent_downloads_of_same_model_do_not_corrupt_each_other(tmp_path, monkeypatch):
    # Regression test: concurrent "Run All" worker threads can race to
    # download the same not-yet-cached model - each must write to its own
    # tmp file, not a shared one, or one curl's partial write can land in
    # the middle of another's.
    barrier = threading.Barrier(4, timeout=5)

    def fake_run(cmd, check, timeout):
        barrier.wait()  # only passes if all 4 calls are mid-flight concurrently
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_bytes(b"model bytes")

    monkeypatch.setattr(aa.subprocess, "run", fake_run)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(
            lambda _: aa._ensure_model("some-model", "https://example.com/some-model.pb", tmp_path),
            range(4),
        ))

    assert all(r == tmp_path / "some-model.pb" for r in results)
    assert (tmp_path / "some-model.pb").read_bytes() == b"model bytes"
    assert not any(tmp_path.glob("*.tmp"))


# --- _ModelCache / _get_model_cache (thread-local model reuse) -------------


class _FakeEffnetModel:
    """Records how many times it's instantiated; returns a fixed-shape
    embedding-like array when called - stands in for
    TensorflowPredictEffnetDiscogs without needing a real graph file."""

    instances = 0

    def __init__(self, graphFilename, output):
        _FakeEffnetModel.instances += 1
        self.graphFilename = graphFilename

    def __call__(self, audio):
        import numpy as np

        return np.zeros((2, 1280), dtype=np.float32)


class _FakeHeadModel:
    instances = 0

    def __init__(self, graphFilename, input, output):
        _FakeHeadModel.instances += 1
        self.graphFilename = graphFilename

    def __call__(self, embeddings):
        import numpy as np

        return np.zeros((2, 2), dtype=np.float32)


@pytest.fixture()
def fake_models_dir(tmp_path):
    (tmp_path / "discogs-effnet-bs64-1.pb").write_bytes(b"fake")
    (tmp_path / "danceability-discogs-effnet-1.pb").write_bytes(b"fake")
    return tmp_path


def test_model_cache_loads_embedding_model_only_once(fake_models_dir, monkeypatch):
    _FakeEffnetModel.instances = 0
    monkeypatch.setattr("essentia.standard.TensorflowPredictEffnetDiscogs", _FakeEffnetModel)
    import numpy as np

    cache = aa._ModelCache(fake_models_dir)
    audio = np.zeros(16000, dtype=np.float32)
    cache.embed(audio)
    cache.embed(audio)
    cache.embed(audio)

    assert _FakeEffnetModel.instances == 1


def test_model_cache_loads_each_head_only_once(fake_models_dir, monkeypatch):
    _FakeHeadModel.instances = 0
    monkeypatch.setattr("essentia.standard.TensorflowPredict2D", _FakeHeadModel)
    import numpy as np

    cache = aa._ModelCache(fake_models_dir)
    head = aa.EFFNET_CLASSIFIERS["danceability"]
    embeddings = np.zeros((2, 1280), dtype=np.float32)
    cache.predict_head("danceability", head, embeddings)
    cache.predict_head("danceability", head, embeddings)

    assert _FakeHeadModel.instances == 1


def test_get_model_cache_returns_same_instance_within_a_thread(fake_models_dir):
    cache1 = aa._get_model_cache(fake_models_dir)
    cache2 = aa._get_model_cache(fake_models_dir)
    assert cache1 is cache2


def test_get_model_cache_returns_a_fresh_instance_per_thread(fake_models_dir):
    caches = {}

    def grab(thread_id):
        caches[thread_id] = aa._get_model_cache(fake_models_dir)

    t1 = threading.Thread(target=grab, args=(1,))
    t2 = threading.Thread(target=grab, args=(2,))
    t1.start()
    t1.join()
    t2.start()
    t2.join()

    assert caches[1] is not caches[2]


def test_get_model_cache_creates_fresh_cache_when_models_dir_changes(tmp_path):
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    dir_a.mkdir()
    dir_b.mkdir()

    cache_a = aa._get_model_cache(dir_a)
    cache_b = aa._get_model_cache(dir_b)

    assert cache_a is not cache_b
    assert cache_a.models_dir == dir_a
    assert cache_b.models_dir == dir_b


# --- decode_audio / generate_waveform (real ffmpeg + Pillow, no essentia needed) -----------


@requires_ffmpeg
def test_decode_audio_returns_normalized_float32_samples(tmp_path):
    import numpy as np

    audio_path = tmp_path / "audio.m4a"
    _make_audio_file(audio_path)

    samples = aa.decode_audio(audio_path, 44100)

    assert samples.dtype == np.float32
    assert len(samples) > 0
    assert np.max(np.abs(samples)) <= 1.0


@requires_ffmpeg
def test_generate_waveform_produces_png_of_requested_size(tmp_path):
    from PIL import Image

    audio_path = tmp_path / "audio.m4a"
    _make_audio_file(audio_path)
    samples = aa.decode_audio(audio_path, 44100)
    out_path = tmp_path / "waveform.png"

    result = aa.generate_waveform(samples, out_path, size=(200, 50))

    assert result == out_path
    assert out_path.exists()
    with Image.open(out_path) as image:
        assert image.size == (200, 50)


@requires_ffmpeg
def test_generate_waveform_draws_something_not_fully_transparent(tmp_path):
    from PIL import Image

    audio_path = tmp_path / "audio.m4a"
    _make_audio_file(audio_path)
    samples = aa.decode_audio(audio_path, 44100)
    out_path = tmp_path / "waveform.png"

    aa.generate_waveform(samples, out_path, size=(200, 50))

    with Image.open(out_path) as image:
        alpha_bytes = image.tobytes()[3::4]
    assert any(b > 0 for b in alpha_bytes)


# --- track_waveform_path -------------------------------------------------------


def test_track_waveform_path_includes_position_and_release_id(tmp_path):
    path = aa.track_waveform_path(tmp_path, 1, "A1", "Some Track")
    assert path.parent == tmp_path / "1"
    assert path.suffix == ".png"
    assert "A1" in path.name


# --- analyze_release_tracks: orchestration (analyze_track faked out) --------


def test_analyze_release_tracks_writes_results_and_cleans_up_scratch_download(conn, tmp_path):
    seed_matched_release(conn)
    audio_dir = tmp_path / "audio"
    waveform_dir = tmp_path / "waveforms"
    models_dir = tmp_path / "models"

    def fake_downloader(url, dest_stem):
        dest_stem.parent.mkdir(parents=True, exist_ok=True)
        path = dest_stem.with_suffix(".m4a")
        path.write_bytes(b"fake audio")
        return path

    def fake_analyze_track(audio_path, waveform_out_path, models_dir):
        waveform_out_path.parent.mkdir(parents=True, exist_ok=True)
        waveform_out_path.write_bytes(b"fake png")
        return aa.AnalysisResult(
            bpm=128.0, musical_key="C", musical_key_scale="major",
            mood_scores={"mood_happy": 0.9}, mood_summary="Happy", waveform_path=waveform_out_path,
        )

    results = aa.analyze_release_tracks(
        conn, 1, audio_dir, waveform_dir, models_dir, downloader=fake_downloader, analyze_track_fn=fake_analyze_track
    )

    assert len(results) == 1
    assert results[0]["status"] == "analyzed"
    assert results[0]["bpm"] == 128.0

    track_row = conn.execute("SELECT * FROM tracks WHERE release_id = 1").fetchone()
    assert track_row["bpm"] == 128.0
    assert track_row["musical_key"] == "C"
    assert track_row["mood_summary"] == "Happy"
    assert track_row["analyzed_at"] is not None

    # Scratch download must be cleaned up - nothing left under audio_dir/_scratch.
    scratch_dir = audio_dir / "_scratch"
    assert not scratch_dir.exists() or not any(scratch_dir.rglob("*"))


def test_analyze_release_tracks_reuses_and_keeps_persisted_download(conn, tmp_path):
    seed_matched_release(conn)
    audio_dir = tmp_path / "audio"
    waveform_dir = tmp_path / "waveforms"
    models_dir = tmp_path / "models"

    from discql import audio_download as ad

    persisted_stem = ad.track_audio_stem(audio_dir, 1, "A1", "Rave-O-Lution")
    persisted_stem.parent.mkdir(parents=True, exist_ok=True)
    persisted_file = persisted_stem.with_suffix(".m4a")
    persisted_file.write_bytes(b"already downloaded")

    used_paths = []

    def fake_analyze_track(audio_path, waveform_out_path, models_dir):
        used_paths.append(audio_path)
        waveform_out_path.parent.mkdir(parents=True, exist_ok=True)
        waveform_out_path.write_bytes(b"fake png")
        return aa.AnalysisResult(
            bpm=120.0, musical_key="A", musical_key_scale="minor",
            mood_scores={}, mood_summary="Neutral", waveform_path=waveform_out_path,
        )

    def unreachable_downloader(url, dest_stem):
        raise AssertionError("should not download - a persisted file already exists")

    aa.analyze_release_tracks(
        conn, 1, audio_dir, waveform_dir, models_dir, downloader=unreachable_downloader,
        analyze_track_fn=fake_analyze_track,
    )

    assert used_paths == [persisted_file]
    assert persisted_file.exists()  # never deleted - user asked to keep it


def test_analyze_release_tracks_skips_already_analyzed_unless_forced(conn, tmp_path):
    seed_matched_release(conn)
    audio_dir = tmp_path / "audio"
    waveform_dir = tmp_path / "waveforms"
    models_dir = tmp_path / "models"
    calls = []

    def fake_downloader(url, dest_stem):
        dest_stem.parent.mkdir(parents=True, exist_ok=True)
        path = dest_stem.with_suffix(".m4a")
        path.write_bytes(b"fake audio")
        return path

    def fake_analyze_track(audio_path, waveform_out_path, models_dir):
        calls.append(audio_path)
        waveform_out_path.parent.mkdir(parents=True, exist_ok=True)
        waveform_out_path.write_bytes(b"fake png")
        return aa.AnalysisResult(
            bpm=100.0, musical_key="D", musical_key_scale="major",
            mood_scores={}, mood_summary="Neutral", waveform_path=waveform_out_path,
        )

    kwargs = dict(downloader=fake_downloader, analyze_track_fn=fake_analyze_track)

    aa.analyze_release_tracks(conn, 1, audio_dir, waveform_dir, models_dir, **kwargs)
    assert len(calls) == 1

    aa.analyze_release_tracks(conn, 1, audio_dir, waveform_dir, models_dir, **kwargs)
    assert len(calls) == 1  # not re-analyzed

    aa.analyze_release_tracks(conn, 1, audio_dir, waveform_dir, models_dir, force=True, **kwargs)
    assert len(calls) == 2


def test_clear_analysis_resets_track_fields_and_deletes_waveform_dir(conn, tmp_path):
    seed_matched_release(conn)
    audio_dir = tmp_path / "audio"
    waveform_dir = tmp_path / "waveforms"
    models_dir = tmp_path / "models"

    def fake_downloader(url, dest_stem):
        dest_stem.parent.mkdir(parents=True, exist_ok=True)
        path = dest_stem.with_suffix(".m4a")
        path.write_bytes(b"fake audio")
        return path

    def fake_analyze_track(audio_path, waveform_out_path, models_dir):
        waveform_out_path.parent.mkdir(parents=True, exist_ok=True)
        waveform_out_path.write_bytes(b"fake png")
        return aa.AnalysisResult(
            bpm=128.0, musical_key="C", musical_key_scale="major",
            mood_scores={"mood_happy": 0.9}, mood_summary="Happy", genre="Electronic",
            style="Electronic---Techno", waveform_path=waveform_out_path,
        )

    aa.analyze_release_tracks(
        conn, 1, audio_dir, waveform_dir, models_dir, downloader=fake_downloader, analyze_track_fn=fake_analyze_track
    )

    release_waveform_dir = waveform_dir / "1"
    assert release_waveform_dir.is_dir()
    track_row = conn.execute("SELECT * FROM tracks WHERE release_id = 1").fetchone()
    assert track_row["bpm"] == 128.0
    assert track_row["style"] == "Electronic---Techno"

    aa.clear_analysis(conn, 1, waveform_dir)

    track_row = conn.execute("SELECT * FROM tracks WHERE release_id = 1").fetchone()
    for column in (
        "bpm", "musical_key", "musical_key_scale", "mood_json",
        "mood_summary", "genre", "style", "waveform_path", "analyzed_at",
    ):
        assert track_row[column] is None
    assert not release_waveform_dir.exists()


def test_clear_analysis_on_release_with_no_waveform_dir_does_not_raise(conn, tmp_path):
    seed_matched_release(conn)
    aa.clear_analysis(conn, 1, tmp_path / "waveforms")  # never created - should be a no-op, not an error


def test_analyze_release_tracks_records_failure_without_raising(conn, tmp_path):
    seed_matched_release(conn)
    audio_dir = tmp_path / "audio"
    waveform_dir = tmp_path / "waveforms"
    models_dir = tmp_path / "models"

    def fake_downloader(url, dest_stem):
        dest_stem.parent.mkdir(parents=True, exist_ok=True)
        path = dest_stem.with_suffix(".m4a")
        path.write_bytes(b"fake audio")
        return path

    def failing_analyze_track(audio_path, waveform_out_path, models_dir):
        raise RuntimeError("essentia exploded")

    errors = []
    results = aa.analyze_release_tracks(
        conn,
        1,
        audio_dir,
        waveform_dir,
        models_dir,
        downloader=fake_downloader,
        analyze_track_fn=failing_analyze_track,
        on_error=lambda pos, exc: errors.append((pos, str(exc))),
    )

    assert results[0]["status"] == "failed"
    assert len(errors) == 1
    assert errors[0][0] == "A1"

    track_row = conn.execute("SELECT bpm, analyzed_at FROM tracks WHERE release_id = 1").fetchone()
    assert track_row["bpm"] is None
    assert track_row["analyzed_at"] is None


# --- analyze_release_tracks: local file match takes priority over YouTube ---


def seed_release_no_video(conn, release_id=1):
    """A release with a tracklist but no YouTube video at all - only a
    local file match (set directly on the track row by each test) can be
    an audio source for it."""
    data = ReleaseData(
        id=release_id,
        title="Test Release",
        year=2020,
        genres=[],
        styles=[],
        formats=[],
        discogs_uri=f"https://discogs.com/release/{release_id}",
        cover_image_url=None,
        artists=[ArtistData(id=10, name="Test Artist")],
        labels=[],
        tracks=[TrackData(position="A1", title="Rave-O-Lution", duration="4:30", artist=None)],
        videos=[],
    )
    upsert_release(conn, data, date_added="2024-01-01T00:00:00", now="2024-02-01T00:00:00")


def test_analyze_release_tracks_uses_local_file_when_no_youtube_match_exists(conn, tmp_path):
    seed_release_no_video(conn)
    local_file = tmp_path / "local" / "track.flac"
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_bytes(b"real audio bytes")
    conn.execute("UPDATE tracks SET local_audio_path = ? WHERE release_id = 1", (str(local_file),))
    conn.commit()

    def unreachable_downloader(url, dest_stem):
        raise AssertionError("should not download - a local file match already exists")

    used_paths = []

    def fake_analyze_track(audio_path, waveform_out_path, models_dir):
        used_paths.append(audio_path)
        waveform_out_path.parent.mkdir(parents=True, exist_ok=True)
        waveform_out_path.write_bytes(b"fake png")
        return aa.AnalysisResult(
            bpm=128.0, musical_key="C", musical_key_scale="major",
            mood_scores={}, mood_summary="Neutral", waveform_path=waveform_out_path,
        )

    results = aa.analyze_release_tracks(
        conn, 1, tmp_path / "audio", tmp_path / "waveforms", tmp_path / "models",
        downloader=unreachable_downloader, analyze_track_fn=fake_analyze_track,
    )

    assert results[0]["status"] == "analyzed"
    assert used_paths == [local_file]
    assert local_file.exists()  # never deleted - it's the user's own permanent library file


def test_analyze_release_tracks_prefers_local_file_over_existing_youtube_match(conn, tmp_path):
    seed_matched_release(conn)  # has a selected YouTube match too
    local_file = tmp_path / "local" / "track.flac"
    local_file.parent.mkdir(parents=True, exist_ok=True)
    local_file.write_bytes(b"real audio bytes")
    conn.execute("UPDATE tracks SET local_audio_path = ? WHERE release_id = 1", (str(local_file),))
    conn.commit()

    def unreachable_downloader(url, dest_stem):
        raise AssertionError("should not download - local file must take priority")

    used_paths = []

    def fake_analyze_track(audio_path, waveform_out_path, models_dir):
        used_paths.append(audio_path)
        waveform_out_path.parent.mkdir(parents=True, exist_ok=True)
        waveform_out_path.write_bytes(b"fake png")
        return aa.AnalysisResult(
            bpm=128.0, musical_key="C", musical_key_scale="major",
            mood_scores={}, mood_summary="Neutral", waveform_path=waveform_out_path,
        )

    aa.analyze_release_tracks(
        conn, 1, tmp_path / "audio", tmp_path / "waveforms", tmp_path / "models",
        downloader=unreachable_downloader, analyze_track_fn=fake_analyze_track,
    )

    assert used_paths == [local_file]


def test_analyze_release_tracks_falls_back_to_youtube_when_local_file_missing(conn, tmp_path):
    seed_matched_release(conn)
    # local_audio_path set but the file itself doesn't exist (e.g. an
    # unmounted NFS share, or the file was moved/deleted since matching).
    conn.execute(
        "UPDATE tracks SET local_audio_path = ? WHERE release_id = 1", (str(tmp_path / "gone.flac"),)
    )
    conn.commit()

    def fake_downloader(url, dest_stem):
        dest_stem.parent.mkdir(parents=True, exist_ok=True)
        path = dest_stem.with_suffix(".m4a")
        path.write_bytes(b"fake audio")
        return path

    def fake_analyze_track(audio_path, waveform_out_path, models_dir):
        waveform_out_path.parent.mkdir(parents=True, exist_ok=True)
        waveform_out_path.write_bytes(b"fake png")
        return aa.AnalysisResult(
            bpm=128.0, musical_key="C", musical_key_scale="major",
            mood_scores={}, mood_summary="Neutral", waveform_path=waveform_out_path,
        )

    results = aa.analyze_release_tracks(
        conn, 1, tmp_path / "audio", tmp_path / "waveforms", tmp_path / "models",
        downloader=fake_downloader, analyze_track_fn=fake_analyze_track,
    )

    assert results[0]["status"] == "analyzed"


# --- analyze_library: collection-wide batch --------------------------------


def _fake_downloader(url, dest_stem):
    dest_stem.parent.mkdir(parents=True, exist_ok=True)
    path = dest_stem.with_suffix(".m4a")
    path.write_bytes(b"fake audio")
    return path


def _fake_analyze_track(audio_path, waveform_out_path, models_dir):
    waveform_out_path.parent.mkdir(parents=True, exist_ok=True)
    waveform_out_path.write_bytes(b"fake png")
    return aa.AnalysisResult(
        bpm=128.0, musical_key="C", musical_key_scale="major",
        mood_scores={}, mood_summary="Neutral", waveform_path=waveform_out_path,
    )


# analyze_library() now runs each release in its own OS process
# (ProcessPoolExecutor - see its docstring for why), so fakes injected into
# it can no longer be monkeypatched onto the aa module (a child process
# re-imports aa fresh and never sees the patch) or be closures/lambdas (not
# picklable). They're passed as explicit analyze_track_fn/
# analyze_release_tracks_fn arguments instead, and must stay plain
# module-level functions - see analyze_release_tracks's docstring.


def _selectively_failing_analyze_release_tracks(*args, **kwargs):
    if args[1] == 1:
        raise RuntimeError("boom")
    return []


def _blocking_analyze_release_tracks(barrier, conn, release_id, *a, **kw):
    barrier.wait(timeout=5)  # only passes if all 3 releases are being analyzed at once


def test_count_pending_releases_counts_releases_with_unanalyzed_tracks(conn, db_path, tmp_path):
    seed_matched_release(conn, release_id=1)
    seed_matched_release(conn, release_id=2)

    assert aa.count_pending_releases(conn) == 2
    aa.analyze_library(
        conn, db_path, tmp_path / "audio", tmp_path / "waveforms", tmp_path / "models",
        downloader=_fake_downloader, analyze_track_fn=_fake_analyze_track,
    )
    assert aa.count_pending_releases(conn) == 0


def test_analyze_library_analyzes_every_matched_release(conn, db_path, tmp_path):
    seed_matched_release(conn, release_id=1)
    seed_matched_release(conn, release_id=2)

    progressed = []
    aa.analyze_library(
        conn, db_path, tmp_path / "audio", tmp_path / "waveforms", tmp_path / "models",
        downloader=_fake_downloader, analyze_track_fn=_fake_analyze_track,
        on_progress=lambda i, t, rid: progressed.append((i, t, rid)),
    )

    rows = conn.execute("SELECT release_id, analyzed_at FROM tracks ORDER BY release_id").fetchall()
    assert [r["release_id"] for r in rows] == [1, 2]
    assert all(r["analyzed_at"] is not None for r in rows)
    assert set(progressed) == {(1, 2, 1), (2, 2, 2)}


def test_analyze_library_skips_already_analyzed_releases_unless_forced(conn, db_path, tmp_path):
    seed_matched_release(conn, release_id=1)
    seed_matched_release(conn, release_id=2)

    aa.analyze_library(
        conn, db_path, tmp_path / "audio", tmp_path / "waveforms", tmp_path / "models",
        downloader=_fake_downloader, analyze_track_fn=_fake_analyze_track,
    )

    progressed = []
    aa.analyze_library(
        conn, db_path, tmp_path / "audio", tmp_path / "waveforms", tmp_path / "models",
        downloader=_fake_downloader, analyze_track_fn=_fake_analyze_track,
        on_progress=lambda i, t, rid: progressed.append((i, t, rid)),
    )
    assert progressed == []

    progressed_forced = []
    aa.analyze_library(
        conn, db_path, tmp_path / "audio", tmp_path / "waveforms", tmp_path / "models",
        downloader=_fake_downloader, analyze_track_fn=_fake_analyze_track, force=True,
        on_progress=lambda i, t, rid: progressed_forced.append((i, t, rid)),
    )
    assert set(progressed_forced) == {(1, 2, 1), (2, 2, 2)}


def test_analyze_library_stops_early_when_should_stop_set(conn, db_path, tmp_path):
    seed_matched_release(conn, release_id=1)
    seed_matched_release(conn, release_id=2)

    progressed = []
    aa.analyze_library(
        conn, db_path, tmp_path / "audio", tmp_path / "waveforms", tmp_path / "models",
        downloader=_fake_downloader, analyze_track_fn=_fake_analyze_track,
        on_progress=lambda i, t, rid: progressed.append((i, t, rid)),
        should_stop=lambda: True,
    )

    assert progressed == []
    rows = conn.execute("SELECT analyzed_at FROM tracks").fetchall()
    assert all(r["analyzed_at"] is None for r in rows)


def test_analyze_library_reports_error_and_continues(conn, db_path, tmp_path):
    seed_matched_release(conn, release_id=1)
    seed_matched_release(conn, release_id=2)

    errors = []
    progressed = []
    aa.analyze_library(
        conn, db_path, tmp_path / "audio", tmp_path / "waveforms", tmp_path / "models",
        downloader=_fake_downloader,
        analyze_release_tracks_fn=_selectively_failing_analyze_release_tracks,
        max_workers=1,
        on_error=lambda rid, exc: errors.append((rid, str(exc))),
        on_progress=lambda i, t, rid: progressed.append((i, t, rid)),
    )

    assert errors == [(1, "boom")]
    assert progressed == [(1, 2, 2)]


def test_analyze_library_can_process_releases_concurrently(conn, db_path, tmp_path):
    seed_matched_release(conn, release_id=1)
    seed_matched_release(conn, release_id=2)
    seed_matched_release(conn, release_id=3)

    barrier = multiprocessing.Barrier(3)

    aa.analyze_library(
        conn, db_path, tmp_path / "audio", tmp_path / "waveforms", tmp_path / "models",
        downloader=_fake_downloader, max_workers=3,
        analyze_release_tracks_fn=functools.partial(_blocking_analyze_release_tracks, barrier),
    )


def test_analyze_library_stops_after_current_release_finishes(conn, db_path, tmp_path):
    seed_matched_release(conn, release_id=1)
    seed_matched_release(conn, release_id=2)
    seed_matched_release(conn, release_id=3)

    stop_flag = {"stop": False}

    def progress(i, t, rid):
        stop_flag["stop"] = True  # request stop right after the first release completes

    aa.analyze_library(
        conn, db_path, tmp_path / "audio", tmp_path / "waveforms", tmp_path / "models",
        downloader=_fake_downloader, analyze_track_fn=_fake_analyze_track, max_workers=1,
        should_stop=lambda: stop_flag["stop"], on_progress=progress,
    )

    rows = conn.execute("SELECT release_id, analyzed_at FROM tracks").fetchall()
    analyzed_releases = {r["release_id"] for r in rows if r["analyzed_at"] is not None}
    assert len(analyzed_releases) == 1
