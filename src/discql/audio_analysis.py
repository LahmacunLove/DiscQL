from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sqlite3
import subprocess
import tempfile
import threading
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from discql import audio_download, db

# Called after each track is analyzed/skipped: (index, total, position)
ProgressCallback = Callable[[int, int, str | None], None]
# Called when a single track's analysis fails: (position, exception)
ErrorCallback = Callable[[str | None, Exception], None]


def _init_analysis_worker() -> None:
    """ProcessPoolExecutor `initializer` for analyze_library's worker pool -
    runs once per spawned worker process, before that worker's first task
    (and therefore before essentia/TensorFlow is ever imported there - both
    are imported lazily inside functions, not at module load time).

    Caps TensorFlow's own internal thread pool to 1 thread. Each worker
    process already gets real OS-level parallelism from ProcessPoolExecutor
    itself (see analyze_library's docstring for why processes rather than
    threads); on top of that, TensorFlow's default SessionOptions spins up
    its own internal thread pool sized to the machine's core count -
    confirmed concretely: a single TensorflowPredictEffnetDiscogs call in
    one process spun up 25 OS threads on this 8-core machine. With
    max_workers processes each doing that at once, the machine ends up
    massively oversubscribed (load average measured at ~20 on an 8-core
    machine with MAX_WORKERS=7, right after switching analyze_library to
    processes). Setting these before the first import brings a single
    worker's thread count down to 4 (verified directly), with no measured
    slowdown per inference - the process-level parallelism above is doing
    the actual multi-core work now, so each individual TensorFlow call
    doesn't need to additionally fan out across cores itself.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")


def _silence_essentia_warnings() -> None:
    """essentia.log.warningActive = False - the "No network created, or
    last created network has been deleted..." warning fires under
    concurrent multi-thread use (RhythmExtractor2013/KeyExtractor running
    alongside the TensorflowPredict-based models in _ModelCache), but
    verified directly (2026-08-20) that it doesn't affect correctness -
    diffed 7 concurrent workers' full BPM/key/mood/genre output against a
    known-good sequential run for the same track, 0/7 mismatches. Global
    switch (all Essentia warnings), not scoped to just this one.

    Also silences infoActive (e.g. "Successfully loaded graph file: ...")
    - harmless and initially left on deliberately (see git history), but
    with one _ModelCache per worker *process* now (see analyze_library's
    docstring for why processes, not threads) each of the 8 models logs
    this once per worker at the start of every run, which multiplied by
    MAX_WORKERS is enough log spam to bury everything else. errorActive
    stays on - real errors should still show.
    """
    import essentia

    essentia.log.warningActive = False
    essentia.log.infoActive = False


def count_pending_releases(conn: sqlite3.Connection) -> int:
    """How many releases a non-force analyze_library() run would still
    touch - same "at least one still-unanalyzed matched track" condition
    used to skip releases by default. "Matched" means either a selected
    YouTube match or a confident local file match (local_audio.py) - either
    is a usable audio source for analysis, see analyze_release_tracks."""
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT r.id) AS n
        FROM releases r
        JOIN tracks t ON t.release_id = r.id
        LEFT JOIN track_video_matches tvm ON tvm.track_id = t.id AND tvm.is_selected = 1
        WHERE r.removed_from_discogs_at IS NULL AND t.analyzed_at IS NULL
          AND (tvm.video_id IS NOT NULL OR t.local_audio_path IS NOT NULL)
        """
    ).fetchone()
    return row["n"]


ESSENTIA_MODELS_BASE_URL = "https://essentia.upf.edu/models"

# Embedding backbone (verified live against essentia.upf.edu/models/
# feature-extractors/discogs-effnet/discogs-effnet-bs64-1.json's "schema"):
# TensorflowPredictEffnetDiscogs computes its own mel-spectrogram internally
# from 16kHz audio, output "PartitionedCall:1" is the 1280-dim
# penultimate-layer embedding all EFFNET_CLASSIFIERS heads below are trained
# on. Kept as its own named constant/URL (rather than folded into
# classify_mood) so a future switch to Discogs-MAEST - a newer,
# better-performing embedding per Essentia's own docs, but one that (as of
# writing) only has published genre heads, not mood ones - only needs a new
# embedding-worker script plus swapping this out, not touching the
# head-scoring logic below.
EFFNET_EMBEDDING_URL_DIR = "feature-extractors/discogs-effnet"
EFFNET_EMBEDDING_MODEL = "discogs-effnet-bs64-1"
EFFNET_EMBEDDING_OUTPUT = "PartitionedCall:1"

# Classification heads trained on the embedding above (TensorflowPredict2D -
# verified live against essentia.upf.edu/models/classification-heads/<name>/
# <name>-discogs-effnet-1.json for each). input_layer/output_layer aren't
# uniform across heads: the 2022 mood heads use "model/Placeholder" ->
# "model/Softmax", but genre_discogs400 (a 2023 export, SavedModel-style
# signature naming) uses "serving_default_model_Placeholder" ->
# "PartitionedCall:0" (a Sigmoid, not Softmax - independent per-style scores
# rather than a single mutually-exclusive pick, which suits a track
# plausibly matching multiple styles). Same binary-collapse convention as
# before: two classes, one of them "non_X"/"not_X", collapse to a single
# named score; genre_discogs400 (400 classes, no such negation) keeps one
# score per class - it's Discogs' own "Genre---Style" taxonomy (e.g.
# "Electronic---Deep House"), the same vocabulary already stored per release
# from the Discogs API itself (genres/styles), just predicted from audio
# instead of read from Discogs.
@dataclass(frozen=True)
class EffnetHead:
    classes: list[str]
    input_layer: str = "model/Placeholder"
    output_layer: str = "model/Softmax"


EFFNET_CLASSIFIERS: dict[str, EffnetHead] = {
    "danceability": EffnetHead(["danceable", "not_danceable"]),
    "mood_happy": EffnetHead(["happy", "non_happy"]),
    "mood_sad": EffnetHead(["non_sad", "sad"]),
    "mood_relaxed": EffnetHead(["non_relaxed", "relaxed"]),
    "mood_aggressive": EffnetHead(["aggressive", "not_aggressive"]),
    "mood_party": EffnetHead(["non_party", "party"]),
    "genre_discogs400": EffnetHead(
        input_layer="serving_default_model_Placeholder",
        output_layer="PartitionedCall:0",
        classes=[
        "Blues---Boogie Woogie", "Blues---Chicago Blues", "Blues---Country Blues",
        "Blues---Delta Blues", "Blues---Electric Blues", "Blues---Harmonica Blues",
        "Blues---Jump Blues", "Blues---Louisiana Blues", "Blues---Modern Electric Blues",
        "Blues---Piano Blues", "Blues---Rhythm & Blues", "Blues---Texas Blues",
        "Brass & Military---Brass Band", "Brass & Military---Marches",
        "Brass & Military---Military", "Children's---Educational",
        "Children's---Nursery Rhymes", "Children's---Story", "Classical---Baroque",
        "Classical---Choral", "Classical---Classical", "Classical---Contemporary",
        "Classical---Impressionist", "Classical---Medieval", "Classical---Modern",
        "Classical---Neo-Classical", "Classical---Neo-Romantic", "Classical---Opera",
        "Classical---Post-Modern", "Classical---Renaissance", "Classical---Romantic",
        "Electronic---Abstract", "Electronic---Acid", "Electronic---Acid House",
        "Electronic---Acid Jazz", "Electronic---Ambient", "Electronic---Bassline",
        "Electronic---Beatdown", "Electronic---Berlin-School", "Electronic---Big Beat",
        "Electronic---Bleep", "Electronic---Breakbeat", "Electronic---Breakcore",
        "Electronic---Breaks", "Electronic---Broken Beat", "Electronic---Chillwave",
        "Electronic---Chiptune", "Electronic---Dance-pop", "Electronic---Dark Ambient",
        "Electronic---Darkwave", "Electronic---Deep House", "Electronic---Deep Techno",
        "Electronic---Disco", "Electronic---Disco Polo", "Electronic---Donk",
        "Electronic---Downtempo", "Electronic---Drone", "Electronic---Drum n Bass",
        "Electronic---Dub", "Electronic---Dub Techno", "Electronic---Dubstep",
        "Electronic---Dungeon Synth", "Electronic---EBM", "Electronic---Electro",
        "Electronic---Electro House", "Electronic---Electroclash", "Electronic---Euro House",
        "Electronic---Euro-Disco", "Electronic---Eurobeat", "Electronic---Eurodance",
        "Electronic---Experimental", "Electronic---Freestyle", "Electronic---Future Jazz",
        "Electronic---Gabber", "Electronic---Garage House", "Electronic---Ghetto",
        "Electronic---Ghetto House", "Electronic---Glitch", "Electronic---Goa Trance",
        "Electronic---Grime", "Electronic---Halftime", "Electronic---Hands Up",
        "Electronic---Happy Hardcore", "Electronic---Hard House", "Electronic---Hard Techno",
        "Electronic---Hard Trance", "Electronic---Hardcore", "Electronic---Hardstyle",
        "Electronic---Hi NRG", "Electronic---Hip Hop", "Electronic---Hip-House",
        "Electronic---House", "Electronic---IDM", "Electronic---Illbient",
        "Electronic---Industrial", "Electronic---Italo House", "Electronic---Italo-Disco",
        "Electronic---Italodance", "Electronic---Jazzdance", "Electronic---Juke",
        "Electronic---Jumpstyle", "Electronic---Jungle", "Electronic---Latin",
        "Electronic---Leftfield", "Electronic---Makina", "Electronic---Minimal",
        "Electronic---Minimal Techno", "Electronic---Modern Classical",
        "Electronic---Musique Concrète", "Electronic---Neofolk", "Electronic---New Age",
        "Electronic---New Beat", "Electronic---New Wave", "Electronic---Noise",
        "Electronic---Nu-Disco", "Electronic---Power Electronics",
        "Electronic---Progressive Breaks", "Electronic---Progressive House",
        "Electronic---Progressive Trance", "Electronic---Psy-Trance",
        "Electronic---Rhythmic Noise", "Electronic---Schranz", "Electronic---Sound Collage",
        "Electronic---Speed Garage", "Electronic---Speedcore", "Electronic---Synth-pop",
        "Electronic---Synthwave", "Electronic---Tech House", "Electronic---Tech Trance",
        "Electronic---Techno", "Electronic---Trance", "Electronic---Tribal",
        "Electronic---Tribal House", "Electronic---Trip Hop", "Electronic---Tropical House",
        "Electronic---UK Garage", "Electronic---Vaporwave", "Folk, World, & Country---African",
        "Folk, World, & Country---Bluegrass", "Folk, World, & Country---Cajun",
        "Folk, World, & Country---Canzone Napoletana",
        "Folk, World, & Country---Catalan Music", "Folk, World, & Country---Celtic",
        "Folk, World, & Country---Country", "Folk, World, & Country---Fado",
        "Folk, World, & Country---Flamenco", "Folk, World, & Country---Folk",
        "Folk, World, & Country---Gospel", "Folk, World, & Country---Highlife",
        "Folk, World, & Country---Hillbilly", "Folk, World, & Country---Hindustani",
        "Folk, World, & Country---Honky Tonk", "Folk, World, & Country---Indian Classical",
        "Folk, World, & Country---Laïkó", "Folk, World, & Country---Nordic",
        "Folk, World, & Country---Pacific", "Folk, World, & Country---Polka",
        "Folk, World, & Country---Raï", "Folk, World, & Country---Romani",
        "Folk, World, & Country---Soukous", "Folk, World, & Country---Séga",
        "Folk, World, & Country---Volksmusik", "Folk, World, & Country---Zouk",
        "Folk, World, & Country---Éntekhno", "Funk / Soul---Afrobeat", "Funk / Soul---Boogie",
        "Funk / Soul---Contemporary R&B", "Funk / Soul---Disco", "Funk / Soul---Free Funk",
        "Funk / Soul---Funk", "Funk / Soul---Gospel", "Funk / Soul---Neo Soul",
        "Funk / Soul---New Jack Swing", "Funk / Soul---P.Funk", "Funk / Soul---Psychedelic",
        "Funk / Soul---Rhythm & Blues", "Funk / Soul---Soul", "Funk / Soul---Swingbeat",
        "Funk / Soul---UK Street Soul", "Hip Hop---Bass Music", "Hip Hop---Boom Bap",
        "Hip Hop---Bounce", "Hip Hop---Britcore", "Hip Hop---Cloud Rap", "Hip Hop---Conscious",
        "Hip Hop---Crunk", "Hip Hop---Cut-up/DJ", "Hip Hop---DJ Battle Tool",
        "Hip Hop---Electro", "Hip Hop---G-Funk", "Hip Hop---Gangsta", "Hip Hop---Grime",
        "Hip Hop---Hardcore Hip-Hop", "Hip Hop---Horrorcore", "Hip Hop---Instrumental",
        "Hip Hop---Jazzy Hip-Hop", "Hip Hop---Miami Bass", "Hip Hop---Pop Rap",
        "Hip Hop---Ragga HipHop", "Hip Hop---RnB/Swing", "Hip Hop---Screw",
        "Hip Hop---Thug Rap", "Hip Hop---Trap", "Hip Hop---Trip Hop", "Hip Hop---Turntablism",
        "Jazz---Afro-Cuban Jazz", "Jazz---Afrobeat", "Jazz---Avant-garde Jazz",
        "Jazz---Big Band", "Jazz---Bop", "Jazz---Bossa Nova", "Jazz---Contemporary Jazz",
        "Jazz---Cool Jazz", "Jazz---Dixieland", "Jazz---Easy Listening",
        "Jazz---Free Improvisation", "Jazz---Free Jazz", "Jazz---Fusion", "Jazz---Gypsy Jazz",
        "Jazz---Hard Bop", "Jazz---Jazz-Funk", "Jazz---Jazz-Rock", "Jazz---Latin Jazz",
        "Jazz---Modal", "Jazz---Post Bop", "Jazz---Ragtime", "Jazz---Smooth Jazz",
        "Jazz---Soul-Jazz", "Jazz---Space-Age", "Jazz---Swing", "Latin---Afro-Cuban",
        "Latin---Baião", "Latin---Batucada", "Latin---Beguine", "Latin---Bolero",
        "Latin---Boogaloo", "Latin---Bossanova", "Latin---Cha-Cha", "Latin---Charanga",
        "Latin---Compas", "Latin---Cubano", "Latin---Cumbia", "Latin---Descarga",
        "Latin---Forró", "Latin---Guaguancó", "Latin---Guajira", "Latin---Guaracha",
        "Latin---MPB", "Latin---Mambo", "Latin---Mariachi", "Latin---Merengue",
        "Latin---Norteño", "Latin---Nueva Cancion", "Latin---Pachanga", "Latin---Porro",
        "Latin---Ranchera", "Latin---Reggaeton", "Latin---Rumba", "Latin---Salsa",
        "Latin---Samba", "Latin---Son", "Latin---Son Montuno", "Latin---Tango",
        "Latin---Tejano", "Latin---Vallenato", "Non-Music---Audiobook", "Non-Music---Comedy",
        "Non-Music---Dialogue", "Non-Music---Education", "Non-Music---Field Recording",
        "Non-Music---Interview", "Non-Music---Monolog", "Non-Music---Poetry",
        "Non-Music---Political", "Non-Music---Promotional", "Non-Music---Radioplay",
        "Non-Music---Religious", "Non-Music---Spoken Word", "Pop---Ballad", "Pop---Bollywood",
        "Pop---Bubblegum", "Pop---Chanson", "Pop---City Pop", "Pop---Europop",
        "Pop---Indie Pop", "Pop---J-pop", "Pop---K-pop", "Pop---Kayōkyoku",
        "Pop---Light Music", "Pop---Music Hall", "Pop---Novelty", "Pop---Parody",
        "Pop---Schlager", "Pop---Vocal", "Reggae---Calypso", "Reggae---Dancehall",
        "Reggae---Dub", "Reggae---Lovers Rock", "Reggae---Ragga", "Reggae---Reggae",
        "Reggae---Reggae-Pop", "Reggae---Rocksteady", "Reggae---Roots Reggae", "Reggae---Ska",
        "Reggae---Soca", "Rock---AOR", "Rock---Acid Rock", "Rock---Acoustic",
        "Rock---Alternative Rock", "Rock---Arena Rock", "Rock---Art Rock",
        "Rock---Atmospheric Black Metal", "Rock---Avantgarde", "Rock---Beat",
        "Rock---Black Metal", "Rock---Blues Rock", "Rock---Brit Pop", "Rock---Classic Rock",
        "Rock---Coldwave", "Rock---Country Rock", "Rock---Crust", "Rock---Death Metal",
        "Rock---Deathcore", "Rock---Deathrock", "Rock---Depressive Black Metal",
        "Rock---Doo Wop", "Rock---Doom Metal", "Rock---Dream Pop", "Rock---Emo",
        "Rock---Ethereal", "Rock---Experimental", "Rock---Folk Metal", "Rock---Folk Rock",
        "Rock---Funeral Doom Metal", "Rock---Funk Metal", "Rock---Garage Rock", "Rock---Glam",
        "Rock---Goregrind", "Rock---Goth Rock", "Rock---Gothic Metal", "Rock---Grindcore",
        "Rock---Grunge", "Rock---Hard Rock", "Rock---Hardcore", "Rock---Heavy Metal",
        "Rock---Indie Rock", "Rock---Industrial", "Rock---Krautrock", "Rock---Lo-Fi",
        "Rock---Lounge", "Rock---Math Rock", "Rock---Melodic Death Metal",
        "Rock---Melodic Hardcore", "Rock---Metalcore", "Rock---Mod", "Rock---Neofolk",
        "Rock---New Wave", "Rock---No Wave", "Rock---Noise", "Rock---Noisecore",
        "Rock---Nu Metal", "Rock---Oi", "Rock---Parody", "Rock---Pop Punk", "Rock---Pop Rock",
        "Rock---Pornogrind", "Rock---Post Rock", "Rock---Post-Hardcore", "Rock---Post-Metal",
        "Rock---Post-Punk", "Rock---Power Metal", "Rock---Power Pop", "Rock---Power Violence",
        "Rock---Prog Rock", "Rock---Progressive Metal", "Rock---Psychedelic Rock",
        "Rock---Psychobilly", "Rock---Pub Rock", "Rock---Punk", "Rock---Rock & Roll",
        "Rock---Rockabilly", "Rock---Shoegaze", "Rock---Ska", "Rock---Sludge Metal",
        "Rock---Soft Rock", "Rock---Southern Rock", "Rock---Space Rock", "Rock---Speed Metal",
        "Rock---Stoner Rock", "Rock---Surf", "Rock---Symphonic Rock",
        "Rock---Technical Death Metal", "Rock---Thrash", "Rock---Twist", "Rock---Viking Metal",
        "Rock---Yé-Yé", "Stage & Screen---Musical", "Stage & Screen---Score",
        "Stage & Screen---Soundtrack", "Stage & Screen---Theme",
        ],
    ),
}

_MOOD_LABELS = {
    "danceability": "Danceable",
    "mood_happy": "Happy",
    "mood_sad": "Sad",
    "mood_relaxed": "Relaxed",
    "mood_aggressive": "Aggressive",
    "mood_party": "Party",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AnalysisResult:
    bpm: float | None
    musical_key: str | None
    musical_key_scale: str | None
    mood_scores: dict[str, float]
    mood_summary: str
    waveform_path: Path
    genre: str | None = None
    style: str | None = None


def track_waveform_path(waveform_dir: Path, release_id: int, position: str | None, title: str) -> Path:
    label = f"{position} - {title}" if position else title
    return waveform_dir / str(release_id) / f"{audio_download.sanitize_filename(label)}.png"


def decode_audio(audio_path: Path, sample_rate: int) -> np.ndarray:
    """Decode+downmix-to-mono+resample via ffmpeg, straight to an in-memory
    float32 PCM array (same normalized [-1, 1] range Essentia's own
    MonoLoader produces). Essentia's bundled audio decoder doesn't reliably
    support every codec yt-dlp can hand back (confirmed concretely: it
    raises "Unsupported codec!" on a downloaded Opus-in-WebM file, a very
    common yt-dlp bestaudio format) - ffmpeg does support it, so it's used
    for every decode in this module rather than Essentia's loader. Centralized
    here (rather than one ffmpeg/MonoLoader call per consumer) so a given
    audio file is only actually decoded once per sample rate it's needed at,
    not once per downstream step (BPM/key, each mood/genre classifier,
    waveform).
    """
    import numpy as np

    result = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(audio_path),
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1", "-ar", str(sample_rate), "-",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype=np.float32)


def extract_bpm_and_key(samples_44100: np.ndarray) -> tuple[float, str, str]:
    """BPM via RhythmExtractor2013, key/scale via KeyExtractor. Both require
    44100 Hz audio (RhythmExtractor2013 explicitly documents this) - expects
    samples_44100 to already be decoded/resampled to that rate (see
    decode_audio / analyze_track).
    """
    import essentia.standard as es

    _silence_essentia_warnings()
    bpm, _ticks, _confidence, _estimates, _intervals = es.RhythmExtractor2013(method="multifeature")(samples_44100)
    key, scale, _strength = es.KeyExtractor()(samples_44100)
    return float(bpm), key, scale


def _ensure_model(filename: str, url: str, models_dir: Path) -> Path:
    path = models_dir / f"{filename}.pb"
    if path.exists():
        return path

    models_dir.mkdir(parents=True, exist_ok=True)
    # A unique tmp filename per call, not a fixed f"{filename}.pb.tmp" -
    # "Run All" can have multiple releases' worker threads hit this
    # concurrently for the same still-missing model, and two curl processes
    # writing to the same tmp path would corrupt each other's download. The
    # final rename can still race harmlessly (same content either way).
    fd, tmp_name = tempfile.mkstemp(dir=models_dir, prefix=f"{filename}.", suffix=".pb.tmp")
    os.close(fd)
    tmp_path = Path(tmp_name)
    # Not requests/urllib: both reproducibly hang reading this server's
    # response for large files (confirmed down to a raw ssl.SSLSocket.recv()
    # stall near the end of the transfer, identical TLS version/cipher as a
    # successful curl request - looks like a Python 3.14 `ssl` module bug,
    # not anything about this codebase or the server). curl reliably works.
    try:
        subprocess.run(["curl", "-sS", "--fail", "-o", str(tmp_path), url], check=True, timeout=120)
        tmp_path.rename(path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return path


def _ensure_embedding_model(models_dir: Path) -> Path:
    url = f"{ESSENTIA_MODELS_BASE_URL}/{EFFNET_EMBEDDING_URL_DIR}/{EFFNET_EMBEDDING_MODEL}.pb"
    return _ensure_model(EFFNET_EMBEDDING_MODEL, url, models_dir)


def _ensure_head_model(name: str, models_dir: Path) -> Path:
    filename = f"{name}-discogs-effnet-1"
    url = f"{ESSENTIA_MODELS_BASE_URL}/classification-heads/{name}/{filename}.pb"
    return _ensure_model(filename, url, models_dir)


def _collapse_predictions(name: str, classes: list[str], averaged: list[float]) -> dict[str, float]:
    """A binary classifier where one class is the "non_X"/"not_X" negation of
    the other collapses to a single named score (the positive class's
    probability); a genuine multi-class classifier (e.g. genre, no such
    negation) keeps one score per class instead.
    """
    negative_prefixes = ("non_", "not_")
    if len(classes) == 2 and any(c.startswith(negative_prefixes) for c in classes):
        positive_index = next(i for i, c in enumerate(classes) if not c.startswith(negative_prefixes))
        return {name: float(averaged[positive_index])}
    return {f"{name}:{cls}": float(averaged[i]) for i, cls in enumerate(classes)}


class _ModelCache:
    """Lazily loads each Essentia/TensorFlow model at most once, then keeps
    the loaded algorithm object around for reuse - see classify_mood for why
    this is safe and _thread_local_cache for why one of these lives per
    worker thread rather than being shared or rebuilt per track."""

    def __init__(self, models_dir: Path):
        _silence_essentia_warnings()
        self.models_dir = models_dir
        self._embedding_algo = None
        self._head_algos: dict[str, object] = {}

    def embed(self, samples_16000):
        import essentia.standard as es

        if self._embedding_algo is None:
            model_path = _ensure_embedding_model(self.models_dir)
            self._embedding_algo = es.TensorflowPredictEffnetDiscogs(
                graphFilename=str(model_path), output=EFFNET_EMBEDDING_OUTPUT
            )
        return self._embedding_algo(samples_16000)

    def predict_head(self, name: str, head: EffnetHead, embeddings) -> list[float]:
        import numpy as np
        import essentia.standard as es

        algo = self._head_algos.get(name)
        if algo is None:
            model_path = _ensure_head_model(name, self.models_dir)
            algo = es.TensorflowPredict2D(
                graphFilename=str(model_path), input=head.input_layer, output=head.output_layer
            )
            self._head_algos[name] = algo
        predictions = algo(embeddings)
        return np.mean(predictions, axis=0).tolist()


_thread_local_cache = threading.local()


def _get_model_cache(models_dir: Path) -> _ModelCache:
    """One _ModelCache per OS thread. In analyze_library's case each worker
    is a separate long-lived OS *process* (ProcessPoolExecutor) that runs one
    task at a time on its own main thread, so this still amounts to one
    cache per worker, reused across every release that worker picks up
    during the run - concurrent.futures pool workers are long-lived across
    submitted tasks either way. A fresh cache per models_dir, in the
    unlikely case that changes mid-run.
    """
    cache = getattr(_thread_local_cache, "cache", None)
    if cache is None or cache.models_dir != models_dir:
        cache = _ModelCache(models_dir)
        _thread_local_cache.cache = cache
    return cache


def classify_mood(samples_44100: np.ndarray, models_dir: Path) -> dict[str, float]:
    """Each model - the embedding extractor and each classification head -
    used to run in its own freshly spawned `python3` subprocess, because
    loading multiple *different* TensorFlow graphs sequentially in one
    process was found to reproducibly hang after about two loads. Verified
    directly (2026-08-20) that this no longer reproduces with the currently
    installed essentia-tensorflow: loading the embedding extractor plus all
    7 classification heads sequentially in one process, repeated across
    several tracks, ran cleanly with no hang - so models are now loaded
    in-process via _get_model_cache()/_ModelCache instead, cutting a real
    ~6s/track (per-subprocess Python+Essentia+TensorFlow import and graph
    load, timed against a real track) down to a small fraction of that once
    a worker's cache is warm. If this ever needs reverting, the prior
    subprocess-per-model implementation is in git history.

    Loading models in-process also means the essentia C extension - which
    never releases the GIL during its algorithm calls, confirmed via `nm -D`
    showing no PyEval_SaveThread/Py_BEGIN_ALLOW_THREADS symbols in
    _essentia*.so - blocks every other Python thread in the same process
    for the full duration of each TensorFlow inference call. That made
    analyze_library's old ThreadPoolExecutor-based parallelism collapse
    to near-sequential (confirmed concretely: 7 workers configured, but the
    process measured ~200% CPU on an 8-core machine, not ~700%) - see
    analyze_library's docstring for why it now uses processes instead.

    Two stages: the embedding extractor (TensorflowPredictEffnetDiscogs, the
    one expensive full-audio forward pass) runs once, producing a
    [n_patches, 1280] embedding matrix; each of EFFNET_CLASSIFIERS' heads
    (TensorflowPredict2D, cheap - just a small dense classifier over that
    embedding) then runs against it.

    The 44.1kHz -> 16kHz resample (both the embedding extractor and every
    head's classes require 16kHz input) happens here via Essentia's own
    Resample algorithm.
    """
    import essentia.standard as es

    samples_16000 = es.Resample(inputSampleRate=44100, outputSampleRate=16000)(samples_44100)

    cache = _get_model_cache(models_dir)
    embeddings = cache.embed(samples_16000)

    scores: dict[str, float] = {}
    for name, head in EFFNET_CLASSIFIERS.items():
        averaged = cache.predict_head(name, head, embeddings)
        scores.update(_collapse_predictions(name, head.classes, averaged))

    return scores


def summarize_mood(scores: dict[str, float], threshold: float = 0.5, max_tags: int = 3) -> str:
    binary_scores = {k: v for k, v in scores.items() if ":" not in k}
    top = sorted((kv for kv in binary_scores.items() if kv[1] >= threshold), key=lambda kv: -kv[1])
    tags = [_MOOD_LABELS.get(k, k) for k, _ in top[:max_tags]]
    return ", ".join(tags) if tags else "Neutral"


def top_style(scores: dict[str, float]) -> str | None:
    """Full "Genre---Style" label (Discogs' own taxonomy - e.g. "Electronic---
    Deep House") of the highest-scoring class in the genre_discogs400 head.
    """
    style_scores = {k.split(":", 1)[1]: v for k, v in scores.items() if k.startswith("genre_discogs400:")}
    if not style_scores:
        return None
    return max(style_scores.items(), key=lambda kv: kv[1])[0]


def top_genre(scores: dict[str, float]) -> str | None:
    """Broad Discogs genre - the part before "---" in top_style()'s pick."""
    style = top_style(scores)
    return style.split("---", 1)[0] if style else None


WAVEFORM_COLOR = (106, 176, 255, 255)  # matches the web UI's --accent blue


def generate_waveform(samples_44100: np.ndarray, out_path: Path, size: tuple[int, int] = (2500, 250)) -> Path:
    """Render a waveform PNG from already-decoded PCM (Pillow line drawing,
    avoiding a system dependency beyond ffmpeg, which is already required
    elsewhere). Drawn in the UI's accent color, not black - a black line on
    transparent background (the old project's approach) would be invisible
    against this app's dark theme.

    Takes the same 44.1kHz array analyze_track() also feeds to BPM/key
    extraction rather than decoding audio_path separately at 22050 Hz - the
    per-pixel min/max downsampling below already collapses far more samples
    per pixel than the difference between those two rates would matter for,
    so reusing the 44.1kHz decode changes nothing visible.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    width, height = size
    samples = samples_44100
    if len(samples) == 0:
        samples = np.zeros(1, dtype=np.float32)

    chunk_size = max(1, len(samples) // width)
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    mid = height // 2
    scale = height / 2  # samples are float32 PCM normalized to [-1, 1]

    for x in range(width):
        chunk = samples[x * chunk_size : (x + 1) * chunk_size]
        if len(chunk) == 0:
            continue
        y1 = mid - int(float(chunk.max()) * scale)
        y2 = mid - int(float(chunk.min()) * scale)
        if y1 == y2:
            y2 += 1
        draw.line([(x, y1), (x, y2)], fill=WAVEFORM_COLOR, width=1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path


def analyze_track(audio_path: Path, waveform_out_path: Path, models_dir: Path) -> AnalysisResult:
    samples_44100 = decode_audio(audio_path, 44100)
    bpm, key, scale = extract_bpm_and_key(samples_44100)
    mood_scores = classify_mood(samples_44100, models_dir)
    mood_summary = summarize_mood(mood_scores)
    genre = top_genre(mood_scores)
    style = top_style(mood_scores)
    generate_waveform(samples_44100, waveform_out_path)
    return AnalysisResult(
        bpm=bpm,
        musical_key=key,
        musical_key_scale=scale,
        mood_scores=mood_scores,
        mood_summary=mood_summary,
        genre=genre,
        style=style,
        waveform_path=waveform_out_path,
    )


def clear_analysis(conn: sqlite3.Connection, release_id: int, waveform_dir: Path) -> None:
    """Wipe analysis results for a release back to a pristine "never
    analyzed" state - resets bpm/musical_key/musical_key_scale/mood_json/
    mood_summary/genre/style/waveform_path/analyzed_at to NULL on its
    tracks, and deletes the waveform PNGs from disk (regenerable analysis
    output, not something to keep around once its DB reference to it is
    gone - same reasoning as analysis-only audio downloads being deleted
    after use, see analyze_release_tracks). Does NOT touch matched videos
    or downloaded audio.
    """
    with conn:
        conn.execute(
            """
            UPDATE tracks
            SET bpm = NULL, musical_key = NULL, musical_key_scale = NULL, mood_json = NULL,
                mood_summary = NULL, genre = NULL, style = NULL, waveform_path = NULL, analyzed_at = NULL
            WHERE release_id = ?
            """,
            (release_id,),
        )
    release_waveform_dir = waveform_dir / str(release_id)
    if release_waveform_dir.is_dir():
        shutil.rmtree(release_waveform_dir, ignore_errors=True)


def _analyzable_tracks(conn: sqlite3.Connection, release_id: int, force: bool) -> list[sqlite3.Row]:
    # LEFT JOINs, not INNER: a track needs *either* a selected YouTube match
    # or a confident local file match (local_audio_path, see local_audio.py)
    # to be analyzable, not necessarily both - see analyze_release_tracks
    # for how local_audio_path takes priority over rv.url when both exist.
    query = """
        SELECT t.id, t.position, t.title, rv.url, t.local_audio_path
        FROM tracks t
        LEFT JOIN track_video_matches tvm ON tvm.track_id = t.id AND tvm.is_selected = 1
        LEFT JOIN release_videos rv ON rv.id = tvm.video_id
        WHERE t.release_id = ? AND (rv.url IS NOT NULL OR t.local_audio_path IS NOT NULL)
    """
    if not force:
        query += " AND t.analyzed_at IS NULL"
    query += " ORDER BY t.id"
    return conn.execute(query, (release_id,)).fetchall()


def analyze_release_tracks(
    conn: sqlite3.Connection,
    release_id: int,
    audio_dir: Path,
    waveform_dir: Path,
    models_dir: Path,
    downloader: audio_download.VideoDownloader = audio_download.default_video_downloader,
    force: bool = False,
    on_progress: ProgressCallback | None = None,
    on_error: ErrorCallback | None = None,
    analyze_track_fn: Callable[[Path, Path, Path], AnalysisResult] = analyze_track,
) -> list[dict]:
    """Analyze (BPM/key/mood/waveform) every track in the release that has a
    usable audio source - either a primary matched video
    (track_video_matches.is_selected=1) or a confident local file match
    (tracks.local_audio_path, see local_audio.py) - and isn't already
    analyzed (unless force=True). A local match always takes priority over
    YouTube when both exist (see the per-track loop below) - it's the
    user's own permanent library file, not something to discard in favor of
    a lower-quality YouTube rip. A track whose audio was already downloaded
    via the standalone "Download tracks" button is reused as-is and left
    alone afterward (the user asked to keep it); anything analysis
    downloads itself goes to a scratch subdirectory and is deleted once
    analysis completes - per CLAUDE.md, only the analysis results are kept
    long-term, not the audio. A matched local file is never deleted either
    way - same treatment as an already-persisted download.

    analyze_track_fn defaults to the real analyze_track - overridable (e.g.
    in tests) same as downloader already is. Needs to stay a plain
    module-level function when analyze_library() submits this whole call
    across a ProcessPoolExecutor boundary (see analyze_library's docstring)
    - a monkeypatched aa.analyze_track wouldn't reach the child process, and
    a closure/lambda isn't picklable at all.
    """
    tracks = _analyzable_tracks(conn, release_id, force)
    total = len(tracks)
    if total:
        title_row = conn.execute("SELECT title FROM releases WHERE id = ?", (release_id,)).fetchone()
        release_title = title_row["title"] if title_row else "?"
        print(f"[audio_analysis] Analyzing release {release_id} ({release_title}) - {total} track{'s' if total != 1 else ''}")
    results = []
    now = _utcnow_iso()

    for index, track in enumerate(tracks, start=1):
        position, title, url = track["position"], track["title"], track["url"]
        local_audio_path = track["local_audio_path"]
        persisted_stem = audio_download.track_audio_stem(audio_dir, release_id, position, title)
        existing = audio_download.find_existing_download(persisted_stem)

        try:
            local_path = Path(local_audio_path) if local_audio_path else None
            if local_path is not None and local_path.is_file():
                # The user's own permanent library file - never deleted,
                # same as an already-persisted "Download tracks" file. Takes
                # priority over a YouTube download even if one also exists.
                audio_path: Path = local_path
                scratch_path: Path | None = None
            elif existing is not None:
                audio_path = existing
                scratch_path = None
            else:
                # Either no local match, or its file went missing (e.g. an
                # unmounted NFS share, or the file was moved/deleted since
                # matching) - fall back to YouTube rather than failing the
                # track outright. _analyzable_tracks only requires *some*
                # source, so a track can reach here with url=None if its
                # only source was a local file that's since gone missing.
                if url is None:
                    raise RuntimeError(
                        "no YouTube match and local file is missing"
                        + (f" ({local_audio_path})" if local_audio_path else "")
                    )
                scratch_stem = audio_download.track_audio_stem(audio_dir / "_scratch", release_id, position, title)
                scratch_stem.parent.mkdir(parents=True, exist_ok=True)
                audio_path = audio_download.download_with_retries(
                    downloader, url, scratch_stem, audio_download.DOWNLOAD_MAX_ATTEMPTS, audio_download.DOWNLOAD_RETRY_DELAY_SECONDS
                )
                scratch_path = audio_path

            waveform_path = track_waveform_path(waveform_dir, release_id, position, title)
            result = analyze_track_fn(audio_path, waveform_path, models_dir)

            conn.execute(
                """
                UPDATE tracks
                SET bpm = ?, musical_key = ?, musical_key_scale = ?, mood_json = ?, mood_summary = ?,
                    genre = ?, style = ?, waveform_path = ?, analyzed_at = ?
                WHERE id = ?
                """,
                (
                    result.bpm,
                    result.musical_key,
                    result.musical_key_scale,
                    json.dumps(result.mood_scores),
                    result.mood_summary,
                    result.genre,
                    result.style,
                    # Just the filename, not the full filesystem path: the web
                    # layer serves it at /waveforms/<release_id>/<filename>,
                    # resolving against config.waveform_dir server-side.
                    result.waveform_path.name,
                    now,
                    track["id"],
                ),
            )
            conn.commit()

            if scratch_path is not None:
                scratch_path.unlink(missing_ok=True)

            results.append(
                {
                    "position": position,
                    "title": title,
                    "status": "analyzed",
                    "bpm": result.bpm,
                    "musical_key": result.musical_key,
                    "mood_summary": result.mood_summary,
                    "genre": result.genre,
                    "style": result.style,
                }
            )
        except Exception as exc:
            if on_error is not None:
                on_error(position, exc)
            results.append({"position": position, "title": title, "status": "failed"})

        if on_progress is not None:
            on_progress(index, total, position)

    # Clean up the scratch directory itself once nothing's left in it.
    scratch_dir = audio_dir / "_scratch" / str(release_id)
    if scratch_dir.is_dir() and not any(scratch_dir.iterdir()):
        shutil.rmtree(scratch_dir, ignore_errors=True)

    return results


def _analyze_one_release(
    db_path: Path,
    release_id: int,
    audio_dir: Path,
    waveform_dir: Path,
    models_dir: Path,
    downloader: audio_download.VideoDownloader,
    force: bool,
    analyze_release_tracks_fn: Callable[..., list[dict]],
    analyze_track_fn: Callable[[Path, Path, Path], AnalysisResult],
) -> None:
    # A dedicated connection per worker process - sqlite3.Connection isn't
    # safe for concurrent/cross-process use, and analyze_library submits
    # this to a ProcessPoolExecutor (see its docstring for why processes
    # rather than threads), so this always runs in its own process anyway.
    conn = db.connect(db_path)
    try:
        analyze_release_tracks_fn(
            conn, release_id, audio_dir, waveform_dir, models_dir,
            downloader=downloader, force=force, analyze_track_fn=analyze_track_fn,
        )
    finally:
        conn.close()


def analyze_library(
    conn: sqlite3.Connection,
    db_path: Path,
    audio_dir: Path,
    waveform_dir: Path,
    models_dir: Path,
    downloader: audio_download.VideoDownloader = audio_download.default_video_downloader,
    force: bool = False,
    max_workers: int = 1,
    on_progress: ProgressCallback | None = None,
    on_error: ErrorCallback | None = None,
    should_stop: Callable[[], bool] | None = None,
    analyze_release_tracks_fn: Callable[..., list[dict]] = analyze_release_tracks,
    analyze_track_fn: Callable[[Path, Path, Path], AnalysisResult] = analyze_track,
) -> None:
    """Run analyze_release_tracks() over every release that has at least one
    track with a selected YouTube match, skipping releases with nothing left
    to analyze unless force=True - the collection-wide counterpart to the
    per-release "Analyze tracks" button, mirroring youtube_matching.
    match_videos()'s collection-wide/force pattern.

    Up to max_workers releases at a time in parallel, each in its own OS
    *process* (ProcessPoolExecutor), not a thread. Threads were tried first
    (matching match_and_download_library's ThreadPoolExecutor) but measured
    no real speedup: the essentia C extension never releases the GIL during
    its algorithm calls (confirmed via `nm -D` on the compiled _essentia*.so
    - no PyEval_SaveThread/Py_BEGIN_ALLOW_THREADS symbols), so one worker
    thread doing TensorFlow inference blocks every other worker thread in
    the same process for that call's full duration - CPU usage measured at
    ~200% on an 8-core machine with max_workers=7, i.e. barely more parallel
    than running one at a time. Separate processes sidestep the GIL
    entirely; each still gets the same per-worker warm model cache as
    before (see _get_model_cache) so the ~6s/track cold-load cost is only
    paid once per worker process per run, not once per track.

    Spawned (multiprocessing.get_context("spawn")), not forked: forking a
    process that may already have TensorFlow loaded (e.g. this same process
    already having served a per-release "Analyze tracks" request) is a
    known source of hangs/crashes, since TF's internal thread state doesn't
    survive fork. Spawn starts each worker as a clean interpreter instead -
    slightly slower to start, but that cost is only paid once per worker
    per run, same as the cold model-load cost above.

    Essentia analysis is itself CPU/GPU-bound per track, so parallelism
    here has less headroom than the network/LLM-bound match+download pass,
    but real multi-core speedup nonetheless (unlike the thread-based
    attempt above).

    Each worker's own TensorFlow thread pool is capped to 1 thread
    (_init_analysis_worker, this pool's `initializer`) - left at its
    default, TensorFlow fans a single inference call out across every core
    on the machine by itself (confirmed: 25 OS threads from one process for
    one call on this 8-core machine), which massively oversubscribes the
    machine once max_workers processes are each doing that at the same
    time (load average measured at ~20 on 8 cores with max_workers=7). The
    process-level parallelism above already uses the machine's cores; each
    individual TensorFlow call doesn't need to fan out across them too.

    If `should_stop` starts returning True, releases still in flight are
    allowed to finish but no new ones are submitted to the pool.

    analyze_release_tracks_fn/analyze_track_fn default to the real
    analyze_release_tracks/analyze_track and exist so tests can inject fakes
    (e.g. avoiding real ffmpeg/Essentia work) - see analyze_release_tracks's
    docstring for why these need to be plain module-level functions
    (picklable-by-reference) rather than monkeypatched or closures, now that
    this crosses a ProcessPoolExecutor boundary.
    """
    release_rows = conn.execute(
        """
        SELECT DISTINCT r.id
        FROM releases r
        JOIN tracks t ON t.release_id = r.id
        LEFT JOIN track_video_matches tvm ON tvm.track_id = t.id AND tvm.is_selected = 1
        WHERE r.removed_from_discogs_at IS NULL
          AND (tvm.video_id IS NOT NULL OR t.local_audio_path IS NOT NULL)
        """
    ).fetchall()

    if not force:
        release_rows = [row for row in release_rows if _analyzable_tracks(conn, row["id"], force=False)]

    releases = [row["id"] for row in release_rows]
    total = len(releases)
    completed = 0
    workers = max(1, max_workers)

    def _stopped() -> bool:
        return should_stop is not None and should_stop()

    with ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init_analysis_worker
    ) as pool:
        pending = list(releases)
        in_flight: dict = {}

        def submit_more() -> None:
            while pending and len(in_flight) < workers and not _stopped():
                release_id = pending.pop(0)
                future = pool.submit(
                    _analyze_one_release, db_path, release_id, audio_dir, waveform_dir, models_dir, downloader, force,
                    analyze_release_tracks_fn, analyze_track_fn,
                )
                in_flight[future] = release_id

        submit_more()
        while in_flight:
            done, _ = wait(list(in_flight.keys()), return_when=FIRST_COMPLETED)
            for future in done:
                release_id = in_flight.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    if on_error is not None:
                        on_error(release_id, exc)
                    continue
                completed += 1
                if on_progress is not None:
                    on_progress(completed, total, release_id)
            submit_more()
