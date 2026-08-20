# DiscQL

A personal tool to mirror a Discogs record collection into a local database,
match tracks to YouTube videos (fuzzy string matching + local LLM assist for
ambiguous cases), and browse the collection through a web GUI. See
[CLAUDE.md](CLAUDE.md) for the full project roadmap.

## Setup

Requires Python 3.14+, plus `ffmpeg` and `curl` available on `PATH` (used
for audio decoding/waveform generation and Essentia model downloads
respectively - see "Audio analysis" below for why `curl` specifically,
not e.g. Python's own `requests`).

```
uv sync
```

Requires the [Ollama](https://ollama.com) CLI installed locally (used to
resolve ambiguous YouTube-to-track matches) with a model such as
`llama3.2:latest` pulled. You don't need to have it running yourself —
YouTube matching starts `ollama serve` itself if it isn't already reachable
(`ollama_client.ensure_ollama_running()`), and stops it again once matching
finishes, once for the whole run (a single release or the whole
library — not once per release). If Ollama was *already* running when
matching started, it's left running afterward too — only an instance this
process itself started gets stopped. If the `ollama` CLI isn't
installed/on PATH, matching still proceeds fuzzy-only (same as if Ollama
were simply unavailable), just without LLM disambiguation for ambiguous
matches.

Then start the web GUI (`uv run discql-web`, see "Usage" below) and open
<http://127.0.0.1:8000> — a fresh install with no Discogs access configured
yet redirects every page to a `/settings` page automatically, where setup
happens in the browser (see "Configuration" below). The CLI
(`discql-sync`) instead prompts for a personal access token interactively
on first use, same as before.

## Configuration

Config and data live outside the project directory, following the XDG base
directory convention:

- **Discogs API token**: `~/.config/discql/config` (respects
  `$XDG_CONFIG_HOME` if set). Two ways to set it up, both save to that file
  automatically so you're only asked once:
  - **Web GUI**: visit `/settings` (a fresh install with nothing configured
    yet redirects there automatically) and either paste in a personal
    access token (get one from
    <https://www.discogs.com/settings/developers>), or connect via OAuth
    instead — register a Discogs Application once at that same URL to get
    a Consumer Key/Secret, paste those into `/settings`, then "Save &
    authorize with Discogs": one click to grant access from then on, no
    more copying tokens by hand. Registering an Application is a one-time,
    per-installation step (there's no shared/bundled key — see
    `discogs_api.py`'s `build_discogs_api` for how a personal token vs.
    OAuth credentials are chosen between).
  - **CLI**: prompts for a personal access token interactively on first
    use (`discql-sync sync`/`add`/`remove`) if nothing's configured yet.
- **SQLite database**: `~/.cache/discql/discogs.db` (respects
  `$XDG_CACHE_HOME` if set), created automatically on first run.

Both locations can be overridden by exporting `DISCOGS_USER_TOKEN` /
`DISCOGS_DB_PATH` as real environment variables.

The fuzzy-matching tunables (see "YouTube matching" below), YouTube
cookies/bitrate settings, local FLAC matching settings, and `MAX_WORKERS`
are likewise configurable, not hardcoded — either through the web GUI's
`/settings` page (a form, no file-editing needed), or by hand: add
`FUZZY_CONFIDENT_THRESHOLD=<value>`, `FUZZY_TIE_MARGIN=<value>` and/or
`FUZZY_DENOISE_TERMS=<comma-separated>` to `~/.config/discql/config`, or
export them as environment variables. All of them are written into the
config file with their defaults on first run if not already present, so
they're always visible there to edit even without using `/settings`.

Age-restricted YouTube videos refuse to serve any metadata at all without an
authenticated session — no `player_client` fallback works around this, and
YouTube's "Sign in to confirm you're not a bot" block (increasingly common
across `player_client`s, not just age-restricted videos) needs the same
fix. Optionally set `YOUTUBE_COOKIES_BROWSER=<browser>` (e.g. `firefox`,
`chrome`, `chromium`, `brave`, `edge`, `safari`, `vivaldi`, `opera`) to have
yt-dlp reuse your logged-in session from that browser
(`--cookies-from-browser`) for both metadata fetching and audio downloads.
Empty/unset by default (opt-in, since it uses your real browser session);
you also need to actually be signed into YouTube in that browser for it to
help.

Providing cookies makes yt-dlp skip the `android`/`android_vr` clients
entirely (they don't support cookie auth), falling back to `web` alone —
which needs its signature/n-challenge solved to resolve any real formats at
all (YouTube's "SABR streaming" push). yt-dlp only does that via a
challenge-solver script it fetches on demand from its own GitHub repo, so
both yt-dlp call sites (`youtube_matching.build_extractor`,
`audio_download.build_downloader`) pass `remote_components: ["ejs:github"]`
whenever cookies are configured — confirmed concretely that without it,
cookie-authenticated requests came back with "Only images are available for
download," no audio formats at all. A no-op when cookies aren't configured.

Audio downloads are capped to `YOUTUBE_AUDIO_MAX_BITRATE_KBPS` (128 by
default) via yt-dlp's format selector (`bestaudio[abr<=N]/bestaudio/best`)
rather than always taking the highest-bitrate stream YouTube offers
(typically ~160kbps Opus) — still one of YouTube's own native streams
picked as-is, no re-encoding, just a smaller one when a lower-bitrate
stream is available. Set it empty to disable the cap and always take the
highest bitrate.

If you have a local FLAC library (e.g. Bandcamp purchases) for some
releases, set `LOCAL_FLAC_DIR` to its root — see "Local FLAC library
matching" below. Empty by default (disabled). `LOCAL_MATCH_CONFIDENT_THRESHOLD`
(0.75 by default) is the fuzzy score above which a folder/file match is
trusted.

## YouTube matching

If a track and video both have a known duration and they differ by more
than 20% of the track's duration, the pair is disqualified outright (score
0) before any title comparison runs at all — a mismatch that large means
it's not the same recording, regardless of how similar the titles look.

Title fuzzy scoring (`rapidfuzz.fuzz.WRatio`) runs with `processor=
rapidfuzz.utils.default_process`, not the library's raw default — WRatio's
own default is case-sensitive (confirmed concretely: a same-titled video in
different case can score as low as ~12/100 despite being an identical
title), which `default_process` fixes by lowercasing and normalizing
punctuation to whitespace before comparing.

Videos are matched to tracklist entries fuzzy-first: a video whose best
fuzzy title/duration score (against the raw YouTube title, or — if that's
ambiguous — against the title with uploader boilerplate stripped) reaches
`FUZZY_CONFIDENT_THRESHOLD`, **and** clears the runner-up track by at least
`FUZZY_TIE_MARGIN` (0.05 by default — guards against e.g. two same-named
mixes scoring near-identically, where trusting whichever wins the tie can
silently commit to the wrong one), is trusted outright. Anything still
below that bar is handed to the local LLM (Ollama) for a judgment call, one
video per call.

The boilerplate stripped before that retry is the release title, artist
name(s), any 4-digit year, plus whatever generic terms are configured in
`FUZZY_DENOISE_TERMS` (comma-separated, default `Original Mix`) — terms an
uploader tacks onto many videos regardless of the actual track name, and
which would otherwise dilute the part of the title that actually
discriminates between tracks.

The final pick among multiple candidate videos for one track is based on a
blend of fuzzy score and LLM confidence (not raw LLM confidence alone) —
LLM confidence from a small local model is both imprecise (a 1.0 vs 0.99
difference is noise) and can be outright wrong, so it can't fully mask a
candidate's own textual fit for that track.

A single LLM classification is also distrusted (falls back to "unrelated")
if it claims a track whose own fuzzy score is clearly weaker (by at least
`FUZZY_TIE_MARGIN`) than a *different* track the same video already scores
higher against — this catches the case where there's no competing video to
blend against, so a confidently wrong LLM verdict would otherwise stand
unchallenged (observed concretely with non-Latin-script tracklists, where a
small English-centric model can default to the first tracklist entry
regardless of actual title/lyrics content).

## Local FLAC library matching

If you already own a digital copy of a release (e.g. bought on Bandcamp),
there's no need to also match/download it from YouTube — `local_audio.py`
matches releases against a local folder tree (`LOCAL_FLAC_DIR`, see
Configuration) instead: fuzzy folder-name-vs-release match first
(`match_release_to_folder`), then fuzzy filename-vs-tracklist match within
that folder (`match_tracks_to_files`) for every track. Fuzzy-only, no LLM
step (unlike YouTube matching) — a match either clears
`LOCAL_MATCH_CONFIDENT_THRESHOLD` or it doesn't, and results are cheap
enough to get right without one.

`scan_local_folders` treats any directory that directly contains an audio
file as one release's folder candidate, recursing arbitrarily deep — this
naturally handles both a flat "one folder per release" layout and a label
folder one level up with per-release subfolders (no audio files of its own).
A flat dump of loose singles with no per-release subfolder at all isn't
discovered as anything meaningful (known gap, not solved yet). Track-to-file
assignment is a greedy highest-score-first match over the full score matrix
(strips a leading track-number prefix like `01 - ` from filenames first) so
the same file is never assigned to two tracks — there's no LLM tie-breaker
here to fall back on like there is for YouTube.

Runs automatically as a cheap pre-step (plain filesystem walk + in-memory
fuzzy scoring, no network) at the start of `pipeline.match_and_download_library`
— i.e. every "Match YT" / "Run All" run — or standalone via
`discql-sync match-local` (`--force` to re-check every release, not just
unchecked ones). A no-op if `LOCAL_FLAC_DIR` is empty/unset or doesn't exist
(e.g. an unmounted NFS share) — never stamps a release as "checked" in that
case, so it's naturally retried once the share is back.

Once a track has a confident local match (`tracks.local_audio_path`),
that file becomes its audio source everywhere: `audio_analysis.
analyze_release_tracks` prefers it over both an already-downloaded YouTube
file and downloading a new one (never deleted afterward — same treatment
as an already-persisted "Download tracks" file, it's the user's own
permanent library, not scratch audio), and a release whose *entire*
tracklist is locally covered has its YouTube matching/download steps
skipped entirely (`pipeline._locally_covered`) — no yt-dlp/Ollama calls
spent on it. This is non-destructive: existing YouTube matches/downloads a
release already has are left alone, only *future* YouTube work is skipped.
If a matched local file later goes missing (moved, or an unmounted share),
analysis transparently falls back to YouTube rather than failing the track.

No web GUI for this yet (backend/CLI only, matching the same "CLI first"
pattern `collection.py`'s add/remove started with) — see CLAUDE.md's Open
TODOs.

## Audio download

A release's detail page has a "Download tracks for this release" button
(shown once at least one track has a matched video) that downloads the
best-quality audio for every matched track — audio-only where YouTube
offers it, no re-encoding (if only a combined video+audio format is
available, the video track is stripped losslessly via an ffmpeg stream
copy, not a re-encode). Downloads are retried up to 3 times on transient
failures (rate limiting, momentary 403s).

The `android_vr` yt-dlp player_client is deliberately *not* in this
download's client fallback list (unlike metadata fetching, which does use
it) — confirmed concretely that its presence anywhere in the list
reproducibly turns the actual media GET into a hard `HTTP Error 403:
Forbidden`, regardless of position, likely fallout from YouTube's
"SABR-only streaming" experiment breaking android/android_vr audio-only
formats (see [yt-dlp#12482](https://github.com/yt-dlp/yt-dlp/issues/12482)).
If `bestaudio` isn't available under the remaining `android`/`web` clients
either (same SABR issue, on a per-video basis), download falls through to
a combined video+audio format and strips the video track as described
above.

Already-downloaded tracks are skipped on repeat clicks. Files go to `DISCOGS_AUDIO_DIR`
(`~/.cache/discql/audio/<release_id>/` by default; a track's audio is
deleted again once `audio_analysis.py` has analyzed it, unless it was
fetched via this standalone button, in which case it's left alone).
Currently per-release only; a collection-wide version is a planned
follow-up.

## Audio analysis

A release's detail page also has an "Analyze tracks for this release"
button that, per matched track: downloads the audio on demand if not
already present (deleting it again afterward unless it was already kept via
the download button above), extracts BPM and musical key
(`essentia.standard.RhythmExtractor2013` / `KeyExtractor`), classifies mood
and genre, and renders a waveform PNG (`WAVEFORM_DIR`) via ffmpeg + Pillow.
BPM/key/mood/genre (the substyle half of the "Genre---Style" pick, e.g.
"Techno" — full pick on hover) are written onto the track and shown as
their own tracklist column; the deduped set of AI-detected genres/styles
across all of a release's tracks is also shown as its own row of badges
below the release's own Discogs genres/styles (visually separated from
them — it's a prediction, not Discogs metadata). A "Clear audio analysis"
button next to the analyze button wipes all of the above back to
unanalyzed (BPM/key/mood/genre/style/waveform) without touching matched
videos or downloaded audio — the same "reset without re-fetching" pattern
as "Clear fuzzy matching" for YouTube matches.

Mood/genre classification is a two-stage Discogs-EffNet pipeline
(`TensorflowPredictEffnetDiscogs` for a 1280-dim audio embedding, then
`TensorflowPredict2D` per classification head on top of it — models
downloaded and cached under `ESSENTIA_MODELS_DIR` on first use):
danceability, happy, sad, relaxed, aggressive, party (binary heads), plus
`genre_discogs400` — a 400-class head predicting Discogs' own
"Genre---Style" taxonomy (e.g. `"Electronic---Deep House"`), the same
vocabulary already used for the genres/styles synced from the Discogs API
itself, just predicted from audio instead of read from Discogs. The genre
tag shown in the UI (`top_genre()`) is the broad genre half of
`genre_discogs400`'s top-scoring class (`top_style()`); the full
"Genre---Style" pick is shown separately as the AI style tag.

Chosen over the older MusiCNN classifiers (danceability/mood/`genre_dortmund`
+ `genre_electronic`, still available in Essentia but effectively
superseded) because Discogs-EffNet's genre taxonomy matches Discogs' own
vocabulary natively — no separate broad/electronic-subgenre two-model
refinement hack needed, and Essentia's own docs recommend the newer
embedding-based models over MusiCNN. A future switch to Discogs-MAEST (an
even newer embedding, "most competitive performance" per Essentia's docs,
but — as of writing — with only genre heads published, no mood ones) would
only require swapping the embedding model constant; the head-scoring logic
is unaffected either way.

Status: implemented and verified end-to-end against a real release (BPM,
key, mood, and waveform all populated correctly and shown in the UI). Two
non-obvious issues had to be worked around:
- Originally, each model (the embedding extractor and each classification
  head) had to run in its own freshly spawned `python3` subprocess — both
  in-process and `ProcessPoolExecutor` reproducibly hung after loading ~2
  different TensorFlow graphs in one process, confirmed directly at the
  time. **Re-verified 2026-08-20 that this no longer reproduces** with the
  currently installed essentia-tensorflow (loading the embedding extractor
  plus all 7 classification heads sequentially in one process, repeated
  across several tracks and under real concurrent multi-thread load, ran
  cleanly with no hang). Models are now loaded in-process instead, one
  `_ModelCache` per worker (`audio_analysis._get_model_cache` /
  `_ModelCache`, thread-local so a pool's reused workers keep their loaded
  models warm across every release/track that worker ever picks up in one
  run) — cut a measured ~6s/track (mostly Python+Essentia+TensorFlow import
  and graph-load cost, paid 8x per track under the old subprocess-per-model
  scheme) down to ~1s cold / ~0.5s warm. If this ever needs reverting, the
  prior implementation is in git history.
  One follow-up from the same 2026-08-20 verification: real concurrent
  runs also logged large volumes of Essentia's `[ WARNING ] No network
  created, or last created network has been deleted...` - reproduced it
  deterministically (needs `RhythmExtractor2013`/`KeyExtractor` running
  concurrently alongside the TensorFlow models, not the TensorFlow models
  alone) and confirmed it's cosmetic: diffed 7 concurrent workers' full
  BPM/key/mood/genre output for the same track against a known-good
  sequential run, 0/7 mismatches. Silenced via `essentia.log.warningActive
  = False` (`_silence_essentia_warnings()`, called once per worker from
  `_ModelCache.__init__` and from `extract_bpm_and_key()`) rather than left
  to flood the log - `errorActive` is untouched, so real errors still show.
  `infoActive` (e.g. "Successfully loaded graph file: ...") was initially
  left on too, but with one `_ModelCache` per worker *process* (see the
  follow-up below) each of the 8 models logs that line once per worker at
  the start of every run - multiplied by `MAX_WORKERS`, enough log spam on
  its own to bury everything else, so `_silence_essentia_warnings()` now
  silences both.
- **Same-day follow-up (2026-08-20):** in-process model loading made
  `analyze_library`'s parallelism collapse to near-sequential in practice -
  measured ~200% CPU on an 8-core machine with `MAX_WORKERS=7`, not ~700%.
  Root cause: the essentia C extension never releases Python's GIL during
  its algorithm calls (confirmed via `nm -D` on the compiled `_essentia*.so`
  - no `PyEval_SaveThread`/`Py_BEGIN_ALLOW_THREADS` symbols at all), so one
  worker thread doing TensorFlow inference blocks every other worker thread
  in the same process for that call's full duration. `analyze_library`
  (only - `match_and_download_library`'s network/LLM-bound work is
  unaffected, since I/O calls release the GIL) now runs each release in its
  own OS *process* (`ProcessPoolExecutor`) instead of a thread, sidestepping
  the GIL entirely while keeping the same per-worker warm model cache
  described above. Spawned (`multiprocessing.get_context("spawn")`), not
  forked - forking a process that may already have TensorFlow loaded (e.g.
  this same process having already served a per-release "Analyze tracks"
  request) is a known source of hangs/crashes, since TF's internal thread
  state doesn't survive fork. Verified end-to-end (real Essentia/TensorFlow
  models, two releases, `max_workers=2`): both spawned workers loaded their
  models and produced correct, independent BPM/key/genre/style results with
  no hang. One consequence for tests: fakes passed into `analyze_library`
  (`analyze_track_fn`/`analyze_release_tracks_fn`) now have to be plain
  module-level functions rather than monkeypatched or closures, since a
  spawned child process re-imports `audio_analysis` fresh and never sees a
  monkeypatch, and can't unpickle a closure/lambda at all.
- Downloading a not-yet-cached model file via Python's `requests` (and
  plain stdlib `urllib.request`) hangs indefinitely partway through reading
  the response — traced down to a raw `ssl.SSLSocket.recv()` stall near the
  end of the transfer, with an identical TLS version/cipher to a `curl`
  request that succeeds instantly, so it looks like a Python 3.14 `ssl`
  module regression rather than anything server- or code-specific. Model
  downloads shell out to `curl` instead.

Essentia also requires audio in a format its own bundled decoder supports;
downloaded files that yt-dlp hands back (e.g. Opus-in-WebM) aren't always
one of them ("Unsupported codec!"), so analysis decodes via the system
ffmpeg instead, straight to an in-memory PCM array (`decode_audio()`) —
purely internal, the persisted downloaded file (if any) is never touched.
That decode happens once per track, at 44.1kHz; BPM/key extraction, the
waveform PNG, and the mood/genre classifiers (after an in-process resample
to the 16kHz they require) all reuse the same array instead of each
re-decoding the file themselves.

### Analyzing a single audio file

`discql-analyze-file` runs the same pipeline (BPM/key via Essentia,
mood/genre via the Discogs-EffNet pipeline above) against one local audio
file, independent of the database/web GUI — useful for testing a new
classifier or debugging without a matched release:

```
uv run discql-analyze-file <path/to/audio-file>
```

Prints BPM, key, mood summary, genre, style, and every raw classifier score
(the 400-class `genre_discogs400` head is truncated to its top 10 in the
formatted report; `--json` includes all 400). `--json` prints
machine-readable output instead of the formatted report; `--waveform-out
<path>` controls where the waveform PNG is written (defaults to `<input
file>.waveform.png`); `--models-dir <path>` overrides where cached Essentia
model files are read from/downloaded to (defaults to `ESSENTIA_MODELS_DIR`,
same as the web GUI flow).

## Run All pipeline

The "Run All" button (`pipeline.run_full_pipeline`) runs YouTube matching,
audio download, and analysis across the whole collection in two passes
rather than interleaving all three per release:

1. **Match + download** (`pipeline.match_and_download_library`), up to
   `workers` releases at a time in parallel (network/LLM-bound - matching
   hits yt-dlp/Ollama, download hits yt-dlp - so concurrency here buys real
   wall-clock speedup). A release's match step is skipped if it's already
   fully classified (unless `force`); its download step is skipped if it's
   already fully analyzed, since there'd be nothing left for phase 2 to use
   it for. Ollama is started once for this pass and stopped again before
   phase 2 begins - analysis doesn't use it, so there's no reason to keep
   it (and whatever GPU/RAM it holds) around while phase 2 runs. This is
   also, standalone, the "Match YT" button's implementation - both share
   the same function, just wired to different UI entry points.
2. **Analyze**, also up to `workers` releases at a time in parallel
   (delegates to `audio_analysis.analyze_library`, same bounded-pool count
   as phase 1, but a process pool rather than a thread pool - see "Audio
   analysis" below for why) - CPU/GPU-bound per track (Essentia/TensorFlow
   inference itself), so parallelism here has less headroom to gain from
   than phase 1's network/LLM-bound work; it still helps when cores are
   available, just don't expect the same near-linear speedup. Each worker
   keeps its own loaded Essentia/TensorFlow models warm across every
   release it processes in the run (`_get_model_cache` - see "Audio
   analysis" below), so the per-track model-loading cost that used to
   dominate is mostly paid once per worker, not once per track. Reuses
   whatever phase 1 already downloaded; a release that's matched but not
   downloaded (shouldn't normally happen, but not guaranteed - e.g. a
   stopped phase 1) still works, since `analyze_release_tracks` falls back
   to downloading on demand.

Phase 2 only starts if phase 1 wasn't stopped early. A non-force run skips
any release that already has nothing left to do in either phase - see
`pipeline.count_pending_releases()`, also used for the single combined "N
releases pending" indicator in the topbar (`GET /pipeline/pending`).

All four background tasks (Sync, Match YT, Analyze audio, Run All) support
cooperative cancellation via a Stop button while running: the release
currently in flight is allowed to finish, but no further ones are started.

"Match YT" and "Run All" both default their `workers` field to
`max(1, cpu_count - 1)` - one core left free for the OS/web server rather
than saturating every core (`web/app.py:default_workers`).

## Usage

Sync your Discogs collection locally:

```
uv run discql-sync sync
```

Other CLI commands: `add <release_id>`, `remove <release_id>`, `list`.

Run the web GUI:

```
uv run discql-web
```

Then open <http://127.0.0.1:8000>. Host/port can be overridden with the
`WEB_HOST` / `WEB_PORT` environment variables.

## Development

```
uv run pytest
```
