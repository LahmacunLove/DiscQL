from __future__ import annotations

import os
import sqlite3
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable
from urllib.parse import urlencode

import discogs_client
import requests
from fastapi import Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from discql import audio_analysis, audio_download, cover_art, crates as crates_module, db, musical_key, ollama_client, pipeline, sticker_selection, stickers, sync, youtube_matching
from discql.config import USER_AGENT, Config, load_config, set_config_values
from discql.discogs_api import build_discogs_api
from discql.web import repository, tasks

BASE_DIR = Path(__file__).parent
PAGE_SIZE = 200


@lru_cache
def get_config() -> Config:
    return load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = db.connect(get_config().db_path)
    try:
        db.migrate(conn)
    finally:
        conn.close()
    yield


app = FastAPI(title="DiscQL", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")
templates.env.filters["format_key"] = lambda key, scale: musical_key.format_key(key, scale, get_config().key_notation)


def static_url(path: str) -> str:
    """Cache-bust static assets with a file-mtime query string, so editing
    e.g. styles.css is picked up on the browser's next normal reload
    instead of silently serving a stale cached copy indefinitely (confirmed
    concretely: a CSS fix was invisible through several plain reloads,
    resolved only once this was added - the server itself already re-reads
    static files fresh on every request, only the browser's own cache was
    stale).
    """
    file_path = BASE_DIR / "static" / path
    version = int(file_path.stat().st_mtime) if file_path.exists() else 0
    return f"/static/{path}?v={version}"


templates.env.globals["static_url"] = static_url


@app.middleware("http")
async def require_discogs_auth(request: Request, call_next):
    """Sends every *top-level page* to /settings until Discogs access is
    configured (personal token or completed OAuth flow - see
    Config.has_discogs_auth) - what makes a fresh install with zero config
    land on setup in the browser instead of, previously, load_config()
    blocking on a terminal getpass() prompt at startup (which doesn't work
    at all for a web server, especially one started without a real
    attached terminal). /settings itself (including its OAuth authorize/
    callback routes, by definition hit *before* auth exists yet) and
    static assets are exempt.

    Also exempts any htmx request (HX-Request header - same signal the
    /releases route already uses to decide full-page vs. partial). This
    is not optional: base.html's topbar auto-loads several small status
    widgets via hx-trigger="load" (sync/youtube/audio/run_all status,
    pending count) - none of them need Discogs auth to render their
    current (possibly idle/empty) state. Redirecting *those* to /settings
    was confirmed concretely to break the page: htmx swaps the redirect's
    response (a full page, complete with base.html's own copy of the same
    hx-trigger="load" widgets) into the tiny status-widget target, and
    htmx re-scans newly-swapped-in content for triggers - so each swap
    spawns more load-triggered requests, each pulling in another full page
    with more such widgets, compounding into a runaway loop of nested
    requests/swaps ("a lot of stuff looping" on the page).
    """
    path = request.url.path
    is_exempt = (
        path.startswith("/settings")
        or path.startswith("/static/")
        or path == "/favicon.ico"
        or request.headers.get("HX-Request") == "true"
    )
    if not is_exempt and not get_config().has_discogs_auth:
        return RedirectResponse(url="/settings", status_code=307)
    return await call_next(request)


def get_db():
    conn = db.connect(get_config().db_path)
    try:
        yield conn
    finally:
        conn.close()


def _availability_status(count: int, available: int, total: int) -> str:
    """"done" if every currently-matched track is covered (count >= available)
    and that's also every track in the tracklist; "partial" if every matched
    track is covered but the tracklist has more (unmatched) tracks beyond
    that; "" if there's still matched-but-not-yet-covered work, or nothing
    is matched at all yet. Used for the release list's download/analysis
    status dots, so e.g. a release where only 2 of 3 tracks ever got a
    YouTube match can still show "fully downloaded/analyzed" for those 2.
    """
    if available <= 0:
        return ""
    if count < available:
        return ""
    return "done" if available == total else "partial"


def _matching_status(matched_count: int, track_count: int, processed: bool) -> str:
    """"" if YouTube matching has never been run for this release yet;
    "partial" if it's run but not every track ended up with a matched
    video (no good candidate for some, or the release just has fewer
    videos than tracks); "done" if every track has one. Without the
    `processed` check, "never attempted" and "attempted but incomplete"
    both looked identical (just "not done")."""
    if not processed or track_count <= 0:
        return ""
    return "done" if matched_count == track_count else "partial"


@app.get("/")
def index() -> RedirectResponse:
    return RedirectResponse(url="/releases")


def _active_sticker_preset(config: Config) -> stickers.LabelPreset:
    return stickers.PRESETS.get(config.sticker_preset, stickers.PRESETS[stickers.DEFAULT_PRESET_KEY])


def _parse_release_filters(
    q: str | None,
    genre: list[str],
    style: list[str],
    year: str | None,
    label: str | None,
    artist: str | None,
    bpm: str | None,
    bpm_tolerance: list[str],
) -> dict:
    """Builds repository.list_releases/count_releases filter kwargs from
    /releases' raw query params - shared with /stickers/select_all so
    "select all filtered" resolves the exact same set of releases. Caller
    must already have dropped empty-checkbox entries from genre/style/
    bpm_tolerance (a submitted-but-empty checkbox group arrives as [""]).
    """
    try:
        year_int = int(year) if year else None
    except ValueError:
        year_int = None
    try:
        bpm_float = float(bpm) if bpm else None
    except ValueError:
        bpm_float = None
    # Checkboxes are additive tolerance bands around the same target BPM
    # (Exact/±8%/±16%, see repository._where_clause), not independent
    # filters - checking more than one is only ever wider, so the widest
    # checked band is what actually determines the match.
    bpm_tolerance_pct = None
    if bpm_float is not None and bpm_tolerance:
        try:
            bpm_tolerance_pct = max(float(t) for t in bpm_tolerance) / 100
        except ValueError:
            bpm_tolerance_pct = None

    return dict(
        q=q, genres=genre, styles=style, year=year_int, label_name=label, artist_name=artist,
        bpm=bpm_float, bpm_tolerance_pct=bpm_tolerance_pct,
    )


@app.get("/releases", response_class=HTMLResponse)
def releases(
    request: Request,
    q: str | None = None,
    genre: list[str] = Query(default=[]),
    style: list[str] = Query(default=[]),
    year: str | None = None,
    label: str | None = None,
    artist: str | None = None,
    bpm: str | None = None,
    bpm_tolerance: list[str] = Query(default=[]),
    sort: str = repository.DEFAULT_SORT,
    view: str = "grid",
    page: int = 1,
    show_all: bool = False,
    continuation: bool = False,
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    if view not in ("grid", "list"):
        view = "grid"
    if sort not in repository.SORT_OPTIONS:
        sort = repository.DEFAULT_SORT
    page = max(page, 1)
    offset = (page - 1) * PAGE_SIZE
    # A submitted-but-empty checkbox group (e.g. the form's genre="" default)
    # arrives as [""], not [] - filter it down to nothing selected.
    genre = [g for g in genre if g]
    style = [s for s in style if s]
    bpm_tolerance = [t for t in bpm_tolerance if t]

    filter_kwargs = _parse_release_filters(q, genre, style, year, label, artist, bpm, bpm_tolerance)

    if show_all:
        items = repository.list_releases(conn, sort=sort, limit=None, **filter_kwargs)
        total = len(items)
        loaded_count = total
        has_more = False
    else:
        items = repository.list_releases(
            conn, sort=sort, limit=PAGE_SIZE, offset=offset, **filter_kwargs
        )
        total = repository.count_releases(conn, **filter_kwargs)
        loaded_count = offset + len(items)
        has_more = loaded_count < total

    # Download count is a filesystem check, not tracked in the DB - only
    # worth doing for the list view, which is the only one that renders it.
    status_by_id: dict[int, dict[str, str]] = {}
    if view == "list":
        audio_dir = get_config().audio_dir
        for item in items:
            downloaded_count = audio_download.count_downloaded_tracks(audio_dir, item.id)
            status_by_id[item.id] = {
                "matching": _matching_status(item.matched_count, item.track_count, item.matching_processed),
                "download": _availability_status(downloaded_count, item.matched_count, item.track_count),
                "analysis": _availability_status(item.analyzed_count, item.matched_count, item.track_count),
            }

    def query_string(**overrides: str | int | list[str] | None) -> str:
        params = {
            "q": q,
            "genre": genre,
            "style": style,
            "year": year,
            "label": label,
            "artist": artist,
            "bpm": bpm,
            "bpm_tolerance": bpm_tolerance,
            "sort": sort,
            "view": view,
        }
        params.update(overrides)
        return urlencode({k: v for k, v in params.items() if v}, doseq=True)

    context = {
        "items": items,
        "status_by_id": status_by_id,
        "q": q or "",
        "selected_genres": genre,
        "selected_styles": style,
        "year": year or "",
        "label": label or "",
        "artist": artist or "",
        "bpm": bpm or "",
        "selected_bpm_tolerance": bpm_tolerance,
        "sort": sort,
        "view": view,
        "page": page,
        "page_size": PAGE_SIZE,
        "total": total,
        "loaded_count": loaded_count,
        "has_more": has_more,
        "continuation": continuation,
        "next_query": query_string(page=page + 1, continuation="true"),
        "all_query": query_string(show_all="true", continuation="true"),
        "genres": repository.list_genres(conn),
        "styles": repository.list_styles(conn),
        "years": repository.list_years(conn),
        "labels": repository.list_labels(conn),
        "artists": repository.list_artists(conn),
        "grid_query": query_string(view="grid"),
        "list_query": query_string(view="list"),
        "all_crates": repository.list_crates_with_counts(conn),
        "crate": None,
        "sticker_selection_ids": sticker_selection.selected_ids(conn),
        "select_all_query": query_string(),
    }

    partial_template = (
        "partials/results_grid.html" if view == "grid" else "partials/results_list.html"
    )

    if request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse(request, partial_template, context)

    context["results_partial"] = partial_template
    return templates.TemplateResponse(request, "releases.html", context)


@app.get("/releases/{release_id}", response_class=HTMLResponse)
def release_detail(
    request: Request,
    release_id: int,
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    release = repository.get_release_detail(conn, release_id)
    if release is None:
        raise HTTPException(status_code=404, detail="Release not found")
    fuzzy_matrix = youtube_matching.debug_score_matrix(conn, release_id)
    return templates.TemplateResponse(
        request,
        "release_detail.html",
        {
            "release": release,
            "fuzzy_matrix": fuzzy_matrix,
            "all_crates": repository.list_crates_with_counts(conn),
            "crate_ids": repository.crate_ids_for_release(conn, release_id),
            "in_sticker_selection": sticker_selection.is_selected(conn, release_id),
        },
    )


@app.post("/releases/{release_id}/match_youtube", response_class=HTMLResponse)
def match_youtube_for_release(
    request: Request,
    release_id: int,
    force: bool = False,
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    release = repository.get_release_detail(conn, release_id)
    if release is None:
        raise HTTPException(status_code=404, detail="Release not found")

    match_error = None
    try:
        with ollama_client.ensure_ollama_running():
            youtube_matching.match_release(
                conn,
                release_id,
                release.title,
                force=force,
                max_workers=youtube_matching.DEFAULT_WORKERS,
                extractor=youtube_matching.build_extractor(get_config().youtube_cookies_browser),
                call_ollama=youtube_matching.call_ollama,
                confident_threshold=get_config().fuzzy_confident_threshold,
                tie_margin=get_config().fuzzy_tie_margin,
                denoise_terms=get_config().fuzzy_denoise_terms,
            )
    except Exception as exc:
        match_error = str(exc)

    release = repository.get_release_detail(conn, release_id)
    fuzzy_matrix = youtube_matching.debug_score_matrix(conn, release_id)
    return templates.TemplateResponse(
        request,
        "partials/release_detail_content.html",
        {"release": release, "fuzzy_matrix": fuzzy_matrix, "match_error": match_error},
    )


@app.post("/releases/{release_id}/clear_youtube_matches", response_class=HTMLResponse)
def clear_youtube_matches_for_release(
    request: Request,
    release_id: int,
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    release = repository.get_release_detail(conn, release_id)
    if release is None:
        raise HTTPException(status_code=404, detail="Release not found")

    youtube_matching.clear_matches(conn, release_id)

    release = repository.get_release_detail(conn, release_id)
    # Not debug_score_matrix(): it's always freshly recomputed from yt_title
    # regardless of match state, so it would still show scores for a release
    # that was just wiped back to "never matched" - clearing should give a
    # genuinely blank slate.
    return templates.TemplateResponse(
        request,
        "partials/release_detail_content.html",
        {"release": release, "fuzzy_matrix": [], "match_error": None},
    )


@app.post("/releases/{release_id}/download_tracks", response_class=HTMLResponse)
def download_tracks_for_release(
    request: Request,
    release_id: int,
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    release = repository.get_release_detail(conn, release_id)
    if release is None:
        raise HTTPException(status_code=404, detail="Release not found")

    download_error = None
    download_results = None
    try:
        download_results = audio_download.download_release_tracks(
            conn,
            release_id,
            get_config().audio_dir,
            downloader=audio_download.build_downloader(
                get_config().youtube_cookies_browser, get_config().youtube_audio_max_bitrate_kbps
            ),
        )
    except Exception as exc:
        download_error = str(exc)

    fuzzy_matrix = youtube_matching.debug_score_matrix(conn, release_id)
    return templates.TemplateResponse(
        request,
        "partials/release_detail_content.html",
        {
            "release": release,
            "fuzzy_matrix": fuzzy_matrix,
            "match_error": None,
            "download_error": download_error,
            "download_results": download_results,
        },
    )


@app.post("/releases/{release_id}/analyze_tracks", response_class=HTMLResponse)
def analyze_tracks_for_release(
    request: Request,
    release_id: int,
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    release = repository.get_release_detail(conn, release_id)
    if release is None:
        raise HTTPException(status_code=404, detail="Release not found")

    config = get_config()
    analyze_error = None
    analyze_results = None
    try:
        analyze_results = audio_analysis.analyze_release_tracks(
            conn,
            release_id,
            config.audio_dir,
            config.waveform_dir,
            config.essentia_models_dir,
            downloader=audio_download.build_downloader(
                config.youtube_cookies_browser, config.youtube_audio_max_bitrate_kbps
            ),
        )
    except Exception as exc:
        analyze_error = str(exc)

    release = repository.get_release_detail(conn, release_id)
    fuzzy_matrix = youtube_matching.debug_score_matrix(conn, release_id)
    return templates.TemplateResponse(
        request,
        "partials/release_detail_content.html",
        {
            "release": release,
            "fuzzy_matrix": fuzzy_matrix,
            "match_error": None,
            "analyze_error": analyze_error,
            "analyze_results": analyze_results,
        },
    )


@app.post("/releases/{release_id}/clear_analysis", response_class=HTMLResponse)
def clear_analysis_for_release(
    request: Request,
    release_id: int,
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    release = repository.get_release_detail(conn, release_id)
    if release is None:
        raise HTTPException(status_code=404, detail="Release not found")

    audio_analysis.clear_analysis(conn, release_id, get_config().waveform_dir)

    release = repository.get_release_detail(conn, release_id)
    fuzzy_matrix = youtube_matching.debug_score_matrix(conn, release_id)
    return templates.TemplateResponse(
        request,
        "partials/release_detail_content.html",
        {"release": release, "fuzzy_matrix": fuzzy_matrix, "match_error": None},
    )


@app.get("/waveforms/{release_id}/{filename}")
def get_waveform(release_id: int, filename: str) -> FileResponse:
    # Resolved against the (possibly test-overridden) configured waveform
    # dir on every request, rather than a fixed StaticFiles mount, since
    # get_config() can change between requests (e.g. across tests). Reject
    # any path-separator/traversal attempt in filename before touching the
    # filesystem — it must name a single file directly inside release_dir.
    if "/" in filename or "\\" in filename or filename in (".", ".."):
        raise HTTPException(status_code=404, detail="Waveform not found")
    release_dir = (get_config().waveform_dir / str(release_id)).resolve()
    path = (release_dir / filename).resolve()
    if path.parent != release_dir or not path.is_file():
        raise HTTPException(status_code=404, detail="Waveform not found")
    return FileResponse(path)


@app.get("/covers/{release_id}")
def get_cover(release_id: int, conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    """Serves the locally cached cover for release_id, downloading+caching
    it from Discogs first if this is the first request for it (see
    cover_art.get_or_fetch_cover) - templates link here instead of directly
    to release.cover_image_url so repeat views don't keep hotlinking
    Discogs' image servers.
    """
    row = conn.execute("SELECT cover_image_url FROM releases WHERE id = ?", (release_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Release not found")

    try:
        path = cover_art.get_or_fetch_cover(get_config().cover_dir, release_id, row["cover_image_url"])
    except requests.RequestException:
        raise HTTPException(status_code=404, detail="Cover unavailable")
    if path is None:
        raise HTTPException(status_code=404, detail="No cover for this release")
    return FileResponse(path)


@app.get("/releases/{release_id}/sticker.pdf")
def get_release_sticker(release_id: int, conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    """Generates (or regenerates - always fresh, cheap) and serves a
    printable sticker PDF for one release (Avery Zweckform L4744REV-65
    format), from already-cached cover/waveform/analysis data - see
    stickers.py.
    """
    release = repository.get_release_detail(conn, release_id)
    if release is None:
        raise HTTPException(status_code=404, detail="Release not found")

    config = get_config()
    path = stickers.generate_release_sticker_pdf(
        release,
        config.cover_dir,
        config.waveform_dir,
        config.sticker_dir,
        config.key_notation,
        config.cjk_font_path,
        _active_sticker_preset(config),
        config.dj_name,
    )
    return FileResponse(path, media_type="application/pdf", filename=f"{release.id}.pdf")


def _make_sync_runner(force_refetch: bool) -> tasks.Runner:
    def run(
        on_progress: tasks.ProgressCallback, on_error: tasks.ErrorCallback, should_stop: tasks.StopCheck
    ) -> None:
        config = get_config()
        conn = db.connect(config.db_path)
        try:
            api = build_discogs_api(config)

            def progress(index: int, total: int, release_id: int, title: str) -> None:
                on_progress(index, total, title)

            def error(release_id: int, exc: Exception) -> None:
                on_error(f"release {release_id}: {exc}")

            sync.sync_collection(
                conn,
                api,
                on_progress=progress,
                on_error=error,
                force_refetch=force_refetch,
                should_stop=should_stop,
            )
        finally:
            conn.close()

    return run


@app.get("/sync/status", response_class=HTMLResponse)
def sync_status(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "partials/sync_status.html", {"task": tasks.get_task("sync")}
    )


@app.post("/sync/stop", response_class=HTMLResponse)
def sync_stop(request: Request) -> HTMLResponse:
    task = tasks.request_stop("sync")
    return templates.TemplateResponse(request, "partials/sync_status.html", {"task": task})


@app.post("/sync/start", response_class=HTMLResponse)
def sync_start(request: Request, force: bool = Form(False)) -> HTMLResponse:
    task = tasks.start_task("sync", _make_sync_runner(force))
    return templates.TemplateResponse(request, "partials/sync_status.html", {"task": task})


def _make_youtube_runner(force: bool, max_workers: int) -> tasks.Runner:
    def run(
        on_progress: tasks.ProgressCallback, on_error: tasks.ErrorCallback, should_stop: tasks.StopCheck
    ) -> None:
        config = get_config()
        conn = db.connect(config.db_path)
        try:
            def progress(index: int, total: int, release_id: int, phase: str) -> None:
                on_progress(index, total, f"release {release_id}")

            def error(release_id: int, exc: Exception) -> None:
                on_error(f"release {release_id}: {exc}")

            pipeline.match_and_download_library(
                conn,
                config.db_path,
                config.audio_dir,
                downloader=audio_download.build_downloader(
                    config.youtube_cookies_browser, config.youtube_audio_max_bitrate_kbps
                ),
                extractor=youtube_matching.build_extractor(config.youtube_cookies_browser),
                call_ollama=youtube_matching.call_ollama,
                force=force,
                max_workers=max_workers,
                confident_threshold=config.fuzzy_confident_threshold,
                tie_margin=config.fuzzy_tie_margin,
                denoise_terms=config.fuzzy_denoise_terms,
                flac_dir=config.local_flac_dir,
                local_confident_threshold=config.local_match_confident_threshold,
                on_progress=progress,
                on_error=error,
                should_stop=should_stop,
            )
        finally:
            conn.close()

    return run


def _pending_count(counter: Callable[[sqlite3.Connection], int]) -> int:
    conn = db.connect(get_config().db_path)
    try:
        return counter(conn)
    finally:
        conn.close()


@app.get("/pipeline/pending", response_class=HTMLResponse)
def pipeline_pending(request: Request) -> HTMLResponse:
    """One combined "N releases pending" indicator for the whole task row,
    rather than each of Match YT / Analyze audio / Run All separately
    showing (mostly the same) number next to its own button."""
    pending = _pending_count(pipeline.count_pending_releases)
    return templates.TemplateResponse(request, "partials/pipeline_pending.html", {"pending": pending})


@app.get("/youtube/status", response_class=HTMLResponse)
def youtube_status(request: Request) -> HTMLResponse:
    task = tasks.get_task("youtube")
    return templates.TemplateResponse(request, "partials/youtube_status.html", {"task": task})


@app.post("/youtube/stop", response_class=HTMLResponse)
def youtube_stop(request: Request) -> HTMLResponse:
    task = tasks.request_stop("youtube")
    return templates.TemplateResponse(request, "partials/youtube_status.html", {"task": task})


@app.post("/youtube/start", response_class=HTMLResponse)
def youtube_start(request: Request, force: bool = Form(False)) -> HTMLResponse:
    task = tasks.start_task("youtube", _make_youtube_runner(force, get_config().max_workers))
    return templates.TemplateResponse(request, "partials/youtube_status.html", {"task": task})


def _make_audio_runner(force: bool) -> tasks.Runner:
    def run(
        on_progress: tasks.ProgressCallback, on_error: tasks.ErrorCallback, should_stop: tasks.StopCheck
    ) -> None:
        config = get_config()
        conn = db.connect(config.db_path)
        try:
            def progress(index: int, total: int, release_id: int) -> None:
                on_progress(index, total, f"release {release_id}")

            def error(release_id: int, exc: Exception) -> None:
                on_error(f"release {release_id}: {exc}")

            audio_analysis.analyze_library(
                conn,
                config.db_path,
                config.audio_dir,
                config.waveform_dir,
                config.essentia_models_dir,
                downloader=audio_download.build_downloader(
                    config.youtube_cookies_browser, config.youtube_audio_max_bitrate_kbps
                ),
                force=force,
                max_workers=config.max_workers,
                on_progress=progress,
                on_error=error,
                should_stop=should_stop,
            )
        finally:
            conn.close()

    return run


@app.get("/audio/status", response_class=HTMLResponse)
def audio_status(request: Request) -> HTMLResponse:
    task = tasks.get_task("audio")
    return templates.TemplateResponse(request, "partials/audio_status.html", {"task": task})


@app.post("/audio/stop", response_class=HTMLResponse)
def audio_stop(request: Request) -> HTMLResponse:
    task = tasks.request_stop("audio")
    return templates.TemplateResponse(request, "partials/audio_status.html", {"task": task})


@app.post("/audio/start", response_class=HTMLResponse)
def audio_start(request: Request, force: bool = Form(False)) -> HTMLResponse:
    task = tasks.start_task("audio", _make_audio_runner(force))
    return templates.TemplateResponse(request, "partials/audio_status.html", {"task": task})


def _make_run_all_runner(force: bool, max_workers: int) -> tasks.Runner:
    def run(
        on_progress: tasks.ProgressCallback, on_error: tasks.ErrorCallback, should_stop: tasks.StopCheck
    ) -> None:
        config = get_config()
        conn = db.connect(config.db_path)
        try:
            phase_labels = {"match": "matching/downloading", "analyze": "analyzing"}

            def progress(index: int, total: int, release_id: int, phase: str) -> None:
                on_progress(index, total, f"{phase_labels.get(phase, phase)}: release {release_id}")

            def error(release_id: int, exc: Exception) -> None:
                on_error(f"release {release_id}: {exc}")

            pipeline.run_full_pipeline(
                conn,
                config.db_path,
                config.audio_dir,
                config.waveform_dir,
                config.essentia_models_dir,
                downloader=audio_download.build_downloader(
                    config.youtube_cookies_browser, config.youtube_audio_max_bitrate_kbps
                ),
                extractor=youtube_matching.build_extractor(config.youtube_cookies_browser),
                call_ollama=youtube_matching.call_ollama,
                force=force,
                max_workers=max_workers,
                confident_threshold=config.fuzzy_confident_threshold,
                tie_margin=config.fuzzy_tie_margin,
                denoise_terms=config.fuzzy_denoise_terms,
                flac_dir=config.local_flac_dir,
                local_confident_threshold=config.local_match_confident_threshold,
                on_progress=progress,
                on_error=error,
                should_stop=should_stop,
            )
        finally:
            conn.close()

    return run


@app.get("/run_all/status", response_class=HTMLResponse)
def run_all_status(request: Request) -> HTMLResponse:
    task = tasks.get_task("run_all")
    return templates.TemplateResponse(request, "partials/run_all_status.html", {"task": task})


@app.post("/run_all/stop", response_class=HTMLResponse)
def run_all_stop(request: Request) -> HTMLResponse:
    task = tasks.request_stop("run_all")
    return templates.TemplateResponse(request, "partials/run_all_status.html", {"task": task})


@app.post("/run_all/start", response_class=HTMLResponse)
def run_all_start(request: Request, force: bool = Form(False)) -> HTMLResponse:
    task = tasks.start_task("run_all", _make_run_all_runner(force, get_config().max_workers))
    return templates.TemplateResponse(request, "partials/run_all_status.html", {"task": task})


@app.get("/crates", response_class=HTMLResponse)
def crates_list(request: Request, conn: sqlite3.Connection = Depends(get_db)) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "crates.html", {"crates": repository.list_crates_with_counts(conn)}
    )


@app.post("/crates")
def create_crate(name: str = Form(...), conn: sqlite3.Connection = Depends(get_db)) -> RedirectResponse:
    crate_id = crates_module.create_crate(conn, name)
    return RedirectResponse(url=f"/crates/{crate_id}", status_code=303)


@app.get("/crates/{crate_id}", response_class=HTMLResponse)
def crate_detail(
    request: Request,
    crate_id: int,
    view: str = "grid",
    conn: sqlite3.Connection = Depends(get_db),
) -> HTMLResponse:
    crate = crates_module.get_crate(conn, crate_id)
    if crate is None:
        raise HTTPException(status_code=404, detail="Crate not found")
    if view not in ("grid", "list"):
        view = "grid"

    items = repository.list_releases(conn, crate_id=crate_id, limit=None)
    status_by_id: dict[int, dict[str, str]] = {}
    if view == "list":
        audio_dir = get_config().audio_dir
        for item in items:
            downloaded_count = audio_download.count_downloaded_tracks(audio_dir, item.id)
            status_by_id[item.id] = {
                "matching": _matching_status(item.matched_count, item.track_count, item.matching_processed),
                "download": _availability_status(downloaded_count, item.matched_count, item.track_count),
                "analysis": _availability_status(item.analyzed_count, item.matched_count, item.track_count),
            }

    partial_template = "partials/results_grid.html" if view == "grid" else "partials/results_list.html"
    return templates.TemplateResponse(
        request,
        "crate_detail.html",
        {
            "crate": crate,
            "items": items,
            "status_by_id": status_by_id,
            "total": len(items),
            "view": view,
            "all_crates": [],
            "has_more": False,
            "continuation": False,
            "results_partial": partial_template,
            "sticker_selection_ids": sticker_selection.selected_ids(conn),
        },
    )


@app.post("/crates/{crate_id}/rename")
def rename_crate(
    crate_id: int, name: str = Form(...), conn: sqlite3.Connection = Depends(get_db)
) -> RedirectResponse:
    crates_module.rename_crate(conn, crate_id, name)
    return RedirectResponse(url=f"/crates/{crate_id}", status_code=303)


@app.post("/crates/{crate_id}/delete")
def delete_crate(crate_id: int, conn: sqlite3.Connection = Depends(get_db)) -> RedirectResponse:
    crates_module.delete_crate(conn, crate_id)
    return RedirectResponse(url="/crates", status_code=303)


@app.post("/crates/add_release")
def add_release_to_crate(
    crate_id: int = Form(...),
    release_id: int = Form(...),
    redirect_to: str = Form("/releases"),
    conn: sqlite3.Connection = Depends(get_db),
) -> RedirectResponse:
    crates_module.add_release(conn, crate_id, release_id)
    return RedirectResponse(url=redirect_to, status_code=303)


@app.post("/crates/{crate_id}/remove_release/{release_id}")
def remove_release_from_crate(
    crate_id: int,
    release_id: int,
    redirect_to: str = Form(None),
    conn: sqlite3.Connection = Depends(get_db),
) -> RedirectResponse:
    crates_module.remove_release(conn, crate_id, release_id)
    return RedirectResponse(url=redirect_to or f"/crates/{crate_id}", status_code=303)


@app.get("/stickers", response_class=HTMLResponse)
def stickers_page(
    request: Request, view: str = "grid", conn: sqlite3.Connection = Depends(get_db)
) -> HTMLResponse:
    if view not in ("grid", "list"):
        view = "grid"

    items = repository.list_releases(conn, in_sticker_selection=True, limit=None)
    status_by_id: dict[int, dict[str, str]] = {}
    if view == "list":
        audio_dir = get_config().audio_dir
        for item in items:
            downloaded_count = audio_download.count_downloaded_tracks(audio_dir, item.id)
            status_by_id[item.id] = {
                "matching": _matching_status(item.matched_count, item.track_count, item.matching_processed),
                "download": _availability_status(downloaded_count, item.matched_count, item.track_count),
                "analysis": _availability_status(item.analyzed_count, item.matched_count, item.track_count),
            }

    partial_template = "partials/results_grid.html" if view == "grid" else "partials/results_list.html"
    return templates.TemplateResponse(
        request,
        "stickers.html",
        {
            "items": items,
            "status_by_id": status_by_id,
            "total": len(items),
            "view": view,
            "all_crates": [],
            "crate": None,
            "has_more": False,
            "continuation": False,
            "results_partial": partial_template,
            "sticker_selection_ids": {item.id for item in items},
            "sticker_view": True,
        },
    )


@app.post("/stickers/add")
def add_release_to_stickers(
    release_id: int = Form(...),
    redirect_to: str = Form("/releases"),
    conn: sqlite3.Connection = Depends(get_db),
) -> RedirectResponse:
    sticker_selection.add_release(conn, release_id)
    return RedirectResponse(url=redirect_to, status_code=303)


@app.post("/stickers/remove/{release_id}")
def remove_release_from_stickers(
    release_id: int,
    redirect_to: str = Form(None),
    conn: sqlite3.Connection = Depends(get_db),
) -> RedirectResponse:
    sticker_selection.remove_release(conn, release_id)
    return RedirectResponse(url=redirect_to or "/stickers", status_code=303)


@app.post("/stickers/select_all")
def select_all_for_stickers(
    q: str | None = None,
    genre: list[str] = Query(default=[]),
    style: list[str] = Query(default=[]),
    year: str | None = None,
    label: str | None = None,
    artist: str | None = None,
    bpm: str | None = None,
    bpm_tolerance: list[str] = Query(default=[]),
    redirect_to: str = Form("/releases"),
    conn: sqlite3.Connection = Depends(get_db),
) -> RedirectResponse:
    genre = [g for g in genre if g]
    style = [s for s in style if s]
    bpm_tolerance = [t for t in bpm_tolerance if t]
    filter_kwargs = _parse_release_filters(q, genre, style, year, label, artist, bpm, bpm_tolerance)
    matching = repository.list_releases(conn, limit=None, **filter_kwargs)
    sticker_selection.add_many(conn, [item.id for item in matching])
    return RedirectResponse(url=redirect_to, status_code=303)


@app.post("/stickers/clear")
def clear_sticker_selection(conn: sqlite3.Connection = Depends(get_db)) -> RedirectResponse:
    sticker_selection.clear(conn)
    return RedirectResponse(url="/stickers", status_code=303)


@app.get("/stickers/sheet.pdf")
def get_sticker_sheet(conn: sqlite3.Connection = Depends(get_db)) -> FileResponse:
    ids = sticker_selection.selected_ids(conn)
    if not ids:
        raise HTTPException(status_code=404, detail="Sticker selection is empty")

    releases_for_sheet = [
        r for r in (repository.get_release_detail(conn, release_id) for release_id in ids) if r is not None
    ]
    config = get_config()
    path = stickers.generate_sticker_sheet_pdf(
        releases_for_sheet,
        config.cover_dir,
        config.waveform_dir,
        config.sticker_dir,
        config.key_notation,
        config.cjk_font_path,
        _active_sticker_preset(config),
        config.dj_name,
    )
    return FileResponse(path, media_type="application/pdf", filename="stickers.pdf")


# OAuth 1.0a needs the same discogs_client.Client instance across two
# separate requests (the request token/secret it gets back from Discogs on
# the first call live on that object, consumed by the second - see
# get_authorize_url()/get_access_token() in the installed library, no way
# to resume with just the token strings on a fresh object). A plain
# module-level variable is enough for this - single-user local app, same
# "in-memory, no real session store" pattern tasks.py's task registry
# already uses. If the server restarts mid-flow the user just clicks
# "authorize" again.
_pending_oauth_client: discogs_client.Client | None = None


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request, saved: bool = False, oauth_error: str | None = None) -> HTMLResponse:
    callback_url = str(request.url.replace(path="/settings/discogs/callback", query=""))
    return templates.TemplateResponse(
        request,
        "settings.html",
        {
            "config": get_config(),
            "saved": saved,
            "oauth_error": oauth_error,
            "oauth_callback_url": callback_url,
            "sticker_presets": stickers.PRESETS,
        },
    )


@app.post("/settings")
def save_settings(
    discogs_user_token: str = Form(""),
    fuzzy_confident_threshold: str = Form(...),
    fuzzy_tie_margin: str = Form(...),
    fuzzy_denoise_terms: str = Form(...),
    youtube_cookies_browser: str = Form(""),
    youtube_audio_max_bitrate_kbps: str = Form(""),
    local_flac_dir: str = Form(""),
    cjk_font_path: str = Form(""),
    local_match_confident_threshold: str = Form(...),
    max_workers: str = Form(...),
    key_notation: str = Form(...),
    sticker_preset: str = Form(...),
    dj_name: str = Form(""),
) -> RedirectResponse:
    values = {
        "FUZZY_CONFIDENT_THRESHOLD": fuzzy_confident_threshold,
        "FUZZY_TIE_MARGIN": fuzzy_tie_margin,
        "FUZZY_DENOISE_TERMS": fuzzy_denoise_terms,
        "YOUTUBE_COOKIES_BROWSER": youtube_cookies_browser,
        "YOUTUBE_AUDIO_MAX_BITRATE_KBPS": youtube_audio_max_bitrate_kbps,
        "LOCAL_FLAC_DIR": local_flac_dir,
        "STICKER_PRESET": sticker_preset,
        "DJ_NAME": dj_name,
        "CJK_FONT_PATH": cjk_font_path,
        "LOCAL_MATCH_CONFIDENT_THRESHOLD": local_match_confident_threshold,
        "MAX_WORKERS": max_workers,
        "KEY_NOTATION": key_notation,
    }
    # Deliberately not pre-filled/always-submitted like the fields above -
    # the token isn't shown back in plaintext once set, so submitting the
    # form without retyping it must not blank it out.
    if discogs_user_token.strip():
        values["DISCOGS_USER_TOKEN"] = discogs_user_token.strip()

    set_config_values(values)
    get_config.cache_clear()
    return RedirectResponse(url="/settings?saved=true", status_code=303)


@app.post("/settings/discogs/authorize")
def start_discogs_oauth(
    request: Request, consumer_key: str = Form(...), consumer_secret: str = Form(...)
) -> RedirectResponse:
    global _pending_oauth_client

    consumer_key = consumer_key.strip()
    consumer_secret = consumer_secret.strip()
    # Saved before attempting the redirect, so a failed/abandoned attempt
    # doesn't lose what was typed in.
    set_config_values({"DISCOGS_CONSUMER_KEY": consumer_key, "DISCOGS_CONSUMER_SECRET": consumer_secret})
    get_config.cache_clear()

    client = discogs_client.Client(USER_AGENT, consumer_key=consumer_key, consumer_secret=consumer_secret)
    callback_url = str(request.url.replace(path="/settings/discogs/callback", query=""))
    try:
        _, _, authorize_url = client.get_authorize_url(callback_url)
    except Exception as exc:
        return RedirectResponse(url=f"/settings?{urlencode({'oauth_error': str(exc)})}", status_code=303)

    _pending_oauth_client = client
    return RedirectResponse(url=authorize_url, status_code=303)


@app.get("/settings/discogs/callback")
def discogs_oauth_callback(oauth_verifier: str | None = None) -> RedirectResponse:
    global _pending_oauth_client

    client = _pending_oauth_client
    _pending_oauth_client = None

    if client is None or not oauth_verifier:
        error = "No pending authorization (did the server restart mid-flow?) - please try again."
        return RedirectResponse(url=f"/settings?{urlencode({'oauth_error': error})}", status_code=303)

    try:
        token, secret = client.get_access_token(oauth_verifier)
    except Exception as exc:
        return RedirectResponse(url=f"/settings?{urlencode({'oauth_error': str(exc)})}", status_code=303)

    set_config_values({"DISCOGS_OAUTH_TOKEN": token, "DISCOGS_OAUTH_TOKEN_SECRET": secret})
    get_config.cache_clear()
    return RedirectResponse(url="/settings?saved=true", status_code=303)


@app.post("/settings/discogs/disconnect")
def disconnect_discogs_oauth() -> RedirectResponse:
    set_config_values(
        {
            "DISCOGS_OAUTH_TOKEN": "",
            "DISCOGS_OAUTH_TOKEN_SECRET": "",
        }
    )
    get_config.cache_clear()
    return RedirectResponse(url="/settings?saved=true", status_code=303)


def main() -> None:
    import uvicorn

    host = os.environ.get("WEB_HOST", "127.0.0.1")
    port = int(os.environ.get("WEB_PORT", "8000"))
    uvicorn.run("discql.web.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
