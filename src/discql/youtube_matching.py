from __future__ import annotations

import json
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Callable

from rapidfuzz import fuzz
from rapidfuzz.utils import default_process as fuzz_process

from discql.ollama_client import OllamaCaller, call_ollama

# Default fuzzy-score threshold above which a match is trusted without ever
# consulting the LLM. Overridable per call (see resolve_release/match_release/
# match_videos) so it can be configured via discql.config instead of
# hardcoded.
CONFIDENT_THRESHOLD = 0.8

# Minimum gap required between a video's best- and second-best-scoring track
# for the confident shortcut above to apply. Guards against e.g. two "Product
# (X Mix)" tracks scoring near-identically on WRatio, where trusting whichever
# one happens to win the tie (rather than letting the LLM disambiguate) can
# silently commit to the wrong track. Also configurable, same mechanism as
# CONFIDENT_THRESHOLD.
TIE_MARGIN = 0.05

# Additional generic terms stripped from a video title during denoising,
# alongside the release title / artist name(s) / year. Unlike those, these
# aren't derived from the release being matched — they're boilerplate that
# recurs across many uploaders/releases (e.g. "Original Mix" tacked onto
# every upload regardless of whether the actual track title says so).
# Configurable via discql.config (FUZZY_DENOISE_TERMS), not hardcoded.
DEFAULT_DENOISE_TERMS: list[str] = ["Original Mix"]

DEFAULT_WORKERS = 3

# Called after each release is processed: (index, total, release_id)
ProgressCallback = Callable[[int, int, int], None]
# Called when a single release's matching fails: (release_id, exception)
ErrorCallback = Callable[[int, Exception], None]

VideoExtractor = Callable[[str], dict]


def count_pending_releases(conn: sqlite3.Connection) -> int:
    """How many releases a non-reprocess match_videos() run would still
    touch - same "at least one still-unclassified video" condition used to
    skip releases by default."""
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT r.id) AS n
        FROM releases r
        JOIN release_videos rv ON rv.release_id = r.id
        WHERE r.removed_from_discogs_at IS NULL AND rv.video_type IS NULL
        """
    ).fetchone()
    return row["n"]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class VideoMetadata:
    title: str | None
    duration: int | None
    uploader: str | None
    description: str | None
    tags: list[str] = field(default_factory=list)
    chapters: list[dict] = field(default_factory=list)


def build_extractor(cookies_browser: str | None = None) -> VideoExtractor:
    """Build a metadata extractor, optionally authenticated via
    --cookies-from-browser (yt-dlp's cookiesfrombrowser option) so
    age-restricted videos — which YouTube otherwise refuses to serve
    metadata for at all, regardless of player_client — can be fetched too.
    """

    def extractor(url: str) -> dict:
        from yt_dlp import YoutubeDL

        opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
            "nocheckcertificate": True,
            "noplaylist": True,
            # We only need metadata (title/duration/description/tags/chapters), not
            # to resolve playable stream URLs, so a client that doesn't require a
            # JS runtime for signature deciphering is fine — avoids both installing
            # a JS runtime and the "No supported JavaScript runtime" warning.
            # A single client can fail on a video that's genuinely available
            # (confirmed via YouTube's oembed endpoint returning real metadata for
            # a video android_vr alone reported as "unavailable") — yt-dlp tries
            # clients in this list in order and uses whichever succeeds, so listing
            # a couple of fallbacks meaningfully reduces false "unavailable" results.
            "extractor_args": {"youtube": {"player_client": ["android_vr", "android", "web"]}},
        }
        if cookies_browser:
            opts["cookiesfrombrowser"] = (cookies_browser,)
            # Providing cookies makes yt-dlp skip android_vr/android entirely
            # (they don't support cookie auth), falling back to "web" alone -
            # which needs its signature/n-challenge solved to resolve any
            # real formats at all (YouTube's "SABR streaming" push). yt-dlp
            # only does that via a challenge-solver script it fetches from
            # its own GitHub repo on demand - opt-in since it's remote code,
            # but harmless/unused when cookies aren't in play (no "web"-only
            # fallback needed then). Confirmed concretely: without this,
            # cookie-authenticated requests came back with "Only images are
            # available for download" - no audio formats at all.
            opts["remote_components"] = ["ejs:github"]
        with YoutubeDL(opts) as ydl:
            return ydl.extract_info(url, download=False)

    return extractor


_default_extractor = build_extractor()


def fetch_video_metadata(url: str, extractor: VideoExtractor = _default_extractor) -> VideoMetadata:
    info = extractor(url)
    if not info:
        # yt-dlp's extract_info() can return None without raising (observed
        # concretely on a real release: a transient YouTube-side block/
        # rate-limit at fetch time, video fetchable again moments later) -
        # treating that the same as "successfully fetched, no metadata at
        # all" (the old `or {}` here) wrote an all-NULL row indistinguishable
        # from an intentionally-skipped video, and silently excluded it from
        # matching forever (match_release only considers videos with
        # yt_title IS NOT NULL) - no error surfaced, no retry without a
        # force re-match. Raising here instead routes it through
        # match_release's existing fetch-failure handling (marks the video
        # unrelated/"metadata fetch failed", reports via on_error) - the
        # same as a genuine exception, not a silent black hole.
        raise RuntimeError("extractor returned no metadata")
    chapters = [
        {"title": c.get("title"), "start_time": c.get("start_time")}
        for c in (info.get("chapters") or [])
    ]
    return VideoMetadata(
        title=info.get("title"),
        duration=info.get("duration"),
        uploader=info.get("uploader"),
        description=info.get("description"),
        tags=list(info.get("tags") or []),
        chapters=chapters,
    )


def parse_track_duration(text: str | None) -> int | None:
    """Convert "8:49" / "1:02:33" style text into seconds."""
    if not text:
        return None
    parts = text.strip().split(":")
    try:
        numbers = [int(p) for p in parts]
    except ValueError:
        return None
    seconds = 0
    for n in numbers:
        seconds = seconds * 60 + n
    return seconds


# If both track and video duration are known and differ by more than this
# fraction of the track's duration, the pair is disqualified outright (score
# 0.0) without ever running title fuzzy matching — a >20% duration mismatch
# means it's not the same recording regardless of how similar the titles
# look, so there's no need to spend a fuzzy (or later LLM) comparison on it.
DURATION_DEVIANCE_THRESHOLD = 0.2


def fuzzy_score(
    track_title: str,
    track_artist: str | None,
    track_duration: int | None,
    video_title: str,
    video_duration: int | None,
) -> float:
    """Combined title+duration match score, normalized to [0, 1]."""
    if track_duration and video_duration:
        diff_ratio = abs(track_duration - video_duration) / track_duration
        if diff_ratio > DURATION_DEVIANCE_THRESHOLD:
            return 0.0

    # processor=fuzz_process: rapidfuzz's WRatio defaults to a raw, case-sensitive
    # comparison (confirmed concretely: "HELLO WORLD TRACK" vs "hello world track"
    # scores ~12/100 despite being the same title) - lowercases and strips
    # punctuation to whitespace before comparing, so case/punctuation differences
    # between Discogs and YouTube titles don't tank the score.
    title_score = fuzz.WRatio(track_title, video_title, processor=fuzz_process)
    if track_artist:
        title_score = max(
            title_score, fuzz.WRatio(f"{track_title} {track_artist}", video_title, processor=fuzz_process)
        )

    if track_duration and video_duration:
        duration_score = max(0.0, 100 - diff_ratio * 100)
        return (title_score * 0.7 + duration_score * 0.3) / 100

    return title_score / 100


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def denoise_video_title(
    video_title: str,
    release_title: str,
    artist_names: list[str],
    extra_terms: list[str] | None = None,
) -> str:
    """Strip release title, artist name(s), any 4-digit year, and any
    configured extra_terms from a video title. Uploaders that post many
    videos for the same release/artist tend to repeat these identically in
    every video's title (e.g. a channel prefixing every upload with "ARTIST
    1983 ..."), which is pure noise for fuzzy title matching — it doesn't
    discriminate between tracks, it just dilutes the part of the string that
    actually does (the track title itself). extra_terms covers generic
    boilerplate that isn't specific to this release (see DEFAULT_DENOISE_TERMS
    / FUZZY_DENOISE_TERMS config).
    """
    text = video_title
    for name in [release_title, *artist_names, *(extra_terms or [])]:
        if name:
            text = re.sub(re.escape(name), " ", text, flags=re.IGNORECASE)
    text = _YEAR_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


_ALNUM_RUN_RE = re.compile(r"[A-Za-z0-9]{2,}")


def _has_meaningful_content(text: str) -> bool:
    """True if text has at least one real word/number left, not just leftover
    punctuation/whitespace. Guards against a video title that's entirely
    composed of the release title/artist name(s) (e.g. just "Artist - Album")
    denoising down to something like "-" — using that degenerate text for
    scoring or an LLM prompt is worse than not denoising at all, since it
    throws away the one signal (however fuzzy-unmatchable) the LLM could
    otherwise reason from.
    """
    return bool(_ALNUM_RUN_RE.search(text))


@dataclass
class TrackInput:
    id: int
    position: str | None
    title: str
    artist: str | None
    duration_seconds: int | None


@dataclass
class VideoInput:
    id: int
    title: str | None
    description: str | None
    duration: int | None
    chapters: list[dict]


@dataclass
class MatchResult:
    video_id: int
    track_id: int | None
    fuzzy_score: float
    video_type: str  # "full_track" | "compilation_or_mix" | "unrelated"
    best_track_position: str | None = None
    llm_confidence: float | None = None
    llm_reasoning: str | None = None
    video_type_reasoning: str | None = None
    is_selected: bool = False


def _selection_key(m: MatchResult) -> float:
    """Ranks a track's candidate videos for is_selected. Always factors in
    fuzzy_score (how well this specific video's title actually fits this
    specific track) alongside llm_confidence, rather than trusting raw LLM
    confidence alone: a small local model's confidence is both miscalibrated
    (1.0 vs 0.99 is noise) and can be outright wrong (high confidence on a
    misclassified track position), so it shouldn't be able to fully mask a
    clearly weaker (or clearly stronger) textual match.
    """
    llm_component = m.llm_confidence if m.llm_confidence is not None else m.fuzzy_score
    return (m.fuzzy_score + llm_component) / 2


def _select_best_per_track(results: list[MatchResult]) -> None:
    by_track: dict[int, list[MatchResult]] = {}
    for r in results:
        if r.track_id is not None:
            by_track.setdefault(r.track_id, []).append(r)
    for matches in by_track.values():
        best = max(matches, key=_selection_key)
        for m in matches:
            m.is_selected = m is best


def _format_duration(seconds: int | None) -> str:
    if seconds is None:
        return "unknown"
    return f"{seconds // 60}:{seconds % 60:02d}"


def _build_single_video_prompt(
    release_title: str,
    artist_names: list[str],
    tracks: list[TrackInput],
    video: VideoInput,
    best_track: TrackInput | None,
    best_score: float,
) -> str:
    tracklist_lines = "\n".join(
        f"- {t.position or '?'}: \"{t.title}\" by {t.artist or ', '.join(artist_names)} "
        f"({_format_duration(t.duration_seconds)})"
        for t in tracks
    )
    description = (video.description or "")[:300].replace("\n", " ")

    # Deliberately NOT telling the model what a simple text-similarity heuristic
    # already guessed: small local models tend to just rubber-stamp whatever
    # hint is given instead of independently comparing the actual title text
    # against the tracklist. It has to work this out from the raw video
    # metadata alone.
    return f"""You are matching a YouTube video to a track on a vinyl record release, for building
a personal music library. Classify it "full_track" only if it is a single continuous recording
of exactly one track from the tracklist below. Note: a tracklist entry like "(Dub Mix)", "(Club Mix)",
"(Radio Mix)", "(Extended Mix)" etc. names ONE remix/version of ONE song — the word "Mix" there does
NOT mean it's a compilation. Only classify "compilation_or_mix" if the VIDEO itself contains short
previews/snippets of multiple DIFFERENT tracklist entries, or is a continuous DJ set/megamix spanning
multiple different songs. If it doesn't correspond to any track at all, or you are not reasonably
confident which track it is, classify it "unrelated".

Release: "{release_title}" by {", ".join(artist_names)}

Tracklist:
{tracklist_lines}

Video to classify: title "{video.title}", duration {_format_duration(video.duration)}, chapters: {len(video.chapters)},
description: "{description}"

Respond with ONLY a single JSON object with exactly these keys:
"classification" (one of "full_track", "compilation_or_mix", "unrelated"),
"matched_track_position" (the tracklist position string if classification is "full_track", else null),
"confidence" (float between 0 and 1), "reasoning" (a short one-sentence explanation)."""


def _classify_one_video_with_llm(
    release_title: str,
    artist_names: list[str],
    tracks: list[TrackInput],
    video: VideoInput,
    best_track: TrackInput | None,
    best_score: float,
    call_ollama: OllamaCaller,
    tie_margin: float = TIE_MARGIN,
) -> MatchResult | None:
    position_to_track = {t.position: t for t in tracks if t.position}
    prompt = _build_single_video_prompt(release_title, artist_names, tracks, video, best_track, best_score)

    try:
        raw = call_ollama(prompt)
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            # Defensive: some models wrap even a single result in a one-element array.
            parsed = parsed[0] if parsed else {}
        classification = parsed.get("classification", "unrelated")
        position = parsed.get("matched_track_position")
        confidence = float(parsed.get("confidence", 0.0))
        reasoning = parsed.get("reasoning")
    except Exception:
        return None

    track = position_to_track.get(position) if position else None
    # The fuzzy score stored must reflect the track the LLM actually claimed,
    # not just the video's own overall best guess (best_score/best_track) -
    # those can be a *different* track entirely. Storing best_score here
    # would overstate fuzzy support for a claim it doesn't actually back,
    # which then lets a wrong-but-confident LLM pick masquerade as having
    # decent fuzzy corroboration in _select_best_per_track.
    if track is not None:
        claimed_fuzzy_score = fuzzy_score(
            track.title, track.artist, track.duration_seconds, video.title or "", video.duration
        )
    else:
        claimed_fuzzy_score = best_score

    # Distrust guard: if the LLM claims a track whose own fuzzy support is
    # clearly weaker than a DIFFERENT track this same video already scores
    # meaningfully higher against, don't commit to the claim — fall back to
    # that stronger fuzzy alternative instead (not "unrelated": it's already
    # the best-scoring candidate across the whole tracklist for this video,
    # just below confident_threshold on its own, which is exactly why this
    # video got escalated to the LLM in the first place). Observed concretely
    # with Arabic-script content, where the model deterministically (not just
    # occasionally) defaulted to the first tracklist entry with a confident
    # but factually wrong justification, regardless of what the video's
    # title/description actually said — while fuzzy scoring, and the video's
    # own description, both pointed at the correct track.
    llm_confidence: float | None = confidence
    if (
        classification == "full_track"
        and track is not None
        and best_track is not None
        and track.id != best_track.id
        and (best_score - claimed_fuzzy_score) >= tie_margin
    ):
        reasoning = (
            f'LLM claimed "{position}" (fuzzy {claimed_fuzzy_score:.2f} for that track), but '
            f'"{best_track.position}" scores clearly higher ({best_score:.2f}) for this video - '
            f'falling back to "{best_track.position}" instead of trusting the LLM claim.'
        )
        track = best_track
        claimed_fuzzy_score = best_score
        llm_confidence = None

    return MatchResult(
        video_id=video.id,
        track_id=track.id if (track and classification == "full_track") else None,
        fuzzy_score=claimed_fuzzy_score,
        video_type=classification,
        best_track_position=best_track.position if best_track else None,
        llm_confidence=llm_confidence,
        llm_reasoning=reasoning,
        video_type_reasoning=reasoning,
    )


def _best_two_scores(
    tracks: list[TrackInput], video_title: str, video_duration: int | None
) -> tuple[TrackInput | None, float, float]:
    """Best-scoring track for a video, plus the runner-up's score. The
    runner-up matters for the confident-shortcut check: a high top score
    that's near-tied with another track (e.g. two "Product (X Mix)" tracks
    that WRatio scores almost identically) shouldn't be trusted outright —
    see TIE_MARGIN.
    """
    best_track: TrackInput | None = None
    best_score = 0.0
    runner_up_score = 0.0
    for track in tracks:
        score = fuzzy_score(track.title, track.artist, track.duration_seconds, video_title, video_duration)
        if score > best_score:
            runner_up_score = best_score
            best_score = score
            best_track = track
        elif score > runner_up_score:
            runner_up_score = score
    return best_track, best_score, runner_up_score


def resolve_release(
    release_title: str,
    artist_names: list[str],
    tracks: list[TrackInput],
    videos: list[VideoInput],
    call_ollama: OllamaCaller = call_ollama,
    force_llm: bool = False,
    max_workers: int = DEFAULT_WORKERS,
    confident_threshold: float = CONFIDENT_THRESHOLD,
    tie_margin: float = TIE_MARGIN,
    denoise_terms: list[str] = DEFAULT_DENOISE_TERMS,
) -> list[MatchResult]:
    """Fuzzy-first matching: a video whose best fuzzy score (raw, or after
    stripping uploader-boilerplate noise — see denoise_video_title) reaches
    confident_threshold, AND clears its runner-up track by at least
    tie_margin, is trusted outright. Everything else — including videos that
    score very low, and videos that score high but are near-tied between two
    tracks (a real failure mode: e.g. "Product (Original Mix)" vs "Product
    (Daydreamers Mix)" can score almost identically against a video title
    that only weakly reflects the mix name) — is escalated to the LLM, since
    fuzzy score alone isn't a reliable enough "obviously unrelated" (or
    "obviously this one track") signal to skip that check. The only
    exception is a release with no tracks at all, which is trivially
    "unrelated" without needing the LLM.

    Each ambiguous video gets its own separate LLM call (never batched —
    small local models are unreliable at returning a complete, valid response
    for more than one item at a time, which silently drops videos from a
    batched call). Ambiguous videos are still classified concurrently,
    bounded by max_workers, since the calls are independent.

    Pass force_llm=True to skip the confident-match shortcut entirely and
    send every video through the LLM regardless of its fuzzy score — a
    (temporary) way to evaluate LLM matching quality/performance across the
    board, not just on genuinely ambiguous cases.
    """
    results: dict[int, MatchResult] = {}
    ambiguous_videos: list[VideoInput] = []
    best_by_video: dict[int, tuple[TrackInput | None, float]] = {}

    # Configured denoise_terms (e.g. "Original Mix") are stripped from track
    # titles too, not just video titles: a bare video title with no mix name
    # at all (a common convention for the plain/original version) should be
    # able to match a track title like "Product (Original Mix)" without the
    # qualifier diluting the comparison on the track side. id/position are
    # preserved via replace(), so these still map back to the real track.
    matching_tracks = [replace(t, title=denoise_video_title(t.title, "", [], denoise_terms)) for t in tracks]

    for video in videos:
        best_track, best_score, runner_up_score = _best_two_scores(
            matching_tracks, video.title or "", video.duration
        )
        best_by_video[video.id] = (best_track, best_score)

        if force_llm:
            ambiguous_videos.append(video)
            continue

        is_clear_winner = (best_score - runner_up_score) >= tie_margin
        if best_track is not None and best_score >= confident_threshold and is_clear_winner:
            results[video.id] = MatchResult(
                video_id=video.id,
                track_id=best_track.id,
                fuzzy_score=best_score,
                video_type="full_track",
                best_track_position=best_track.position,
            )
            continue

        if best_track is None:
            # Nothing plausible to match against: either the release has no
            # tracks at all, or every track's duration deviates too much from
            # this video's (see DURATION_DEVIANCE_THRESHOLD in fuzzy_score)
            # to be worth an LLM call.
            results[video.id] = MatchResult(
                video_id=video.id,
                track_id=None,
                fuzzy_score=best_score,
                video_type="unrelated",
                best_track_position=None,
            )
            continue

        # Ambiguous on the raw title: retry with uploader boilerplate (release
        # title / artist name / year) stripped, since that's often repeated
        # identically across every video from the same channel and dilutes
        # the track-specific part of the fuzzy comparison.
        video_for_llm = video
        denoised_title = denoise_video_title(video.title or "", release_title, artist_names, denoise_terms)
        if (
            denoised_title
            and denoised_title != (video.title or "")
            and _has_meaningful_content(denoised_title)
        ):
            denoised_best_track, denoised_best_score, denoised_runner_up_score = _best_two_scores(
                matching_tracks, denoised_title, video.duration
            )
            denoised_is_clear_winner = (denoised_best_score - denoised_runner_up_score) >= tie_margin

            if (
                denoised_best_track is not None
                and denoised_best_score >= confident_threshold
                and denoised_is_clear_winner
            ):
                results[video.id] = MatchResult(
                    video_id=video.id,
                    track_id=denoised_best_track.id,
                    fuzzy_score=denoised_best_score,
                    video_type="full_track",
                    best_track_position=denoised_best_track.position,
                )
                continue

            # Still below the confidence threshold after denoising: escalate to the LLM using the
            # denoised title/best-guess, both usually more informative than
            # the noisy raw ones.
            best_by_video[video.id] = (denoised_best_track, denoised_best_score)
            video_for_llm = replace(video, title=denoised_title)

        ambiguous_videos.append(video_for_llm)

    if ambiguous_videos:
        llm_results: dict[int, MatchResult] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _classify_one_video_with_llm,
                    release_title,
                    artist_names,
                    tracks,
                    video,
                    *best_by_video[video.id],
                    call_ollama,
                    tie_margin,
                ): video.id
                for video in ambiguous_videos
            }
            for future in as_completed(futures):
                video_id = futures[future]
                result = future.result()
                if result is not None:
                    llm_results[video_id] = result

        for video in ambiguous_videos:
            if video.id in llm_results:
                results[video.id] = llm_results[video.id]
            else:
                # Ollama unavailable/failed for this video: it was already below
                # confident_threshold on fuzzy score alone (that's why it got here),
                # so without an LLM verdict there's no reliable basis to commit it
                # to a track — leave it unassigned rather than guessing.
                fallback_track, fallback_score = best_by_video[video.id]
                results[video.id] = MatchResult(
                    video_id=video.id,
                    track_id=None,
                    fuzzy_score=fallback_score,
                    video_type="unrelated",
                    best_track_position=fallback_track.position if fallback_track else None,
                )

    resolved = list(results.values())
    _select_best_per_track(resolved)
    return resolved


def match_videos(
    conn: sqlite3.Connection,
    on_progress: ProgressCallback | None = None,
    on_error: ErrorCallback | None = None,
    force: bool = False,
    reprocess: bool | None = None,
    max_workers: int = DEFAULT_WORKERS,
    extractor: VideoExtractor = _default_extractor,
    call_ollama: OllamaCaller = call_ollama,
    confident_threshold: float = CONFIDENT_THRESHOLD,
    tie_margin: float = TIE_MARGIN,
    denoise_terms: list[str] = DEFAULT_DENOISE_TERMS,
    should_stop: Callable[[], bool] | None = None,
) -> None:
    """`force` controls whether YouTube metadata is re-fetched for videos that
    already have it; `reprocess` controls whether releases that are already
    fully classified get reconsidered at all. They're independent so you can
    e.g. recompute classification for everyone using only cached metadata
    (force=False, reprocess=True) — fast, no yt-dlp calls. If `reprocess` is
    left as None it defaults to `force` (old force=True/False behavior,
    unchanged for existing callers).

    If `should_stop` starts returning True, the release currently being
    matched is finished but no further releases are started.
    """
    if reprocess is None:
        reprocess = force

    release_rows = conn.execute(
        """
        SELECT DISTINCT r.id, r.title
        FROM releases r
        JOIN release_videos rv ON rv.release_id = r.id
        WHERE r.removed_from_discogs_at IS NULL
        """
    ).fetchall()

    if not reprocess:
        release_rows = [
            row
            for row in release_rows
            if conn.execute(
                "SELECT 1 FROM release_videos WHERE release_id = ? AND video_type IS NULL LIMIT 1",
                (row["id"],),
            ).fetchone()
            is not None
        ]

    total = len(release_rows)
    completed = 0

    for row in release_rows:
        if should_stop is not None and should_stop():
            break
        release_id = row["id"]
        try:
            match_release(
                conn,
                release_id,
                row["title"],
                force,
                max_workers,
                extractor,
                call_ollama,
                on_error,
                confident_threshold=confident_threshold,
                tie_margin=tie_margin,
                denoise_terms=denoise_terms,
            )
        except Exception as exc:
            if on_error is not None:
                on_error(release_id, exc)
            continue
        completed += 1
        if on_progress is not None:
            on_progress(completed, total, release_id)


def match_release(
    conn: sqlite3.Connection,
    release_id: int,
    release_title: str,
    force: bool,
    max_workers: int,
    extractor: VideoExtractor,
    call_ollama: OllamaCaller,
    on_error: ErrorCallback | None = None,
    force_llm: bool = False,
    confident_threshold: float = CONFIDENT_THRESHOLD,
    tie_margin: float = TIE_MARGIN,
    denoise_terms: list[str] = DEFAULT_DENOISE_TERMS,
) -> None:
    artist_rows = conn.execute(
        """
        SELECT a.name FROM release_artists ra
        JOIN artists a ON a.id = ra.artist_id
        WHERE ra.release_id = ?
        """,
        (release_id,),
    ).fetchall()
    artist_names = [row["name"] for row in artist_rows]

    track_rows = conn.execute(
        "SELECT id, position, title, artist, duration FROM tracks WHERE release_id = ?",
        (release_id,),
    ).fetchall()
    tracks = [
        TrackInput(
            id=row["id"],
            position=row["position"],
            title=row["title"],
            artist=row["artist"],
            duration_seconds=parse_track_duration(row["duration"]),
        )
        for row in track_rows
    ]

    video_rows = conn.execute(
        "SELECT id, url, yt_fetched_at FROM release_videos WHERE release_id = ?",
        (release_id,),
    ).fetchall()

    to_fetch = [row for row in video_rows if force or row["yt_fetched_at"] is None]
    now = _utcnow_iso()

    if to_fetch:
        urls_by_id = {row["id"]: row["url"] for row in to_fetch}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fetch_video_metadata, row["url"], extractor): row["id"]
                for row in to_fetch
            }
            for future in as_completed(futures):
                video_id = futures[future]
                try:
                    metadata = future.result()
                except Exception as exc:
                    if on_error is not None:
                        on_error(release_id, RuntimeError(f"video {video_id} ({urls_by_id[video_id]}): {exc}"))
                    # Stamp yt_fetched_at and video_type anyway so a permanently-broken URL
                    # isn't retried forever on every non-force run; yt_title stays NULL so
                    # it's excluded from matching consideration.
                    with conn:
                        conn.execute(
                            "UPDATE release_videos SET yt_fetched_at = ?, video_type = 'unrelated', "
                            "video_type_reasoning = 'metadata fetch failed' WHERE id = ?",
                            (now, video_id),
                        )
                    continue
                with conn:
                    conn.execute(
                        """
                        UPDATE release_videos
                        SET yt_title = ?, yt_duration = ?, yt_uploader = ?, yt_description = ?,
                            yt_tags_json = ?, yt_chapters_json = ?, yt_fetched_at = ?
                        WHERE id = ?
                        """,
                        (
                            metadata.title,
                            metadata.duration,
                            metadata.uploader,
                            metadata.description,
                            json.dumps(metadata.tags),
                            json.dumps(metadata.chapters),
                            now,
                            video_id,
                        ),
                    )

    video_rows = conn.execute(
        "SELECT id, yt_title, yt_duration, yt_description, yt_chapters_json FROM release_videos WHERE release_id = ?",
        (release_id,),
    ).fetchall()
    videos = [
        VideoInput(
            id=row["id"],
            title=row["yt_title"],
            description=row["yt_description"],
            duration=row["yt_duration"],
            chapters=json.loads(row["yt_chapters_json"]) if row["yt_chapters_json"] else [],
        )
        for row in video_rows
        if row["yt_title"] is not None
    ]

    results = resolve_release(
        release_title,
        artist_names,
        tracks,
        videos,
        call_ollama=call_ollama,
        force_llm=force_llm,
        max_workers=max_workers,
        confident_threshold=confident_threshold,
        tie_margin=tie_margin,
        denoise_terms=denoise_terms,
    )

    with conn:
        for result in results:
            conn.execute(
                """
                UPDATE release_videos
                SET video_type = ?, video_type_reasoning = ?, best_fuzzy_score = ?, best_fuzzy_track_position = ?
                WHERE id = ?
                """,
                (
                    result.video_type,
                    result.video_type_reasoning,
                    result.fuzzy_score,
                    result.best_track_position,
                    result.video_id,
                ),
            )
        track_ids = [t.id for t in tracks]
        if track_ids:
            placeholders = ",".join("?" * len(track_ids))
            conn.execute(
                f"DELETE FROM track_video_matches WHERE track_id IN ({placeholders})", track_ids
            )
        for result in results:
            if result.track_id is None:
                continue
            conn.execute(
                """
                INSERT INTO track_video_matches
                    (track_id, video_id, fuzzy_score, llm_confidence, llm_reasoning, is_selected, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.track_id,
                    result.video_id,
                    result.fuzzy_score,
                    result.llm_confidence,
                    result.llm_reasoning,
                    int(result.is_selected),
                    now,
                ),
            )

        # Backfill a track's own duration from its matched video's when
        # Discogs' tracklist data is missing it — a common gap, and duration
        # is useful data to have (e.g. for the confident/tie-margin fuzzy
        # checks on a future re-match). Only from the *selected* (primary)
        # match, never an alternate candidate.
        tracks_by_id = {t.id: t for t in tracks}
        videos_by_id = {v.id: v for v in videos}
        for result in results:
            if result.track_id is None or not result.is_selected:
                continue
            track = tracks_by_id.get(result.track_id)
            video = videos_by_id.get(result.video_id)
            if track is None or video is None:
                continue
            if track.duration_seconds is not None or not video.duration:
                continue
            conn.execute(
                "UPDATE tracks SET duration = ? WHERE id = ?",
                (_format_duration(video.duration), track.id),
            )


def debug_score_matrix(conn: sqlite3.Connection, release_id: int) -> list[dict]:
    """Full track x video fuzzy-score matrix for a release, for debugging match
    quality — shows every score considered, not just the winning pick.
    Temporary developer aid; safe to delete along with its call site/template
    section once matching quality has been verified.
    """
    track_rows = conn.execute(
        "SELECT id, position, title, artist, duration FROM tracks WHERE release_id = ? ORDER BY id",
        (release_id,),
    ).fetchall()
    tracks = [
        TrackInput(
            id=row["id"],
            position=row["position"],
            title=row["title"],
            artist=row["artist"],
            duration_seconds=parse_track_duration(row["duration"]),
        )
        for row in track_rows
    ]

    video_rows = conn.execute(
        """
        SELECT id, yt_title, yt_duration
        FROM release_videos
        WHERE release_id = ? AND yt_title IS NOT NULL
        ORDER BY id
        """,
        (release_id,),
    ).fetchall()

    selected_by_video = {
        row["video_id"]: (row["track_id"], row["llm_confidence"] is not None)
        for row in conn.execute(
            """
            SELECT tvm.video_id, tvm.track_id, tvm.llm_confidence
            FROM track_video_matches tvm
            JOIN tracks t ON t.id = tvm.track_id
            WHERE t.release_id = ? AND tvm.is_selected = 1
            """,
            (release_id,),
        ).fetchall()
    }

    matrix = []
    for video in video_rows:
        selected_track_id, selected_by_llm = selected_by_video.get(video["id"], (None, False))
        cells = [
            {
                "track_id": track.id,
                "score": fuzzy_score(
                    track.title, track.artist, track.duration_seconds, video["yt_title"], video["yt_duration"]
                ),
                "is_selected": track.id == selected_track_id,
                "is_llm_selected": track.id == selected_track_id and selected_by_llm,
            }
            for track in tracks
        ]
        matrix.append({"video_id": video["id"], "video_title": video["yt_title"], "cells": cells})
    return matrix


def clear_matches(conn: sqlite3.Connection, release_id: int) -> None:
    """Wipe matching results for a release back to a pristine "never matched"
    state — deletes its track_video_matches rows and resets video_type/
    video_type_reasoning/best_fuzzy_score/best_fuzzy_track_position to NULL
    on its release_videos rows. Does NOT touch fetched YouTube metadata
    (yt_title/yt_duration/etc.) or delete the release_videos rows themselves.
    """
    with conn:
        conn.execute(
            """
            DELETE FROM track_video_matches
            WHERE track_id IN (SELECT id FROM tracks WHERE release_id = ?)
            """,
            (release_id,),
        )
        conn.execute(
            """
            UPDATE release_videos
            SET video_type = NULL, video_type_reasoning = NULL,
                best_fuzzy_score = NULL, best_fuzzy_track_position = NULL
            WHERE release_id = ?
            """,
            (release_id,),
        )
