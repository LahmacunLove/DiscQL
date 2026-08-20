from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field

_ARTIST_DISAMBIGUATION_RE = re.compile(r"\s\(\d+\)(?=,|$)")


def _strip_artist_disambiguation(value: str) -> str:
    """Discogs suffixes same-named artists with a disambiguation number,
    e.g. "Klex (3)" - an internal Discogs identifier, not part of the
    artist's actual name, so it's stripped for display. The lookahead
    (comma or end-of-string) makes this safe to run over an already
    comma-joined multi-artist string too."""
    return _ARTIST_DISAMBIGUATION_RE.sub("", value)

_ARTIST_SORT_KEY = """(
    SELECT MIN(a.name COLLATE NOCASE)
    FROM release_artists ra
    JOIN artists a ON a.id = ra.artist_id
    WHERE ra.release_id = r.id
)"""
_LABEL_SORT_KEY = """(
    SELECT MIN(l.name COLLATE NOCASE)
    FROM release_labels rl
    JOIN labels l ON l.id = rl.label_id
    WHERE rl.release_id = r.id
)"""

SORT_OPTIONS = {
    "date_added_desc": "r.date_added DESC",
    "date_added_asc": "r.date_added ASC",
    "title_asc": "r.title COLLATE NOCASE ASC",
    "year_desc": "r.year DESC",
    "year_asc": "r.year ASC",
    "artist_asc": f"{_ARTIST_SORT_KEY} ASC",
    "label_asc": f"{_LABEL_SORT_KEY} ASC",
}
DEFAULT_SORT = "date_added_desc"


@dataclass
class ReleaseSummary:
    id: int
    title: str
    year: int | None
    cover_image_url: str | None
    artist_names: str
    genres: list[str] = field(default_factory=list)
    track_count: int = 0
    matched_count: int = 0
    analyzed_count: int = 0
    matching_processed_count: int = 0

    @property
    def matching_processed(self) -> bool:
        """Whether YouTube matching has actually been run for this release
        (at least one of its videos has been classified) - distinct from
        fully_matched, since a release can be fully processed yet still
        have zero matched tracks (no good candidate video for any of
        them)."""
        return self.matching_processed_count > 0

    @property
    def fully_matched(self) -> bool:
        return self.track_count > 0 and self.matched_count == self.track_count

    @property
    def fully_analyzed(self) -> bool:
        return self.track_count > 0 and self.analyzed_count == self.track_count


@dataclass
class CrateSummary:
    id: int
    name: str
    created_at: str
    item_count: int = 0


@dataclass
class LabelInfo:
    id: int
    name: str
    catalog_number: str


@dataclass
class ArtistRef:
    id: int
    name: str


@dataclass
class TrackInfo:
    position: str | None
    title: str
    duration: str | None
    artist: str | None = None
    matched_video_url: str | None = None
    matched_video_title: str | None = None
    bpm: float | None = None
    musical_key: str | None = None
    musical_key_scale: str | None = None
    mood_summary: str | None = None
    genre: str | None = None
    style: str | None = None
    waveform_path: str | None = None


@dataclass
class VideoInfo:
    title: str | None
    url: str
    duration: int | None
    video_type: str | None = None
    video_type_reasoning: str | None = None
    matched_track_position: str | None = None
    fuzzy_score: float | None = None
    best_fuzzy_track_position: str | None = None
    llm_confidence: float | None = None
    is_selected: bool = False


@dataclass
class CreditInfo:
    artist_id: int
    artist_name: str
    role: str


@dataclass
class ReleaseDetail:
    id: int
    title: str
    year: int | None
    cover_image_url: str | None
    discogs_uri: str | None
    date_added: str | None
    country: str | None
    images: list[dict]
    artist_names: list[str]
    artists: list[ArtistRef]
    labels: list[LabelInfo]
    genres: list[str]
    styles: list[str]
    ai_genres: list[str]
    ai_styles: list[str]
    tracks: list[TrackInfo]
    tracks_have_distinct_artists: bool
    videos: list[VideoInfo]
    credits: list[CreditInfo]


def _in_params(prefix: str, values: list[str], params: dict) -> str:
    """Registers values under distinct :prefix_N keys in params and returns
    the "(:prefix_0, :prefix_1, ...)" placeholder list for an IN clause.
    """
    keys = []
    for i, value in enumerate(values):
        key = f"{prefix}_{i}"
        params[key] = value
        keys.append(f":{key}")
    return f"({', '.join(keys)})"


def _where_clause(
    q: str | None,
    genres: list[str] | None,
    year: int | None = None,
    styles: list[str] | None = None,
    label_name: str | None = None,
    artist_name: str | None = None,
    bpm: float | None = None,
    bpm_tolerance_pct: float | None = None,
    crate_id: int | None = None,
) -> tuple[str, dict]:
    clauses = ["r.removed_from_discogs_at IS NULL"]
    params: dict = {}

    def like_param(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return f"%{escaped}%"

    if q:
        clauses.append(
            """
            (strip_accents(r.title) LIKE strip_accents(:q) ESCAPE '\\' OR EXISTS (
                SELECT 1 FROM release_artists ra2
                JOIN artists a2 ON a2.id = ra2.artist_id
                WHERE ra2.release_id = r.id AND strip_accents(a2.name) LIKE strip_accents(:q) ESCAPE '\\'
            ))
            """
        )
        params["q"] = like_param(q)

    if genres:
        in_list = _in_params("genre", genres, params)
        clauses.append(f"EXISTS (SELECT 1 FROM json_each(r.genres_json) je WHERE je.value IN {in_list})")

    if styles:
        in_list = _in_params("style", styles, params)
        clauses.append(f"EXISTS (SELECT 1 FROM json_each(r.styles_json) je2 WHERE je2.value IN {in_list})")

    if year:
        clauses.append("r.year = :year")
        params["year"] = year

    if label_name:
        clauses.append(
            """
            EXISTS (
                SELECT 1 FROM release_labels rl
                JOIN labels l ON l.id = rl.label_id
                WHERE rl.release_id = r.id AND strip_accents(l.name) LIKE strip_accents(:label_name) ESCAPE '\\'
            )
            """
        )
        params["label_name"] = like_param(label_name)

    if artist_name:
        clauses.append(
            """
            EXISTS (
                SELECT 1 FROM release_artists ra3
                JOIN artists a3 ON a3.id = ra3.artist_id
                WHERE ra3.release_id = r.id AND strip_accents(a3.name) LIKE strip_accents(:artist_name) ESCAPE '\\'
            )
            """
        )
        params["artist_name"] = like_param(artist_name)

    # Crate-mode-style BPM filter (CLAUDE.md section 6): a release matches
    # if *any* of its tracks' analyzed BPM falls within a tolerance band
    # around the target value. bpm_tolerance_pct is a fraction (0.08 for
    # "±8%"), not a percent - computed here in Python rather than via SQL
    # ROUND()/percentage math, so the query stays a plain BETWEEN and the
    # band logic is easy to unit-test directly. 0 (the "Exact" option)
    # isn't literal float equality (analyzed BPM is rarely a whole number,
    # e.g. 127.98) - a fixed ±0.5 BPM band around the target instead, which
    # in practice means "rounds to the same whole-number BPM."
    if bpm is not None and bpm_tolerance_pct is not None:
        if bpm_tolerance_pct > 0:
            bpm_low, bpm_high = bpm * (1 - bpm_tolerance_pct), bpm * (1 + bpm_tolerance_pct)
        else:
            bpm_low, bpm_high = bpm - 0.5, bpm + 0.5
        clauses.append(
            "EXISTS (SELECT 1 FROM tracks t4 WHERE t4.release_id = r.id AND t4.bpm BETWEEN :bpm_low AND :bpm_high)"
        )
        params["bpm_low"] = bpm_low
        params["bpm_high"] = bpm_high

    if crate_id is not None:
        clauses.append(
            "EXISTS (SELECT 1 FROM crate_releases cr WHERE cr.crate_id = :crate_id AND cr.release_id = r.id)"
        )
        params["crate_id"] = crate_id

    return " AND ".join(clauses), params


def list_releases(
    conn: sqlite3.Connection,
    *,
    q: str | None = None,
    genres: list[str] | None = None,
    year: int | None = None,
    styles: list[str] | None = None,
    label_name: str | None = None,
    artist_name: str | None = None,
    bpm: float | None = None,
    bpm_tolerance_pct: float | None = None,
    crate_id: int | None = None,
    sort: str = DEFAULT_SORT,
    limit: int | None = 48,
    offset: int = 0,
) -> list[ReleaseSummary]:
    where_sql, params = _where_clause(
        q, genres, year, styles, label_name, artist_name, bpm, bpm_tolerance_pct, crate_id
    )
    order_sql = SORT_OPTIONS.get(sort, SORT_OPTIONS[DEFAULT_SORT])

    limit_sql = ""
    if limit is not None:
        limit_sql = "LIMIT :limit OFFSET :offset"
        params["limit"] = limit
        params["offset"] = offset

    rows = conn.execute(
        f"""
        SELECT
            r.id, r.title, r.year, r.cover_image_url, r.genres_json,
            (
                SELECT group_concat(a.name, ', ')
                FROM release_artists ra
                JOIN artists a ON a.id = ra.artist_id
                WHERE ra.release_id = r.id
            ) AS artist_names,
            (SELECT COUNT(*) FROM tracks t WHERE t.release_id = r.id) AS track_count,
            (
                SELECT COUNT(*) FROM tracks t
                JOIN track_video_matches tvm ON tvm.track_id = t.id AND tvm.is_selected = 1
                WHERE t.release_id = r.id
            ) AS matched_count,
            (SELECT COUNT(*) FROM tracks t WHERE t.release_id = r.id AND t.analyzed_at IS NOT NULL) AS analyzed_count,
            (
                SELECT COUNT(*) FROM release_videos rv
                WHERE rv.release_id = r.id AND rv.video_type IS NOT NULL
            ) AS matching_processed_count
        FROM releases r
        WHERE {where_sql}
        ORDER BY {order_sql}
        {limit_sql}
        """,
        params,
    ).fetchall()

    return [
        ReleaseSummary(
            id=row["id"],
            title=row["title"],
            year=row["year"],
            cover_image_url=row["cover_image_url"],
            artist_names=_strip_artist_disambiguation(row["artist_names"]) if row["artist_names"] else "",
            genres=json.loads(row["genres_json"]) if row["genres_json"] else [],
            track_count=row["track_count"],
            matched_count=row["matched_count"],
            analyzed_count=row["analyzed_count"],
            matching_processed_count=row["matching_processed_count"],
        )
        for row in rows
    ]


def count_releases(
    conn: sqlite3.Connection,
    *,
    q: str | None = None,
    genres: list[str] | None = None,
    year: int | None = None,
    styles: list[str] | None = None,
    label_name: str | None = None,
    artist_name: str | None = None,
    bpm: float | None = None,
    bpm_tolerance_pct: float | None = None,
    crate_id: int | None = None,
) -> int:
    where_sql, params = _where_clause(
        q, genres, year, styles, label_name, artist_name, bpm, bpm_tolerance_pct, crate_id
    )
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM releases r WHERE {where_sql}", params
    ).fetchone()
    return row["n"]


def list_genres(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT je.value AS genre
        FROM releases r, json_each(r.genres_json) je
        WHERE r.removed_from_discogs_at IS NULL
        ORDER BY genre COLLATE NOCASE
        """
    ).fetchall()
    return [row["genre"] for row in rows]


def list_styles(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT je.value AS style
        FROM releases r, json_each(r.styles_json) je
        WHERE r.removed_from_discogs_at IS NULL
        ORDER BY style COLLATE NOCASE
        """
    ).fetchall()
    return [row["style"] for row in rows]


def list_years(conn: sqlite3.Connection) -> list[int]:
    rows = conn.execute(
        """
        SELECT DISTINCT year
        FROM releases
        WHERE removed_from_discogs_at IS NULL AND year IS NOT NULL
        ORDER BY year DESC
        """
    ).fetchall()
    return [row["year"] for row in rows]


def list_labels(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT l.name
        FROM labels l
        JOIN release_labels rl ON rl.label_id = l.id
        JOIN releases r ON r.id = rl.release_id
        WHERE r.removed_from_discogs_at IS NULL
        ORDER BY l.name COLLATE NOCASE
        """
    ).fetchall()
    return [row["name"] for row in rows]


def list_artists(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT a.name
        FROM artists a
        JOIN release_artists ra ON ra.artist_id = a.id
        JOIN releases r ON r.id = ra.release_id
        WHERE r.removed_from_discogs_at IS NULL
        ORDER BY a.name COLLATE NOCASE
        """
    ).fetchall()
    # dict.fromkeys dedupes distinct Discogs artist IDs that share a display
    # name once their disambiguation suffix is stripped (e.g. "Cher (2)" and
    # "Cher (3)"), while preserving the SQL-side alphabetical order.
    return list(dict.fromkeys(_strip_artist_disambiguation(row["name"]) for row in rows))


def list_crates_with_counts(conn: sqlite3.Connection) -> list[CrateSummary]:
    rows = conn.execute(
        """
        SELECT c.id, c.name, c.created_at, COUNT(cr.release_id) AS item_count
        FROM crates c
        LEFT JOIN crate_releases cr ON cr.crate_id = c.id
        GROUP BY c.id
        ORDER BY c.created_at DESC
        """
    ).fetchall()
    return [
        CrateSummary(id=row["id"], name=row["name"], created_at=row["created_at"], item_count=row["item_count"])
        for row in rows
    ]


def crate_ids_for_release(conn: sqlite3.Connection, release_id: int) -> set[int]:
    rows = conn.execute(
        "SELECT crate_id FROM crate_releases WHERE release_id = ?", (release_id,)
    ).fetchall()
    return {row["crate_id"] for row in rows}


def get_release_detail(conn: sqlite3.Connection, release_id: int) -> ReleaseDetail | None:
    release_row = conn.execute(
        """
        SELECT id, title, year, cover_image_url, discogs_uri, date_added,
               genres_json, styles_json, country, images_json
        FROM releases
        WHERE id = ?
        """,
        (release_id,),
    ).fetchone()
    if release_row is None:
        return None

    artist_rows = conn.execute(
        """
        SELECT a.id, a.name
        FROM release_artists ra
        JOIN artists a ON a.id = ra.artist_id
        WHERE ra.release_id = ?
        ORDER BY a.name COLLATE NOCASE
        """,
        (release_id,),
    ).fetchall()

    label_rows = conn.execute(
        """
        SELECT l.id, l.name, rl.catalog_number
        FROM release_labels rl
        JOIN labels l ON l.id = rl.label_id
        WHERE rl.release_id = ?
        ORDER BY l.name COLLATE NOCASE
        """,
        (release_id,),
    ).fetchall()

    track_rows = conn.execute(
        """
        SELECT t.position, t.title, t.duration, t.artist,
               t.bpm, t.musical_key, t.musical_key_scale, t.mood_summary, t.genre, t.style, t.waveform_path,
               mv.url AS matched_video_url, mv.yt_title AS matched_video_title
        FROM tracks t
        LEFT JOIN track_video_matches tvm ON tvm.track_id = t.id AND tvm.is_selected = 1
        LEFT JOIN release_videos mv ON mv.id = tvm.video_id
        WHERE t.release_id = ?
        ORDER BY t.id
        """,
        (release_id,),
    ).fetchall()

    video_rows = conn.execute(
        """
        SELECT rv.title, rv.url, rv.duration, rv.video_type, rv.video_type_reasoning,
               rv.best_fuzzy_score, rv.best_fuzzy_track_position,
               t.position AS matched_track_position, tvm.llm_confidence, tvm.is_selected
        FROM release_videos rv
        LEFT JOIN track_video_matches tvm ON tvm.video_id = rv.id
        LEFT JOIN tracks t ON t.id = tvm.track_id
        WHERE rv.release_id = ?
        ORDER BY rv.id
        """,
        (release_id,),
    ).fetchall()

    credit_rows = conn.execute(
        """
        SELECT a.id AS artist_id, a.name AS artist_name, rc.role
        FROM release_credits rc
        JOIN artists a ON a.id = rc.artist_id
        WHERE rc.release_id = ?
        ORDER BY rc.role COLLATE NOCASE, a.name COLLATE NOCASE
        """,
        (release_id,),
    ).fetchall()

    artist_names = [_strip_artist_disambiguation(row["name"]) for row in artist_rows]
    artists = [ArtistRef(id=row["id"], name=_strip_artist_disambiguation(row["name"])) for row in artist_rows]
    release_artist_label = ", ".join(artist_names)
    tracks = [
        TrackInfo(
            position=row["position"],
            title=row["title"],
            duration=row["duration"],
            artist=_strip_artist_disambiguation(row["artist"]) if row["artist"] else None,
            matched_video_url=row["matched_video_url"],
            matched_video_title=row["matched_video_title"],
            bpm=row["bpm"],
            musical_key=row["musical_key"],
            musical_key_scale=row["musical_key_scale"],
            mood_summary=row["mood_summary"],
            genre=row["genre"],
            style=row["style"],
            waveform_path=row["waveform_path"],
        )
        for row in track_rows
    ]
    tracks_have_distinct_artists = any(
        track.artist and track.artist != release_artist_label for track in tracks
    )
    # Deduped, order-preserved AI-detected genres/styles across all analyzed
    # tracks - shown alongside Discogs' own genres/styles in the UI.
    ai_genres = list(dict.fromkeys(track.genre for track in tracks if track.genre))
    ai_styles = list(dict.fromkeys(track.style for track in tracks if track.style))

    return ReleaseDetail(
        id=release_row["id"],
        title=release_row["title"],
        year=release_row["year"],
        cover_image_url=release_row["cover_image_url"],
        discogs_uri=release_row["discogs_uri"],
        date_added=release_row["date_added"],
        country=release_row["country"],
        images=json.loads(release_row["images_json"]) if release_row["images_json"] else [],
        artist_names=artist_names,
        artists=artists,
        labels=[
            LabelInfo(id=row["id"], name=row["name"], catalog_number=row["catalog_number"]) for row in label_rows
        ],
        genres=json.loads(release_row["genres_json"]) if release_row["genres_json"] else [],
        styles=json.loads(release_row["styles_json"]) if release_row["styles_json"] else [],
        ai_genres=ai_genres,
        ai_styles=ai_styles,
        tracks=tracks,
        tracks_have_distinct_artists=tracks_have_distinct_artists,
        videos=[
            VideoInfo(
                title=row["title"],
                url=row["url"],
                duration=row["duration"],
                video_type=row["video_type"],
                video_type_reasoning=row["video_type_reasoning"],
                matched_track_position=row["matched_track_position"],
                fuzzy_score=row["best_fuzzy_score"],
                best_fuzzy_track_position=row["best_fuzzy_track_position"],
                llm_confidence=row["llm_confidence"],
                is_selected=bool(row["is_selected"]),
            )
            for row in video_rows
        ],
        credits=[
            CreditInfo(
                artist_id=row["artist_id"],
                artist_name=_strip_artist_disambiguation(row["artist_name"]),
                role=row["role"],
            )
            for row in credit_rows
        ],
    )
