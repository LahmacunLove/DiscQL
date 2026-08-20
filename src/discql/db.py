from __future__ import annotations

import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path

MIGRATIONS_PACKAGE = "discql.migrations"
MIGRATION_FILENAME_RE = re.compile(r"^(\d+)_.*\.sql$")


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Letters like o/slash (ø) or ae (æ) have no Unicode decomposition into a
# plain base letter + combining mark - NFKD leaves them untouched - so they
# need an explicit fold on top of it.
_EXTRA_FOLDS = str.maketrans(
    {
        "ø": "o", "Ø": "O",
        "æ": "ae", "Æ": "AE",
        "œ": "oe", "Œ": "OE",
        "ł": "l", "Ł": "L",
        "đ": "d", "Đ": "D",
        "ð": "d", "Ð": "D",
        "þ": "th", "Þ": "Th",
        "ß": "ss",
    }
)


def strip_accents(value: str | None) -> str | None:
    """Lowercased, diacritic-stripped, umlaut-folded form of value, used as a
    SQLite function so search fields can match regardless of accents the
    user's keyboard can't easily type - e.g. "Kulør" -> "kulor", and
    "Müller"/"Mueller"/"Muller" all fold to "muller" since ae/oe/ue are
    also valid ASCII spellings of ä/ö/ü, not just their bare vowel.
    """
    if value is None:
        return None
    decomposed = unicodedata.normalize("NFKD", value.translate(_EXTRA_FOLDS))
    folded = "".join(c for c in decomposed if not unicodedata.combining(c)).lower()
    for digraph, vowel in (("ae", "a"), ("oe", "o"), ("ue", "u")):
        folded = folded.replace(digraph, vowel)
    return folded


def connect(db_path: Path) -> sqlite3.Connection:
    # check_same_thread=False: FastAPI's sync-dependency generators run their
    # __enter__/endpoint-body/__exit__ phases as separate anyio threadpool
    # jobs, which aren't guaranteed to land on the same OS thread under
    # concurrent load - confirmed concretely once the cover-art endpoint
    # started getting many parallel <img> requests per page (grid view),
    # which single in-flight requests to other endpoints never triggered.
    # Each request still gets its own dedicated connection, so this doesn't
    # introduce cross-request sharing - it just tolerates the thread hopping
    # within one request's own lifecycle.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL + busy_timeout: the "Run All" pipeline opens one connection per
    # worker thread to process several releases concurrently (see
    # pipeline.py) - the default rollback-journal mode serializes writers
    # and raises "database is locked" immediately on contention, where WAL
    # lets readers proceed during a write and busy_timeout makes a
    # contended write retry for a while before giving up instead of
    # failing outright.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.row_factory = sqlite3.Row
    conn.create_function("strip_accents", 1, strip_accents, deterministic=True)
    return conn


def _discover_migrations() -> list[tuple[int, str]]:
    migrations = []
    for entry in resources.files(MIGRATIONS_PACKAGE).iterdir():
        match = MIGRATION_FILENAME_RE.match(entry.name)
        if match:
            migrations.append((int(match.group(1)), entry.name))
    return sorted(migrations)


def migrate(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        )
        """
    )
    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}

    for version, filename in _discover_migrations():
        if version in applied:
            continue
        sql = resources.files(MIGRATIONS_PACKAGE).joinpath(filename).read_text()
        with conn:
            conn.executescript(sql)
            conn.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, utcnow_iso()),
            )
