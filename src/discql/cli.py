from __future__ import annotations

import argparse
import sys

from discql import collection, db, local_audio, ollama_client, sync, youtube_matching
from discql.config import ensure_token_interactively, load_config
from discql.discogs_api import build_discogs_api


def _print_progress(index: int, total: int, release_id: int, title: str) -> None:
    print(f"[{index}/{total}] {release_id}: {title}")


def _print_error(release_id: int, exc: Exception) -> None:
    print(f"  ! failed to sync release {release_id}: {exc}", file=sys.stderr)


def cmd_sync(args: argparse.Namespace) -> None:
    config = load_config()
    if not config.has_discogs_auth:
        ensure_token_interactively()
        config = load_config()
    conn = db.connect(config.db_path)
    db.migrate(conn)
    api = build_discogs_api(config)
    print(f"Fetching collection for {api.username}...")
    if args.limit is not None:
        print(f"(dev mode: limited to first {args.limit} releases, no soft-delete reconciliation)")
    if args.force_refetch:
        print("(force-refetch: refreshing all matched releases from Discogs)")
    else:
        print("(skipping releases already known locally; use --force-refetch to refresh them)")
    sync.sync_collection(
        conn,
        api,
        on_progress=_print_progress,
        on_error=_print_error,
        limit=args.limit,
        force_refetch=args.force_refetch,
    )
    print("Sync complete.")


def cmd_add(args: argparse.Namespace) -> None:
    config = load_config()
    if not config.has_discogs_auth:
        ensure_token_interactively()
        config = load_config()
    conn = db.connect(config.db_path)
    db.migrate(conn)
    api = build_discogs_api(config)
    collection.add_release(conn, api, args.release_id)
    print(f"Added release {args.release_id}.")


def cmd_remove(args: argparse.Namespace) -> None:
    config = load_config()
    if not config.has_discogs_auth:
        ensure_token_interactively()
        config = load_config()
    conn = db.connect(config.db_path)
    db.migrate(conn)
    api = build_discogs_api(config)
    collection.remove_release(conn, api, args.release_id)
    print(f"Removed release {args.release_id}.")


def cmd_list(args: argparse.Namespace) -> None:
    config = load_config()
    conn = db.connect(config.db_path)
    db.migrate(conn)
    rows = conn.execute(
        """
        SELECT id, title, year, date_added, removed_from_discogs_at
        FROM releases
        ORDER BY date_added
        """
    ).fetchall()
    for row in rows:
        status = " [removed]" if row["removed_from_discogs_at"] else ""
        print(f"{row['id']:>10}  {row['year'] or '----':>4}  {row['title']}{status}")
    print(f"\n{len(rows)} release(s).")


def cmd_match_local(args: argparse.Namespace) -> None:
    config = load_config()
    conn = db.connect(config.db_path)
    db.migrate(conn)
    flac_dir = config.local_flac_dir
    if flac_dir is None:
        print("LOCAL_FLAC_DIR is not configured (empty) - nothing to do.")
        return
    if not flac_dir.is_dir():
        print(f"LOCAL_FLAC_DIR ({flac_dir}) doesn't exist or isn't mounted - nothing to do.")
        return

    def progress(index: int, total: int, release_id: int) -> None:
        print(f"[{index}/{total}] checked release {release_id}")

    def error(release_id: int, exc: Exception) -> None:
        print(f"  ! failed to local-match release {release_id}: {exc}", file=sys.stderr)

    local_audio.match_library(
        conn,
        flac_dir,
        confident_threshold=config.local_match_confident_threshold,
        force=args.force,
        on_progress=progress,
        on_error=error,
    )

    matched = conn.execute("SELECT COUNT(*) AS n FROM releases WHERE local_folder_path IS NOT NULL").fetchone()["n"]
    print(f"Done. {matched} release(s) have a confident local folder match.")


def cmd_retry_failed_matches(args: argparse.Namespace) -> None:
    """Force re-match only releases with a stuck/failed YouTube video fetch
    - not the whole collection. Two patterns count as "stuck":
    1. yt_fetched_at set but yt_title/video_type still NULL - the pre-fix
       fetch_video_metadata() bug (extractor() returning None silently
       treated as "fetched, nothing there" instead of a failure) leaves
       rows in exactly this state, permanently excluded from matching.
    2. video_type_reasoning = 'metadata fetch failed' - a genuine
       (typically transient) fetch failure, correctly marked as such but
       never retried without force either.
    Both are worth a one-off forced retry: many of these turn out to be
    fetchable again by the time this runs (a temporary YouTube-side
    rate-limit/block, not a permanently gone video).
    """
    config = load_config()
    conn = db.connect(config.db_path)
    db.migrate(conn)

    release_rows = conn.execute(
        """
        SELECT DISTINCT r.id, r.title
        FROM releases r
        JOIN release_videos rv ON rv.release_id = r.id
        WHERE (rv.yt_fetched_at IS NOT NULL AND rv.yt_title IS NULL AND rv.video_type IS NULL)
           OR rv.video_type_reasoning = 'metadata fetch failed'
        """
    ).fetchall()

    if not release_rows:
        print("No releases with a stuck/failed YouTube video fetch found.")
        return

    print(f"{len(release_rows)} release(s) with a stuck/failed video fetch - force re-matching just these:")
    for row in release_rows:
        print(f"  {row['id']}: {row['title']}")

    extractor = youtube_matching.build_extractor(config.youtube_cookies_browser)

    with ollama_client.ensure_ollama_running():
        for row in release_rows:
            try:
                youtube_matching.match_release(
                    conn,
                    row["id"],
                    row["title"],
                    force=True,
                    max_workers=youtube_matching.DEFAULT_WORKERS,
                    extractor=extractor,
                    call_ollama=youtube_matching.call_ollama,
                    confident_threshold=config.fuzzy_confident_threshold,
                    tie_margin=config.fuzzy_tie_margin,
                    denoise_terms=config.fuzzy_denoise_terms,
                )
            except Exception as exc:
                print(f"  ! failed to re-match release {row['id']}: {exc}", file=sys.stderr)

    print("Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="discql-sync")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync_parser = subparsers.add_parser("sync", help="Mirror the Discogs collection locally")
    sync_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only sync the first N releases (dev mode, skips soft-delete reconciliation)",
    )
    sync_parser.add_argument(
        "--force-refetch",
        action="store_true",
        help="Refresh releases even if already present locally (by default, "
        "already-known active releases are skipped)",
    )
    sync_parser.set_defaults(func=cmd_sync)

    add_parser = subparsers.add_parser("add", help="Add a release to the Discogs collection")
    add_parser.add_argument("release_id", type=int)
    add_parser.set_defaults(func=cmd_add)

    remove_parser = subparsers.add_parser(
        "remove", help="Remove a release from the Discogs collection"
    )
    remove_parser.add_argument("release_id", type=int)
    remove_parser.set_defaults(func=cmd_remove)

    list_parser = subparsers.add_parser("list", help="List locally mirrored releases")
    list_parser.set_defaults(func=cmd_list)

    match_local_parser = subparsers.add_parser(
        "match-local", help="Match releases against a local FLAC library (LOCAL_FLAC_DIR)"
    )
    match_local_parser.add_argument(
        "--force", action="store_true", help="Re-check every release, not just unchecked ones"
    )
    match_local_parser.set_defaults(func=cmd_match_local)

    retry_failed_parser = subparsers.add_parser(
        "retry-failed-matches",
        help="Force re-match only releases with a stuck/failed YouTube video fetch (not the whole collection)",
    )
    retry_failed_parser.set_defaults(func=cmd_retry_failed_matches)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
