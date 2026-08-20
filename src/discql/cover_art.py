from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests

from discql.config import USER_AGENT

DEFAULT_EXTENSION = ".jpg"


def find_cached_cover(cover_dir: Path, release_id: int) -> Path | None:
    """Return the already-cached cover file for release_id (any extension), if any."""
    if not cover_dir.is_dir():
        return None
    for path in cover_dir.iterdir():
        if path.stem == str(release_id):
            return path
    return None


def _extension_from_url(url: str) -> str:
    suffix = Path(urlparse(url).path).suffix
    return suffix if suffix else DEFAULT_EXTENSION


def fetch_and_cache_cover(cover_dir: Path, release_id: int, cover_url: str, timeout: float = 10.0) -> Path:
    """Download cover_url and save it under cover_dir as <release_id><ext>,
    overwriting any previous cache entry for this release (caller should
    check find_cached_cover() first if a fresh download isn't wanted).
    """
    response = requests.get(cover_url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    cover_dir.mkdir(parents=True, exist_ok=True)
    path = cover_dir / f"{release_id}{_extension_from_url(cover_url)}"
    path.write_bytes(response.content)
    return path


def get_or_fetch_cover(cover_dir: Path, release_id: int, cover_url: str | None) -> Path | None:
    """The locally cached cover for release_id if present; otherwise
    downloads+caches cover_url (Discogs' own hosted image) if given. None if
    there's no cover available at all (nothing cached, no cover_url either).
    """
    cached = find_cached_cover(cover_dir, release_id)
    if cached is not None:
        return cached
    if not cover_url:
        return None
    return fetch_and_cache_cover(cover_dir, release_id, cover_url)
