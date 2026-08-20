from __future__ import annotations

import pytest
import requests

from discql import cover_art


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


# --- find_cached_cover ----------------------------------------------------


def test_find_cached_cover_none_when_dir_missing(tmp_path):
    assert cover_art.find_cached_cover(tmp_path / "nope", 1) is None


def test_find_cached_cover_none_when_no_matching_file(tmp_path):
    (tmp_path / "2.jpg").write_bytes(b"other release")
    assert cover_art.find_cached_cover(tmp_path, 1) is None


def test_find_cached_cover_finds_file_by_stem_regardless_of_extension(tmp_path):
    path = tmp_path / "1.png"
    path.write_bytes(b"cover bytes")
    assert cover_art.find_cached_cover(tmp_path, 1) == path


# --- fetch_and_cache_cover --------------------------------------------------


def test_fetch_and_cache_cover_saves_bytes_with_extension_from_url(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cover_art.requests, "get", lambda url, timeout, headers: _FakeResponse(b"jpeg bytes")
    )

    path = cover_art.fetch_and_cache_cover(tmp_path, 42, "https://img.discogs.com/abc/R-1-cover.jpeg")

    assert path == tmp_path / "42.jpeg"
    assert path.read_bytes() == b"jpeg bytes"


def test_fetch_and_cache_cover_defaults_extension_when_url_has_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cover_art.requests, "get", lambda url, timeout, headers: _FakeResponse(b"bytes"))

    path = cover_art.fetch_and_cache_cover(tmp_path, 7, "https://img.discogs.com/abc/no-extension")

    assert path == tmp_path / "7.jpg"


def test_fetch_and_cache_cover_creates_cover_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cover_art.requests, "get", lambda url, timeout, headers: _FakeResponse(b"bytes"))
    nested_dir = tmp_path / "does" / "not" / "exist" / "yet"

    cover_art.fetch_and_cache_cover(nested_dir, 1, "https://example.com/cover.jpg")

    assert nested_dir.is_dir()


def test_fetch_and_cache_cover_raises_on_http_error(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cover_art.requests, "get", lambda url, timeout, headers: _FakeResponse(b"", status_code=404)
    )

    with pytest.raises(requests.HTTPError):
        cover_art.fetch_and_cache_cover(tmp_path, 1, "https://example.com/gone.jpg")


# --- get_or_fetch_cover ------------------------------------------------------


def test_get_or_fetch_cover_returns_cached_without_fetching(tmp_path, monkeypatch):
    cached = tmp_path / "5.jpg"
    cached.write_bytes(b"already here")

    def unreachable_get(url, timeout, headers):
        raise AssertionError("should not fetch - already cached")

    monkeypatch.setattr(cover_art.requests, "get", unreachable_get)

    result = cover_art.get_or_fetch_cover(tmp_path, 5, "https://example.com/should-be-ignored.jpg")

    assert result == cached


def test_get_or_fetch_cover_fetches_when_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr(cover_art.requests, "get", lambda url, timeout, headers: _FakeResponse(b"fresh bytes"))

    result = cover_art.get_or_fetch_cover(tmp_path, 9, "https://example.com/cover.png")

    assert result == tmp_path / "9.png"
    assert result.read_bytes() == b"fresh bytes"


def test_get_or_fetch_cover_none_when_not_cached_and_no_url(tmp_path):
    assert cover_art.get_or_fetch_cover(tmp_path, 1, None) is None
