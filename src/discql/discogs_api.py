from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import discogs_client
from discogs_client.models import CollectionItemInstance, Release

from discql.config import Config, USER_AGENT

ALL_FOLDER_ID = 0


@dataclass
class TrackData:
    position: str | None
    title: str
    duration: str | None
    artist: str | None


@dataclass
class ArtistData:
    id: int
    name: str


@dataclass
class LabelData:
    id: int
    name: str
    catalog_number: str


@dataclass
class VideoData:
    title: str | None
    url: str
    duration: int | None
    description: str | None


@dataclass
class CreditData:
    artist_id: int
    artist_name: str
    role: str
    tracks: str
    name_variation: str


@dataclass
class ReleaseData:
    id: int
    title: str
    year: int | None
    genres: list[str] = field(default_factory=list)
    styles: list[str] = field(default_factory=list)
    formats: list[dict] = field(default_factory=list)
    discogs_uri: str | None = None
    cover_image_url: str | None = None
    date_added: str | None = None
    country: str | None = None
    images: list[dict] = field(default_factory=list)
    artists: list[ArtistData] = field(default_factory=list)
    labels: list[LabelData] = field(default_factory=list)
    tracks: list[TrackData] = field(default_factory=list)
    videos: list[VideoData] = field(default_factory=list)
    credits: list[CreditData] = field(default_factory=list)


class DiscogsApi:
    def __init__(self, user_token: str | None = None, oauth_credentials: tuple[str, str, str, str] | None = None):
        """Exactly one of user_token (simple personal access token) or
        oauth_credentials (consumer_key, consumer_secret, token, secret -
        from completing the OAuth 1.0a browser authorize flow, see
        web/app.py's /settings/discogs/* routes) should be given - prefer
        build_discogs_api(config) over constructing this directly, it
        picks the right one for you based on what's configured.
        """
        if oauth_credentials:
            consumer_key, consumer_secret, token, secret = oauth_credentials
            self._client = discogs_client.Client(
                USER_AGENT, consumer_key=consumer_key, consumer_secret=consumer_secret, token=token, secret=secret
            )
        else:
            self._client = discogs_client.Client(USER_AGENT, user_token=user_token)
        self._username: str | None = None

    @property
    def username(self) -> str:
        if self._username is None:
            self._username = self._client.identity().username
        return self._username

    def _collection_folder(self):
        return self._client.user(self.username).collection_folders[ALL_FOLDER_ID]

    def fetch_collection_items(self) -> Iterator[tuple[int, str | None]]:
        """Yield (release_id, date_added) for every release in the collection."""
        for item in self._collection_folder().releases:
            date_added = item.date_added.isoformat() if item.date_added else None
            yield item.release.id, date_added

    def fetch_release_detail(self, release_id: int) -> ReleaseData:
        release = self._client.release(release_id)
        return _release_to_data(release)

    def add_release_to_collection(self, release_id: int) -> None:
        folder = self._collection_folder()
        folder.add_release(release_id)

    def remove_release_from_collection(self, release_id: int) -> None:
        folder = self._collection_folder()
        instance = _find_collection_instance(folder, release_id)
        if instance is None:
            raise ValueError(f"Release {release_id} is not in the collection")
        folder.remove_release(instance)


def build_discogs_api(config: Config) -> DiscogsApi:
    """The one place that decides which auth mode DiscogsApi should use -
    OAuth if config.has_discogs_auth's OAuth credentials are all present,
    else the simple personal token. Every call site (cli.py's cmd_sync/
    cmd_add/cmd_remove, web/app.py's sync runner) should go through this
    rather than constructing DiscogsApi directly."""
    if (
        config.discogs_consumer_key
        and config.discogs_consumer_secret
        and config.discogs_oauth_token
        and config.discogs_oauth_token_secret
    ):
        return DiscogsApi(
            oauth_credentials=(
                config.discogs_consumer_key,
                config.discogs_consumer_secret,
                config.discogs_oauth_token,
                config.discogs_oauth_token_secret,
            )
        )
    return DiscogsApi(user_token=config.discogs_user_token)


def _find_collection_instance(folder, release_id: int) -> CollectionItemInstance | None:
    for item in folder.releases:
        if item.release.id == release_id:
            return item
    return None


def _dedupe_videos(raw_videos) -> list[VideoData]:
    """Discogs' own video list sometimes contains genuine duplicate URLs (verified
    against the live API) — likely from multiple community contributions or
    format variants being merged into one release. Dedupe by URL, keeping the
    first occurrence, so we don't mirror the duplication downstream.
    """
    seen_urls: set[str] = set()
    videos = []
    for video in raw_videos:
        if video.url in seen_urls:
            continue
        seen_urls.add(video.url)
        videos.append(
            VideoData(
                title=video.title,
                url=video.url,
                duration=video.duration,
                description=video.description,
            )
        )
    return videos


def _release_to_data(release: Release) -> ReleaseData:
    raw_images = release.images or []
    cover_image_url = raw_images[0]["uri"] if raw_images else None
    images = [
        {
            "type": img.get("type"),
            "uri": img.get("uri"),
            "uri150": img.get("uri150"),
            "width": img.get("width"),
            "height": img.get("height"),
        }
        for img in raw_images
    ]

    artists = [ArtistData(id=a.id, name=a.name) for a in (release.artists or [])]
    labels = [
        LabelData(id=label.id, name=label.name, catalog_number=label.catno or "")
        for label in (release.labels or [])
    ]
    tracks = [
        TrackData(
            position=track.position,
            title=track.title,
            duration=track.duration,
            artist=", ".join(a.name for a in track.artists) if track.artists else None,
        )
        for track in (release.tracklist or [])
    ]
    videos = _dedupe_videos(release.videos or [])
    # Artist objects are "primary" API objects: accessing a *declared* field that's
    # missing from the embedded JSON (e.g. .role on a plain artists-list entry)
    # silently triggers a real extra HTTP request. role/tracks/anv only exist on
    # credits entries, so read them via .data.get(...) to bypass that entirely.
    credits = [
        CreditData(
            artist_id=credit.id,
            artist_name=credit.name,
            role=credit.data.get("role", ""),
            tracks=credit.data.get("tracks", ""),
            name_variation=credit.data.get("anv", ""),
        )
        for credit in (release.credits or [])
    ]

    return ReleaseData(
        id=release.id,
        title=release.title,
        year=release.year,
        genres=list(release.genres or []),
        styles=list(release.styles or []),
        formats=list(release.formats or []),
        discogs_uri=release.url,
        cover_image_url=cover_image_url,
        country=release.country,
        images=images,
        artists=artists,
        labels=labels,
        tracks=tracks,
        videos=videos,
        credits=credits,
    )
