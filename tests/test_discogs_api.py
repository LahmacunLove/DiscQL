from __future__ import annotations

from pathlib import Path

from discql import discogs_api
from discql.config import Config


def make_config(**overrides) -> Config:
    defaults = dict(
        discogs_user_token="",
        db_path=Path("/tmp/db"),
        fuzzy_confident_threshold=0.8,
        fuzzy_tie_margin=0.05,
        fuzzy_denoise_terms=["Original Mix"],
        audio_dir=Path("/tmp/audio"),
        youtube_cookies_browser=None,
        youtube_audio_max_bitrate_kbps=128,
        waveform_dir=Path("/tmp/waveforms"),
        essentia_models_dir=Path("/tmp/models"),
        cover_dir=Path("/tmp/covers"),
        max_workers=4,
        local_flac_dir=None,
        local_match_confident_threshold=0.75,
        discogs_consumer_key=None,
        discogs_consumer_secret=None,
        discogs_oauth_token=None,
        discogs_oauth_token_secret=None,
    )
    defaults.update(overrides)
    return Config(**defaults)


def test_build_discogs_api_uses_personal_token_when_no_oauth_configured(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        discogs_api,
        "DiscogsApi",
        lambda user_token=None, oauth_credentials=None: captured.update(
            user_token=user_token, oauth_credentials=oauth_credentials
        ),
    )

    discogs_api.build_discogs_api(make_config(discogs_user_token="plain-token"))

    assert captured == {"user_token": "plain-token", "oauth_credentials": None}


def test_build_discogs_api_prefers_oauth_when_fully_configured(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        discogs_api,
        "DiscogsApi",
        lambda user_token=None, oauth_credentials=None: captured.update(
            user_token=user_token, oauth_credentials=oauth_credentials
        ),
    )

    config = make_config(
        discogs_user_token="plain-token",  # present too - OAuth should still win
        discogs_consumer_key="ck",
        discogs_consumer_secret="cs",
        discogs_oauth_token="tok",
        discogs_oauth_token_secret="sec",
    )
    discogs_api.build_discogs_api(config)

    assert captured == {"user_token": None, "oauth_credentials": ("ck", "cs", "tok", "sec")}


def test_build_discogs_api_falls_back_to_personal_token_when_oauth_partially_configured(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        discogs_api,
        "DiscogsApi",
        lambda user_token=None, oauth_credentials=None: captured.update(
            user_token=user_token, oauth_credentials=oauth_credentials
        ),
    )

    # Only 3 of 4 OAuth fields set - not enough, must not be treated as configured.
    config = make_config(
        discogs_user_token="plain-token",
        discogs_consumer_key="ck",
        discogs_consumer_secret="cs",
        discogs_oauth_token="tok",
        discogs_oauth_token_secret=None,
    )
    discogs_api.build_discogs_api(config)

    assert captured == {"user_token": "plain-token", "oauth_credentials": None}


def test_discogs_api_init_with_oauth_credentials_constructs_client_correctly(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, user_agent, **kwargs):
            captured["user_agent"] = user_agent
            captured.update(kwargs)

    monkeypatch.setattr(discogs_api.discogs_client, "Client", FakeClient)

    discogs_api.DiscogsApi(oauth_credentials=("ck", "cs", "tok", "sec"))

    assert captured["consumer_key"] == "ck"
    assert captured["consumer_secret"] == "cs"
    assert captured["token"] == "tok"
    assert captured["secret"] == "sec"


def test_discogs_api_init_with_user_token_constructs_client_correctly(monkeypatch):
    captured = {}

    class FakeClient:
        def __init__(self, user_agent, **kwargs):
            captured["user_agent"] = user_agent
            captured.update(kwargs)

    monkeypatch.setattr(discogs_api.discogs_client, "Client", FakeClient)

    discogs_api.DiscogsApi(user_token="plain-token")

    assert captured["user_token"] == "plain-token"
    assert "consumer_key" not in captured
