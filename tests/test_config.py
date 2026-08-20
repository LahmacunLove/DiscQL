from __future__ import annotations

from discql import config as config_module


def _isolate_dirs(monkeypatch, tmp_path):
    # Avoid picking up the real project-root .env (with a real token) via
    # load_dotenv()'s cwd-upward search.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_module, "CONFIG_DIR", tmp_path / "config")
    monkeypatch.setattr(config_module, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.delenv("DISCOGS_USER_TOKEN", raising=False)
    monkeypatch.delenv("DISCOGS_DB_PATH", raising=False)
    monkeypatch.delenv("FUZZY_CONFIDENT_THRESHOLD", raising=False)
    monkeypatch.delenv("FUZZY_TIE_MARGIN", raising=False)
    monkeypatch.delenv("FUZZY_DENOISE_TERMS", raising=False)
    monkeypatch.delenv("DISCOGS_AUDIO_DIR", raising=False)
    monkeypatch.delenv("YOUTUBE_COOKIES_BROWSER", raising=False)
    monkeypatch.delenv("YOUTUBE_AUDIO_MAX_BITRATE_KBPS", raising=False)
    monkeypatch.delenv("WAVEFORM_DIR", raising=False)
    monkeypatch.delenv("ESSENTIA_MODELS_DIR", raising=False)
    monkeypatch.delenv("COVER_DIR", raising=False)
    monkeypatch.delenv("MAX_WORKERS", raising=False)
    monkeypatch.delenv("LOCAL_FLAC_DIR", raising=False)
    monkeypatch.delenv("LOCAL_MATCH_CONFIDENT_THRESHOLD", raising=False)
    monkeypatch.delenv("DISCOGS_CONSUMER_KEY", raising=False)
    monkeypatch.delenv("DISCOGS_CONSUMER_SECRET", raising=False)
    monkeypatch.delenv("DISCOGS_OAUTH_TOKEN", raising=False)
    monkeypatch.delenv("DISCOGS_OAUTH_TOKEN_SECRET", raising=False)


def test_load_config_uses_env_token_and_default_cache_db_path(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")

    cfg = config_module.load_config()

    assert cfg.discogs_user_token == "env-token"
    assert cfg.db_path == tmp_path / "cache" / "discogs.db"
    assert cfg.db_path.parent.is_dir()
    assert cfg.fuzzy_confident_threshold == config_module.DEFAULT_FUZZY_CONFIDENT_THRESHOLD
    assert cfg.fuzzy_tie_margin == config_module.DEFAULT_FUZZY_TIE_MARGIN
    assert cfg.fuzzy_denoise_terms == ["Original Mix"]
    assert isinstance(cfg.fuzzy_denoise_terms, list)
    assert all(isinstance(term, str) for term in cfg.fuzzy_denoise_terms)
    assert cfg.audio_dir == tmp_path / "cache" / "audio"
    assert cfg.youtube_cookies_browser is None
    assert cfg.youtube_audio_max_bitrate_kbps == int(config_module.DEFAULT_YOUTUBE_AUDIO_MAX_BITRATE_KBPS)
    assert cfg.waveform_dir == tmp_path / "cache" / "waveforms"
    assert cfg.essentia_models_dir == tmp_path / "cache" / "essentia_models"
    assert cfg.cover_dir == tmp_path / "cache" / "covers"
    assert cfg.max_workers == config_module._default_max_workers()


def test_default_max_workers_leaves_one_core_free(monkeypatch):
    monkeypatch.setattr(config_module.os, "cpu_count", lambda: 9)
    assert config_module._default_max_workers() == 8


def test_default_max_workers_never_goes_below_one(monkeypatch):
    monkeypatch.setattr(config_module.os, "cpu_count", lambda: 1)
    assert config_module._default_max_workers() == 1
    monkeypatch.setattr(config_module.os, "cpu_count", lambda: None)
    assert config_module._default_max_workers() == 1


def test_load_config_respects_custom_max_workers_env(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")
    monkeypatch.setenv("MAX_WORKERS", "3")

    cfg = config_module.load_config()

    assert cfg.max_workers == 3


def test_load_config_writes_default_max_workers_into_new_config_file(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")
    monkeypatch.setattr(config_module.os, "cpu_count", lambda: 5)

    cfg = config_module.load_config()

    assert cfg.max_workers == 4
    saved = (tmp_path / "config" / "config").read_text()
    assert "MAX_WORKERS=4" in saved


def test_load_config_respects_custom_waveform_and_models_dir_env(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")
    custom_waveform = tmp_path / "elsewhere" / "waveforms"
    custom_models = tmp_path / "elsewhere" / "models"
    custom_covers = tmp_path / "elsewhere" / "covers"
    monkeypatch.setenv("WAVEFORM_DIR", str(custom_waveform))
    monkeypatch.setenv("ESSENTIA_MODELS_DIR", str(custom_models))
    monkeypatch.setenv("COVER_DIR", str(custom_covers))

    cfg = config_module.load_config()

    assert cfg.waveform_dir == custom_waveform
    assert cfg.essentia_models_dir == custom_models
    assert cfg.cover_dir == custom_covers


def test_load_config_respects_custom_audio_dir_env(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")
    custom_dir = tmp_path / "elsewhere" / "audio"
    monkeypatch.setenv("DISCOGS_AUDIO_DIR", str(custom_dir))

    cfg = config_module.load_config()

    assert cfg.audio_dir == custom_dir


def test_load_config_respects_custom_cookies_browser_env(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")
    monkeypatch.setenv("YOUTUBE_COOKIES_BROWSER", "firefox")

    cfg = config_module.load_config()

    assert cfg.youtube_cookies_browser == "firefox"


def test_load_config_writes_default_empty_cookies_browser_into_config_file(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")

    cfg = config_module.load_config()

    assert cfg.youtube_cookies_browser is None
    saved = (tmp_path / "config" / "config").read_text()
    assert "YOUTUBE_COOKIES_BROWSER=" in saved


def test_load_config_respects_custom_max_bitrate_env(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")
    monkeypatch.setenv("YOUTUBE_AUDIO_MAX_BITRATE_KBPS", "96")

    cfg = config_module.load_config()

    assert cfg.youtube_audio_max_bitrate_kbps == 96


def test_load_config_empty_max_bitrate_env_disables_cap(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")
    monkeypatch.setenv("YOUTUBE_AUDIO_MAX_BITRATE_KBPS", "")

    cfg = config_module.load_config()

    assert cfg.youtube_audio_max_bitrate_kbps is None


def test_load_config_respects_custom_denoise_terms_env(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")
    monkeypatch.setenv("FUZZY_DENOISE_TERMS", "Original Mix, Radio Edit ,Extended Mix")

    cfg = config_module.load_config()

    assert cfg.fuzzy_denoise_terms == ["Original Mix", "Radio Edit", "Extended Mix"]


def test_load_config_respects_custom_fuzzy_threshold_env(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")
    monkeypatch.setenv("FUZZY_CONFIDENT_THRESHOLD", "0.65")

    cfg = config_module.load_config()

    assert cfg.fuzzy_confident_threshold == 0.65


def test_load_config_respects_custom_tie_margin_env(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")
    monkeypatch.setenv("FUZZY_TIE_MARGIN", "0.1")

    cfg = config_module.load_config()

    assert cfg.fuzzy_tie_margin == 0.1


def test_load_config_respects_custom_db_path_env(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")
    custom_path = tmp_path / "elsewhere" / "my.db"
    monkeypatch.setenv("DISCOGS_DB_PATH", str(custom_path))

    cfg = config_module.load_config()

    assert cfg.db_path == custom_path
    assert cfg.db_path.parent.is_dir()


def test_load_config_writes_default_threshold_into_new_config_file(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")

    cfg = config_module.load_config()

    assert cfg.fuzzy_confident_threshold == config_module.DEFAULT_FUZZY_CONFIDENT_THRESHOLD
    assert cfg.fuzzy_tie_margin == config_module.DEFAULT_FUZZY_TIE_MARGIN
    assert cfg.fuzzy_denoise_terms == ["Original Mix"]
    saved = (tmp_path / "config" / "config").read_text()
    assert f"FUZZY_CONFIDENT_THRESHOLD={config_module.DEFAULT_FUZZY_CONFIDENT_THRESHOLD}" in saved
    assert f"FUZZY_TIE_MARGIN={config_module.DEFAULT_FUZZY_TIE_MARGIN}" in saved
    assert f"FUZZY_DENOISE_TERMS={config_module.DEFAULT_FUZZY_DENOISE_TERMS}" in saved


def test_load_config_adds_missing_threshold_to_existing_config_file(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "config").write_text("DISCOGS_USER_TOKEN=file-token\n")

    cfg = config_module.load_config()

    assert cfg.discogs_user_token == "file-token"
    assert cfg.fuzzy_confident_threshold == config_module.DEFAULT_FUZZY_CONFIDENT_THRESHOLD
    saved = (config_dir / "config").read_text()
    assert "DISCOGS_USER_TOKEN=file-token" in saved
    assert "FUZZY_CONFIDENT_THRESHOLD=" in saved


def test_load_config_does_not_duplicate_existing_threshold(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "config").write_text("DISCOGS_USER_TOKEN=file-token\nFUZZY_CONFIDENT_THRESHOLD=0.55\n")

    cfg = config_module.load_config()

    assert cfg.fuzzy_confident_threshold == 0.55
    saved = (config_dir / "config").read_text()
    # The line itself isn't duplicated (a mention inside another key's
    # comment, like FUZZY_TIE_MARGIN's, is fine and expected).
    assert saved.count("FUZZY_CONFIDENT_THRESHOLD=0.55") == 1


def test_load_config_reads_token_from_config_file(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "config").write_text("DISCOGS_USER_TOKEN=file-token\n")

    cfg = config_module.load_config()

    assert cfg.discogs_user_token == "file-token"


def test_load_config_never_prompts_when_no_token_found(monkeypatch, tmp_path):
    """load_config() must never block on stdin - the web app starts with
    zero config and sends the user to /settings instead. See
    ensure_token_interactively() for the CLI's own explicit prompt step."""
    _isolate_dirs(monkeypatch, tmp_path)

    def unreachable_getpass(prompt=""):
        raise AssertionError("load_config() must not prompt")

    monkeypatch.setattr(config_module, "getpass", unreachable_getpass)

    cfg = config_module.load_config()

    assert cfg.discogs_user_token == ""
    assert cfg.has_discogs_auth is False


def test_ensure_token_interactively_prompts_and_saves(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(config_module, "getpass", lambda prompt="": "typed-token")

    token = config_module.ensure_token_interactively()

    assert token == "typed-token"
    saved = (tmp_path / "config" / "config").read_text()
    assert "DISCOGS_USER_TOKEN=typed-token" in saved
    assert config_module.load_config().discogs_user_token == "typed-token"


def test_ensure_token_interactively_reprompts_on_blank_input(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    responses = iter(["  ", "", "second-try"])
    monkeypatch.setattr(config_module, "getpass", lambda prompt="": next(responses))

    token = config_module.ensure_token_interactively()

    assert token == "second-try"


def test_set_config_values_appends_new_keys_to_a_fresh_file(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)

    config_module.set_config_values({"MAX_WORKERS": "3", "YOUTUBE_COOKIES_BROWSER": "firefox"})

    saved = (tmp_path / "config" / "config").read_text()
    assert "MAX_WORKERS=3" in saved
    assert "YOUTUBE_COOKIES_BROWSER=firefox" in saved


def test_set_config_values_replaces_existing_key_in_place(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    config_file = tmp_path / "config" / "config"
    config_file.parent.mkdir(parents=True)
    config_file.write_text("# a comment\nMAX_WORKERS=3\n# another comment\nFUZZY_TIE_MARGIN=0.05\n")

    config_module.set_config_values({"MAX_WORKERS": "7"})

    saved = config_file.read_text()
    lines = saved.splitlines()
    assert lines == ["# a comment", "MAX_WORKERS=7", "# another comment", "FUZZY_TIE_MARGIN=0.05"]


def test_set_config_values_used_by_load_config_on_next_call(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")

    config_module.set_config_values({"MAX_WORKERS": "5"})
    # load_config() reads from the file via load_dotenv(), which only sets
    # os.environ keys not already present - MAX_WORKERS wasn't set via
    # monkeypatch.setenv here, so the freshly-written file value applies.
    cfg = config_module.load_config()

    assert cfg.max_workers == 5


def test_has_discogs_auth_false_with_nothing_configured(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)

    assert config_module.load_config().has_discogs_auth is False


def test_has_discogs_auth_true_with_personal_token(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    monkeypatch.setenv("DISCOGS_USER_TOKEN", "env-token")

    assert config_module.load_config().has_discogs_auth is True


def test_has_discogs_auth_true_only_when_all_four_oauth_fields_are_set(monkeypatch, tmp_path):
    _isolate_dirs(monkeypatch, tmp_path)
    oauth_env = {
        "DISCOGS_CONSUMER_KEY": "ck",
        "DISCOGS_CONSUMER_SECRET": "cs",
        "DISCOGS_OAUTH_TOKEN": "tok",
        "DISCOGS_OAUTH_TOKEN_SECRET": "sec",
    }
    for key in oauth_env:
        assert config_module.load_config().has_discogs_auth is False
        monkeypatch.setenv(key, oauth_env[key])

    assert config_module.load_config().has_discogs_auth is True
