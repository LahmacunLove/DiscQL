from __future__ import annotations

import os
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path

from dotenv import load_dotenv

USER_AGENT = "discql/0.1 +https://github.com/ferdifuchs"

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser() / "discql"
CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", "~/.cache")).expanduser() / "discql"

# Fuzzy score (0-1) above which a YouTube-to-track match is trusted without
# consulting the LLM. Adjustable without touching code: add
# FUZZY_CONFIDENT_THRESHOLD=<value> to the config file, or export it as an
# environment variable.
DEFAULT_FUZZY_CONFIDENT_THRESHOLD = 0.8

# Minimum gap required between a video's best- and second-best-scoring track
# for the confident threshold above to apply, instead of escalating to the
# LLM. Also adjustable via FUZZY_TIE_MARGIN in the config file or environment.
DEFAULT_FUZZY_TIE_MARGIN = 0.05

# Comma-separated generic terms stripped from a video title before fuzzy
# matching, alongside the release title/artist name(s)/year (see
# denoise_video_title). Adjustable via FUZZY_DENOISE_TERMS in the config
# file or environment, e.g. FUZZY_DENOISE_TERMS=Original Mix,Radio Edit
DEFAULT_FUZZY_DENOISE_TERMS = "Original Mix"


# Local FLAC library root (e.g. Bandcamp purchases) to match against the
# collection before falling back to YouTube - see local_audio.py. Empty by
# default (disabled, same "empty = disabled" convention as
# YOUTUBE_COOKIES_BROWSER below) - no machine-specific path belongs in the
# code default. Set via LOCAL_FLAC_DIR in the config file or environment.
DEFAULT_LOCAL_FLAC_DIR = ""

# Fuzzy score (0-1) above which a local folder is trusted as a release's
# match. Separate from FUZZY_CONFIDENT_THRESHOLD (YouTube) since folder-name
# noise (catalog numbers, repeated artist/title, underscores) has a
# different profile than video-title noise. Adjustable via
# LOCAL_MATCH_CONFIDENT_THRESHOLD in the config file or environment.
DEFAULT_LOCAL_MATCH_CONFIDENT_THRESHOLD = 0.75


# Browser to pull YouTube cookies from (yt-dlp's --cookies-from-browser),
# e.g. "firefox", "chrome", "chromium", "brave", "edge", "safari", "vivaldi",
# "opera". Empty by default (disabled) - opt-in since it uses your real
# logged-in browser session. Needed for age-restricted videos, which YouTube
# otherwise refuses to serve metadata for at all (no per-client workaround).
DEFAULT_YOUTUBE_COOKIES_BROWSER = ""

# Caps which audio-only stream yt-dlp downloads (format selector
# "bestaudio[abr<=N]"), rather than always the highest-bitrate one available
# (typically ~160kbps Opus). Still one of YouTube's own native streams, no
# re-encoding - just a smaller one when a lower-bitrate stream exists.
# Adjustable via YOUTUBE_AUDIO_MAX_BITRATE_KBPS in the config file or
# environment; leave empty to disable (always take the highest bitrate).
DEFAULT_YOUTUBE_AUDIO_MAX_BITRATE_KBPS = "128"

# Waveform images, audio, models, and covers all live under the persistent
# XDG cache dir (as subdirs of CACHE_DIR, computed inside load_config()
# rather than precomputed here - CACHE_DIR itself may be monkeypatched,
# e.g. in tests, after this module is imported).


def _default_max_workers() -> int:
    """How many releases "Match YT" / "Run All" process in parallel by
    default: one core left free for the OS/web server rather than
    saturating every core. Computed fresh (not a module-level constant)
    since it's written into the config file with the actual number for
    this machine on first run - see load_config()."""
    return max(1, (os.cpu_count() or 2) - 1)


@dataclass(frozen=True)
class Config:
    discogs_user_token: str
    db_path: Path
    fuzzy_confident_threshold: float
    fuzzy_tie_margin: float
    fuzzy_denoise_terms: list[str]
    audio_dir: Path
    youtube_cookies_browser: str | None
    youtube_audio_max_bitrate_kbps: int | None
    waveform_dir: Path
    essentia_models_dir: Path
    cover_dir: Path
    max_workers: int
    local_flac_dir: Path | None
    local_match_confident_threshold: float
    discogs_consumer_key: str | None
    discogs_consumer_secret: str | None
    discogs_oauth_token: str | None
    discogs_oauth_token_secret: str | None

    @property
    def has_discogs_auth(self) -> bool:
        """Whether Discogs access is configured at all - either a simple
        personal token, or a full OAuth 1.0a credential set (consumer
        key/secret from a registered Discogs Application, plus the access
        token/secret from completing the browser authorize flow - see
        web/app.py's /settings/discogs/* routes). Used to decide whether
        the web app should redirect to /settings, and whether the CLI
        needs to prompt for a token."""
        return bool(self.discogs_user_token) or bool(
            self.discogs_consumer_key
            and self.discogs_consumer_secret
            and self.discogs_oauth_token
            and self.discogs_oauth_token_secret
        )


def _config_file() -> Path:
    return CONFIG_DIR / "config"


def _ensure_default_in_config_file(key: str, comment: str, default: float | str) -> None:
    """Write `key=default` into the config file if it isn't there yet,
    generating the file (and its directory) on first run if needed. This
    makes the value a visible, editable line in the config file instead of
    an invisible code default.
    """
    config_file = _config_file()
    existing = config_file.read_text() if config_file.exists() else ""
    if key in existing:
        return

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with config_file.open("a") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(f"\n{comment}\n{key}={default}\n")


def set_config_values(values: dict[str, str]) -> None:
    """Upsert one or more KEY=value lines into the config file - replaces
    an existing line for a key in place (preserving surrounding comments
    and line order), appends a new line for any key not already present.
    Creates the file (and its directory) if it doesn't exist yet, same as
    _ensure_default_in_config_file. Used by the web settings page to
    persist edits (including the Discogs token/OAuth credentials) - unlike
    _ensure_default_in_config_file, this always writes the given value,
    not just when the key is missing.

    Also removes each key from os.environ (see the end of this function) -
    load_dotenv() (called by load_config()) defaults to *not* overriding
    already-set environment variables, which is right for its normal job
    (an explicitly-exported shell env var should win over the file), but
    wrong here: once a long-running process (the web app) has called
    load_config() once, the values it read are already sitting in
    os.environ, so a later load_dotenv() call would never pick up a change
    this function just wrote to the file - confirmed concretely: saving a
    new MAX_WORKERS via the settings page had no effect until the process
    restarted, without this.
    """
    config_file = _config_file()
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    lines = config_file.read_text().splitlines() if config_file.exists() else []

    remaining = dict(values)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0]
        if key in remaining:
            lines[i] = f"{key}={remaining.pop(key)}"

    for key, value in remaining.items():
        lines.append(f"{key}={value}")

    config_file.write_text("\n".join(lines) + "\n")
    config_file.chmod(0o600)

    for key in values:
        os.environ.pop(key, None)


def ensure_token_interactively() -> str:
    """CLI-only: blocks on a terminal prompt for the Discogs API token if
    none is configured yet, saves it, and returns it. load_config() itself
    never prompts (see its own comments) - the web app needs to be able to
    start without ever blocking on stdin (it sends a fresh install to the
    /settings page in the browser instead), so this has to be an explicit,
    separate step a caller opts into - see cli.py's call sites.
    """
    print("No Discogs API token configured.")
    print("Create a personal access token at https://www.discogs.com/settings/developers")
    token = ""
    while not token:
        token = getpass("Discogs API token: ").strip()

    set_config_values({"DISCOGS_USER_TOKEN": token})
    print(f"Saved token to {_config_file()}")
    return token


def load_config() -> Config:
    _ensure_default_in_config_file(
        "FUZZY_CONFIDENT_THRESHOLD",
        "# Fuzzy score (0-1) above which a YouTube-to-track match is trusted\n"
        "# without asking the LLM. Below this, a video is escalated to the LLM\n"
        "# (after first retrying with uploader boilerplate stripped from its title).",
        DEFAULT_FUZZY_CONFIDENT_THRESHOLD,
    )
    _ensure_default_in_config_file(
        "FUZZY_TIE_MARGIN",
        "# Minimum gap required between a video's best- and second-best-scoring\n"
        "# track for FUZZY_CONFIDENT_THRESHOLD to apply outright. Below this gap,\n"
        "# the match is treated as ambiguous (denoise retry / LLM) even if the top\n"
        "# score alone would otherwise be confident.",
        DEFAULT_FUZZY_TIE_MARGIN,
    )
    _ensure_default_in_config_file(
        "FUZZY_DENOISE_TERMS",
        "# Comma-separated generic terms stripped from a video title before fuzzy\n"
        "# matching (alongside the release title/artist name(s)/year), e.g. terms\n"
        "# uploaders tack onto every video regardless of the actual track name.",
        DEFAULT_FUZZY_DENOISE_TERMS,
    )
    _ensure_default_in_config_file(
        "YOUTUBE_COOKIES_BROWSER",
        "# Browser to pull YouTube cookies from (yt-dlp --cookies-from-browser),\n"
        "# e.g. firefox / chrome / chromium / brave / edge / safari / vivaldi /\n"
        "# opera. Leave empty to disable. Needed for age-restricted videos, which\n"
        "# YouTube otherwise refuses to serve metadata for at all.",
        DEFAULT_YOUTUBE_COOKIES_BROWSER,
    )
    _ensure_default_in_config_file(
        "YOUTUBE_AUDIO_MAX_BITRATE_KBPS",
        "# Caps which audio-only stream yt-dlp downloads, rather than always\n"
        "# the highest-bitrate one available (typically ~160kbps Opus). Still\n"
        "# one of YouTube's own native streams, no re-encoding - just a smaller\n"
        "# one when a lower-bitrate stream exists. Leave empty to disable\n"
        "# (always take the highest bitrate).",
        DEFAULT_YOUTUBE_AUDIO_MAX_BITRATE_KBPS,
    )
    _ensure_default_in_config_file(
        "LOCAL_FLAC_DIR",
        "# Local FLAC library root (e.g. Bandcamp purchases) to match against the\n"
        "# collection before falling back to YouTube - see local_audio.py. Empty\n"
        "# disables the feature entirely. Leave empty if you don't have one.",
        DEFAULT_LOCAL_FLAC_DIR,
    )
    _ensure_default_in_config_file(
        "LOCAL_MATCH_CONFIDENT_THRESHOLD",
        "# Fuzzy score (0-1) above which a local folder is trusted as a release's\n"
        "# match, and a local file is trusted as a track's match.",
        DEFAULT_LOCAL_MATCH_CONFIDENT_THRESHOLD,
    )
    _ensure_default_in_config_file(
        "MAX_WORKERS",
        "# How many releases \"Match YT\" / \"Run All\" match+download in parallel.\n"
        "# Defaulted below to this machine's core count minus one, so it isn't\n"
        "# forgotten as an invisible code default - edit freely.",
        _default_max_workers(),
    )
    load_dotenv(_config_file())

    # Never prompts - see ensure_token_interactively()'s docstring for why.
    # A missing token isn't an error here: Config.has_discogs_auth /
    # OAuth credentials (discogs_consumer_key etc., below) are the actual
    # "is Discogs access configured" check callers should use.
    token = os.environ.get("DISCOGS_USER_TOKEN", "").strip()

    db_path = Path(os.environ.get("DISCOGS_DB_PATH", str(CACHE_DIR / "discogs.db"))).expanduser()
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Downloaded track audio is normally interim storage - kept if fetched
    # via the standalone "Download tracks" button, but deleted again once
    # audio_analysis.py has finished analyzing a track (see CLAUDE.md: only
    # the analysis results are kept long-term, not the audio). Still lives
    # under the persistent cache dir rather than /tmp, since "Run All" now
    # downloads a batch of releases' audio up front (see pipeline.py) before
    # analyzing any of it - /tmp being tmpfs-backed or cleared mid-run would
    # lose that interim audio before analysis gets to it.
    audio_dir = Path(os.environ.get("DISCOGS_AUDIO_DIR", str(CACHE_DIR / "audio"))).expanduser()
    waveform_dir = Path(os.environ.get("WAVEFORM_DIR", str(CACHE_DIR / "waveforms"))).expanduser()
    essentia_models_dir = Path(
        os.environ.get("ESSENTIA_MODELS_DIR", str(CACHE_DIR / "essentia_models"))
    ).expanduser()
    # Cover art is fetched lazily and cached on first request (see
    # cover_art.get_or_fetch_cover), not eagerly during sync - a genuine
    # cache of an external resource, not a generated analysis result, but
    # kept long-term like waveforms/models rather than in the temporary
    # audio_dir, since re-fetching every release's cover on every request
    # would be wasteful and there's no "delete after use" reason to.
    cover_dir = Path(os.environ.get("COVER_DIR", str(CACHE_DIR / "covers"))).expanduser()

    fuzzy_confident_threshold = float(
        os.environ.get("FUZZY_CONFIDENT_THRESHOLD", DEFAULT_FUZZY_CONFIDENT_THRESHOLD)
    )
    fuzzy_tie_margin = float(os.environ.get("FUZZY_TIE_MARGIN", DEFAULT_FUZZY_TIE_MARGIN))
    fuzzy_denoise_terms = [
        term.strip()
        for term in os.environ.get("FUZZY_DENOISE_TERMS", DEFAULT_FUZZY_DENOISE_TERMS).split(",")
        if term.strip()
    ]
    youtube_cookies_browser = (
        os.environ.get("YOUTUBE_COOKIES_BROWSER", DEFAULT_YOUTUBE_COOKIES_BROWSER).strip() or None
    )
    youtube_audio_max_bitrate_raw = os.environ.get(
        "YOUTUBE_AUDIO_MAX_BITRATE_KBPS", DEFAULT_YOUTUBE_AUDIO_MAX_BITRATE_KBPS
    ).strip()
    youtube_audio_max_bitrate_kbps = int(youtube_audio_max_bitrate_raw) if youtube_audio_max_bitrate_raw else None
    max_workers = max(1, int(os.environ.get("MAX_WORKERS", _default_max_workers())))

    local_flac_dir_raw = os.environ.get("LOCAL_FLAC_DIR", DEFAULT_LOCAL_FLAC_DIR).strip()
    local_flac_dir = Path(local_flac_dir_raw).expanduser() if local_flac_dir_raw else None
    local_match_confident_threshold = float(
        os.environ.get("LOCAL_MATCH_CONFIDENT_THRESHOLD", DEFAULT_LOCAL_MATCH_CONFIDENT_THRESHOLD)
    )

    # OAuth credentials - no code default/config-file scaffolding like the
    # settings above (see _ensure_default_in_config_file calls): they don't
    # exist until the user either goes through the browser authorize flow
    # (web/app.py's /settings/discogs/* routes write these directly via
    # set_config_values) or pastes them in manually. Absent unless set.
    discogs_consumer_key = os.environ.get("DISCOGS_CONSUMER_KEY", "").strip() or None
    discogs_consumer_secret = os.environ.get("DISCOGS_CONSUMER_SECRET", "").strip() or None
    discogs_oauth_token = os.environ.get("DISCOGS_OAUTH_TOKEN", "").strip() or None
    discogs_oauth_token_secret = os.environ.get("DISCOGS_OAUTH_TOKEN_SECRET", "").strip() or None

    return Config(
        discogs_user_token=token,
        db_path=db_path,
        fuzzy_confident_threshold=fuzzy_confident_threshold,
        fuzzy_tie_margin=fuzzy_tie_margin,
        fuzzy_denoise_terms=fuzzy_denoise_terms,
        audio_dir=audio_dir,
        youtube_cookies_browser=youtube_cookies_browser,
        youtube_audio_max_bitrate_kbps=youtube_audio_max_bitrate_kbps,
        waveform_dir=waveform_dir,
        essentia_models_dir=essentia_models_dir,
        cover_dir=cover_dir,
        max_workers=max_workers,
        local_flac_dir=local_flac_dir,
        local_match_confident_threshold=local_match_confident_threshold,
        discogs_consumer_key=discogs_consumer_key,
        discogs_consumer_secret=discogs_consumer_secret,
        discogs_oauth_token=discogs_oauth_token,
        discogs_oauth_token_secret=discogs_oauth_token_secret,
    )
