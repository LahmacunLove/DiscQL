from __future__ import annotations

import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from typing import Callable, Iterator

import requests

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

OllamaCaller = Callable[[str], str]


def get_ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", DEFAULT_HOST)


def get_ollama_model() -> str:
    return os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)


def call_ollama(
    prompt: str,
    model: str | None = None,
    host: str | None = None,
    timeout: float = 120,
) -> str:
    """POST a prompt to a local Ollama server, requesting JSON-formatted output.

    Returns the raw response text (the caller is responsible for json.loads-ing
    it and handling malformed output — Ollama's format="json" mode constrains
    the model to emit syntactically valid JSON, but not necessarily the shape
    the caller asked for in the prompt).
    """
    response = requests.post(
        f"{host or get_ollama_host()}/api/generate",
        json={
            "model": model or get_ollama_model(),
            "prompt": prompt,
            "format": "json",
            "stream": False,
            # Greedy decoding: matching decisions should be reproducible run
            # to run given the same inputs, not vary with sampling noise.
            "options": {"temperature": 0},
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()["response"]


def is_ollama_reachable(host: str | None = None, timeout: float = 2.0) -> bool:
    try:
        response = requests.get(f"{host or get_ollama_host()}/api/tags", timeout=timeout)
        return response.ok
    except requests.RequestException:
        return False


def start_ollama(host: str | None = None, startup_timeout: float = 30.0) -> subprocess.Popen | None:
    """Start `ollama serve` in the background if it isn't already reachable,
    and block until it responds (LLM calls right after this returns would
    otherwise hit a server still booting). Returns the Popen handle if this
    call actually started a new process - the caller then owns it and must
    stop_ollama() it when done - or None if Ollama was already running, in
    which case it must NOT be stopped afterward: it wasn't started by us, so
    it isn't ours to shut down (could be serving something else entirely).
    """
    if is_ollama_reachable(host):
        return None

    if shutil.which("ollama") is None:
        raise RuntimeError("Ollama isn't running and the `ollama` CLI isn't installed/on PATH")

    process = subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        if is_ollama_reachable(host):
            return process
        if process.poll() is not None:
            raise RuntimeError(f"`ollama serve` exited immediately (exit code {process.returncode})")
        time.sleep(0.2)

    process.terminate()
    raise RuntimeError(f"Ollama did not become reachable within {startup_timeout}s of starting it")


def stop_ollama(process: subprocess.Popen | None) -> None:
    """Stop an Ollama server previously started via start_ollama(). A no-op
    if process is None - see start_ollama for why that means "not ours to
    stop", not "nothing to do here, but check anyway".
    """
    if process is None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@contextmanager
def ensure_ollama_running(host: str | None = None, startup_timeout: float = 30.0) -> Iterator[None]:
    """Start Ollama for the duration of the wrapped block if it isn't
    already running, and stop it again afterward - but only if this call is
    the one that started it (see start_ollama/stop_ollama). Wrap the
    *outermost* call that might need LLM matching (a whole-library match
    run, not each release within it) so Ollama is started/stopped once, not
    once per release.

    If Ollama can't be started at all (CLI missing, never became reachable),
    this doesn't raise - matching already tolerates Ollama being unavailable
    (falls back to fuzzy-only results, see resolve_release), so failing to
    auto-start it should degrade the same way as it not being installed at
    all, not abort matching outright.
    """
    process: subprocess.Popen | None = None
    try:
        process = start_ollama(host, startup_timeout)
    except RuntimeError as exc:
        print(f"ensure_ollama_running: {exc} - continuing without it (fuzzy-only matching)")
    try:
        yield
    finally:
        stop_ollama(process)
