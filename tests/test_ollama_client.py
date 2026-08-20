from __future__ import annotations

import subprocess

import pytest
import requests

from discql import ollama_client as oc


class _FakeResponse:
    def __init__(self, ok: bool):
        self.ok = ok


class _FakeProcess:
    def __init__(self, poll_results=None):
        self._poll_results = list(poll_results or [])
        self.terminated = False
        self.killed = False
        self.wait_calls = 0
        self.returncode = None

    def poll(self):
        if self._poll_results:
            self.returncode = self._poll_results.pop(0)
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.wait_calls += 1


# --- is_ollama_reachable ------------------------------------------------------


def test_is_ollama_reachable_true_when_request_succeeds(monkeypatch):
    monkeypatch.setattr(oc.requests, "get", lambda url, timeout: _FakeResponse(True))
    assert oc.is_ollama_reachable() is True


def test_is_ollama_reachable_false_on_connection_error(monkeypatch):
    def raise_connection_error(url, timeout):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(oc.requests, "get", raise_connection_error)
    assert oc.is_ollama_reachable() is False


def test_is_ollama_reachable_false_when_response_not_ok(monkeypatch):
    monkeypatch.setattr(oc.requests, "get", lambda url, timeout: _FakeResponse(False))
    assert oc.is_ollama_reachable() is False


# --- start_ollama --------------------------------------------------------------


def test_start_ollama_returns_none_if_already_reachable(monkeypatch):
    monkeypatch.setattr(oc, "is_ollama_reachable", lambda host=None: True)
    popen_calls = []
    monkeypatch.setattr(oc.subprocess, "Popen", lambda *a, **kw: popen_calls.append((a, kw)))

    result = oc.start_ollama()

    assert result is None
    assert popen_calls == []


def test_start_ollama_raises_if_cli_missing(monkeypatch):
    monkeypatch.setattr(oc, "is_ollama_reachable", lambda host=None: False)
    monkeypatch.setattr(oc.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="ollama"):
        oc.start_ollama()


def test_start_ollama_spawns_process_and_waits_until_reachable(monkeypatch):
    monkeypatch.setattr(oc.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(oc.time, "sleep", lambda seconds: None)

    reachable_calls = {"count": 0}

    def fake_reachable(host=None):
        reachable_calls["count"] += 1
        return reachable_calls["count"] >= 3  # not reachable for the first two checks

    monkeypatch.setattr(oc, "is_ollama_reachable", fake_reachable)

    fake_process = _FakeProcess()
    monkeypatch.setattr(oc.subprocess, "Popen", lambda *a, **kw: fake_process)

    result = oc.start_ollama(startup_timeout=5.0)

    assert result is fake_process
    assert reachable_calls["count"] >= 3


def test_start_ollama_raises_if_process_exits_immediately(monkeypatch):
    monkeypatch.setattr(oc.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(oc.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(oc, "is_ollama_reachable", lambda host=None: False)

    fake_process = _FakeProcess(poll_results=[1])
    monkeypatch.setattr(oc.subprocess, "Popen", lambda *a, **kw: fake_process)

    with pytest.raises(RuntimeError, match="exit code"):
        oc.start_ollama(startup_timeout=5.0)


def test_start_ollama_raises_and_terminates_if_never_reachable_within_timeout(monkeypatch):
    monkeypatch.setattr(oc.shutil, "which", lambda name: "/usr/bin/ollama")
    monkeypatch.setattr(oc.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(oc, "is_ollama_reachable", lambda host=None: False)

    fake_process = _FakeProcess()
    monkeypatch.setattr(oc.subprocess, "Popen", lambda *a, **kw: fake_process)

    with pytest.raises(RuntimeError, match="did not become reachable"):
        oc.start_ollama(startup_timeout=0.05)

    assert fake_process.terminated is True


# --- stop_ollama -----------------------------------------------------------


def test_stop_ollama_none_is_noop():
    oc.stop_ollama(None)  # must not raise


def test_stop_ollama_terminates_and_waits():
    fake_process = _FakeProcess()
    oc.stop_ollama(fake_process)
    assert fake_process.terminated is True
    assert fake_process.wait_calls == 1
    assert fake_process.killed is False


def test_stop_ollama_kills_if_terminate_times_out():
    fake_process = _FakeProcess()

    calls = {"count": 0}

    def wait(timeout=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise subprocess.TimeoutExpired(cmd="ollama serve", timeout=timeout)

    fake_process.wait = wait

    oc.stop_ollama(fake_process)

    assert fake_process.terminated is True
    assert fake_process.killed is True
    assert calls["count"] == 2


# --- ensure_ollama_running ---------------------------------------------------


def test_ensure_ollama_running_starts_and_stops_when_it_started(monkeypatch):
    fake_process = object()
    monkeypatch.setattr(oc, "start_ollama", lambda host=None, startup_timeout=30.0: fake_process)
    stop_calls = []
    monkeypatch.setattr(oc, "stop_ollama", lambda process: stop_calls.append(process))

    ran = False
    with oc.ensure_ollama_running():
        ran = True

    assert ran is True
    assert stop_calls == [fake_process]


def test_ensure_ollama_running_does_not_stop_when_already_running(monkeypatch):
    monkeypatch.setattr(oc, "start_ollama", lambda host=None, startup_timeout=30.0: None)
    stop_calls = []
    monkeypatch.setattr(oc, "stop_ollama", lambda process: stop_calls.append(process))

    with oc.ensure_ollama_running():
        pass

    assert stop_calls == [None]  # stop_ollama itself treats None as "not ours, no-op"


def test_ensure_ollama_running_swallows_start_failure_and_still_yields(monkeypatch, capsys):
    def failing_start(host=None, startup_timeout=30.0):
        raise RuntimeError("ollama CLI not found")

    monkeypatch.setattr(oc, "start_ollama", failing_start)
    stop_calls = []
    monkeypatch.setattr(oc, "stop_ollama", lambda process: stop_calls.append(process))

    ran = False
    with oc.ensure_ollama_running():
        ran = True

    assert ran is True
    assert stop_calls == [None]
    assert "ollama CLI not found" in capsys.readouterr().out
