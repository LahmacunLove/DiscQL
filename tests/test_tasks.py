from __future__ import annotations

import threading
import time

import pytest

from discql.web import tasks


def _wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("condition not met in time")


def test_start_task_runs_in_background_and_reports_progress():
    def run(on_progress, on_error, should_stop):
        on_progress(1, 2, "first")
        on_progress(2, 2, "second")

    tasks.start_task("test-progress", run)

    _wait_until(lambda: tasks.get_task("test-progress").status == tasks.TaskStatus.DONE)

    state = tasks.get_task("test-progress")
    assert state.current == 2
    assert state.total == 2
    assert state.message == "second"
    assert state.started_at is not None
    assert state.finished_at is not None


def test_start_task_is_noop_while_already_running():
    started = threading.Event()
    release = threading.Event()
    calls = []

    def run(on_progress, on_error, should_stop):
        calls.append(1)
        started.set()
        release.wait(timeout=2)

    tasks.start_task("test-dedupe", run)
    started.wait(timeout=2)
    tasks.start_task("test-dedupe", run)  # should not start a second run
    release.set()

    _wait_until(lambda: tasks.get_task("test-dedupe").status == tasks.TaskStatus.DONE)
    assert calls == [1]


def test_start_task_records_error_status_and_message():
    def run(on_progress, on_error, should_stop):
        raise RuntimeError("boom")

    tasks.start_task("test-error", run)

    _wait_until(lambda: tasks.get_task("test-error").status == tasks.TaskStatus.ERROR)

    state = tasks.get_task("test-error")
    assert "boom" in state.message


def test_on_error_callback_appends_to_errors_list():
    def run(on_progress, on_error, should_stop):
        on_error("release 1 failed")
        on_error("release 2 failed")

    tasks.start_task("test-partial-errors", run)

    _wait_until(lambda: tasks.get_task("test-partial-errors").status == tasks.TaskStatus.DONE)

    state = tasks.get_task("test-partial-errors")
    assert state.errors == ["release 1 failed", "release 2 failed"]


def test_get_task_returns_idle_state_for_unknown_kind():
    state = tasks.get_task("never-started")
    assert state.status == tasks.TaskStatus.IDLE


def test_request_stop_sets_cancelled_status_once_runner_returns():
    started = threading.Event()
    seen_stop = threading.Event()

    def run(on_progress, on_error, should_stop):
        started.set()
        while not should_stop():
            time.sleep(0.01)
        seen_stop.set()

    tasks.start_task("test-stop", run)
    started.wait(timeout=2)
    tasks.request_stop("test-stop")

    _wait_until(lambda: tasks.get_task("test-stop").status == tasks.TaskStatus.CANCELLED)
    assert seen_stop.is_set()


def test_request_stop_is_noop_when_not_running():
    state = tasks.request_stop("test-stop-idle")
    assert state.status == tasks.TaskStatus.IDLE
