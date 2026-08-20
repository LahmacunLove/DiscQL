from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Callable

ProgressCallback = Callable[[int, int, str], None]
ErrorCallback = Callable[[str], None]
StopCheck = Callable[[], bool]
Runner = Callable[[ProgressCallback, ErrorCallback, StopCheck], None]


class TaskStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class TaskState:
    kind: str
    status: TaskStatus = TaskStatus.IDLE
    current: int = 0
    total: int = 0
    message: str = ""
    errors: list[str] = field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None


_lock = threading.Lock()
_tasks: dict[str, TaskState] = {}
_stop_events: dict[str, threading.Event] = {}


def get_task(kind: str) -> TaskState:
    """Background tasks are single-instance per kind (personal, single-user tool)."""
    with _lock:
        return _tasks.setdefault(kind, TaskState(kind=kind))


def _update(kind: str, **changes: object) -> None:
    with _lock:
        state = _tasks.setdefault(kind, TaskState(kind=kind))
        for key, value in changes.items():
            setattr(state, key, value)


def start_task(kind: str, run: Runner) -> TaskState:
    """Start `run(on_progress, on_error, should_stop)` in a background thread.

    A no-op if a task of this kind is already running, so e.g. clicking
    "Discogs Sync" twice doesn't spawn two overlapping syncs.
    """
    with _lock:
        state = _tasks.setdefault(kind, TaskState(kind=kind))
        if state.status == TaskStatus.RUNNING:
            return state
        state.status = TaskStatus.RUNNING
        state.current = 0
        state.total = 0
        state.message = ""
        state.errors = []
        state.started_at = datetime.now(timezone.utc)
        state.finished_at = None
        stop_event = threading.Event()
        _stop_events[kind] = stop_event

    def on_progress(current: int, total: int, message: str = "") -> None:
        _update(kind, current=current, total=total, message=message)

    def on_error(message: str) -> None:
        with _lock:
            _tasks[kind].errors.append(message)

    def should_stop() -> bool:
        return stop_event.is_set()

    def runner() -> None:
        try:
            run(on_progress, on_error, should_stop)
            final_status = TaskStatus.CANCELLED if stop_event.is_set() else TaskStatus.DONE
            _update(kind, status=final_status, finished_at=datetime.now(timezone.utc))
        except Exception as exc:
            _update(
                kind,
                status=TaskStatus.ERROR,
                message=str(exc),
                finished_at=datetime.now(timezone.utc),
            )

    threading.Thread(target=runner, daemon=True).start()
    return state


def request_stop(kind: str) -> TaskState:
    """Ask a running task to stop cooperatively - it finishes whatever unit
    of work (e.g. a release) is currently in flight, then exits early rather
    than starting the next one. A no-op if the task isn't running."""
    with _lock:
        state = _tasks.setdefault(kind, TaskState(kind=kind))
        if state.status == TaskStatus.RUNNING:
            event = _stop_events.get(kind)
            if event is not None:
                event.set()
            state.message = "Stopping…"
        return state
