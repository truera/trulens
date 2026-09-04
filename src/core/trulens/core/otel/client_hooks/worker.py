"""Detached exporter worker for the coding-agent hook journal."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import BinaryIO, Iterator, Optional, Sequence

from trulens.core.otel.client_hooks import journal as journal_module

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def worker_log_path(directory: Optional[Path] = None) -> Path:
    root = directory or journal_module.default_journal_dir()
    return root / "worker.log"


def worker_lock_path(directory: Optional[Path] = None) -> Path:
    root = directory or journal_module.default_journal_dir()
    return root / "worker.lock"


def worker_command() -> Sequence[str]:
    return [
        sys.executable,
        "-m",
        "trulens.core.otel.client_hooks",
        "worker",
    ]


def ensure_worker() -> bool:
    """Start a detached exporter worker and return whether launch succeeded."""

    directory = journal_module.default_journal_dir()
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_path = worker_log_path(directory)
    try:
        log_file = log_path.open("ab")
        os.chmod(log_path, 0o600)
        kwargs = {
            "stdin": subprocess.DEVNULL,
            "stdout": log_file,
            "stderr": subprocess.STDOUT,
            "close_fds": True,
        }
        if os.name == "nt":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | subprocess.DETACHED_PROCESS
            )
        else:
            kwargs["start_new_session"] = True
        subprocess.Popen(worker_command(), **kwargs)
        return True
    except OSError:
        return False
    finally:
        if "log_file" in locals():
            log_file.close()


def _try_lock(file: BinaryIO) -> bool:
    try:
        if os.name == "nt":
            msvcrt.locking(file.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(file: BinaryIO) -> None:
    if os.name == "nt":
        file.seek(0)
        msvcrt.locking(file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file.fileno(), fcntl.LOCK_UN)


@contextmanager
def singleton_worker(
    directory: Optional[Path] = None,
) -> Iterator[bool]:
    """Yield whether this process acquired the journal worker lock."""

    path = worker_lock_path(directory)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with path.open("a+b") as lock_file:
        os.chmod(path, 0o600)
        acquired = _try_lock(lock_file)
        try:
            yield acquired
        finally:
            if acquired:
                _unlock(lock_file)


def is_worker_running(directory: Optional[Path] = None) -> bool:
    """Return whether another process currently owns the worker lock."""

    with singleton_worker(directory) as acquired:
        return not acquired


def run_worker() -> int:
    """Drain the journal until it remains idle for a bounded grace period."""

    from trulens.core.otel.client_hooks import service

    idle_seconds = float(os.environ.get("TRULENS_WORKER_IDLE_SECONDS", "2"))
    poll_seconds = float(os.environ.get("TRULENS_WORKER_POLL_SECONDS", "0.25"))
    hook_service = service.HookService()
    with singleton_worker(hook_service.journal.directory) as acquired:
        if not acquired:
            return 0
        idle_since: Optional[float] = None
        while True:
            hook_service.flush()
            if hook_service.journal.has_exportable_turns(
                stale_after=hook_service.stale_after
            ):
                idle_since = None
                delay = hook_service.journal.next_retry_delay()
                time.sleep(max(poll_seconds, min(delay or poll_seconds, 1.0)))
                continue
            if idle_since is None:
                idle_since = time.monotonic()
            elif time.monotonic() - idle_since >= idle_seconds:
                return 0
            time.sleep(poll_seconds)
