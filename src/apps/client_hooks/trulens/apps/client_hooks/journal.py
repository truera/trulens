"""Crash-tolerant local event journal for short-lived hook processes."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import hashlib
import json
import os
from pathlib import Path
import secrets
import tempfile
from typing import Any, Dict, Iterator, List, Optional, Tuple

from trulens.apps.client_hooks import models

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def _lock(file_descriptor: int) -> None:
    if os.name == "nt":
        msvcrt.locking(file_descriptor, msvcrt.LK_LOCK, 1)
    else:
        fcntl.flock(file_descriptor, fcntl.LOCK_EX)


def _unlock(file_descriptor: int) -> None:
    if os.name == "nt":
        msvcrt.locking(file_descriptor, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(file_descriptor, fcntl.LOCK_UN)


def default_journal_dir() -> Path:
    """Return the configured or platform-neutral journal directory."""

    configured = os.environ.get("TRULENS_HOOKS_JOURNAL_DIR")
    if configured:
        return Path(configured).expanduser()
    state_home = os.environ.get("XDG_STATE_HOME")
    if state_home:
        return Path(state_home).expanduser() / "trulens" / "client-hooks"
    return Path.home() / ".trulens" / "client-hooks"


class EventJournal:
    """Persist and correlate hook events across independent processes."""

    def __init__(self, directory: Optional[Path] = None) -> None:
        self.directory = directory or default_journal_dir()

    def _key(self, client: str, conversation_id: str) -> str:
        value = f"{client}:{conversation_id}".encode()
        return hashlib.sha256(value).hexdigest()

    @contextmanager
    def _locked_state(
        self, client: str, conversation_id: str
    ) -> Iterator[Tuple[Path, Dict[str, Any]]]:
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        key = self._key(client, conversation_id)
        state_path = self.directory / f"{key}.json"
        lock_path = self.directory / f"{key}.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            os.chmod(lock_path, 0o600)
            _lock(lock_file.fileno())
            state: Dict[str, Any] = {
                "active_turn": None,
                "turns": {},
            }
            if state_path.exists():
                try:
                    state = json.loads(state_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    state = {"active_turn": None, "turns": {}}
            yield state_path, state
            self._write_atomic(state_path, state)
            _unlock(lock_file.fileno())

    @staticmethod
    def _write_atomic(path: Path, state: Dict[str, Any]) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(state, output, separators=(",", ":"), sort_keys=True)
                output.flush()
                os.fsync(output.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @staticmethod
    def _starts_turn(event: models.HookEvent) -> bool:
        return event.event_name.lower() in {
            "userpromptsubmit",
            "beforesubmitprompt",
        }

    def append(self, event: models.HookEvent) -> Tuple[str, bool]:
        """Append an event and return its correlated turn and terminal state."""

        with self._locked_state(event.client, event.conversation_id) as (
            _,
            state,
        ):
            turn_id = event.turn_id
            if turn_id is None and self._starts_turn(event):
                active_turn_id = state.get("active_turn")
                active_turn = state["turns"].get(active_turn_id, {})
                if event.event_id in active_turn.get("event_ids", []):
                    turn_id = active_turn_id
                else:
                    turn_id = f"turn:{secrets.token_hex(16)}"
            if turn_id is None:
                turn_id = state.get("active_turn")
            if turn_id is None:
                turn_id = f"session:{event.conversation_id}"
            turn_id = str(turn_id)
            if self._starts_turn(event) or state.get("active_turn") is None:
                state["active_turn"] = turn_id
            turn = state["turns"].setdefault(
                turn_id,
                {
                    "events": [],
                    "event_ids": [],
                    "exported": False,
                    "export_attempts": 0,
                    "claimed_until": None,
                    "next_retry_at": None,
                    "updated_at": event.observed_at.isoformat(),
                },
            )
            if event.event_id not in turn["event_ids"]:
                serialized = event.to_dict()
                serialized["turn_id"] = turn_id
                turn["events"].append(serialized)
                turn["event_ids"].append(event.event_id)
            turn["updated_at"] = event.observed_at.isoformat()
            if event.terminal and state.get("active_turn") == turn_id:
                state["active_turn"] = None
            return turn_id, event.terminal

    def get_turn(
        self, client: str, conversation_id: str, turn_id: str
    ) -> List[models.HookEvent]:
        """Read all events for one turn in observation order."""

        with self._locked_state(client, conversation_id) as (_, state):
            turn = state["turns"].get(turn_id, {})
            return sorted(
                [
                    models.HookEvent.from_dict(value)
                    for value in turn.get("events", [])
                ],
                key=lambda event: event.observed_at,
            )

    def mark_exported(
        self, client: str, conversation_id: str, turn_id: str
    ) -> None:
        """Mark a turn exported after its complete span batch succeeds."""

        with self._locked_state(client, conversation_id) as (_, state):
            turn = state["turns"].get(turn_id)
            if turn is not None:
                turn["exported"] = True
                turn["claimed_until"] = None
                turn["next_retry_at"] = None

    def claim_pending_turns(
        self,
        client: str,
        conversation_id: str,
        *,
        stale_after: timedelta = timedelta(hours=24),
        lease_for: timedelta = timedelta(minutes=5),
    ) -> List[str]:
        """Atomically claim complete or stale turns for one exporter."""

        now = datetime.now(timezone.utc)
        with self._locked_state(client, conversation_id) as (_, state):
            result = []
            for turn_id, turn in state["turns"].items():
                if turn.get("exported"):
                    continue
                claimed_until = turn.get("claimed_until")
                if (
                    claimed_until
                    and datetime.fromisoformat(claimed_until) > now
                ):
                    continue
                next_retry_at = turn.get("next_retry_at")
                if (
                    next_retry_at
                    and datetime.fromisoformat(next_retry_at) > now
                ):
                    continue
                events = [
                    models.HookEvent.from_dict(value)
                    for value in turn.get("events", [])
                ]
                updated_at = datetime.fromisoformat(turn["updated_at"])
                if any(event.terminal for event in events) or (
                    now - updated_at >= stale_after
                ):
                    turn["claimed_until"] = (now + lease_for).isoformat()
                    result.append(turn_id)
            return result

    def pending_turns(
        self,
        client: str,
        conversation_id: str,
        *,
        stale_after: timedelta = timedelta(hours=24),
    ) -> List[str]:
        """Return complete or stale unexported turns without claiming them."""

        now = datetime.now(timezone.utc)
        with self._locked_state(client, conversation_id) as (_, state):
            result = []
            for turn_id, turn in state["turns"].items():
                if turn.get("exported"):
                    continue
                events = [
                    models.HookEvent.from_dict(value)
                    for value in turn.get("events", [])
                ]
                updated_at = datetime.fromisoformat(turn["updated_at"])
                if any(event.terminal for event in events) or (
                    now - updated_at >= stale_after
                ):
                    result.append(turn_id)
            return result

    def release_claim(
        self,
        client: str,
        conversation_id: str,
        turn_id: str,
        *,
        failed: bool,
    ) -> None:
        """Release an export claim, applying bounded backoff after failures."""

        with self._locked_state(client, conversation_id) as (_, state):
            turn = state["turns"].get(turn_id)
            if turn is None:
                return
            turn["claimed_until"] = None
            if not failed:
                turn["next_retry_at"] = None
                return
            attempts = int(turn.get("export_attempts", 0)) + 1
            turn["export_attempts"] = attempts
            delay = timedelta(seconds=min(300, 2 ** min(attempts, 8)))
            turn["next_retry_at"] = (
                datetime.now(timezone.utc) + delay
            ).isoformat()
