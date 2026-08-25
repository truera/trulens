"""Client-neutral hook event models."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timezone
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class HookEvent:
    """A normalized event emitted by a coding-agent client."""

    client: str
    event_name: str
    event_id: str
    conversation_id: str
    observed_at: datetime
    turn_id: Optional[str] = None
    operation_id: Optional[str] = None
    phase: str = "instant"
    category: str = "workflow"
    terminal: bool = False
    failed: bool = False
    model: Optional[str] = None
    tool_name: Optional[str] = None
    server_name: Optional[str] = None
    duration_ms: Optional[float] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost: Optional[float] = None
    prompt: Optional[Any] = None
    response: Optional[Any] = None
    tool_input: Optional[Any] = None
    tool_output: Optional[Any] = None
    paths: Optional[Any] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the event for durable journaling."""

        result = asdict(self)
        result["observed_at"] = self.observed_at.isoformat()
        return result

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "HookEvent":
        """Deserialize an event from durable storage."""

        value = dict(value)
        value["observed_at"] = datetime.fromisoformat(value["observed_at"])
        return cls(**value)


def utc_now() -> datetime:
    """Return the current timezone-aware UTC time."""

    return datetime.now(timezone.utc)
