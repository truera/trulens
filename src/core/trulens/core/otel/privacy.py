"""Shared privacy helpers for OpenTelemetry payload capture."""

from __future__ import annotations

import os
import re
from typing import Any, Mapping

_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|credential|password|"
    r"private[_-]?key|secret|token$)",
    re.IGNORECASE,
)


def is_truthy_env_var(name: str) -> bool:
    """Return whether an environment variable contains a truthy value."""

    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def redact_sensitive_data(value: Any) -> Any:
    """Recursively redact values whose keys identify credentials."""

    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif hasattr(value, "dict") and callable(value.dict):
        value = value.dict()
    if isinstance(value, Mapping):
        return {
            str(key): "[REDACTED]"
            if _SECRET_KEY.search(str(key))
            else redact_sensitive_data(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_sensitive_data(item) for item in value]
    return value
