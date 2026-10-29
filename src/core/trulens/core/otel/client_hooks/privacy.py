"""Privacy and size controls for coding-agent hook events."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from typing import Any, Dict

from trulens.core.otel import privacy as otel_privacy
from trulens.core.otel.client_hooks import models

_SAFE_METADATA_KEYS = {
    "is_interrupt",
    "loop_count",
    "permission_mode",
    "sandbox",
    "source",
    "cursor_version",
    "client_version",
}


def _bounded(value: Any, max_bytes: int) -> Any:
    if value is None:
        return None
    redacted = otel_privacy.redact_sensitive_data(value)
    encoded = json.dumps(
        redacted, default=str, sort_keys=True, separators=(",", ":")
    )
    if len(encoded.encode()) <= max_bytes:
        return redacted
    return (
        encoded.encode()[:max_bytes].decode(errors="ignore") + "...[TRUNCATED]"
    )


class CapturePolicy:
    """Apply metadata-first capture rules before durable storage."""

    def __init__(
        self,
        *,
        capture_content: bool = False,
        capture_tool_payloads: bool = False,
        capture_diffs: bool = False,
        capture_paths: bool = False,
        max_field_bytes: int = 16_384,
    ) -> None:
        self.capture_content = capture_content
        self.capture_tool_payloads = capture_tool_payloads
        self.capture_diffs = capture_diffs
        self.capture_paths = capture_paths
        self.max_field_bytes = max_field_bytes

    @classmethod
    def from_environment(cls) -> "CapturePolicy":
        """Build a capture policy from TruLens hook environment variables."""

        max_bytes = os.environ.get("TRULENS_MAX_FIELD_BYTES", "16384")
        try:
            parsed_max_bytes = max(256, int(max_bytes))
        except ValueError:
            parsed_max_bytes = 16_384
        return cls(
            capture_content=otel_privacy.is_truthy_env_var(
                "TRULENS_CAPTURE_CONTENT"
            ),
            capture_tool_payloads=otel_privacy.is_truthy_env_var(
                "TRULENS_CAPTURE_TOOL_PAYLOADS"
            ),
            capture_diffs=otel_privacy.is_truthy_env_var(
                "TRULENS_CAPTURE_DIFFS"
            ),
            capture_paths=otel_privacy.is_truthy_env_var(
                "TRULENS_CAPTURE_PATHS"
            ),
            max_field_bytes=parsed_max_bytes,
        )

    def apply(self, event: models.HookEvent) -> models.HookEvent:
        """Return a sanitized copy of a normalized hook event."""

        safe_metadata: Dict[str, Any] = {
            key: _bounded(value, self.max_field_bytes)
            for key, value in event.metadata.items()
            if key in _SAFE_METADATA_KEYS
        }
        return replace(
            event,
            prompt=_bounded(event.prompt, self.max_field_bytes)
            if self.capture_content
            else None,
            response=_bounded(event.response, self.max_field_bytes)
            if self.capture_content
            else None,
            tool_input=_bounded(event.tool_input, self.max_field_bytes)
            if self.capture_tool_payloads
            else None,
            tool_output=_bounded(event.tool_output, self.max_field_bytes)
            if self.capture_tool_payloads
            else None,
            diff=_bounded(event.diff, self.max_field_bytes)
            if self.capture_diffs
            else None,
            paths=_bounded(event.paths, self.max_field_bytes)
            if self.capture_paths
            else None,
            error=_bounded(event.error, self.max_field_bytes)
            if self.capture_tool_payloads
            else "[error content not captured]"
            if event.error
            else None,
            metadata=safe_metadata,
        )
