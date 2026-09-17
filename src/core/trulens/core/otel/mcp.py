"""Framework-independent instrumentation for the official MCP Python SDK.

The SDK is optional. Importing this module has no side effects; callers opt in
by invoking :func:`instrument_mcp` once during application setup.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any

from trulens.core.otel.function_call_context_manager import (
    create_function_call_context_manager,
)
from trulens.core.otel.instrument import _finalize_span
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
import wrapt

_PATCH_FLAG = "__trulens_mcp_call_tool_instrumented__"
_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|authorization|cookie|credential|password|private[_-]?key|secret|token)",
    re.IGNORECASE,
)
_DEFAULT_MAX_BYTES = 16_384


def _capture_enabled() -> bool:
    return os.environ.get("TRULENS_CAPTURE_TOOL_PAYLOADS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _max_bytes() -> int:
    try:
        return max(256, int(os.environ.get("TRULENS_MAX_FIELD_BYTES", "16384")))
    except ValueError:
        return _DEFAULT_MAX_BYTES


def _redact(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    elif hasattr(value, "dict") and callable(value.dict):
        value = value.dict()
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if _SECRET_KEY.search(str(key))
            else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


def _serialize(value: Any) -> str:
    if not _capture_enabled():
        return "[content not captured]"
    try:
        encoded = json.dumps(
            _redact(value), default=str, sort_keys=True, separators=(",", ":")
        )
    except Exception:
        return "[content unavailable]"
    limit = _max_bytes()
    if len(encoded.encode()) <= limit:
        return encoded
    return encoded.encode()[:limit].decode(errors="ignore") + "...[TRUNCATED]"


def _server_name(session: Any) -> str:
    """Read a server identity only from metadata owned by the SDK session."""

    for attr in ("server_name", "_server_name"):
        value = getattr(session, attr, None)
        if isinstance(value, str) and value:
            return value
    for attr in ("server_info", "_server_info"):
        value = getattr(session, attr, None)
        if isinstance(value, dict):
            name = value.get("name")
        else:
            name = getattr(value, "name", None)
        if isinstance(name, str) and name:
            return name
    return "unknown"


def _mcp_attributes(
    result: Any,
    exception: Exception | None,
    *args: Any,
    **kwargs: Any,
) -> dict[str, Any]:
    session = args[0] if args else None
    tool_name = kwargs.get("name")
    arguments = kwargs.get("arguments")
    if tool_name is None and len(args) > 1:
        tool_name = args[1]
    if arguments is None and len(args) > 2:
        arguments = args[2]
    tool_name = tool_name or "unknown"
    return {
        SpanAttributes.MCP.TOOL_NAME: str(tool_name),
        SpanAttributes.MCP.SERVER_NAME: _server_name(session),
        SpanAttributes.MCP.INPUT_ARGUMENTS: _serialize(arguments),
        SpanAttributes.MCP.OUTPUT_CONTENT: _serialize(result),
        SpanAttributes.MCP.OUTPUT_IS_ERROR: bool(
            exception is not None or getattr(result, "isError", False)
        ),
    }


def _wrap_call_tool(client_session: type) -> None:
    if getattr(client_session, _PATCH_FLAG, False):
        return
    original = getattr(client_session, "call_tool")

    @wrapt.decorator
    async def call_tool_wrapper(wrapped, instance, args, kwargs):
        if not instrument.enabled:
            return await wrapped(*args, **kwargs)
        from opentelemetry.baggage import get_baggage

        if get_baggage("__trulens_recording__") is None:
            return await wrapped(*args, **kwargs)

        span_name = f"{client_session.__module__}.{client_session.__qualname__}.call_tool"
        started = time.perf_counter()
        result = None
        exception: Exception | None = None
        cancelled_error: asyncio.CancelledError | None = None
        with create_function_call_context_manager(
            True, span_name, span_type=SpanAttributes.SpanType.MCP
        ) as span:
            try:
                result = await wrapped(*args, **kwargs)
            except asyncio.CancelledError as error:
                cancelled_error = error
            except Exception as error:
                exception = error
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0

                def attributes(ret, error, *attribute_args, **attribute_kwargs):
                    resolved = _mcp_attributes(
                        ret, error, *attribute_args, **attribute_kwargs
                    )
                    resolved[SpanAttributes.MCP.EXECUTION_TIME_MS] = elapsed_ms
                    return resolved

                _finalize_span(
                    span,
                    SpanAttributes.SpanType.MCP,
                    span_name,
                    wrapped,
                    exception,
                    attributes,
                    instance,
                    args,
                    kwargs,
                    result,
                )
        if cancelled_error is not None:
            raise cancelled_error
        if exception is not None:
            raise exception
        return result

    wrapped_call_tool = call_tool_wrapper(original)
    setattr(wrapped_call_tool, _PATCH_FLAG, True)
    setattr(original, _PATCH_FLAG, True)
    setattr(client_session, _PATCH_FLAG, True)
    setattr(client_session, "call_tool", wrapped_call_tool)


def instrument_mcp() -> None:
    """Instrument the official MCP SDK's async ``ClientSession.call_tool``.

    The optional SDK is imported only when this function is called. Repeated
    calls are safe, and importing this module never patches the SDK.

    Raises:
        ImportError: If the optional ``mcp`` package is not installed.
    """

    try:
        from mcp import ClientSession
    except ImportError as exc:
        raise ImportError(
            "MCP instrumentation requires the optional 'mcp' package. "
            "Install it with 'pip install mcp'."
        ) from exc
    _wrap_call_tool(ClientSession)
