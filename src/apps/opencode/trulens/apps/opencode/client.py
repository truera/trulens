"""Minimal OpenCode-native plugin specification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from trulens.core.otel.client_hooks.clients import ClientSpec
from trulens.core.otel.client_hooks.clients import FieldAliases

_HOOK_EVENTS = (
    "chat.message",
    "tool.execute.before",
    "tool.execute.after",
    "experimental.text.complete",
    "session.idle",
    "session.error",
    "file.edited",
)


def _first(payload: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return value
    return None


def _nested(payload: Mapping[str, Any], *path: str) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _prompt_from_parts(parts: Any) -> Any:
    if not isinstance(parts, list):
        return None
    texts = [
        part.get("text")
        for part in parts
        if isinstance(part, Mapping) and part.get("text")
    ]
    return "\n".join(texts) or None


def extract_overrides(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Flatten OpenCode camelCase and nested event payloads."""

    event = payload.get("event")
    properties = event.get("properties") if isinstance(event, Mapping) else {}
    if not isinstance(properties, Mapping):
        properties = {}
    event_name = payload.get("hook_event_name")
    if event_name is None and isinstance(event, Mapping):
        event_name = event.get("type")
    model = _first(payload, "model")
    if isinstance(model, Mapping):
        model = model.get("modelID") or model.get("model_id")
    prompt = _first(payload, "prompt") or _prompt_from_parts(
        payload.get("parts")
    )
    if prompt is None:
        message = payload.get("message")
        if isinstance(message, Mapping):
            prompt = message.get("content") or _prompt_from_parts(
                message.get("parts")
            )
    overrides = {
        "conversation_id": _first(
            payload,
            "session_id",
            "sessionID",
        )
        or properties.get("sessionID")
        or _nested(payload, "event", "properties", "sessionID"),
        "turn_id": _first(payload, "message_id", "messageID"),
        "operation_id": _first(payload, "call_id", "callID", "tool_call_id"),
        "event_name": event_name,
        "tool_name": _first(payload, "tool_name", "tool"),
        "prompt": prompt,
        "response": _first(payload, "response", "text")
        or _nested(payload, "output", "text"),
        "tool_input": _first(payload, "tool_input", "args")
        or _nested(payload, "output", "args"),
        "tool_output": _first(payload, "tool_output")
        or _nested(payload, "output", "output"),
    }
    if model:
        overrides["model"] = model
    return {key: value for key, value in overrides.items() if value is not None}


def _plugin(command: str) -> str:
    encoded = json.dumps(command)
    return f"""// managed_by: trulens-client-hooks
const COMMAND = {encoded}

async function ingest(payload) {{
  try {{
    const proc = Bun.spawn(["sh", "-c", COMMAND], {{
      stdin: "pipe",
      stdout: "ignore",
      stderr: "pipe",
    }})
    proc.stdin.write(JSON.stringify(payload))
    proc.stdin.end()
    await proc.exited
  }} catch (_error) {{
    // Fail open: telemetry must never block OpenCode.
  }}
}}

function textFromParts(parts) {{
  if (!Array.isArray(parts)) {{
    return undefined
  }}
  return parts
    .map((part) => (part && part.text) || "")
    .filter(Boolean)
    .join("\\n") || undefined
}}

export const TruLensClientHooks = async ({{ directory }}) => {{
  let lastSessionId
  const send = async (payload) => {{
    if (payload.session_id) {{
      lastSessionId = payload.session_id
    }}
    await ingest({{ cwd: directory, ...payload }})
  }}
  return {{
    "chat.message": async (input, output) => {{
      await send({{
        session_id: input.sessionID,
        message_id: input.messageID,
        hook_event_name: "chat.message",
        model: input.model && input.model.modelID,
        prompt: textFromParts(output && output.parts),
      }})
    }},
    "tool.execute.before": async (input, output) => {{
      await send({{
        session_id: input.sessionID,
        hook_event_name: "tool.execute.before",
        tool_name: input.tool,
        tool_call_id: input.callID,
        tool_input: output && output.args,
      }})
    }},
    "tool.execute.after": async (input, output) => {{
      await send({{
        session_id: input.sessionID,
        hook_event_name: "tool.execute.after",
        tool_name: input.tool,
        tool_call_id: input.callID,
        tool_input: input.args,
        tool_output: output && output.output,
      }})
    }},
    "experimental.text.complete": async (input, output) => {{
      await send({{
        session_id: input.sessionID,
        message_id: input.messageID,
        hook_event_name: "experimental.text.complete",
        text: output && output.text,
      }})
    }},
    event: async ({{ event }}) => {{
      const type = event && event.type
      const properties = (event && event.properties) || {{}}
      const sessionId =
        properties.sessionID || event.sessionID || lastSessionId
      if (type === "session.idle" || type === "session.error") {{
        await send({{
          session_id: sessionId,
          hook_event_name: type,
          status: type === "session.error" ? "error" : "completed",
          error:
            type === "session.error"
              ? properties.error || event.error
              : undefined,
        }})
      }}
      if (type === "file.edited") {{
        await send({{
          session_id: sessionId,
          hook_event_name: "file.edited",
          file_path: properties.file || event.path,
        }})
      }}
    }},
  }}
}}
"""


client_spec = ClientSpec(
    name="opencode",
    aliases=("open-code",),
    user_config_path=Path("~/.config/opencode/plugins/trulens-client-hooks.js"),
    project_config_path=Path(".opencode/plugins/trulens-client-hooks.js"),
    hook_events=_HOOK_EVENTS,
    field_aliases=FieldAliases(
        conversation=("session_id", "sessionID"),
        turn=("message_id", "messageID"),
        operation=("call_id", "callID", "tool_call_id"),
        response=("text", "response", "output"),
    ),
    plugin_builder=_plugin,
    extract_overrides=extract_overrides,
)
