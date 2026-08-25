# trulens-apps-client-hooks

TruLens instrumentation for coding agents that expose command hooks. The
adapter currently supports Claude Code and Cursor.

## Install

```bash
pip install trulens-apps-client-hooks
```

Generate the configuration for your client:

```bash
python -m trulens.apps.client_hooks config --client claude
python -m trulens.apps.client_hooks config --client cursor
```

The generated hook command uses the installed `trulens-client-hooks` executable
when available, avoiding dependence on the editor's default Python interpreter.

Add the Claude output to `~/.claude/settings.json` or the Cursor output to
`~/.cursor/hooks.json`. Existing hooks must be merged rather than replaced.

## Destination

Local SQLite storage is the default:

```bash
export TRULENS_HOOKS_DESTINATION=local
export TRULENS_HOOKS_DATABASE_PATH="$HOME/.trulens/client-hooks.sqlite"
```

To send completed traces to Snowflake AI Observability, install the Snowflake
connector and name a connection from `connections.toml`:

```bash
pip install trulens-connectors-snowflake
export TRULENS_HOOKS_DESTINATION=snowflake
export TRULENS_HOOKS_SNOWFLAKE_CONNECTION=my_connection
```

Validate the destination settings before enabling hooks:

```bash
python -m trulens.apps.client_hooks validate
```

## Privacy

The default configuration records lifecycle metadata, timestamps, status,
model, tool names, durations, and opaque correlation IDs. It does not persist
prompts, responses, commands, tool arguments/results, paths, or transcripts.

Content capture is opt-in:

```bash
export TRULENS_HOOKS_CAPTURE_CONTENT=true
export TRULENS_HOOKS_CAPTURE_TOOL_PAYLOADS=true
export TRULENS_HOOKS_CAPTURE_PATHS=true
export TRULENS_HOOKS_MAX_FIELD_BYTES=16384
```

Values under keys resembling tokens, passwords, credentials, cookies, private
keys, or authorization headers are redacted before journaling. Captured fields
are bounded by `TRULENS_HOOKS_MAX_FIELD_BYTES`.

## Trace Model

- Each user prompt or Cursor generation becomes one `RECORD_ROOT` span.
- The coding-agent turn becomes an `AGENT` child span.
- Shell, file, search, and generic tool calls become `TOOL` spans.
- MCP calls become `MCP` spans.
- Subagent lifecycle events become `AGENT` spans.
- Other lifecycle events become `WORKFLOW_STEP` spans.

Hook processes are short-lived. Events are therefore written to a locked local
journal and assembled into completed OpenTelemetry spans when the client emits a
turn completion or failure event. Export failures stay pending for retry. Hooks
always fail open and return valid JSON so observability cannot block the client.

The journal defaults to `~/.trulens/client-hooks`. Override it with
`TRULENS_HOOKS_JOURNAL_DIR`. Incomplete turns become error records after 24
hours; configure that with `TRULENS_HOOKS_STALE_AFTER_HOURS`.
