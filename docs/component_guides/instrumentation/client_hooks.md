# Instrument Claude Code and Cursor

`trulens-apps-client-hooks` turns coding-agent lifecycle hooks into TruLens
OpenTelemetry traces. It supports Claude Code and Cursor without wrapping or
modifying either client.

## Install and configure

```bash
pip install trulens-apps-client-hooks
python -m trulens.apps.client_hooks config --client claude
python -m trulens.apps.client_hooks config --client cursor
```

Generated snippets use the installed `trulens-client-hooks` executable when it
is available, so the editor does not need to resolve the correct Python
environment itself.

Merge the generated JSON into `~/.claude/settings.json` or
`~/.cursor/hooks.json`. Validate environment configuration with:

```bash
python -m trulens.apps.client_hooks validate
```

Local persistence is enabled by default. To use Snowflake AI Observability:

```bash
pip install trulens-connectors-snowflake
export TRULENS_HOOKS_DESTINATION=snowflake
export TRULENS_HOOKS_SNOWFLAKE_CONNECTION=my_connection
```

The connection name refers to a normal Snowflake `connections.toml` entry, so
credentials do not need to be included in hook configuration.

## Captured spans

| Client activity | TruLens span type |
| --- | --- |
| User prompt or generation | `RECORD_ROOT` |
| Coding-agent turn | `AGENT` |
| Shell, read, edit, search, or generic tool | `TOOL` |
| MCP invocation | `MCP` |
| Subagent | `AGENT` |
| Other lifecycle event | `WORKFLOW_STEP` |

The client session or conversation ID is stored as `CONVERSATION_ID`, linking
multiple prompt records into one interaction.

## Privacy defaults

Only metadata is captured by default. Prompts, responses, commands, tool
payloads, file paths, and transcript paths are omitted. Enable specific content
classes explicitly:

```bash
export TRULENS_HOOKS_CAPTURE_CONTENT=true
export TRULENS_HOOKS_CAPTURE_TOOL_PAYLOADS=true
export TRULENS_HOOKS_CAPTURE_PATHS=true
```

Secret-like mapping keys are redacted and captured values are size-limited.
Review your organization's data handling requirements before enabling content.

## Lifecycle behavior

Each hook is a separate process, so the adapter journals sanitized events and
creates completed spans at turn boundaries. The journal is locked for concurrent
hooks, deduplicates repeated events, and retains failed exports for retry.
Instrumentation failures are written to stderr and never block the client.
