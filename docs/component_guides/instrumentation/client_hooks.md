# Instrument Cursor and Claude Code

TruLens can capture coding-agent lifecycle hooks without wrapping application
code. The shared runtime ships in `trulens-core`; each coding client is a thin
plugin containing only its native hook and configuration contract.

## Install a client

For Cursor:

```bash
pip install trulens-core trulens-apps-cursor
trulens-client-hooks install cursor --dry-run
trulens-client-hooks install cursor
```

For Claude Code:

```bash
pip install trulens-core trulens-apps-claude
trulens-client-hooks install claude-code --dry-run
trulens-client-hooks install claude-code
```

Use `--project` to install into the current repository instead of the user's
global client configuration. The installer preserves unrelated hooks, creates a
backup, and is idempotent.

## Choose a destination

The default is a local SQLite database. Set a database URL for a custom SQLite
file or PostgreSQL:

```bash
export TRULENS_HOOKS_DESTINATION=database
export TRULENS_HOOKS_DATABASE_URL="postgresql+psycopg://trulens@localhost/traces"
```

For Snowflake, use a named `connections.toml` profile:

```bash
export TRULENS_HOOKS_DESTINATION=snowflake
export TRULENS_HOOKS_SNOWFLAKE_CONNECTION=my_connection
```

For OTLP gRPC:

```bash
export TRULENS_HOOKS_DESTINATION=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
```

## Configure privacy

Lifecycle metadata is captured by default. Source-bearing content is opt-in:

```bash
export TRULENS_HOOKS_CAPTURE_CONTENT=true
export TRULENS_HOOKS_CAPTURE_TOOL_PAYLOADS=true
export TRULENS_HOOKS_CAPTURE_DIFFS=true
export TRULENS_HOOKS_CAPTURE_PATHS=true
```

Diffs include Cursor `afterFileEdit` old/new pairs and explicit patches. They
can contain source code or credentials, so they remain independently opt-in.
Values are redacted and size-bounded before durable journaling.

## Trace semantics

Native Cursor `conversation_id` and Claude Code `session_id` map to the existing
TruLens `SpanAttributes.CONVERSATION_ID`. Each prompt gets distinct `RECORD_ID`
and `INPUT_ID` values. There is no separate thread identity contract.

Each turn emits the same semantic conventions as other TruLens apps:

```text
Agent span
├── Request/response RECORD_ROOT span
├── Tool, edit, shell, MCP, and subagent spans
└── Response-generation span
```

Existing input/output and trace-level selectors work without client-specific
logic. Coding-agent-only metadata such as client name, native hook event, editor
version, workspace, and diff is defined centrally in `trulens-otel-semconv`.

## Validate and inspect

```bash
trulens-client-hooks clients
trulens-client-hooks validate
trulens-client-hooks status cursor
trulens-client-hooks flush
```

Every hook invocation retries eligible exports across the journal. `flush`
also retries completed turns and exports stale turns without waiting for another
client event, which is useful after reconnecting an unavailable destination.

Remove only the TruLens-managed hook entries while preserving other hooks:

```bash
trulens-client-hooks uninstall cursor
```

Runtime hooks always fail open: telemetry errors are written to stderr and never
block the coding client.
