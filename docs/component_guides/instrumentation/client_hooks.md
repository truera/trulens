# Instrument Cursor, Claude Code, and OpenCode

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

For OpenCode:

```bash
pip install trulens-core trulens-apps-opencode
trulens-client-hooks install opencode --dry-run
trulens-client-hooks install opencode
```

OpenCode does not use JSON command hooks. Install writes a managed plugin to
`~/.config/opencode/plugins/trulens-client-hooks.js` (or
`.opencode/plugins/trulens-client-hooks.js` with `--project`). The plugin
forwards native OpenCode lifecycle events to `trulens-client-hooks ingest`.

Use `--project` to install into the current repository instead of the user's
global client configuration. The installer preserves unrelated hooks, creates a
backup, and is idempotent.

## Choose a destination

The default is a local SQLite database. Set a database URL for a custom SQLite
file or PostgreSQL:

```bash
export TRULENS_DESTINATION=database
export TRULENS_DATABASE_URL="postgresql+psycopg://trulens@localhost/traces"
```

For Snowflake, use a named `connections.toml` profile:

```bash
export TRULENS_DESTINATION=snowflake
export TRULENS_SNOWFLAKE_CONNECTION=my_connection
export TRULENS_SNOWFLAKE_DATABASE=TRULENS_TRACES
export TRULENS_SNOWFLAKE_SCHEMA=CLIENT_HOOKS
```

By default, the native client determines identity: the app name is `cursor`,
`claude`, or `opencode`, the app version is detected from the native client,
and the run name is the native conversation/session ID. Cursor supplies its
version in hook payloads, Claude Code supplies it in the native transcript, and
the OpenCode plugin detects it from the installed CLI.

`TRULENS_APP_NAME`, `TRULENS_APP_VERSION`, and `TRULENS_RUN_NAME` remain
available as explicit overrides.

For OTLP gRPC:

```bash
export TRULENS_DESTINATION=otlp
export OTEL_EXPORTER_OTLP_ENDPOINT="http://localhost:4317"
```

## Configure privacy

Lifecycle metadata is captured by default. Source-bearing content is opt-in:

```bash
export TRULENS_CAPTURE_CONTENT=true
export TRULENS_CAPTURE_TOOL_PAYLOADS=true
export TRULENS_CAPTURE_DIFFS=true
export TRULENS_CAPTURE_PATHS=true
```

Diffs include Cursor `afterFileEdit` old/new pairs and explicit patches. They
can contain source code or credentials, so they remain independently opt-in.
Values are redacted and size-bounded before durable journaling.

## Run lifecycle

Exporting spans is not enough to make a turn observable. A run's status is
derived from its invocation metadata, not from the presence of spans, so a run
whose ingestion never started renders as perpetually in-progress even though its
spans arrived.

Each conversation maps to one run, and each exported turn contributes one
completed invocation to it:

```text
Run (run name = native conversation/session ID)
├── invocation for turn 1  -> COMPLETED
├── invocation for turn 2  -> COMPLETED
└── invocation for turn 3  -> COMPLETED
```

For every turn the exporter creates the run if it does not exist, exports the
turn's spans, then starts ingestion for that turn. Run creation comes first
because the spans carry the run name, and ingestion comes last so the ingestion
window does not open before the spans it waits for have been sent. The run
therefore becomes terminal after the first turn and stays terminal as later
turns arrive, because status resolves against the most recent invocation.

Runs are created in `LOG_INGESTION` mode: spans are assembled from journalled
native events rather than by invoking a Python app.

A turn is only marked exported once both its span export and its ingestion
succeed. If either fails the turn is released for retry with the journal's usual
backoff, since spans without ingestion would leave the run in-progress forever.

Destinations with no run concept, such as plain OTLP, still receive spans; there
is simply no run to complete. To export spans without managing runs at all:

```bash
export TRULENS_MANAGE_RUNS=false
```

Turns then never reach a terminal run status, so this is intended for debugging
the span path in isolation.

## Trace semantics

Native Cursor `conversation_id`, Claude Code `session_id`, and OpenCode
`sessionID` map to the existing TruLens
`SpanAttributes.CONVERSATION_ID`. Each prompt gets distinct `RECORD_ID` and
`INPUT_ID` values. There is no separate thread identity contract.

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
See [Evaluating Coding-Agent Hook Traces](coding_agent_evals.md) for how
post-hoc metrics would run on those records.

Official OTEL GenAI conventions are used for model inference, structured
messages, token usage, and tool execution. TruLens record/evaluation fields and
coding-agent/MCP concepts without an OTEL equivalent remain under the
`ai.observability.*` namespace. Custom data is not emitted into reserved OTEL
namespaces.

## Validate and inspect

```bash
trulens-client-hooks clients
trulens-client-hooks validate
trulens-client-hooks status cursor
trulens-client-hooks flush
```

Each hook invocation returns after writing to a locked local journal. A detached
singleton worker exports completed turns and retries transient destination
failures without blocking the coding client. Configure worker behavior with:

```bash
export TRULENS_EXPORT_LEASE_SECONDS=60
export TRULENS_WORKER_IDLE_SECONDS=2
```

`status` reports worker, pending, claimed, and retry state. `flush` remains a
synchronous recovery command for troubleshooting after a machine or process
crash.

Remove only the TruLens-managed hook entries while preserving other hooks:

```bash
trulens-client-hooks uninstall cursor
```

Runtime hooks always fail open: telemetry errors are written to stderr and never
block the coding client.
