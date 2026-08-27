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

## Identity and correlation

TruLens preserves the native client identity instead of creating a separate
thread identifier. The mapping is:

| TruLens field | Value |
| --- | --- |
| `SpanAttributes.CONVERSATION_ID` | Native conversation or session ID |
| `SpanAttributes.RUN_NAME` | The same native conversation or session ID |
| `SpanAttributes.INPUT_ID` | Correlated native turn ID |
| `SpanAttributes.RECORD_ID` | `<client>:<conversation-id>:<turn-id>` |
| OpenTelemetry trace ID | Deterministically derived from `RECORD_ID` |

The native conversation fields are Cursor `conversation_id`, Claude Code
`session_id`, and OpenCode `sessionID`. A conversation therefore maps to one
TruLens conversation and, for destinations that support runs, one run. Each user
prompt within it maps to a distinct turn and `RECORD_ROOT` span.

The durable journal correlates prompt, tool, edit, shell, MCP, subagent,
response, and terminal events into that turn. TruLens uses a native turn ID when
the client supplies one. Otherwise it generates a stable `turn:<hex>` ID when
the prompt arrives. Later events without a turn ID join the active turn, and the
terminal event closes it. OpenCode response events are explicitly correlated to
the active prompt turn because OpenCode gives the response a different message
ID.

Identity is deterministic across retries. Re-exporting the same journalled turn
reuses its run name, record ID, input ID, and trace ID instead of creating a new
record.

## Run lifecycle

Run lifecycle — creating a run and driving it to a terminal status — is a
Snowflake AI Observability concept. It is managed **only** when hook spans are
exported to a Snowflake destination. OSS (local database/Postgres) and plain
OTLP destinations export spans but skip run lifecycle entirely: no run is
created and no ingestion is started.

For a Snowflake destination, each conversation maps to one run. Each exported
turn contributes one invocation containing one input record:

```text
Run (run name = native conversation/session ID)
├── invocation for turn 1  -> COMPLETED
├── invocation for turn 2  -> COMPLETED
└── invocation for turn 3  -> COMPLETED
```

For every turn the exporter creates the run if it does not exist, exports the
turn's spans, then starts ingestion for that turn. Run creation comes first
because the spans carry the run name, and ingestion comes last so the ingestion
window does not open before the spans it waits for have been sent. When ingestion
can start, the run becomes terminal after the first turn and stays terminal as
later turns arrive because status resolves against the most recent invocation.

Runs are created in `LOG_INGESTION` mode: spans are assembled from journalled
native events rather than by invoking a Python app. Run identity is idempotent on
`(app_name, app_version, run_name)`, where the app version is the detected native
client version.

Snowflake run completion starts an ingestion task after span export. This step
requires `CREATE TASK` on the target schema and `EXECUTE TASK` on the account. If
those privileges are unavailable, TruLens retains the successfully exported
trace and logs a warning, but the run may remain non-terminal in the UI. A span
export failure still releases the turn for retry with the journal's normal
backoff.

Destinations without a Snowflake run store — OSS databases, plain OTLP, or a
session with no connector — still receive spans; there is simply no run to create
or complete, and this is treated as a quiet, expected state rather than an error.
To additionally disable run management on a Snowflake destination (for example,
to debug the span path in isolation):

```bash
export TRULENS_MANAGE_RUNS=false
```

With run management disabled, turns never reach a terminal run status.


## Trace semantics

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
