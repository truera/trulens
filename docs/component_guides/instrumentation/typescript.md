# TypeScript SDK

The TruLens TypeScript SDK lets you instrument Node.js AI applications with
OpenTelemetry spans that follow TruLens
[semantic conventions](../../otel/semantic_conventions.md). Traces are stored in
any TruLens-connected database and rendered in the same dashboard used by Python
apps.

!!! info "Scope (v1)"

    The TypeScript SDK covers **instrumentation only**: decorating functions,
    emitting OTEL spans, and exporting them. Evaluation (feedback functions),
    the dashboard, and database connectors remain in Python. This means you
    instrument in TypeScript and evaluate/view in Python.

## Architecture

```mermaid
flowchart LR
    subgraph tsApp [TypeScript App]
        inst["instrument() / createTruApp()"]
        autoInst["Auto-instrumentation (OpenAI)"]
        exp[OTLP Exporter]
    end

    subgraph pyReceiver [Python TruSession]
        recv[OTLP Receiver]
        enrich[Cost Enrichment]
        db[(SQLite / Postgres)]
    end

    dashboard[TruLens Dashboard]

    inst --> exp
    autoInst --> exp
    exp -->|"OTLP/HTTP"| recv
    recv --> enrich
    enrich --> db
    db --> dashboard
```

Alternatively, for Snowflake AI Observability, spans can be exported directly
without a Python receiver:

```mermaid
flowchart LR
    subgraph tsApp [TypeScript App]
        inst["instrument() / createTruApp()"]
        sfExp[Snowflake Exporter]
    end

    sf[(Snowflake)]
    dashboard[TruLens Dashboard]

    inst --> sfExp
    sfExp -->|direct| sf
    sf --> dashboard
```

## Packages

| Package | Description |
|---|---|
| `@trulens/core` | Core SDK: `TruSession`, `instrument()`, `instrumentDecorator()`, `withRecord()`, `createTruApp()` |
| `@trulens/semconv` | Semantic conventions (pure port of Python `trace.py`) |
| `@trulens/instrumentation-openai` | Auto-instrumentation for the OpenAI Node.js SDK |
| `@trulens/connectors-snowflake` | Direct Snowflake span exporter |

`@trulens/core` re-exports everything from `@trulens/semconv`, so most apps
only need a single import.

## Installation

```bash
npm install @trulens/core @trulens/instrumentation-openai \
            @opentelemetry/exporter-trace-otlp-http
# or
pnpm add @trulens/core @trulens/instrumentation-openai \
         @opentelemetry/exporter-trace-otlp-http
```

For the direct Snowflake path:

```bash
npm install @trulens/connectors-snowflake
```

## Session initialisation

`TruSession.init()` is the entry point. It registers the app with the Python
receiver (so it appears in the dashboard), sets up the OTEL tracer provider,
and enables any auto-instrumentations.

=== "OTLP path (SQLite / Postgres)"

    ```typescript
    import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
    import { TruSession } from "@trulens/core";
    import { OpenAIInstrumentation } from "@trulens/instrumentation-openai";

    const session = await TruSession.init({
      appName: "my-app",
      appVersion: "v1",
      exporter: new OTLPTraceExporter({
        url: "http://localhost:4318/v1/traces",
      }),
      endpoint: "http://localhost:4318",
      instrumentations: [new OpenAIInstrumentation()],
    });
    ```

=== "Direct Snowflake"

    ```typescript
    import {
      SnowflakeConnector,
      TruLensSnowflakeSpanExporter,
    } from "@trulens/connectors-snowflake";
    import { TruSession } from "@trulens/core";
    import { OpenAIInstrumentation } from "@trulens/instrumentation-openai";

    const connector = new SnowflakeConnector({
      account: "myorg-myaccount",
      username: "myuser",
      password: "mypassword",
      database: "MY_DB",
      schema: "MY_SCHEMA",
    });

    const session = await TruSession.init({
      appName: "my-app",
      appVersion: "v1",
      exporter: new TruLensSnowflakeSpanExporter({ connector }),
      instrumentations: [new OpenAIInstrumentation()],
    });
    ```

!!! note

    `TruSession.init()` is async because it makes an HTTP call to register the
    app before tracing begins. Omit `endpoint` to skip registration (useful in
    tests or the Snowflake path).

### Auto-instrumentation

The `instrumentations` option accepts an array of OTEL instrumentations. When
provided, they are registered with the session's `TracerProvider` and enabled
automatically.

`@trulens/instrumentation-openai` patches `openai.chat.completions.create()`
so every call produces a `GENERATION` span with:

- `SpanAttributes.COST.MODEL` — the model name
- `SpanAttributes.COST.NUM_PROMPT_TOKENS` — prompt tokens
- `SpanAttributes.COST.NUM_COMPLETION_TOKENS` — completion tokens
- `SpanAttributes.COST.NUM_TOKENS` — total tokens

This means you **don't need to manually instrument LLM calls** or change
return types to expose token counts.

## Instrumenting functions

### `instrument(fn, options)`

Wraps a function so every call produces an OTEL span with TruLens attributes.
Works with both sync and async functions.

```typescript
import { instrument } from "@trulens/core";
import { SpanAttributes, SpanType } from "@trulens/core";

const retrieve = instrument(
  async (query: string): Promise<string[]> => {
    return await vectorSearch(query);
  },
  {
    spanName: "retrieve",
    spanType: SpanType.RETRIEVAL,
    attributes: (ret, _err, query) => ({
      [SpanAttributes.RETRIEVAL.QUERY_TEXT]: query as string,
      [SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]: JSON.stringify(ret),
    }),
  }
);
```

**Options:**

| Option | Type | Default | Description |
|---|---|---|---|
| `spanName` | `string` | `fn.name` | Name shown in the dashboard trace tree |
| `spanType` | `SpanType` | `UNKNOWN` | Semantic span type (`RETRIEVAL`, `GENERATION`, etc.) |
| `attributes` | object or function | — | Static map or `(ret, err, ...args) => Record<string, unknown>` |

!!! tip "Always set `spanName` for arrow functions"

    Arrow functions are anonymous in JavaScript, so the span name defaults to
    `"anonymous"`. Pass `spanName` explicitly to get meaningful names in the
    dashboard.

### `instrumentDecorator(options)`

Method decorator for TypeScript 5+ (Stage 3 decorators). Automatically infers
the span name from the method name — no `spanName` needed:

```typescript
import { instrumentDecorator as instrument } from "@trulens/core";

class MyRAG {
  @instrument({ spanType: SpanType.RETRIEVAL })
  async retrieve(query: string): Promise<string[]> {
    // completely untouched business logic
  }
}
```

### `withRecord(fn, options)`

Creates a `RECORD_ROOT` span — the top-level span the dashboard uses as the
entry point for a complete app execution:

```typescript
import { withRecord } from "@trulens/core";

const answer = await withRecord(
  () => rag.query(userQuestion),
  { input: userQuestion }
);
```

All `instrument()`-ed calls inside `fn` become children of the record root and
share the same `RECORD_ID`.

**Options:**

| Option | Type | Description |
|---|---|---|
| `input` | `unknown` | Main input to the app (shown in dashboard) |
| `groundTruthOutput` | `unknown` | Expected output, if known |
| `runName` | `string` | Groups records into a named run |

### `createTruApp(target, options)`

Wraps an app object so its "main" method is automatically traced with a
`RECORD_ROOT` span — no per-call `withRecord()` needed:

```typescript
import { createTruApp } from "@trulens/core";

const rag = new SimpleRAG();
const app = createTruApp(rag, {
  mainMethod: "query",
  mainInput: (question: string) => question,
});

// Calling app.query("What is RAG?") automatically creates a RECORD_ROOT span
const answer = await app.query("What is RAG?");
```

This mirrors the Python `TruBasicApp` / `TruChain` pattern where you wrap the
app once, not per-call.

**Options:**

| Option | Type | Description |
|---|---|---|
| `mainMethod` | `string` | Method name to wrap with `RECORD_ROOT` |
| `mainInput` | function | Extracts the input from the method's arguments |
| `recordOptions` | `WithRecordOptions` | Additional per-record options (`runName`, `groundTruthOutput`) |

## Capturing token usage and cost

There are two approaches for capturing token/cost data:

### Auto-instrumentation (recommended)

Register `OpenAIInstrumentation` in `TruSession.init()`. Every
`openai.chat.completions.create()` call is automatically traced with model
name and token counts. The Python receiver computes USD cost server-side.

```typescript
// No manual instrumentation needed for LLM calls
async generate(query: string, contexts: string[]): Promise<string> {
  const response = await this.openai.chat.completions.create({ ... });
  return response.choices[0]?.message.content ?? "";
}
```

### Manual attributes

For non-OpenAI providers or custom cost tracking, use the attribute resolver:

```typescript
const generate = instrument(
  async (query: string, contexts: string[]) => {
    const response = await someProvider.complete({ ... });
    return {
      text: response.text,
      model: response.model,
      promptTokens: response.usage.input,
      completionTokens: response.usage.output,
      totalTokens: response.usage.total,
    };
  },
  {
    spanName: "generate",
    spanType: SpanType.GENERATION,
    attributes: (ret, _err) => ({
      [SpanAttributes.COST.MODEL]: ret?.model ?? "",
      [SpanAttributes.COST.NUM_TOKENS]: ret?.totalTokens ?? 0,
      [SpanAttributes.COST.NUM_PROMPT_TOKENS]: ret?.promptTokens ?? 0,
      [SpanAttributes.COST.NUM_COMPLETION_TOKENS]: ret?.completionTokens ?? 0,
    }),
  }
);
```

## Semantic conventions

The TypeScript SDK uses the same `ai.observability.*` attribute keys as the
Python SDK. See the full [semantic conventions reference](../../otel/semantic_conventions.md).

Commonly used attributes:

| Namespace | Key attributes |
|---|---|
| `SpanAttributes.RETRIEVAL` | `QUERY_TEXT`, `RETRIEVED_CONTEXTS`, `NUM_CONTEXTS` |
| `SpanAttributes.COST` | `MODEL`, `NUM_TOKENS`, `NUM_PROMPT_TOKENS`, `NUM_COMPLETION_TOKENS`, `COST`, `CURRENCY` |
| `SpanAttributes.RECORD_ROOT` | `INPUT`, `OUTPUT`, `ERROR` |
| `SpanAttributes.CALL` | `FUNCTION`, `KWARGS`, `RETURN`, `ERROR` |

## Flushing spans

Before the process exits, flush pending spans:

```typescript
await session.shutdown();
```

The `BatchSpanProcessor` buffers spans for efficiency. If you skip this call,
recent spans may be lost.

## Starting the Python OTLP receiver

The receiver is built into `TruSession`:

```bash
python -c "
from trulens.core import TruSession

session = TruSession()
session.start_otlp_receiver(port=4318)
print('Listening on :4318')
input()
"
```

For PostgreSQL, pass `database_url`:

```python
session = TruSession(database_url="postgresql://user:pass@host/db")
```

## Dashboard compatibility

TypeScript-emitted spans are structurally identical to Python-emitted spans.
The TruLens dashboard requires **no changes** to display them — the same
Leaderboard, Records, and Trace Details views work out of the box.

## Evaluating TypeScript traces with Python feedback functions

Although feedback functions are defined in Python, they can evaluate spans
produced by the TypeScript SDK. The workflow is:

1. Run your TypeScript app (spans are stored in the database).
2. Define feedback functions in Python using `Selector` to pick span attributes.
3. Run evaluation — TruLens reads the stored spans and computes scores.

```python
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector
from trulens.providers.openai import OpenAI
from trulens.otel.semconv.trace import SpanAttributes

provider = OpenAI()

f_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)

f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on({
        "context": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=True,
        ),
    })
    .on_output()
)
```

## Quickstart

See the [TypeScript Quickstart](../../getting_started/quickstarts/typescript_quickstart.md)
for a complete, copy-pasteable example.
