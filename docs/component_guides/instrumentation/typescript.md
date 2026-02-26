# TypeScript SDK

The TruLens TypeScript SDK lets you instrument Node.js AI applications with
OpenTelemetry spans that follow TruLens
[semantic conventions](../../otel/semantic_conventions.md). Traces are stored in
any TruLens-connected database and rendered in the same dashboard used by Python
apps.

!!! info "Scope"

    The TypeScript SDK covers **instrumentation and local tracing**: decorating
    functions, emitting OTEL spans, and persisting them to SQLite via an
    embedded receiver. Evaluation (feedback / metrics) and the dashboard remain
    in Python. This means you instrument and trace in TypeScript, then evaluate
    and view in Python.

## Architecture

The recommended path for local development uses **connector mode**, where
`TruSession` starts an embedded OTLP receiver that writes directly to SQLite
-- no Python process needed for tracing:

``` mermaid
graph LR
  A["instrument() / createTruApp()"] --> B["OTLP Exporter"]
  C["Auto-instrumentation"] --> B
  B -- localhost --> D["TruLensReceiver"]
  D --> E[("SQLite")]
  E --> F["Python Evals + Dashboard"]
```

For Snowflake AI Observability, spans can be exported directly. See
[Logging in Snowflake](../../component_guides/logging/where_to_log/log_in_snowflake.md#typescript-sdk)
for setup details.

## Packages

| Package | Description |
|---|---|
| `@trulens/core` | Core SDK: `TruSession`, `instrument()`, `instrumentDecorator()`, `withRecord()`, `createTruApp()`, `SQLiteConnector`, `TruLensReceiver`, `DBConnector` |
| `@trulens/semconv` | Semantic conventions (pure port of Python `trace.py`) |
| `@trulens/instrumentation-openai` | Auto-instrumentation for the OpenAI Node.js SDK |
| `@trulens/instrumentation-langchain` | Auto-instrumentation for LangChain.js via callbacks |
| `@trulens/connectors-snowflake` | Direct Snowflake span exporter and run manager (see [Logging in Snowflake](../../component_guides/logging/where_to_log/log_in_snowflake.md#typescript-sdk)) |

`@trulens/core` re-exports everything from `@trulens/semconv`, so most apps
only need a single import.

## Installation

The packages are not yet published to npm. Build from the monorepo:

```bash
cd typescript
pnpm install --no-frozen-lockfile
pnpm --filter @trulens/semconv build
pnpm --filter @trulens/core build
pnpm --filter @trulens/instrumentation-openai build   # if using OpenAI
pnpm --filter @trulens/instrumentation-langchain build # if using LangChain.js
```

## Session initialisation

`TruSession.init()` is the entry point. It sets up the OTEL tracer provider,
registers the app, and enables any auto-instrumentations.

=== "Connector mode (SQLite) -- recommended"

    ```typescript
    import { TruSession, SQLiteConnector } from "@trulens/core";
    import { OpenAIInstrumentation } from "@trulens/instrumentation-openai";

    const session = await TruSession.init({
      appName: "my-app",
      appVersion: "v1",
      connector: new SQLiteConnector(),
      instrumentations: [new OpenAIInstrumentation()],
    });
    ```

    This starts an embedded `TruLensReceiver` and writes spans to
    `default.sqlite` in the current directory. No Python process needed for
    tracing.

=== "LangChain.js auto-instrumentation"

    ```typescript
    import { TruSession, SQLiteConnector } from "@trulens/core";
    import { LangChainInstrumentation } from "@trulens/instrumentation-langchain";

    const session = await TruSession.init({
      appName: "langchain-rag",
      appVersion: "v1",
      connector: new SQLiteConnector(),
      instrumentations: [new LangChainInstrumentation()],
    });
    ```

    `LangChainInstrumentation` hooks into LangChain's `CallbackManager` so
    every chain, retriever, and LLM call is traced automatically.

### Auto-instrumentation

The `instrumentations` option accepts an array of OTEL instrumentations. When
provided, they are registered with the session's `TracerProvider` and enabled
automatically.

`@trulens/instrumentation-openai` patches `openai.chat.completions.create()`
so every call produces a `GENERATION` span with:

- `SpanAttributes.COST.MODEL` -- the model name
- `SpanAttributes.COST.NUM_PROMPT_TOKENS` -- prompt tokens
- `SpanAttributes.COST.NUM_COMPLETION_TOKENS` -- completion tokens
- `SpanAttributes.COST.NUM_TOKENS` -- total tokens

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
      [SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]: ret as string[],
    }),
  }
);
```

**Options:**

| Option | Type | Default | Description |
|---|---|---|---|
| `spanName` | `string` | `fn.name` | Name shown in the dashboard trace tree |
| `spanType` | `SpanType` | `UNKNOWN` | Semantic span type (`RETRIEVAL`, `GENERATION`, etc.) |
| `attributes` | object or function | -- | Static map or `(ret, err, ...args) => Record<string, unknown>` |

!!! tip "Always set `spanName` for arrow functions"

    Arrow functions are anonymous in JavaScript, so the span name defaults to
    `"anonymous"`. Pass `spanName` explicitly to get meaningful names in the
    dashboard.

### `instrumentDecorator(options)`

Method decorator for TypeScript 5+ (Stage 3 decorators). Automatically infers
the span name from the method name -- no `spanName` needed:

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

Creates a `RECORD_ROOT` span -- the top-level span the dashboard uses as the
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
`RECORD_ROOT` span -- no per-call `withRecord()` needed:

```typescript
import { createTruApp } from "@trulens/core";

const rag = new SimpleRAG();
const app = createTruApp(rag, {
  mainMethod: "query",
  mainInput: (question: string) => question,
});

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
name and token counts.

```typescript
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

!!! note "Array attributes"

    The TypeScript SDK passes arrays (e.g. `RETRIEVED_CONTEXTS`) as native
    OTEL array attributes (`string[]`), not as JSON strings. This means
    Python-side `Selector` with `collect_list=False` correctly iterates over
    individual elements.

## Flushing spans

Before the process exits, flush pending spans:

```typescript
await session.shutdown();
```

The `BatchSpanProcessor` buffers spans for efficiency. If you skip this call,
recent spans may be lost.

## Pluggable database connectors

The `DBConnector` interface allows swapping the storage backend. The built-in
`SQLiteConnector` writes to a local `.sqlite` file compatible with Python's
`TruSession` and dashboard. Custom connectors (e.g. PostgreSQL) can implement
the same interface:

```typescript
interface DBConnector {
  addApp(app: AppDefinition): string;
  addEvents(events: EventRecord[]): string[];
  close(): void;
}
```

## Dashboard compatibility

TypeScript-emitted spans are structurally identical to Python-emitted spans.
The TruLens dashboard requires **no changes** to display them -- the same
Leaderboard, Records, and Trace Details views work out of the box.

The `SQLiteConnector` creates all tables the dashboard expects (`trulens_apps`,
`trulens_events`, `trulens_records`, `trulens_feedback_defs`,
`trulens_feedbacks`) and stamps the alembic version, so Python's `TruSession`
can open the same file without migration.

## Evaluating TypeScript traces with Python

Although metrics are defined in Python, they can evaluate spans produced by
the TypeScript SDK. The workflow is:

1. Run your TypeScript app (spans are stored in `default.sqlite`).
2. Define metrics in Python using `Selector` to pick span attributes.
3. Run evaluation -- TruLens reads the stored spans and computes scores.

```python
from trulens.core import TruSession, Metric
from trulens.core.feedback.selector import Selector
from trulens.providers.openai import OpenAI
from trulens.otel.semconv.trace import SpanAttributes

session = TruSession()
provider = OpenAI()

f_relevance = Metric(
    implementation=provider.relevance_with_cot_reasons,
    name="Answer Relevance",
    selectors={
        "prompt": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
        ),
        "response": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
        ),
    },
)

f_groundedness = Metric(
    implementation=provider.groundedness_measure_with_cot_reasons,
    name="Groundedness",
    selectors={
        "source": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=True,
        ),
        "statement": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
        ),
    },
)

events = session.get_events(app_name="my-app", app_version="v1")
session.compute_feedbacks_on_events(events, [f_relevance, f_groundedness])
```

## Quickstarts

- [TypeScript Quickstart](../../getting_started/quickstarts/typescript_quickstart.md) --
  instrument a RAG app with OpenAI, evaluate with the RAG triad, view in the dashboard
- [LangChain.js Quickstart](../../getting_started/quickstarts/langchain_typescript_quickstart.md) --
  auto-instrument a LangChain LCEL chain with zero TruLens imports in app code
