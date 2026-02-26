# TruLens TypeScript SDK

TypeScript instrumentation SDK for TruLens. Lets Node.js AI apps emit
OpenTelemetry spans with TruLens semantic conventions, then store them in any
TruLens-connected database (SQLite, PostgreSQL, Snowflake, etc.) and view them
in the TruLens dashboard.

## Packages

| Package | Description |
|---|---|
| `@trulens/semconv` | Semantic conventions — pure port of the Python `trace.py` |
| `@trulens/core` | Core SDK: `TruSession`, `instrument()`, `instrumentDecorator()`, `withRecord()`, `createTruApp()` |
| `@trulens/instrumentation-openai` | Auto-instrumentation for the OpenAI Node.js SDK |
| `@trulens/connectors-snowflake` | Direct Snowflake span exporter (no Python server needed) |

## Quick start

### Prerequisites

- Node.js 18+
- pnpm (`npm install -g pnpm`)
- A running Python TruSession (for the OTLP path) **or** Snowflake credentials (for the direct path)

### Install

```bash
npm install @trulens/core @trulens/instrumentation-openai \
            @opentelemetry/exporter-trace-otlp-http
# or
pnpm add @trulens/core @trulens/instrumentation-openai \
         @opentelemetry/exporter-trace-otlp-http
```

### 1. Initialise TruSession with auto-instrumentation

```typescript
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { TruSession } from "@trulens/core";
import { OpenAIInstrumentation } from "@trulens/instrumentation-openai";

const session = await TruSession.init({
  appName: "my-rag-app",
  appVersion: "v1",
  exporter: new OTLPTraceExporter({ url: "http://localhost:4318/v1/traces" }),
  endpoint: "http://localhost:4318",
  instrumentations: [new OpenAIInstrumentation()],
});
```

With `OpenAIInstrumentation`, every `openai.chat.completions.create()` call
automatically produces a `GENERATION` span with model, token counts, and cost
data — no manual instrumentation needed for LLM calls.

### 2. Instrument your app

**Decorator style** (recommended for class methods):

```typescript
import { instrumentDecorator as instrument, createTruApp } from "@trulens/core";
import { SpanAttributes, SpanType } from "@trulens/core";

class MyRAG {
  @instrument({ spanType: SpanType.RETRIEVAL })
  async retrieve(query: string): Promise<string[]> {
    return fetchRelevantDocs(query);
  }

  // No @instrument needed — OpenAI auto-instrumentation handles this
  async generate(query: string, contexts: string[]): Promise<string> {
    const response = await this.openai.chat.completions.create({ ... });
    return response.choices[0]?.message.content ?? "";
  }

  async query(question: string): Promise<string> {
    const contexts = await this.retrieve(question);
    return this.generate(question, contexts);
  }
}
```

**Wrapper style** (works with standalone functions):

```typescript
import { instrument } from "@trulens/core";

const retrieve = instrument(
  async (query: string): Promise<string[]> => {
    return fetchRelevantDocs(query);
  },
  {
    spanName: "retrieve",
    spanType: SpanType.RETRIEVAL,
    attributes: (ret, _err, query) => ({
      [SpanAttributes.RETRIEVAL.QUERY_TEXT]: query,
      [SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]: JSON.stringify(ret),
    }),
  }
);
```

### 3. Wrap with `createTruApp` or `withRecord`

**`createTruApp`** wraps a designated method with a `RECORD_ROOT` span
automatically — recommended for class-based apps:

```typescript
import { createTruApp } from "@trulens/core";

const rag = new MyRAG();
const app = createTruApp(rag, {
  mainMethod: "query",
  mainInput: (question: string) => question,
});

const answer = await app.query("What is RAG?");
```

**`withRecord`** for ad-hoc wrapping:

```typescript
import { withRecord } from "@trulens/core";

const answer = await withRecord(
  () => rag.query(userQuestion),
  { input: userQuestion }
);
```

### 4. Flush spans on exit

```typescript
await session.shutdown();
```

## Running the demo

The `examples/rag-demo/` directory contains a complete working example.

### Step 1 — Start the Python TruSession OTLP receiver

```bash
python -c "
from trulens.core import TruSession

session = TruSession()
session.start_otlp_receiver(port=4318)
print('OTLP receiver listening on http://localhost:4318')
input('Press Enter to stop...')
"
```

### Step 2 — Run the TypeScript demo

```bash
cd typescript/examples/rag-demo
pnpm install
OPENAI_API_KEY=sk-... pnpm start
```

### Step 3 — View traces in the dashboard

```bash
poetry run python -c "from trulens.dashboard import run_dashboard; run_dashboard(port=8501)"
```

Open http://localhost:8501 — you should see three records from the demo,
each with `RETRIEVAL` and `GENERATION` child spans.

## Development

```bash
cd typescript
pnpm install        # install all workspace deps
pnpm build          # build all packages
pnpm test           # run all unit tests
pnpm format         # prettier
pnpm lint           # eslint
```

## Dashboard compatibility

TypeScript-emitted spans are identical in structure to Python-emitted spans
(same OTEL format, same `ai.observability.*` attribute keys). The TruLens
Streamlit dashboard requires no changes to display them.
