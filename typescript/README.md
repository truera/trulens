# TruLens TypeScript SDK

TypeScript instrumentation SDK for TruLens. Lets Node.js AI apps emit
OpenTelemetry spans with TruLens semantic conventions, then store them in any
TruLens-connected database (SQLite, PostgreSQL, Snowflake, etc.) and view them
in the TruLens dashboard.

## Packages

| Package | Description |
|---|---|
| `@trulens/semconv` | Semantic conventions — pure port of the Python `trace.py` |
| `@trulens/core` | Core SDK: `TruSession`, `instrument()`, `withRecord()` |
| `@trulens/connectors-snowflake` | Direct Snowflake span exporter (no Python server needed) |

## Quick start

### Prerequisites

- Node.js 18+
- pnpm (`npm install -g pnpm`)
- A running Python TruSession (for the OTLP path) **or** Snowflake credentials (for the direct path)

### Install

```bash
npm install @trulens/core
# or
pnpm add @trulens/core
```

### 1. Initialise TruSession

**OTLP path** — sends spans to a running Python `TruSession` (SQLite, Postgres, etc.):

```typescript
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { TruSession } from "@trulens/core";

const session = TruSession.init({
  appName: "my-rag-app",
  appVersion: "v1",
  exporter: new OTLPTraceExporter({ url: "http://localhost:4318/v1/traces" }),
});
```

**Direct Snowflake path** — no Python server needed:

```typescript
import { SnowflakeConnector, TruLensSnowflakeSpanExporter } from "@trulens/connectors-snowflake";
import { TruSession } from "@trulens/core";

const connector = new SnowflakeConnector({
  account: "myorg-myaccount",
  username: "myuser",
  password: "mypassword",
  database: "MY_DB",
  schema: "MY_SCHEMA",
});

const session = TruSession.init({
  appName: "my-rag-app",
  appVersion: "v1",
  exporter: new TruLensSnowflakeSpanExporter({ connector }),
});
```

### 2. Instrument functions

**Wrapper style** (works everywhere, no decorator config needed):

```typescript
import { instrument } from "@trulens/core";
import { SpanAttributes, SpanType } from "@trulens/semconv";

const retrieve = instrument(
  async (query: string): Promise<string[]> => {
    return fetchRelevantDocs(query);
  },
  {
    spanType: SpanType.RETRIEVAL,
    attributes: (ret, _err, query) => ({
      [SpanAttributes.RETRIEVAL.QUERY_TEXT]: query,
      [SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]: JSON.stringify(ret),
    }),
  }
);
```

**Decorator style** (TypeScript 5+, requires `experimentalDecorators: false` in tsconfig):

```typescript
import { instrumentDecorator as instrument } from "@trulens/core";
import { SpanType } from "@trulens/semconv";

class MyRAG {
  @instrument({ spanType: SpanType.RETRIEVAL })
  async retrieve(query: string): Promise<string[]> {
    return fetchRelevantDocs(query);
  }
}
```

### 3. Wrap top-level calls with `withRecord()`

`withRecord()` creates a `RECORD_ROOT` span — the entry point the TruLens
dashboard uses to represent a complete app execution.

```typescript
import { withRecord } from "@trulens/core";

const answer = await withRecord(
  () => rag.query(userQuestion),
  { input: userQuestion }
);
```

### 4. Flush spans on exit

```typescript
// Call before process.exit() to ensure all buffered spans are sent.
await session.shutdown();
```

## Running the demo

The `examples/rag-demo/` directory contains a complete working example.

### Step 1 — Start the Python TruSession OTLP receiver

```bash
python -c "
from trulens.core import TruSession
session = TruSession()          # defaults to SQLite
session.start_otlp_receiver(port=4318)
print('OTLP receiver listening on http://localhost:4318/v1/traces')
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
# In a new terminal
trulens-dashboard
```

Open http://localhost:8501 — you should see three records from the demo,
each with `RETRIEVAL` and `GENERATION` child spans.

### Using the direct Snowflake path instead

Edit `src/trulens.ts` and replace the exporter:

```typescript
import { SnowflakeConnector, TruLensSnowflakeSpanExporter } from "@trulens/connectors-snowflake";

const connector = new SnowflakeConnector({ /* your credentials */ });
const exporter = new TruLensSnowflakeSpanExporter({ connector });
```

No Python server needed — spans go directly to Snowflake AI Observability.

## Attribute resolvers

The `attributes` option in `instrument()` controls which span attributes are
set. It accepts either:

**A callback** `(ret, error, ...args) => Record<string, unknown>` — mirrors
the Python lambda form. Use this when you need full access to arguments:

```typescript
instrument(myFn, {
  attributes: (ret, err, query, options) => ({
    [SpanAttributes.RETRIEVAL.QUERY_TEXT]: query,
    [SpanAttributes.RETRIEVAL.NUM_CONTEXTS]: options?.topK ?? 3,
    [SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]: JSON.stringify(ret),
  }),
});
```

**A string-keyed object** for simple positional arg → attribute mappings:

```typescript
instrument(myFn, {
  attributes: {
    [SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]: "return",
    // "return" is a special key that maps to the function's return value.
    // Other values are treated as positional arg indices ("0", "1", ...).
  },
});
```

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
