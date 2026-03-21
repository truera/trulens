# TypeScript Quickstart

More and more LLM apps — agents, copilots, RAG pipelines — are being built in
TypeScript. TruLens now brings the same evaluation-driven observability to
TypeScript developers: automatic LLM call tracing with token and cost tracking,
lightweight decorators for pipeline steps, and a shared dashboard where Python
and TypeScript traces live side by side — all powered by OpenTelemetry.

This quickstart runs a pre-built RAG demo, evaluates the traces, and shows the
results in the TruLens dashboard. **All steps run in a single terminal from the
repo root.**

## Prerequisites

- **Node.js 18+** and **pnpm** (`npm install -g pnpm`)
- **Python 3.9+** and **Poetry** (`pip install poetry`)
- An **OpenAI API key**

## Step 1 — Set up the environment

All commands assume you start from the root of the TruLens repository.

```bash
export OPENAI_API_KEY=sk-...

# Python environment (needed for evaluations and dashboard)
python -m venv .venv
source .venv/bin/activate               # on Windows: .venv\Scripts\activate
poetry install --with dev
```

## Step 2 — Build the TypeScript SDK packages

```bash
cd typescript
pnpm install --no-frozen-lockfile
pnpm --filter @trulens/semconv build
pnpm --filter @trulens/core build
pnpm --filter @trulens/instrumentation-openai build
```

## Step 3 — Run the RAG demo

The repo includes a ready-to-run demo at `typescript/examples/rag-demo/`.

```bash
cd examples/rag-demo
pnpm start
```

You'll see an interactive prompt — try a few questions:

```
TruSession ready — app="trulens-rag-demo"
Spans written to default.sqlite
Type a question and press Enter. Ctrl+C or empty line to quit.

Q: What is TruLens?
A: TruLens is an open-source library for evaluating and tracking LLM apps...

Q: How does RAG work?
A: RAG combines retrieval with LLM generation...

Q:                          <-- press Enter on an empty line to quit
Flushing spans…
Done.
```

Spans are now saved to `default.sqlite` in the current directory.

## Step 4 — Run evaluations on your traces

The TypeScript app has exited and you're still in `typescript/examples/rag-demo/`.
The Python virtualenv is already active from Step 1. Run the evaluations
directly:

```bash
python << 'PYEOF'
from trulens.core import TruSession, Metric
from trulens.core.feedback.selector import Selector
from trulens.providers.openai import OpenAI
from trulens.otel.semconv.trace import SpanAttributes

# Reads default.sqlite from CWD — the same file the TypeScript app wrote to
session = TruSession()
provider = OpenAI()

f_answer_relevance = Metric(
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

f_context_relevance = Metric(
    implementation=provider.context_relevance_with_cot_reasons,
    name="Context Relevance",
    selectors={
        "question": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
        ),
        "context": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=False,
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

events = session.get_events(
    app_name="trulens-rag-demo",
    app_version="v1",
)
print(f"Found {len(events)} events")

session.compute_feedbacks_on_events(
    events,
    [f_answer_relevance, f_context_relevance, f_groundedness],
)
print("Evaluations complete.")
PYEOF
```

This defines the RAG triad — answer relevance, context relevance, and
groundedness — and runs them against the traces your TypeScript app produced.
Results are written back to the same database.

- **Answer Relevance** — how well the response addresses the question
- **Context Relevance** — how relevant each retrieved context is to the question
- **Groundedness** — how well the response is supported by the retrieved contexts

## Step 5 — View in the dashboard

Still in the same terminal and directory:

```bash
python -c "
from trulens.dashboard import run_dashboard
run_dashboard(port=8501)
"
```

Open [http://localhost:8501](http://localhost:8501). You should see:

- **Leaderboard**: your `trulens-rag-demo` app with total tokens, cost, and
  evaluation scores
- **Records**: one record per question you asked, each containing `retrieve` and
  `openai.chat.completions.create` child spans with their captured attributes,
  plus scores for answer relevance, context relevance, and groundedness

---

## Appendix — Build the RAG demo from scratch

The steps above use the pre-built `rag-demo` example. This section walks
through creating the same app from scratch so you can understand every piece.

!!! note "Local workspace vs. npm"

```
The `@trulens/*` packages are not yet published to npm. Until they are,
new projects must live inside the `typescript/examples/` directory so pnpm
workspace resolution can find them.
```

### A1 — Create the project

Starting from the repo root (after completing Steps 1 and 2):

```bash
cd typescript
mkdir -p examples/my-rag-demo
cd examples/my-rag-demo

cat > package.json << 'EOF'
{
  "name": "my-rag-demo",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "start": "tsx src/main.ts"
  },
  "dependencies": {
    "@trulens/core": "workspace:*",
    "@trulens/instrumentation-openai": "workspace:*",
    "@trulens/semconv": "workspace:*",
    "openai": "^4.0.0"
  },
  "devDependencies": {
    "tsx": "^4.0.0",
    "typescript": "^5.5.0"
  }
}
EOF

cat > tsconfig.json << 'EOF'
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "strict": true,
    "esModuleInterop": true,
    "outDir": "dist"
  }
}
EOF

mkdir -p src
cd ../..
pnpm install --no-frozen-lockfile
cd examples/my-rag-demo
```

### A2 — `src/trulens.ts` — session setup

```typescript
import { TruSession, SQLiteConnector } from "@trulens/core";
import { OpenAIInstrumentation } from "@trulens/instrumentation-openai";

export async function initSession(): Promise<TruSession> {
  return TruSession.init({
    appName: "trulens-rag-demo",
    appVersion: "v1",
    connector: new SQLiteConnector(),
    instrumentations: [new OpenAIInstrumentation()],
  });
}
```

`TruSession.init()` with a `connector` starts an embedded OTLP receiver that
writes spans directly to SQLite — no separate Python process needed for tracing.
The OpenAI auto-instrumentation captures every LLM call with model, token
counts, and cost data.

### A3 — `src/rag.ts` — RAG pipeline

```typescript
import OpenAI from "openai";
import { instrumentDecorator as instrument, createTruApp } from "@trulens/core";
import { SpanAttributes, SpanType } from "@trulens/core";

const DOCS = [
  "TruLens is an open-source library for evaluating and tracking LLM apps.",
  "RAG combines retrieval with LLM generation for grounded answers.",
  "OpenTelemetry is a vendor-neutral observability framework.",
];

class SimpleRAG {
  private openai = new OpenAI();

  @instrument<[string], Promise<string[]>>({
    spanType: SpanType.RETRIEVAL,
    attributes: (ret, _err, query) => ({
      [SpanAttributes.RETRIEVAL.QUERY_TEXT]: query as string,
      [SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]: ret as string[],
    }),
  })
  async retrieve(query: string): Promise<string[]> {
    const tokens = new Set(
      query.toLowerCase().split(/\W+/).filter((t) => t.length > 3)
    );
    const matches = DOCS.filter((d) =>
      d.toLowerCase().split(/\W+/).some((t) => tokens.has(t))
    );
    return matches.length > 0 ? matches : [DOCS[0]!];
  }

  async generate(query: string, contexts: string[]): Promise<string> {
    const contextBlock = contexts.map((c, i) => `[${i + 1}] ${c}`).join("\n\n");
    const response = await this.openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [
        { role: "system", content: `Answer using only this context:\n${contextBlock}` },
        { role: "user", content: query },
      ],
      max_tokens: 256,
    });
    return response.choices[0]?.message.content ?? "";
  }

  async query(question: string): Promise<string> {
    const contexts = await this.retrieve(question);
    return this.generate(question, contexts);
  }
}

const rag = new SimpleRAG();
export const app = createTruApp(rag, {
  mainMethod: "query",
  mainInput: (question: string) => question,
});
```

Key points:

- **`@instrument(options)`** decorates a method to emit an OTEL span on each
  call. The span name is inferred from the method name.
- **OpenAI auto-instrumentation** (from `trulens.ts`) captures `GENERATION`
  spans with token/cost data for every `openai.chat.completions.create()` call —
  no manual instrumentation needed.
- **`createTruApp(target, options)`** wraps the main entry method with a
  `RECORD_ROOT` span, so you don't need `withRecord()` on every call.

### A4 — `src/main.ts` — entry point

```typescript
import * as readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { initSession } from "./trulens.js";
import { app } from "./rag.js";

async function main() {
  const session = await initSession();
  console.log(`TruSession ready — app="${session.appName}"`);
  console.log('Type a question and press Enter. Ctrl+C or empty line to quit.\n');

  const rl = readline.createInterface({ input: stdin, output: stdout });

  try {
    while (true) {
      const question = await rl.question("Q: ");
      if (!question.trim()) break;
      console.log(`A: ${await app.query(question)}\n`);
    }
  } finally {
    rl.close();
    console.log("\nFlushing spans…");
    await session.shutdown();
    console.log("Done.");
  }
}

main()
  .catch(console.error)
  .finally(() => process.exit(0));
```

Run with `pnpm start`, then continue with Steps 4 and 5 above.

## What's next

- Read the full [TypeScript SDK documentation](../../component_guides/instrumentation/typescript.md)
  for architecture details and the complete API surface.
- Explore [feedback functions](../../component_guides/evaluation/index.md)
  to add custom evaluations beyond the RAG triad.
- Compare different app versions using the dashboard leaderboard.
