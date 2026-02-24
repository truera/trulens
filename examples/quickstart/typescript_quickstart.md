# TypeScript Quickstart

More and more LLM apps — agents, copilots, RAG pipelines — are being built in
TypeScript. TruLens now brings the same evaluation-driven observability to
TypeScript developers: automatic LLM call tracing with token and cost tracking,
lightweight decorators for pipeline steps, and a shared dashboard where Python
and TypeScript traces live side by side — all powered by OpenTelemetry.

This quickstart instruments a simple RAG app, traces retrieval and generation,
and shows the results in the TruLens dashboard.

## Prerequisites

- **Node.js 18+** and **pnpm** (or npm/yarn)
- **Python 3.9+** and **Poetry** (`pip install poetry`)
- An **OpenAI API key**

## Step 1 — Set up the Python environment and start the OTLP receiver

The TypeScript SDK sends OpenTelemetry spans to a lightweight receiver built
into the Python `TruSession`. The OTLP receiver is part of the TruLens
development branch, so you need to install from source.

### 1a. Create a fresh virtual environment and install TruLens

From the root of the TruLens repository:

```bash
# Create and activate a fresh virtual environment
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

# Install TruLens from source (includes the OTLP receiver)
poetry install --with dev
```

!!! tip "Why from source?"

```
The `start_otlp_receiver()` method and the `/v1/register` endpoint are
new features in this branch. They are not yet available in the PyPI
release of `trulens`. Once released, `pip install trulens` will work.
```

### 1b. Start the OTLP receiver

In the same terminal (with the virtualenv active):

```bash
poetry run python -c "
from trulens.core import TruSession

session = TruSession()
session.start_otlp_receiver(port=4318)
print('OTLP receiver listening on http://localhost:4318')
input('Press Enter to stop...')
"
```

This stores traces in a local SQLite database by default. For PostgreSQL or
Snowflake, pass a `database_url` to `TruSession`.

Leave this terminal running — the TypeScript app will send spans to it.

## Step 2 — Build the TypeScript SDK packages

Open a **new terminal** (leave the OTLP receiver running). The TypeScript
packages live in the `typescript/` directory of the repo and are managed with
pnpm workspaces.

### 2a. Install pnpm (if not already installed)

```bash
npm install -g pnpm
```

### 2b. Install dependencies and build

From the root of the TruLens repository:

```bash
cd typescript
pnpm install --no-frozen-lockfile
pnpm --filter @trulens/semconv build
pnpm --filter @trulens/core build
pnpm --filter @trulens/instrumentation-openai build
```

### 2c. Run the included demo (fastest path)

The repo includes a ready-to-run demo at `typescript/examples/rag-demo/`.
You can skip straight to it:

```bash
cd examples/rag-demo
OPENAI_API_KEY=sk-... pnpm start
```

If this works, skip to [Step 6](#step-6--view-in-the-dashboard). The rest of
the steps below walk through building the same thing from scratch.

### 2d. Create your own project (from scratch)

!!! note "Local workspace vs. npm"

```
The `@trulens/*` packages are not yet published to npm. Until they are,
new projects must live inside the `typescript/examples/` directory so pnpm
workspace resolution can find them. Once published, you can create a
project anywhere and use `pnpm add @trulens/core` directly.
```

Create a project inside the monorepo's `typescript/examples/` directory (so
pnpm workspace resolution can find the local `@trulens/*` packages):

```bash
cd /path/to/trulens/typescript          # <-- adjust to your repo location
mkdir -p examples/my-app
cd examples/my-app

cat > package.json << 'EOF'
{
  "name": "my-trulens-app",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "start": "tsx src/main.ts"
  },
  "dependencies": {
    "@opentelemetry/exporter-trace-otlp-http": "^0.57.0",
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

# Go back to typescript/ root and install so pnpm links everything
cd ../..
pnpm install --no-frozen-lockfile
cd examples/my-app
```

## Step 3 — Initialise TruSession with auto-instrumentation

```bash
cat > src/trulens.ts << 'TSEOF'
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { TruSession } from "@trulens/core";
import { OpenAIInstrumentation } from "@trulens/instrumentation-openai";

const ENDPOINT = process.env.TRULENS_ENDPOINT ?? "http://localhost:4318";

export async function initSession(): Promise<TruSession> {
  return TruSession.init({
    appName: "my-rag-app",
    appVersion: "v1",
    exporter: new OTLPTraceExporter({ url: `${ENDPOINT}/v1/traces` }),
    endpoint: ENDPOINT,
    // Auto-instrument every openai.chat.completions.create() call
    instrumentations: [new OpenAIInstrumentation()],
  });
}
TSEOF
```

`TruSession.init()` registers the app with the Python receiver so it appears in
the dashboard, configures the OpenTelemetry tracer provider, and enables the
OpenAI auto-instrumentation so every LLM call is traced automatically with
model, token counts, and cost data.

## Step 4 — Build a RAG pipeline

```bash
cat > src/rag.ts << 'TSEOF'
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

  // The decorator infers the span name from the method name ("retrieve")
  @instrument<[string], Promise<string[]>>({
    spanType: SpanType.RETRIEVAL,
    attributes: (ret, _err, query) => ({
      [SpanAttributes.RETRIEVAL.QUERY_TEXT]: query as string,
      [SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]: JSON.stringify(ret),
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

  // No @instrument needed — OpenAI auto-instrumentation captures the
  // GENERATION span with model, token counts, and cost automatically.
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

// createTruApp wraps the "query" method with a RECORD_ROOT span automatically
const rag = new SimpleRAG();
export const app = createTruApp(rag, {
  mainMethod: "query",
  mainInput: (question: string) => question,
});
TSEOF
```

Key points:

- `**@instrument(options)**` decorates a method to emit an OTEL span on each
call. The span name is inferred from the method name.
- **OpenAI auto-instrumentation** (registered in step 3) captures `GENERATION`
spans with token/cost data for every `openai.chat.completions.create()` call —
no manual instrumentation needed.
- `**createTruApp(target, options)`** wraps the main entry method with a
`RECORD_ROOT` span, so you don't need `withRecord()` on every call.
- **Token/cost attributes** (`SpanAttributes.COST.`*) are sent from TypeScript;
the Python receiver computes the USD cost automatically.

## Step 5 — Run it

```bash
cat > src/main.ts << 'TSEOF'
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

main().catch(console.error);
TSEOF
```

Run with:

```bash
OPENAI_API_KEY=sk-... pnpm start
```

You'll see an interactive prompt — try a few questions:

```
Q: What is TruLens?
A: TruLens is an open-source library for evaluating and tracking LLM apps...

Q: How does RAG work?
A: RAG combines retrieval with LLM generation...

Q:                          <-- press Enter on an empty line to quit
Flushing spans…
Done.
```

## Step 6 — View in the dashboard

In another terminal, start the dashboard from the repo root using Poetry:

```bash
cd /path/to/trulens                     # <-- adjust to your repo location
poetry run python -c "
from trulens.dashboard import run_dashboard
run_dashboard(port=8501)
"
```

Open [http://localhost:8501](http://localhost:8501). You should see:

- **Leaderboard**: your app with total tokens and cost
- **Records**: one record per question you asked, each containing `retrieve` and
`openai.chat.completions.create` child spans with their captured attributes

## What's next

- Read the full [TypeScript SDK documentation](../../component_guides/instrumentation/typescript.md)
for architecture details and the complete API surface.
- Add [feedback functions](../../component_guides/evaluation/index.md)
(Python-side) to evaluate the traced spans automatically.
