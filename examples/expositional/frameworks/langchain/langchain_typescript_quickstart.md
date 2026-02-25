# LangChain.js + TruLens TypeScript Quickstart

This quickstart builds a RAG pipeline with LangChain.js (LCEL) and instruments
it with TruLens — **without adding any TruLens-specific code to the app
itself**. The `@trulens/instrumentation-langchain` package hooks into
LangChain's native callback system to emit OpenTelemetry spans for every chain,
retriever, and LLM call automatically.

By the end you will have:

- A working LangChain LCEL RAG chain in TypeScript
- Automatic tracing of retriever, LLM, and chain spans with token/cost data
- Evaluation scores (answer relevance, context relevance, groundedness)
- Everything visible in the TruLens dashboard

## Prerequisites

- **Node.js 18+** and **pnpm** (or npm/yarn)
- **Python 3.9+** and **Poetry** (`pip install poetry`)
- An **OpenAI API key**

Set your API key once so both TypeScript and Python steps can use it:

```bash
export OPENAI_API_KEY=sk-...
```

## Step 1 — Set up the Python environment and start the OTLP receiver

The TypeScript SDK sends OpenTelemetry spans to a lightweight receiver built
into the Python `TruSession`. The OTLP receiver is part of the TruLens
development branch, so you need to install from source.

### 1a. Create a fresh virtual environment and install TruLens

From the root of the TruLens repository:

```bash
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

poetry install --with dev
```

### 1b. Start the OTLP receiver

In the same terminal (with the virtualenv active):

```bash
poetry run python -c "
from trulens.core import TruSession

session = TruSession()
session.reset_database()
session.start_otlp_receiver(port=4318)
print('OTLP receiver listening on http://localhost:4318')
input('Press Enter to stop...')
"
```

Leave this terminal running — the TypeScript app will send spans to it.

## Step 2 — Build the TypeScript SDK packages

Open a **new terminal**. The TypeScript packages live in the `typescript/`
directory of the repo and are managed with pnpm workspaces.

```bash
npm install -g pnpm    # if not already installed

cd typescript
pnpm install --no-frozen-lockfile
pnpm --filter @trulens/semconv build
pnpm --filter @trulens/core build
pnpm --filter @trulens/instrumentation-langchain build
```

## Step 3 — Create a project

Create a project inside the monorepo's `typescript/examples/` directory so
pnpm workspace resolution can find the local `@trulens/*` packages:

```bash
cd /path/to/trulens/typescript
mkdir -p examples/langchain-rag
cd examples/langchain-rag

cat > package.json << 'EOF'
{
  "name": "langchain-rag-demo",
  "version": "0.1.0",
  "private": true,
  "type": "module",
  "scripts": {
    "start": "tsx src/main.ts"
  },
  "dependencies": {
    "@langchain/core": "^0.3.0",
    "@langchain/openai": "^0.4.0",
    "langchain": "^0.3.0",
    "@opentelemetry/exporter-trace-otlp-http": "^0.57.0",
    "@trulens/core": "workspace:*",
    "@trulens/instrumentation-langchain": "workspace:*",
    "@trulens/semconv": "workspace:*"
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
cd examples/langchain-rag
```

## Step 4 — Initialise TruSession with LangChain auto-instrumentation

```bash
cat > src/trulens.ts << 'TSEOF'
import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import { TruSession } from "@trulens/core";
import { LangChainInstrumentation } from "@trulens/instrumentation-langchain";

const ENDPOINT = process.env.TRULENS_ENDPOINT ?? "http://localhost:4318";

export async function initSession(): Promise<TruSession> {
  return TruSession.init({
    appName: "langchain-rag",
    appVersion: "v1",
    exporter: new OTLPTraceExporter({ url: `${ENDPOINT}/v1/traces` }),
    endpoint: ENDPOINT,
    instrumentations: [new LangChainInstrumentation()],
  });
}
TSEOF
```

`TruSession.init()` registers the app with the Python receiver (via `endpoint`)
and activates the `LangChainInstrumentation`, which hooks into LangChain's
`CallbackManager` so every chain, retriever, and LLM call is traced
automatically — no decorators needed on your app code.

## Step 5 — Build a LangChain LCEL RAG chain

```bash
cat > src/rag.ts << 'TSEOF'
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import type { Document } from "@langchain/core/documents";

const DOCS = [
  "TruLens is an open-source library for evaluating and tracking LLM apps.",
  "RAG combines retrieval with LLM generation for grounded answers.",
  "OpenTelemetry is a vendor-neutral observability framework.",
  "LangChain Expression Language (LCEL) lets you compose chains declaratively.",
];

function formatDocs(docs: Document[]): string {
  return docs.map((d, i) => `[${i + 1}] ${d.pageContent}`).join("\n\n");
}

export async function buildChain() {
  const embeddings = new OpenAIEmbeddings();
  const vectorStore = await MemoryVectorStore.fromTexts(
    DOCS,
    DOCS.map((_, i) => ({ id: i })),
    embeddings,
  );
  const retriever = vectorStore.asRetriever({ k: 2 });

  const prompt = ChatPromptTemplate.fromTemplate(
    `Answer the question using only the following context:

{context}

Question: {question}`,
  );

  const llm = new ChatOpenAI({ model: "gpt-4o-mini", temperature: 0 });

  const chain = RunnableSequence.from([
    {
      context: retriever.pipe(formatDocs),
      question: new RunnablePassthrough(),
    },
    prompt,
    llm,
    new StringOutputParser(),
  ]);

  return chain;
}
TSEOF
```

Notice: **zero TruLens imports** in the app logic above. The chain is pure
LangChain LCEL — retriever, prompt, LLM, and output parser piped together.
The entry point (`main.ts`) still uses `createTruApp` to produce the top-level
`RECORD_ROOT` span, but TruLens observes all inner steps (retriever, LLM,
chain) entirely through LangChain's callback system.

## Step 6 — Run it

```bash
cat > src/main.ts << 'TSEOF'
import * as readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { initSession } from "./trulens.js";
import { buildChain } from "./rag.js";
import { createTruApp } from "@trulens/core";

async function main() {
  const session = await initSession();
  console.log(`TruSession ready — app="${session.appName}"`);

  const chain = await buildChain();
  console.log("LangChain RAG chain built.\n");

  const app = createTruApp(
    { query: (q: string) => chain.invoke(q) },
    { mainMethod: "query", mainInput: (q: string) => q },
  );

  console.log("Type a question and press Enter. Empty line or Ctrl+C to quit.\n");
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
pnpm start
```

Try a few questions:

```
Q: What is TruLens?
A: TruLens is an open-source library for evaluating and tracking LLM apps...

Q: How does RAG work?
A: RAG combines retrieval with LLM generation...

Q:
Flushing spans…
Done.
```

## Step 7 — Run evaluations on your traces

Evaluations run in Python using the same `TruSession` that stores the spans.
Open a new terminal (with the Poetry virtualenv active) and run:

```bash
cd /path/to/trulens
poetry run python << 'PYEOF'
from trulens.core import TruSession, Metric
from trulens.core.feedback.selector import Selector
from trulens.providers.openai import OpenAI
from trulens.otel.semconv.trace import SpanAttributes

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
    app_name="langchain-rag",
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

## Step 8 — View in the dashboard

```bash
cd /path/to/trulens
poetry run python -c "
from trulens.dashboard import run_dashboard
run_dashboard(port=8501)
"
```

Open [http://localhost:8501](http://localhost:8501). You should see:

- **Leaderboard**: your `langchain-rag` app with token counts, cost, and
  evaluation scores
- **Records**: one record per question, each containing child spans for the
  retriever and LLM with their captured attributes

## Comparison with the raw TypeScript quickstart

Both quickstarts use the same `TruSession.init()` for app registration and
`createTruApp()` for the top-level `RECORD_ROOT` span. The difference is how
the inner pipeline steps (retrieval, generation) get traced:

- **Raw OpenAI quickstart** — app code needs `@instrument` decorators on
  each pipeline method (e.g. `retrieve`, `generate`) to emit retrieval spans.
  `OpenAIInstrumentation` auto-captures LLM calls.
- **LangChain quickstart** — `LangChainInstrumentation` hooks into the
  callback system so retriever, LLM, and chain spans are all captured
  automatically. The app logic (`rag.ts`) has zero TruLens imports.
