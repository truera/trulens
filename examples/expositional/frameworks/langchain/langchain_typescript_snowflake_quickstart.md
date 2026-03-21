# LangChain.js + Snowflake AI Observability — Pure TypeScript

This quickstart builds the same LangChain.js RAG pipeline from the
[local quickstart](langchain_typescript_quickstart.md) but **exports traces
directly to Snowflake** and triggers **server-side evaluations** — all from
TypeScript, no Python required.

By the end you will have:

- A working LangChain LCEL RAG chain in TypeScript
- Traces exported directly to a Snowflake AI Observability stage
- A Snowflake Run with server-side metrics (e.g. `answer_relevance`)
- Everything visible in the **Snowflake AI Observability UI**

## Prerequisites

- **Node.js 18+** and **pnpm** (or npm/yarn)
- An **OpenAI API key**
- A **Snowflake account** with AI Observability enabled

Set your environment variables:

```bash
export OPENAI_API_KEY=sk-...

# Snowflake connection
export SNOWFLAKE_ACCOUNT=<your-account-identifier>
export SNOWFLAKE_USER=<your-username>
export SNOWFLAKE_PASSWORD=<your-password>
export SNOWFLAKE_DATABASE=<your-database>
export SNOWFLAKE_SCHEMA=<your-schema>
export SNOWFLAKE_WAREHOUSE=<your-warehouse>
```

Or place them in a `.env` file in the project root (loaded automatically via `dotenv`).

## Step 1 — Build the TypeScript SDK packages

```bash
# Install pnpm if you don't have it already
npm install -g pnpm

cd typescript
pnpm install --no-frozen-lockfile
pnpm --filter @trulens/semconv build
pnpm --filter @trulens/core build
pnpm --filter @trulens/instrumentation-langchain build
pnpm --filter @trulens/connectors-snowflake build
```

## Step 2 — Create the project

```bash
cd /path/to/trulens/typescript
mkdir -p examples/langchain-rag-snowflake
cd examples/langchain-rag-snowflake

cat > package.json << 'EOF'
{
  "name": "langchain-rag-snowflake-demo",
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
    "@trulens/core": "workspace:*",
    "@trulens/instrumentation-langchain": "workspace:*",
    "@trulens/connectors-snowflake": "workspace:*",
    "@trulens/semconv": "workspace:*",
    "dotenv": "^17.0.0"
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
    "skipLibCheck": true,
    "outDir": "dist"
  }
}
EOF

mkdir -p src

cd ../..
pnpm install --no-frozen-lockfile
cd examples/langchain-rag-snowflake
```

## Step 3 — Connect to Snowflake and verify

The connection is created once in a shared module and reused by the rest of
the app.

```bash
cat > src/snowflake-connection.ts << 'TSEOF'
import "dotenv/config";
import { SnowflakeConnector } from "@trulens/connectors-snowflake";

export const sfConnector = new SnowflakeConnector({
  account: process.env.SNOWFLAKE_ACCOUNT!,
  username: process.env.SNOWFLAKE_USER!,
  password: process.env.SNOWFLAKE_PASSWORD,
  database: process.env.SNOWFLAKE_DATABASE!,
  schema: process.env.SNOWFLAKE_SCHEMA!,
  warehouse: process.env.SNOWFLAKE_WAREHOUSE!,
});

/**
 * Connect and verify with a simple query.
 */
export async function connectAndVerify(): Promise<void> {
  console.log("Connecting to Snowflake...");

  const rows = (await sfConnector.execute(
    'SELECT CURRENT_USER() AS "user", CURRENT_DATABASE() AS "db", CURRENT_SCHEMA() AS "schema"',
  )) as Array<Record<string, string>>;

  const r = rows[0] ?? {};
  console.log(
    `Connected as ${r.user} | ${r.db}.${r.schema}\n`,
  );
}
TSEOF
```

Test the connection before proceeding:

```bash
cat > src/test-connection.ts << 'TSEOF'
import { sfConnector, connectAndVerify } from "./snowflake-connection.js";

async function main() {
  try {
    await connectAndVerify();
    console.log("Snowflake connection is working.");
  } catch (err) {
    console.error("Connection failed:", err);
  } finally {
    await sfConnector.close();
  }
}

main();
TSEOF

npx tsx src/test-connection.ts
```

Expected output:

```
Connecting to Snowflake...
Connected as JREINI | TRULENS_DEMOS.TYPESCRIPT_LANGCHAIN

Snowflake connection is working.
```

## Step 4 — Initialise TruSession with Snowflake

This wires up the `SnowflakeRunManager` and `TruLensSnowflakeSpanExporter`,
reusing the same connector from Step 3.

```bash
cat > src/trulens-snowflake.ts << 'TSEOF'
import { TruSession } from "@trulens/core";
import { LangChainInstrumentation } from "@trulens/instrumentation-langchain";
import {
  TruLensSnowflakeSpanExporter,
  SnowflakeRunManager,
} from "@trulens/connectors-snowflake";
import { sfConnector } from "./snowflake-connection.js";

const APP_NAME = "langchain-typescript-rag";
const APP_VERSION = "v1";
export const RUN_NAME = `langchain_ts_rag_run_${Date.now()}`;

export const runManager = new SnowflakeRunManager({
  connector: sfConnector,
});

export async function initSession(): Promise<TruSession> {
  return TruSession.init({
    appName: APP_NAME,
    appVersion: APP_VERSION,
    runName: RUN_NAME,
    exporter: new TruLensSnowflakeSpanExporter({
      connector: sfConnector,
    }),
    instrumentations: [new LangChainInstrumentation()],

    onInit: async () => {
      const objectName = await runManager.ensureExternalAgent(
        APP_NAME,
        APP_VERSION,
      );
      console.log(`External Agent: ${objectName}`);

      await runManager.createRun({
        objectName: APP_NAME,
        appVersion: APP_VERSION,
        runName: RUN_NAME,
        datasetName: "langchain_ts_rag_queries",
        sourceType: "DATAFRAME",
      });
      console.log(`Run created: ${RUN_NAME}`);
    },

    onShutdown: async (count) => {
      console.log(
        `Finalizing run (${count} records traced)...`,
      );
      await runManager.finalizeRun({
        objectName: APP_NAME,
        appVersion: APP_VERSION,
        runName: RUN_NAME,
        inputRecordsCount: count,
      });
    },
  });
}
TSEOF
```

What happens here:

1. **`onInit`** creates the External Agent + Run entity in Snowflake before
   any traces are sent.
2. **`TruLensSnowflakeSpanExporter`** serialises spans to protobuf, uploads
   them to a Snowflake stage, and calls `SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN`
   per batch.
3. **`onShutdown`** finalises the run (triggers `START_INGESTION`).
4. **`runName`** is automatically stamped on every `RECORD_ROOT` span so
   Snowflake can associate traces with the run.

## Step 5 — Build the LangChain LCEL RAG chain

Zero TruLens imports — the knowledge base covers how the TruLens TypeScript SDK
works, its Snowflake integration, span types, and more. This lets the demo
answer questions about itself.

```bash
cat > src/rag.ts << 'TSEOF'
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { StringOutputParser } from "@langchain/core/output_parsers";
import type { Document } from "@langchain/core/documents";

const DOCS = [
  // --- Overview ---
  "TruLens is an open-source library for evaluating and tracking LLM applications. The TypeScript SDK (@trulens/core, @trulens/semconv, @trulens/connectors-snowflake, @trulens/instrumentation-langchain) brings these capabilities to Node.js developers and can export traces directly to Snowflake without any Python code.",

  // --- TruSession lifecycle ---
  "TruSession is the main entry point. TruSession.init() creates a singleton that sets up an OpenTelemetry NodeTracerProvider with a BatchSpanProcessor wired to the configured SpanExporter. It attaches resource attributes (app_name, app_version, app_id) to every span produced by this provider. The init method calls the optional onInit hook (used to create External Agents and Runs on Snowflake), then enables all registered instrumentations. TruSession.shutdown() disables instrumentations, calls provider.forceFlush() to drain the BatchSpanProcessor queue, invokes the onShutdown hook with the record count, and finally calls provider.shutdown().",

  // --- createTruApp and withRecord ---
  "createTruApp() wraps an application object with a JavaScript Proxy so that calling the designated mainMethod (e.g. 'query') automatically invokes withRecord(). withRecord() creates a RECORD_ROOT span, generates a unique record_id (UUID), stores it as both a span attribute (ai.observability.record_id) and in OTEL Baggage so child spans inherit it, and increments TruSession.inputRecordsCount. The run_name from TruSession is also stamped on the RECORD_ROOT span as ai.observability.run.name.",

  // --- Span types and semantic conventions ---
  "TruLens defines span types via @trulens/semconv SpanAttributes. Available types: RECORD_ROOT (top-level invocation), RETRIEVAL (vector search), GENERATION (LLM call), TOOL, AGENT, WORKFLOW_STEP, GRAPH_NODE, RERANKING, and UNKNOWN. Each span carries ai.observability.span_type, ai.observability.record_id, and optionally ai.observability.run.name. Resource-level attributes include trulens.app.name, trulens.app.version, and trulens.app.id.",

  // --- LangChain.js auto-instrumentation ---
  "The @trulens/instrumentation-langchain package patches LangChain's CallbackManager._configureSync to inject a TruLensLangChainTracer into every callback chain. This tracer extends BaseTracer and maps LangChain run types to TruLens span types: 'llm' and 'chat_model' become GENERATION, 'retriever' becomes RETRIEVAL, 'tool' becomes TOOL, and 'chain' becomes WORKFLOW_STEP. On run creation, the tracer opens an OTEL span nested under the parent run's span context. On run end, it extracts outputs: for GENERATION spans it captures token usage (totalTokens, promptTokens, completionTokens) and model name; for RETRIEVAL spans it captures the retrieved documents and query text.",

  // --- RECORD_ID propagation across callbacks ---
  "LangChain callbacks run outside the normal OTEL async context, so child spans can lose the baggage-propagated record_id. The TruLensLangChainTracer solves this by maintaining an internal runRecordIds cache keyed by LangChain run ID. When a span starts, the tracer first checks OTEL baggage for record_id, then falls back to the parent run's cached value. This ensures every span in the trace carries the same record_id even when baggage propagation breaks.",

  // --- Protobuf serialization for Snowflake ---
  "The TruLensSnowflakeSpanExporter serializes spans for Snowflake ingestion. Each ReadableSpan is individually converted to an OTLP protobuf message using sdkSpanToOtlpSpan() from @opentelemetry/otlp-transformer, then encoded via the protobuf Span.encode().finish() method. Each encoded span is preceded by a varint length delimiter (base-128 encoding of the byte length). All spans are concatenated into a single .pb file. This per-span length-delimited format matches what the Python TruLens exporter produces and what Snowflake's ingestion expects.",

  // --- Stage upload and ingestion ---
  "After writing the .pb file to a temp directory, the exporter uploads it to Snowflake in two SQL steps: (1) CREATE TEMP STAGE IF NOT EXISTS trulens_spans — creates a session-scoped temporary stage; (2) PUT file:///tmp/trulens_spans_<timestamp>.pb @trulens_spans — uploads the file (Snowflake auto-compresses to .gz). After upload, the local temp file is deleted. The exporter then calls SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN with the stage file path (@DB.SCHEMA.trulens_spans/<filename>.gz), object_name (fully qualified agent name), object_version, run_name, and the evaluation phase INGESTION_MULTIPLE_BATCHES. This tells Snowflake to read and parse the protobuf spans from the stage.",

  // --- Span grouping in the exporter ---
  "Before export, spans are grouped by (app_name, app_version, run_name, input_records_count) extracted from resource and span attributes. Each group is exported as a separate .pb file with its own PUT and SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN call. This ensures spans are correctly associated with the right External Agent version and Run in Snowflake.",

  // --- Cross-batch RECORD_ID / RUN_NAME propagation ---
  "The BatchSpanProcessor may split a single trace across multiple export batches — the RECORD_ROOT span might arrive in one batch while child spans arrive in the next. The exporter maintains a persistent _traceAttrCache (Map<traceId, {recordId, runName}>) that survives across export calls. On each batch, propagateTraceAttrs() first scans all spans to populate the cache with any record_id or run_name values found, then iterates again to inject missing values into child spans. To preserve ReadableSpan prototype methods (especially spanContext()), enriched spans are created using Object.create(span, { attributes: { value: mergedAttrs } }) rather than object spread.",

  // --- External Agent creation ---
  "Before any traces are sent, the SnowflakeRunManager creates a Snowflake External Agent via SQL: CREATE EXTERNAL AGENT \"<APP_NAME>\" WITH VERSION \"<version>\". If the agent already exists (checked via SHOW EXTERNAL AGENTS), it adds the version with ALTER EXTERNAL AGENT IF EXISTS \"<name>\" ADD VERSION \"<version>\". The fully qualified name (DATABASE.SCHEMA.AGENT_NAME) is returned and used in all subsequent Run API calls.",

  // --- Run creation ---
  "A Run is created by calling SELECT SYSTEM$AIML_RUN_OPERATION('CREATE', <json_payload>) where the JSON payload contains: object_name (fully qualified agent name), object_type ('External Agent'), object_version, run_name (unique per session, e.g. langchain_rag_run_<timestamp>), run_metadata (labels, llm_judge_name defaulting to 'llama3.1-70b', mode 'APP_INVOCATION'), and source_info (dataset name, source_type 'DATAFRAME'). The Run entity is what ties together all the ingested traces and computed metrics.",

  // --- Run finalization (START_INGESTION) ---
  "After all queries are done and spans are flushed, the onShutdown hook calls runManager.finalizeRun(). This calls SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN with evaluation phase START_INGESTION, signaling Snowflake to begin processing the uploaded protobuf files. The input_record_count parameter tells Snowflake how many records to expect. After finalization, waitForIngestion() polls SYSTEM$AIML_RUN_OPERATION('GET') every 3 seconds until the run status transitions to INVOCATION_COMPLETED (or COMPLETED, PARTIALLY_COMPLETED, COMPUTATION_IN_PROGRESS).",

  // --- Run status parsing ---
  "Run status is determined by parsing the JSON response from SYSTEM$AIML_RUN_OPERATION('GET'). The response contains run_metadata with invocations (keyed by timestamp) and metrics sections. If no invocations exist, the status is CREATED. The latest invocation's completion_status.status maps to: COMPLETED → INVOCATION_COMPLETED, PARTIALLY_COMPLETED → INVOCATION_PARTIALLY_COMPLETED, STARTED → INVOCATION_IN_PROGRESS, FAILED → FAILED. If metrics exist, the overall status considers whether all metrics completed, failed, or are in progress.",

  // --- Server-side metric computation ---
  "After ingestion completes, runManager.computeMetrics() calls SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN with evaluation phase COMPUTE_METRICS and an array of metric names (e.g. ['answer_relevance']). Available metrics include answer_relevance (is the answer relevant to the question), context_relevance (are retrieved contexts relevant), and groundedness (is the answer grounded in the contexts). Snowflake runs these using built-in LLM judges (default model: llama3.1-70b). Computation is asynchronous — the SQL call returns immediately and metrics appear in the AI Observability UI when done.",

  // --- SnowflakeConnector connection management ---
  "The SnowflakeConnector uses the snowflake-sdk npm package. Connections are lazily created on first use via getConnection(). A promise-based mutex (_connecting) prevents race conditions: if multiple callers invoke getConnection() concurrently (e.g. the BatchSpanProcessor export and the main thread), only one connection attempt is made. The first caller sets _connecting to the _connect() promise; subsequent callers await the same promise. Once resolved, the connection is cached in _connection. The connector configures snowflake.configure({ logLevel: 'WARN' }) to suppress verbose INFO-level SDK logs.",

  // --- Authentication options ---
  "SnowflakeConnector supports multiple authentication methods via the authenticator option: 'SNOWFLAKE' (default, password-based), 'SNOWFLAKE_JWT' (key-pair with privateKey/privateKeyPath), 'OAUTH' (pre-obtained token), 'OAUTH_AUTHORIZATION_CODE' (browser-based OAuth flow), and 'EXTERNALBROWSER' (SSO via browser). For browser-based flows (EXTERNALBROWSER, OAUTH_AUTHORIZATION_CODE), the connector uses connectAsync() instead of connect() and sets disableConsoleLogin=false to use the direct console login URL.",

  // --- isTruLensSpan filter ---
  "The exporter filters incoming spans using isTruLensSpan(), which checks whether a span has either trulens.app.name in its resource attributes or ai.observability.span_type in its span attributes. This prevents non-TruLens spans (e.g. from other OTEL instrumentations) from being exported to Snowflake. In dryRun mode, all spans pass through unfiltered.",

  // --- End-to-end flow ---
  "The complete flow: (1) TruSession.init() creates NodeTracerProvider + BatchSpanProcessor + Exporter, calls onInit to create External Agent and Run in Snowflake; (2) createTruApp() wraps the app so each call creates a RECORD_ROOT span with record_id and run_name; (3) LangChainInstrumentation auto-creates child spans for each LangChain component; (4) BatchSpanProcessor batches spans and calls exporter.export(); (5) exporter propagates record_id/run_name, serializes to protobuf, PUTs to stage, calls INGESTION_MULTIPLE_BATCHES; (6) session.shutdown() flushes remaining spans and calls onShutdown which triggers START_INGESTION; (7) waitForIngestion() polls until done; (8) computeMetrics() triggers server-side evaluation.",
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
  const retriever = vectorStore.asRetriever({ k: 4 });

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

## Step 6 — Run the app and trigger evaluations

```bash
cat > src/main.ts << 'TSEOF'
import * as readline from "node:readline/promises";
import { stdin, stdout } from "node:process";
import { sfConnector, connectAndVerify } from "./snowflake-connection.js";
import { initSession, runManager, RUN_NAME } from "./trulens-snowflake.js";
import { buildChain } from "./rag.js";
import { createTruApp } from "@trulens/core";

async function main() {
  await connectAndVerify();

  // Initialise TruSession (creates External Agent + Run)
  const session = await initSession();
  console.log(`TruSession ready — app="${session.appName}"`);

  const chain = await buildChain();
  console.log("LangChain RAG chain built.\n");

  const app = createTruApp(
    { query: (q: string) => chain.invoke(q) },
    { mainMethod: "query", mainInput: (q: string) => q },
  );

  console.log("Ask about the TruLens TypeScript SDK. Try:");
  console.log("  - How are spans serialized and uploaded to Snowflake?");
  console.log("  - How does record_id propagation work across batches?");
  console.log("  - What SQL calls create a Snowflake Run?");
  console.log("  - How does the LangChain tracer map run types to span types?");
  console.log("  - What happens during session.shutdown()?\n");
  const rl = readline.createInterface({ input: stdin, output: stdout });

  try {
    while (true) {
      const question = await rl.question("Q: ");
      if (!question.trim()) break;
      console.log(`A: ${await app.query(question)}\n`);
    }
  } finally {
    rl.close();
  }

  console.log("\nFlushing spans and finalizing run...");
  await session.shutdown();

  console.log("Waiting for Snowflake ingestion to complete...");
  await runManager.waitForIngestion({
    objectName: "langchain-typescript-rag",
    runName: RUN_NAME,
  });
  console.log("Ingestion complete.");

  console.log("Computing metrics: answer_relevance, context_relevance, groundedness...");
  await runManager.computeMetrics({
    objectName: "langchain-typescript-rag",
    appVersion: "v1",
    runName: RUN_NAME,
    metrics: ["answer_relevance", "context_relevance", "groundedness"],
  });

  console.log("Waiting for metric computation to complete...");
  const finalStatus = await runManager.waitForMetrics({
    objectName: "langchain-typescript-rag",
    runName: RUN_NAME,
  });
  console.log(`Metrics ${finalStatus.toLowerCase()}.`);

  await sfConnector.close();
  console.log("\nDone! Open Snowflake AI Observability UI to see traces and evaluations.");
}

main().catch(console.error);
TSEOF
```

Run with:

```bash
pnpm start
```

Try a few questions, then press Enter on an empty line to finish:

```
Connecting to Snowflake...
Connected as JREINI | TRULENS_DEMOS.TYPESCRIPT_LANGCHAIN

External Agent: TRULENS_DEMOS.TYPESCRIPT_LANGCHAIN.LANGCHAIN-TYPESCRIPT-RAG
Run created: langchain_ts_rag_run_1740000000000
TruSession ready — app="langchain-typescript-rag"
LangChain RAG chain built.

Ask about the TruLens TypeScript SDK. Try:
  - How are spans serialized and uploaded to Snowflake?
  - How does record_id propagation work across batches?
  - What SQL calls create a Snowflake Run?
  - How does the LangChain tracer map run types to span types?
  - What happens during session.shutdown()?

Q: How are spans uploaded to Snowflake?
A: Each span is individually serialized as a protobuf message preceded by
   a varint length delimiter, concatenated into a .pb file, uploaded via
   PUT to a temporary stage, then ingested via
   SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN with INGESTION_MULTIPLE_BATCHES.

Q:

Flushing spans and finalizing run...
Finalizing run (1 records traced)...
Waiting for Snowflake ingestion to complete...
Ingestion complete.
Computing metrics: answer_relevance, context_relevance, groundedness...
Waiting for metric computation to complete...
Metrics completed.

Done! Open Snowflake AI Observability UI to see traces and evaluations.
```

## Step 7 — View in Snowflake AI Observability

Navigate to the Snowflake AI Observability UI in your Snowflake account.
You should see:

- **External Agent**: `LANGCHAIN-TYPESCRIPT-RAG` with version `v1`
- **Run**: `langchain_ts_rag_run_...` with invocation records
- **Traces**: one trace per query, each containing child spans for the
retriever and LLM calls with captured attributes (token usage, retrieved
contexts, etc.)
- **Metrics**: `answer_relevance` scores for each record (computed
asynchronously by Snowflake)

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│  TypeScript App                                                 │
│                                                                 │
│  TruSession.init()                                              │
│  ├─ onInit hook:                                                │
│  │   ├─ CREATE EXTERNAL AGENT "LANGCHAIN-TYPESCRIPT-RAG" …      │
│  │   └─ SYSTEM$AIML_RUN_OPERATION('CREATE', run_config)         │
│  │                                                              │
│  App runs (LangChain auto-instrumented)                         │
│  └─ TruLensSnowflakeSpanExporter                               │
│      ├─ PUT protobuf to stage                                   │
│      └─ SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN (per batch)         │
│                                                                 │
│  session.shutdown()                                             │
│  └─ onShutdown hook:                                            │
│      └─ SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN (START_INGESTION)   │
│                                                                 │
│  runManager.waitForIngestion()                                  │
│  └─ polls SYSTEM$AIML_RUN_OPERATION('GET') until ready          │
│                                                                 │
│  runManager.computeMetrics([...])                               │
│  └─ SYSTEM$EXECUTE_AI_OBSERVABILITY_RUN (COMPUTE_METRICS)       │
│                                                                 │
│  runManager.waitForMetrics()                                    │
│  └─ polls SYSTEM$AIML_RUN_OPERATION('GET') until metrics done   │
└─────────────────────────────────────────────────────────────────┘
```

## Comparison with the local quickstart

- **Local quickstart** — spans go to a Python OTLP receiver, evaluations
run in Python using `Metric` + `Selector`, results visible in the
Streamlit dashboard.
- **Snowflake quickstart** — spans go directly to Snowflake, evaluations
are triggered server-side (Snowflake-hosted), results visible in the
native Snowflake AI Observability UI. No Python required.

Both quickstarts share the same app code (`rag.ts`) and the same
`LangChainInstrumentation` for automatic tracing.

## Available server-side metrics

When calling `computeMetrics`, you can pass any Snowflake-hosted metric name:

- `answer_relevance` — how relevant is the answer to the question
- `context_relevance` — how relevant are the retrieved contexts to the question
- `groundedness` — is the answer grounded in the retrieved contexts

These are evaluated entirely on Snowflake's side using built-in LLM judges.
