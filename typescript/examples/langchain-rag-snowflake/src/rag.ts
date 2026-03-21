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
