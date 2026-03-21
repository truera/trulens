import {
  beforeAll,
  beforeEach,
  describe,
  expect,
  it,
} from "vitest";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
  NodeTracerProvider,
} from "@opentelemetry/sdk-trace-node";
import { SpanAttributes, SpanType } from "@trulens/semconv";
import { randomUUID } from "node:crypto";

import { TruLensLangChainTracer } from "../tracer.js";

const exporter = new InMemorySpanExporter();

beforeAll(() => {
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
});

beforeEach(() => {
  exporter.reset();
});

// ---------------------------------------------------------------------------
// Helpers to build minimal mock objects matching LangChain's interfaces
// ---------------------------------------------------------------------------

function serialized(name: string) {
  return { lc: 1, type: "not_implemented", id: [name] };
}

describe("TruLensLangChainTracer", () => {
  it("creates a RETRIEVAL span for retriever runs", async () => {
    const tracer = new TruLensLangChainTracer();
    const runId = randomUUID();

    await tracer.handleRetrieverStart(
      serialized("TestRetriever"),
      "What is RAG?",
      runId
    );

    await tracer.handleRetrieverEnd(
      [
        { pageContent: "RAG is ...", metadata: {} },
        { pageContent: "It works by ...", metadata: {} },
      ],
      runId
    );

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0]!;
    expect(span.attributes[SpanAttributes.SPAN_TYPE]).toBe(
      SpanType.RETRIEVAL
    );
    expect(span.attributes[SpanAttributes.RETRIEVAL.QUERY_TEXT]).toBe(
      "What is RAG?"
    );
    const contexts = span.attributes[
      SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS
    ] as string[];
    expect(contexts).toEqual(["RAG is ...", "It works by ..."]);
    expect(span.attributes[SpanAttributes.RETRIEVAL.NUM_CONTEXTS]).toBe(2);
  });

  it("creates a GENERATION span for LLM runs with token usage", async () => {
    const tracer = new TruLensLangChainTracer();
    const runId = randomUUID();

    await tracer.handleLLMStart(
      serialized("ChatOpenAI"),
      ["Hello!"],
      runId,
      undefined,
      { invocation_params: { model: "gpt-4o" } }
    );

    await tracer.handleLLMEnd(
      {
        generations: [[{ text: "Hi!" }]],
        llmOutput: {
          tokenUsage: {
            totalTokens: 50,
            promptTokens: 20,
            completionTokens: 30,
          },
        },
      },
      runId
    );

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);

    const span = spans[0]!;
    expect(span.attributes[SpanAttributes.SPAN_TYPE]).toBe(
      SpanType.GENERATION
    );
    expect(span.attributes[SpanAttributes.COST.MODEL]).toBe("gpt-4o");
    expect(span.attributes[SpanAttributes.COST.NUM_TOKENS]).toBe(50);
    expect(span.attributes[SpanAttributes.COST.NUM_PROMPT_TOKENS]).toBe(20);
    expect(span.attributes[SpanAttributes.COST.NUM_COMPLETION_TOKENS]).toBe(
      30
    );
  });

  it("creates a TOOL span for tool runs", async () => {
    const tracer = new TruLensLangChainTracer();
    const runId = randomUUID();

    await tracer.handleToolStart(
      serialized("Calculator"),
      "2 + 2",
      runId
    );

    await tracer.handleToolEnd("4", runId);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes[SpanAttributes.SPAN_TYPE]).toBe(
      SpanType.TOOL
    );
  });

  it("creates a WORKFLOW_STEP span for chain runs", async () => {
    const tracer = new TruLensLangChainTracer();
    const runId = randomUUID();

    await tracer.handleChainStart(
      serialized("RetrievalQAChain"),
      { input: "test" },
      runId
    );

    await tracer.handleChainEnd({ output: "result" }, runId);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes[SpanAttributes.SPAN_TYPE]).toBe(
      SpanType.WORKFLOW_STEP
    );
  });

  it("records errors from failed runs", async () => {
    const tracer = new TruLensLangChainTracer();
    const runId = randomUUID();

    await tracer.handleChainStart(
      serialized("FailChain"),
      { input: "fail" },
      runId
    );

    await tracer.handleChainError(new Error("chain failed"), runId);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes[SpanAttributes.CALL.ERROR]).toContain(
      "chain failed"
    );
  });

  it("nests child spans under parent spans", async () => {
    const tracer = new TruLensLangChainTracer();
    const parentId = randomUUID();
    const childId = randomUUID();

    // Start parent chain
    await tracer.handleChainStart(
      serialized("MainChain"),
      { input: "test" },
      parentId
    );

    // Start child retriever under parent
    await tracer.handleRetrieverStart(
      serialized("Retriever"),
      "query",
      childId,
      parentId
    );

    // End child then parent
    await tracer.handleRetrieverEnd(
      [{ pageContent: "doc", metadata: {} }],
      childId
    );
    await tracer.handleChainEnd({ output: "done" }, parentId);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(2);

    const parentSpan = spans.find((s) => s.name === "MainChain")!;
    const childSpan = spans.find((s) => s.name === "Retriever")!;

    expect(parentSpan).toBeDefined();
    expect(childSpan).toBeDefined();
    expect(childSpan.parentSpanId).toBe(parentSpan.spanContext().spanId);
  });

  it("records input kwargs and output return", async () => {
    const tracer = new TruLensLangChainTracer();
    const runId = randomUUID();

    await tracer.handleChainStart(
      serialized("MyChain"),
      { question: "What?" },
      runId
    );

    await tracer.handleChainEnd({ answer: "42" }, runId);

    const spans = exporter.getFinishedSpans();
    const span = spans[0]!;
    expect(JSON.parse(String(span.attributes[SpanAttributes.CALL.KWARGS]))).toEqual({
      question: "What?",
    });
    expect(JSON.parse(String(span.attributes[SpanAttributes.CALL.RETURN]))).toEqual({
      answer: "42",
    });
  });

  it("sets CALL.FUNCTION to the run name", async () => {
    const tracer = new TruLensLangChainTracer();
    const runId = randomUUID();

    await tracer.handleToolStart(
      serialized("WebSearch"),
      "query",
      runId
    );
    await tracer.handleToolEnd("result", runId);

    const spans = exporter.getFinishedSpans();
    expect(spans[0]!.attributes[SpanAttributes.CALL.FUNCTION]).toBe(
      "WebSearch"
    );
  });
});
