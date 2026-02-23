import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
  NodeTracerProvider,
} from "@opentelemetry/sdk-trace-node";

import { instrument, withRecord } from "../instrument.js";
import { SpanAttributes, SpanType } from "@trulens/semconv";

// ---------------------------------------------------------------------------
// Test setup: set up the provider once — teardown/re-register cycling causes
// OTEL global state corruption for async spans.
// ---------------------------------------------------------------------------

const exporter = new InMemorySpanExporter();

beforeAll(() => {
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
});

// Reset finished spans before each test so they don't accumulate.
beforeEach(() => {
  exporter.reset();
});

// ---------------------------------------------------------------------------
// instrument()
// ---------------------------------------------------------------------------

describe("instrument()", () => {
  it("creates a span for a sync function", () => {
    const add = instrument((a: number, b: number) => a + b, {
      spanType: SpanType.TOOL,
    });

    const result = add(2, 3);
    expect(result).toBe(5);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    const span = spans[0]!;
    expect(span.attributes[SpanAttributes.SPAN_TYPE]).toBe(SpanType.TOOL);
  });

  it("creates a span for an async function", async () => {
    const fetchData = instrument(async (url: string) => `data from ${url}`, {
      spanType: SpanType.RETRIEVAL,
    });

    const result = await fetchData("https://example.com");
    expect(result).toBe("data from https://example.com");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes[SpanAttributes.SPAN_TYPE]).toBe(
      SpanType.RETRIEVAL
    );
  });

  it("records an error on a sync function and re-throws", () => {
    const boom = instrument(
      () => {
        throw new Error("kaboom");
      },
      { spanType: SpanType.TOOL }
    );

    expect(() => boom()).toThrow("kaboom");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes[SpanAttributes.CALL.ERROR]).toContain("kaboom");
  });

  it("records an error on an async function and re-throws", async () => {
    const boom = instrument(
      async () => {
        throw new Error("async-kaboom");
      },
      { spanType: SpanType.TOOL }
    );

    await expect(boom()).rejects.toThrow("async-kaboom");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes[SpanAttributes.CALL.ERROR]).toContain(
      "async-kaboom"
    );
  });

  it("resolves lambda attributes (ret + named args)", async () => {
    const retrieve = instrument(
      async (_query: string): Promise<string[]> => ["ctx1", "ctx2"],
      {
        spanType: SpanType.RETRIEVAL,
        attributes: (ret, _err, _query) => ({
          [SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]: JSON.stringify(ret),
        }),
      }
    );

    await retrieve("what is RAG?");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(
      spans[0]!.attributes[SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]
    ).toBe(JSON.stringify(["ctx1", "ctx2"]));
  });

  it("resolves 'return' string key to the function return value", () => {
    const fn = instrument((a: number) => a * 2, {
      spanType: SpanType.UNKNOWN,
      attributes: {
        "my.result": "return",
      },
    });

    fn(21);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes["my.result"]).toBe(42);
  });
});

// ---------------------------------------------------------------------------
// withRecord()
// ---------------------------------------------------------------------------

describe("withRecord()", () => {
  it("creates a RECORD_ROOT span with input and output", async () => {
    const output = await withRecord(async () => "hello world", {
      input: "say hello",
    });

    expect(output).toBe("hello world");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    const span = spans[0]!;
    expect(span.attributes[SpanAttributes.SPAN_TYPE]).toBe(
      SpanType.RECORD_ROOT
    );
    expect(span.attributes[SpanAttributes.RECORD_ROOT.INPUT]).toBe("say hello");
    expect(span.attributes[SpanAttributes.RECORD_ROOT.OUTPUT]).toBe(
      "hello world"
    );
  });

  it("records errors and re-throws", async () => {
    await expect(
      withRecord(async () => {
        throw new Error("boom");
      })
    ).rejects.toThrow("boom");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(
      spans[0]!.attributes[SpanAttributes.RECORD_ROOT.ERROR]
    ).toContain("boom");
  });

  it("nested instrument() spans are children of the RECORD_ROOT span", async () => {
    const retrieve = instrument(async () => ["ctx"], {
      spanType: SpanType.RETRIEVAL,
    });

    await withRecord(async () => {
      await retrieve();
      return "answer";
    });

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(2);

    const rootSpan = spans.find(
      (s) =>
        s.attributes[SpanAttributes.SPAN_TYPE] === SpanType.RECORD_ROOT
    )!;
    const retrieveSpan = spans.find(
      (s) =>
        s.attributes[SpanAttributes.SPAN_TYPE] === SpanType.RETRIEVAL
    )!;

    expect(rootSpan).toBeDefined();
    expect(retrieveSpan).toBeDefined();
    expect(retrieveSpan.parentSpanId).toBe(
      rootSpan.spanContext().spanId
    );
  });
});
