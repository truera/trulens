import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
  NodeTracerProvider,
} from "@opentelemetry/sdk-trace-node";

import {
  instrument,
  instrumentDecorator,
  withRecord,
} from "../instrument.js";
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
// serializeAttrValue (tested indirectly via instrument / withRecord)
// ---------------------------------------------------------------------------

describe("serializeAttrValue (via instrument)", () => {
  it("passes string[] arrays natively", async () => {
    const fn = instrument(
      async (): Promise<string[]> => ["a", "b", "c"],
      {
        spanType: SpanType.RETRIEVAL,
        attributes: (ret) => ({
          [SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]: ret,
        }),
      }
    );

    await fn();

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    const val = spans[0]!.attributes[SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS];
    expect(val).toEqual(["a", "b", "c"]);
  });

  it("passes number[] arrays natively", () => {
    const fn = instrument(
      () => [1, 2, 3],
      {
        spanType: SpanType.UNKNOWN,
        attributes: { "my.nums": "return" },
      }
    );

    fn();

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes["my.nums"]).toEqual([1, 2, 3]);
  });

  it("passes boolean[] arrays natively", () => {
    const fn = instrument(
      () => [true, false, true],
      {
        spanType: SpanType.UNKNOWN,
        attributes: { "my.bools": "return" },
      }
    );

    fn();

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes["my.bools"]).toEqual([true, false, true]);
  });

  it("serialises mixed-type arrays as string[]", () => {
    const fn = instrument(
      () => [1, "two", { n: 3 }],
      {
        spanType: SpanType.UNKNOWN,
        attributes: (ret) => ({ "my.mixed": ret }),
      }
    );

    fn();

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    const val = spans[0]!.attributes["my.mixed"];
    // Mixed arrays fall through to the map() branch that stringifies non-strings
    expect(val).toEqual(["1", "two", '{"n":3}']);
  });

  it("JSON-stringifies plain objects", () => {
    const fn = instrument(
      () => ({ key: "value", nested: { a: 1 } }),
      {
        spanType: SpanType.UNKNOWN,
        attributes: (ret) => ({ "my.obj": ret }),
      }
    );

    fn();

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    const val = spans[0]!.attributes["my.obj"];
    expect(val).toBe('{"key":"value","nested":{"a":1}}');
  });

  it("passes primitive string through unchanged", () => {
    const fn = instrument(
      () => "hello",
      {
        spanType: SpanType.UNKNOWN,
        attributes: { "my.str": "return" },
      }
    );

    fn();

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes["my.str"]).toBe("hello");
  });

  it("passes primitive number through unchanged", () => {
    const fn = instrument(
      () => 42,
      {
        spanType: SpanType.UNKNOWN,
        attributes: { "my.num": "return" },
      }
    );

    fn();

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes["my.num"]).toBe(42);
  });

  it("passes primitive boolean through unchanged", () => {
    const fn = instrument(
      () => true,
      {
        spanType: SpanType.UNKNOWN,
        attributes: { "my.bool": "return" },
      }
    );

    fn();

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes["my.bool"]).toBe(true);
  });

  it("skips null and undefined attribute values", () => {
    const fn = instrument(
      () => null,
      {
        spanType: SpanType.UNKNOWN,
        attributes: (ret) => ({ "my.null": ret, "my.undef": undefined }),
      }
    );

    fn();

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes).not.toHaveProperty("my.null");
    expect(spans[0]!.attributes).not.toHaveProperty("my.undef");
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

  it("propagates RECORD_ID via baggage to child instrument() spans", async () => {
    const child = instrument(async () => "ok", {
      spanType: SpanType.TOOL,
    });

    await withRecord(async () => {
      await child();
      return "done";
    });

    const spans = exporter.getFinishedSpans();
    const rootSpan = spans.find(
      (s) => s.attributes[SpanAttributes.SPAN_TYPE] === SpanType.RECORD_ROOT
    )!;
    const childSpan = spans.find(
      (s) => s.attributes[SpanAttributes.SPAN_TYPE] === SpanType.TOOL
    )!;

    expect(rootSpan.attributes[SpanAttributes.RECORD_ID]).toBeDefined();
    expect(childSpan.attributes[SpanAttributes.RECORD_ID]).toBe(
      rootSpan.attributes[SpanAttributes.RECORD_ID]
    );
  });

  it("propagates RUN_NAME via baggage to child instrument() spans", async () => {
    const child = instrument(async () => "ok", {
      spanType: SpanType.TOOL,
    });

    await withRecord(
      async () => {
        await child();
        return "done";
      },
      { runName: "test-run-42" }
    );

    const spans = exporter.getFinishedSpans();
    const rootSpan = spans.find(
      (s) => s.attributes[SpanAttributes.SPAN_TYPE] === SpanType.RECORD_ROOT
    )!;
    const childSpan = spans.find(
      (s) => s.attributes[SpanAttributes.SPAN_TYPE] === SpanType.TOOL
    )!;

    expect(rootSpan.attributes[SpanAttributes.RUN_NAME]).toBe("test-run-42");
    expect(childSpan.attributes[SpanAttributes.RUN_NAME]).toBe("test-run-42");
  });

  it("generates a unique RECORD_ID per withRecord call", async () => {
    const out1 = await withRecord(async () => "a");
    const id1 = exporter.getFinishedSpans()[0]!.attributes[SpanAttributes.RECORD_ID];
    exporter.reset();

    const out2 = await withRecord(async () => "b");
    const id2 = exporter.getFinishedSpans()[0]!.attributes[SpanAttributes.RECORD_ID];

    expect(out1).toBe("a");
    expect(out2).toBe("b");
    expect(id1).not.toBe(id2);
  });
});

// ---------------------------------------------------------------------------
// instrumentDecorator()
// ---------------------------------------------------------------------------

describe("instrumentDecorator()", () => {
  it("creates a span with the method name as span name", async () => {
    class MyRAG {
      @instrumentDecorator({ spanType: SpanType.RETRIEVAL })
      async retrieve(query: string): Promise<string[]> {
        return [`result for ${query}`];
      }
    }

    const rag = new MyRAG();
    const result = await rag.retrieve("test");
    expect(result).toEqual(["result for test"]);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.name).toBe("retrieve");
    expect(spans[0]!.attributes[SpanAttributes.SPAN_TYPE]).toBe(
      SpanType.RETRIEVAL
    );
  });

  it("uses explicit spanName when provided", () => {
    class Svc {
      @instrumentDecorator({
        spanType: SpanType.TOOL,
        spanName: "custom-name",
      })
      doWork(): string {
        return "done";
      }
    }

    const svc = new Svc();
    svc.doWork();

    const spans = exporter.getFinishedSpans();
    expect(spans[0]!.name).toBe("custom-name");
  });

  it("preserves 'this' context", () => {
    class Counter {
      count = 0;

      @instrumentDecorator({ spanType: SpanType.TOOL })
      increment(): number {
        this.count += 1;
        return this.count;
      }
    }

    const c = new Counter();
    expect(c.increment()).toBe(1);
    expect(c.increment()).toBe(2);
    expect(c.count).toBe(2);
  });

  it("records errors and re-throws", () => {
    class Faulty {
      @instrumentDecorator({ spanType: SpanType.TOOL })
      explode(): void {
        throw new Error("decorator-boom");
      }
    }

    const f = new Faulty();
    expect(() => f.explode()).toThrow("decorator-boom");

    const spans = exporter.getFinishedSpans();
    expect(spans[0]!.attributes[SpanAttributes.CALL.ERROR]).toContain(
      "decorator-boom"
    );
  });
});
