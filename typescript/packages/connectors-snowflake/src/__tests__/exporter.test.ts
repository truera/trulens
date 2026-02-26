import { describe, expect, it } from "vitest";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-node";
import { ResourceAttributes, SpanAttributes } from "@trulens/semconv";

import {
  encodeVarint,
  isTruLensSpan,
  propagateTraceAttrs,
} from "../exporter.js";

// ---------------------------------------------------------------------------
// Mock ReadableSpan builder
// ---------------------------------------------------------------------------

function mockSpan(opts: {
  traceId: string;
  spanId?: string;
  attributes?: Record<string, unknown>;
  resourceAttributes?: Record<string, unknown>;
}): ReadableSpan {
  return {
    spanContext: () => ({
      traceId: opts.traceId,
      spanId: opts.spanId ?? "span-1",
      traceFlags: 1,
    }),
    attributes: opts.attributes ?? {},
    resource: {
      attributes: opts.resourceAttributes ?? {},
    },
  } as unknown as ReadableSpan;
}

// ---------------------------------------------------------------------------
// isTruLensSpan
// ---------------------------------------------------------------------------

describe("isTruLensSpan()", () => {
  it("returns true when resource has APP_NAME", () => {
    const span = mockSpan({
      traceId: "t1",
      resourceAttributes: { [ResourceAttributes.APP_NAME]: "my-app" },
    });
    expect(isTruLensSpan(span)).toBe(true);
  });

  it("returns true when span has SPAN_TYPE attribute", () => {
    const span = mockSpan({
      traceId: "t1",
      attributes: { [SpanAttributes.SPAN_TYPE]: "tool" },
    });
    expect(isTruLensSpan(span)).toBe(true);
  });

  it("returns false when neither APP_NAME nor SPAN_TYPE is present", () => {
    const span = mockSpan({
      traceId: "t1",
      attributes: { "some.other": "value" },
      resourceAttributes: { "service.name": "test" },
    });
    expect(isTruLensSpan(span)).toBe(false);
  });
});

// ---------------------------------------------------------------------------
// propagateTraceAttrs
// ---------------------------------------------------------------------------

describe("propagateTraceAttrs()", () => {
  it("copies RECORD_ID from root span to child spans in the same trace", () => {
    const root = mockSpan({
      traceId: "trace-1",
      spanId: "root",
      attributes: {
        [SpanAttributes.RECORD_ID]: "rec-123",
        [SpanAttributes.SPAN_TYPE]: "record_root",
      },
    });
    const child = mockSpan({
      traceId: "trace-1",
      spanId: "child",
      attributes: { [SpanAttributes.SPAN_TYPE]: "tool" },
    });

    const cache = new Map<string, { recordId?: string; runName?: string }>();
    const result = propagateTraceAttrs([root, child], cache);

    expect(result[1]!.attributes[SpanAttributes.RECORD_ID]).toBe("rec-123");
  });

  it("copies RUN_NAME from root span to child spans", () => {
    const root = mockSpan({
      traceId: "trace-2",
      spanId: "root",
      attributes: {
        [SpanAttributes.RUN_NAME]: "run-42",
      },
    });
    const child = mockSpan({
      traceId: "trace-2",
      spanId: "child",
      attributes: {},
    });

    const cache = new Map<string, { recordId?: string; runName?: string }>();
    const result = propagateTraceAttrs([root, child], cache);

    expect(result[1]!.attributes[SpanAttributes.RUN_NAME]).toBe("run-42");
  });

  it("does not overwrite existing RECORD_ID on child spans", () => {
    const root = mockSpan({
      traceId: "trace-3",
      spanId: "root",
      attributes: { [SpanAttributes.RECORD_ID]: "from-root" },
    });
    const child = mockSpan({
      traceId: "trace-3",
      spanId: "child",
      attributes: { [SpanAttributes.RECORD_ID]: "already-set" },
    });

    const cache = new Map<string, { recordId?: string; runName?: string }>();
    const result = propagateTraceAttrs([root, child], cache);

    expect(result[1]!.attributes[SpanAttributes.RECORD_ID]).toBe(
      "already-set"
    );
  });

  it("uses persistent cache across batches", () => {
    const cache = new Map<string, { recordId?: string; runName?: string }>();

    // Batch 1: root arrives first
    const root = mockSpan({
      traceId: "trace-4",
      attributes: { [SpanAttributes.RECORD_ID]: "rec-x" },
    });
    propagateTraceAttrs([root], cache);

    // Batch 2: child arrives later
    const child = mockSpan({
      traceId: "trace-4",
      spanId: "child",
      attributes: {},
    });
    const result = propagateTraceAttrs([child], cache);

    expect(result[0]!.attributes[SpanAttributes.RECORD_ID]).toBe("rec-x");
  });

  it("returns spans unchanged when no cache entry exists", () => {
    const span = mockSpan({
      traceId: "trace-orphan",
      attributes: { foo: "bar" },
    });
    const cache = new Map<string, { recordId?: string; runName?: string }>();
    const result = propagateTraceAttrs([span], cache);

    expect(result[0]).toBe(span); // same reference
  });
});

// ---------------------------------------------------------------------------
// encodeVarint
// ---------------------------------------------------------------------------

describe("encodeVarint()", () => {
  it("encodes 0 as a single byte [0x00]", () => {
    expect(encodeVarint(0)).toEqual(Buffer.from([0x00]));
  });

  it("encodes 1 as [0x01]", () => {
    expect(encodeVarint(1)).toEqual(Buffer.from([0x01]));
  });

  it("encodes 127 as [0x7f]", () => {
    expect(encodeVarint(127)).toEqual(Buffer.from([0x7f]));
  });

  it("encodes 128 as [0x80, 0x01]", () => {
    expect(encodeVarint(128)).toEqual(Buffer.from([0x80, 0x01]));
  });

  it("encodes 300 as [0xac, 0x02]", () => {
    // 300 = 0b100101100
    // low 7 bits: 0101100 = 0x2c, set continuation bit -> 0xac
    // remaining: 10 = 0x02
    expect(encodeVarint(300)).toEqual(Buffer.from([0xac, 0x02]));
  });

  it("encodes 16384 as [0x80, 0x80, 0x01]", () => {
    expect(encodeVarint(16384)).toEqual(Buffer.from([0x80, 0x80, 0x01]));
  });
});
