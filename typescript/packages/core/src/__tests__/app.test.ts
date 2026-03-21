import { beforeAll, beforeEach, describe, expect, it } from "vitest";
import {
  InMemorySpanExporter,
  SimpleSpanProcessor,
  NodeTracerProvider,
} from "@opentelemetry/sdk-trace-node";

import { createTruApp } from "../app.js";
import { SpanAttributes, SpanType } from "@trulens/semconv";

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
// Test target: a simple app object
// ---------------------------------------------------------------------------

class SimpleApp {
  name = "SimpleApp";

  async query(question: string): Promise<string> {
    return `Answer to: ${question}`;
  }

  sync(): string {
    return "sync-result";
  }
}

describe("createTruApp()", () => {
  it("wraps the main method with a RECORD_ROOT span", async () => {
    const raw = new SimpleApp();
    const app = createTruApp(raw, {
      mainMethod: "query",
      mainInput: (q) => q,
    });

    const answer = await app.query("What is RAG?");
    expect(answer).toBe("Answer to: What is RAG?");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    const span = spans[0]!;
    expect(span.attributes[SpanAttributes.SPAN_TYPE]).toBe(
      SpanType.RECORD_ROOT
    );
    expect(span.attributes[SpanAttributes.RECORD_ROOT.INPUT]).toBe(
      "What is RAG?"
    );
    expect(span.attributes[SpanAttributes.RECORD_ROOT.OUTPUT]).toBe(
      "Answer to: What is RAG?"
    );
  });

  it("passes through non-main properties unchanged", () => {
    const raw = new SimpleApp();
    const app = createTruApp(raw, {
      mainMethod: "query",
      mainInput: (q) => q,
    });

    expect(app.name).toBe("SimpleApp");
  });

  it("passes through non-main methods unchanged", () => {
    const raw = new SimpleApp();
    const app = createTruApp(raw, {
      mainMethod: "query",
      mainInput: (q) => q,
    });

    expect(app.sync()).toBe("sync-result");
  });

  it("defaults mainInput to JSON.stringify(args)", async () => {
    const raw = new SimpleApp();
    const app = createTruApp(raw, { mainMethod: "query" });

    await app.query("hello");

    const spans = exporter.getFinishedSpans();
    expect(spans[0]!.attributes[SpanAttributes.RECORD_ROOT.INPUT]).toBe(
      '["hello"]'
    );
  });

  it("records RECORD_ID on the RECORD_ROOT span", async () => {
    const raw = new SimpleApp();
    const app = createTruApp(raw, {
      mainMethod: "query",
      mainInput: (q) => q,
    });

    await app.query("test");

    const spans = exporter.getFinishedSpans();
    const recordId = spans[0]!.attributes[SpanAttributes.RECORD_ID];
    expect(recordId).toBeDefined();
    expect(typeof recordId).toBe("string");
    // UUID format
    expect(String(recordId)).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
    );
  });
});
