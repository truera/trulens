import {
  beforeAll,
  beforeEach,
  afterEach,
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

import { OpenAIInstrumentation } from "../instrumentation.js";

const exporter = new InMemorySpanExporter();

beforeAll(() => {
  const provider = new NodeTracerProvider();
  provider.addSpanProcessor(new SimpleSpanProcessor(exporter));
  provider.register();
});

beforeEach(() => {
  exporter.reset();
});

describe("OpenAIInstrumentation", () => {
  let instr: OpenAIInstrumentation;

  afterEach(() => {
    instr?.disable();
  });

  it("patches OpenAI.Chat.Completions.prototype.create on enable()", async () => {
    instr = new OpenAIInstrumentation();
    await instr.enable();

    // Import the openai module and verify the patch is in place
    const openai = await import("openai");
    const CompletionsClass =
      (openai as any).OpenAI?.Chat?.Completions ??
      (openai as any).default?.Chat?.Completions;

    expect(CompletionsClass.prototype.create.name).toBe("patchedCreate");
  });

  it("restores original create on disable()", async () => {
    instr = new OpenAIInstrumentation();
    await instr.enable();

    const openai = await import("openai");
    const CompletionsClass =
      (openai as any).OpenAI?.Chat?.Completions ??
      (openai as any).default?.Chat?.Completions;

    // Capture the patched version
    expect(CompletionsClass.prototype.create.name).toBe("patchedCreate");

    instr.disable();
    expect(CompletionsClass.prototype.create.name).not.toBe("patchedCreate");
  });

  it("is idempotent — double enable() doesn't break", async () => {
    instr = new OpenAIInstrumentation();
    await instr.enable();
    await instr.enable(); // should be a no-op
    instr.disable();
  });

  it("creates a GENERATION span when create() is called", async () => {
    instr = new OpenAIInstrumentation();
    await instr.enable();

    const openai = await import("openai");
    const CompletionsClass =
      (openai as any).OpenAI?.Chat?.Completions ??
      (openai as any).default?.Chat?.Completions;

    // Build a mock instance with the patched prototype
    const mockInstance = Object.create(CompletionsClass.prototype);
    // Replace the patched create with a version that calls original → mock
    const originalCreate = async (body: any) => ({
      model: body.model ?? "gpt-4",
      choices: [{ message: { content: "Hello!" } }],
      usage: {
        total_tokens: 100,
        prompt_tokens: 30,
        completion_tokens: 70,
      },
    });

    // Patch the internal _original to point to our mock
    (instr as any)._original = originalCreate;

    const response = await mockInstance.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: "Hi" }],
    });

    expect(response.usage.total_tokens).toBe(100);

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    const span = spans[0]!;

    expect(span.attributes[SpanAttributes.SPAN_TYPE]).toBe(
      SpanType.GENERATION
    );
    expect(span.attributes[SpanAttributes.COST.MODEL]).toBe("gpt-4o");
    expect(span.attributes[SpanAttributes.COST.NUM_TOKENS]).toBe(100);
    expect(span.attributes[SpanAttributes.COST.NUM_PROMPT_TOKENS]).toBe(30);
    expect(span.attributes[SpanAttributes.COST.NUM_COMPLETION_TOKENS]).toBe(70);
  });

  it("records errors from create() and re-throws", async () => {
    instr = new OpenAIInstrumentation();
    await instr.enable();

    const openai = await import("openai");
    const CompletionsClass =
      (openai as any).OpenAI?.Chat?.Completions ??
      (openai as any).default?.Chat?.Completions;

    const mockInstance = Object.create(CompletionsClass.prototype);
    (instr as any)._original = async () => {
      throw new Error("API error");
    };

    await expect(
      mockInstance.create({ model: "gpt-4", messages: [] })
    ).rejects.toThrow("API error");

    const spans = exporter.getFinishedSpans();
    expect(spans).toHaveLength(1);
    expect(spans[0]!.attributes[SpanAttributes.CALL.ERROR]).toContain(
      "API error"
    );
  });

  it("accepts a custom chatSpanName", async () => {
    instr = new OpenAIInstrumentation({ chatSpanName: "my-custom-span" });
    await instr.enable();

    const openai = await import("openai");
    const CompletionsClass =
      (openai as any).OpenAI?.Chat?.Completions ??
      (openai as any).default?.Chat?.Completions;

    const mockInstance = Object.create(CompletionsClass.prototype);
    (instr as any)._original = async () => ({ choices: [] });

    await mockInstance.create({ model: "gpt-4", messages: [] });

    const spans = exporter.getFinishedSpans();
    expect(spans[0]!.name).toBe("my-custom-span");
  });

  it("setTracerProvider is a no-op", () => {
    instr = new OpenAIInstrumentation();
    expect(() => instr.setTracerProvider({} as any)).not.toThrow();
  });
});
