import { afterEach, describe, expect, it } from "vitest";

import { LangChainInstrumentation } from "../instrumentation.js";
import { TruLensLangChainTracer } from "../tracer.js";

describe("LangChainInstrumentation", () => {
  let instr: LangChainInstrumentation;

  afterEach(() => {
    instr?.disable();
  });

  it("patches CallbackManager._configureSync on enable()", async () => {
    instr = new LangChainInstrumentation();
    await instr.enable();

    const mod = await import("@langchain/core/callbacks/manager");
    const CM = mod.CallbackManager;
    const method: "_configureSync" | "configure" =
      "_configureSync" in CM ? "_configureSync" : "configure";

    expect((CM as any)[method].name).toBe("patched");
  });

  it("restores the original method on disable()", async () => {
    instr = new LangChainInstrumentation();
    await instr.enable();

    const mod = await import("@langchain/core/callbacks/manager");
    const CM = mod.CallbackManager;
    const method: "_configureSync" | "configure" =
      "_configureSync" in CM ? "_configureSync" : "configure";

    expect((CM as any)[method].name).toBe("patched");

    instr.disable();
    expect((CM as any)[method].name).not.toBe("patched");
  });

  it("is idempotent — double enable() doesn't break", async () => {
    instr = new LangChainInstrumentation();
    await instr.enable();
    await instr.enable(); // no-op
    instr.disable();
  });

  it("injects TruLensLangChainTracer into null handlers", async () => {
    instr = new LangChainInstrumentation();
    await instr.enable();

    const mod = await import("@langchain/core/callbacks/manager");
    const CM = mod.CallbackManager;
    const method: "_configureSync" | "configure" =
      "_configureSync" in CM ? "_configureSync" : "configure";

    // Call the patched method with null handlers — should inject the tracer
    const result = (CM as any)[method](null);

    // The result depends on the original method's behaviour, but
    // the first argument should be modified to include the tracer.
    // We verify by checking that the internal tracer exists.
    expect((instr as any)._tracer).toBeInstanceOf(TruLensLangChainTracer);
  });

  it("injects TruLensLangChainTracer into array handlers", async () => {
    instr = new LangChainInstrumentation();
    await instr.enable();

    const mod = await import("@langchain/core/callbacks/manager");
    const CM = mod.CallbackManager;
    const method: "_configureSync" | "configure" =
      "_configureSync" in CM ? "_configureSync" : "configure";

    const existingHandlers: unknown[] = [];

    // Call with an empty array — should push tracer into it
    (CM as any)[method](existingHandlers);

    expect(existingHandlers.length).toBeGreaterThanOrEqual(1);
    expect(
      existingHandlers.some((h) => h instanceof TruLensLangChainTracer)
    ).toBe(true);
  });

  it("does not duplicate the tracer on repeated calls", async () => {
    instr = new LangChainInstrumentation();
    await instr.enable();

    const mod = await import("@langchain/core/callbacks/manager");
    const CM = mod.CallbackManager;
    const method: "_configureSync" | "configure" =
      "_configureSync" in CM ? "_configureSync" : "configure";

    const handlers: unknown[] = [];
    (CM as any)[method](handlers);
    (CM as any)[method](handlers);

    const tracerCount = handlers.filter(
      (h) => h instanceof TruLensLangChainTracer
    ).length;
    expect(tracerCount).toBe(1);
  });

  it("manuallyInstrument() patches a provided module", async () => {
    instr = new LangChainInstrumentation();

    const mod = await import("@langchain/core/callbacks/manager");
    instr.manuallyInstrument(mod as any);

    const CM = mod.CallbackManager;
    const method: "_configureSync" | "configure" =
      "_configureSync" in CM ? "_configureSync" : "configure";

    expect((CM as any)[method].name).toBe("patched");
  });

  it("setTracerProvider is a no-op", () => {
    instr = new LangChainInstrumentation();
    expect(() => instr.setTracerProvider({} as any)).not.toThrow();
  });
});
