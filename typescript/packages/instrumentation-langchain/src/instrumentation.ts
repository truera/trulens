/**
 * LangChainInstrumentation — auto-instruments LangChain.js by injecting
 * a TruLensLangChainTracer into the LangChain CallbackManager.
 *
 * Two usage modes:
 *
 * 1. **Auto** — pass to `TruSession.init({ instrumentations: [...] })`.
 *    `enable()` patches `CallbackManager._configureSync` so every
 *    LangChain operation is traced without any per-call changes.
 *
 * 2. **Manual** — call `manuallyInstrument(CallbackManagerModule)` if
 *    you prefer explicit control (mirrors OpenInference's API).
 *
 * Follows the same ESM-safe prototype-patching pattern used by
 * `@trulens/instrumentation-openai`.
 */

import type { TracerProvider } from "@opentelemetry/api";
import { TruLensLangChainTracer } from "./tracer.js";

const PKG_NAME = "@trulens/instrumentation-langchain";

type CallbackManagerModule = {
  CallbackManager: {
    _configureSync?: (...args: any[]) => any;
    configure?: (...args: any[]) => any;
  };
};

export interface LangChainInstrumentationConfig {
  // Reserved for future options (e.g. ignoreChain, maskInputs, etc.)
}

export class LangChainInstrumentation {
  private _config: LangChainInstrumentationConfig;
  private _original: ((...args: any[]) => any) | null = null;
  private _patchTarget: CallbackManagerModule | null = null;
  private _patchMethod: "_configureSync" | "configure" | null = null;
  private _tracer: TruLensLangChainTracer;

  constructor(config: LangChainInstrumentationConfig = {}) {
    this._config = config;
    this._tracer = new TruLensLangChainTracer();
  }

  /**
   * Dynamically import `@langchain/core/callbacks/manager` and patch
   * `CallbackManager._configureSync` (or `configure` on older versions)
   * to inject the TruLens tracer into every callback chain.
   */
  async enable(): Promise<void> {
    if (this._original) return;

    let mod: CallbackManagerModule;
    try {
      mod = await import(
        "@langchain/core/callbacks/manager"
      ) as unknown as CallbackManagerModule;
    } catch {
      console.warn(
        `[${PKG_NAME}] Could not import("@langchain/core/callbacks/manager"). ` +
          "LangChain auto-instrumentation is disabled."
      );
      return;
    }

    this._patch(mod);
  }

  /**
   * Manually instrument a pre-imported CallbackManager module.
   * Useful when you want to avoid relying on dynamic import.
   *
   * @example
   * ```ts
   * import * as CBM from "@langchain/core/callbacks/manager";
   * const instr = new LangChainInstrumentation();
   * instr.manuallyInstrument(CBM);
   * ```
   */
  manuallyInstrument(mod: CallbackManagerModule): void {
    this._patch(mod);
  }

  /** Restore the original method. */
  disable(): void {
    if (
      this._patchTarget &&
      this._original &&
      this._patchMethod
    ) {
      (this._patchTarget.CallbackManager as any)[this._patchMethod] =
        this._original;
      this._original = null;
      this._patchTarget = null;
      this._patchMethod = null;
    }
  }

  /**
   * Accepts a TracerProvider for API compatibility with TruSession.
   * The tracer uses the globally registered provider automatically.
   */
  setTracerProvider(_provider: TracerProvider): void {
    // no-op — the TruLensLangChainTracer uses trace.getTracer() internally
  }

  // -----------------------------------------------------------------
  // Internal patching
  // -----------------------------------------------------------------

  private _patch(mod: CallbackManagerModule): void {
    const CM = mod.CallbackManager;
    if (!CM) {
      console.warn(
        `[${PKG_NAME}] CallbackManager not found in module. ` +
          "LangChain auto-instrumentation is disabled."
      );
      return;
    }

    // Prefer _configureSync (LC >= 0.2); fall back to configure
    const methodName: "_configureSync" | "configure" =
      "_configureSync" in CM ? "_configureSync" : "configure";

    const original = (CM as any)[methodName] as
      | ((...args: any[]) => any)
      | undefined;
    if (!original) {
      console.warn(
        `[${PKG_NAME}] CallbackManager.${methodName} not found. ` +
          "LangChain auto-instrumentation is disabled."
      );
      return;
    }

    if (this._original) return; // already patched

    this._original = original;
    this._patchTarget = mod;
    this._patchMethod = methodName;

    const tracer = this._tracer;

    (CM as any)[methodName] = function patched(
      this: any,
      ...args: any[]
    ) {
      // args[0] is inheritableHandlers — inject our tracer
      const handlers = args[0];
      args[0] = injectTracer(tracer, handlers);
      return original.apply(this, args);
    };
  }
}

// -----------------------------------------------------------------
// Handler injection helper
// -----------------------------------------------------------------

function injectTracer(
  tracer: TruLensLangChainTracer,
  handlers: unknown
): unknown {
  if (handlers == null) {
    return [tracer];
  }

  if (Array.isArray(handlers)) {
    const alreadyPresent = handlers.some(
      (h) => h instanceof TruLensLangChainTracer
    );
    if (!alreadyPresent) {
      handlers.push(tracer);
    }
    return handlers;
  }

  // CallbackManager instance — has .handlers / .inheritableHandlers
  const mgr = handlers as {
    handlers: any[];
    inheritableHandlers: any[];
    addHandler?: (h: any, inherit: boolean) => void;
  };

  const alreadyPresent =
    mgr.handlers?.some(
      (h: any) => h instanceof TruLensLangChainTracer
    ) ||
    mgr.inheritableHandlers?.some(
      (h: any) => h instanceof TruLensLangChainTracer
    );

  if (!alreadyPresent && typeof mgr.addHandler === "function") {
    mgr.addHandler(tracer, true);
  }

  return handlers;
}
