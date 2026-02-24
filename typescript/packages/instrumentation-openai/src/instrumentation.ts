/**
 * Auto-instrumentation for the OpenAI Node.js SDK.
 *
 * Patches `OpenAI.Chat.Completions.prototype.create` so that every call
 * automatically produces a GENERATION span carrying TruLens semantic-
 * convention attributes for model name, token counts, and cost.
 *
 * Unlike typical OTEL instrumentations that hook into `require()` via
 * `require-in-the-middle`, this implementation patches the prototype
 * directly at enable-time. This makes it work with both ESM (`import`)
 * and CJS (`require`) — critical because most TypeScript tooling (tsx,
 * ts-node, vitest) uses ESM.
 */

import {
  context,
  propagation,
  SpanStatusCode,
  trace,
  type Span,
} from "@opentelemetry/api";
import { SpanAttributes, SpanType } from "@trulens/semconv";

const PKG_NAME = "@trulens/instrumentation-openai";

export interface OpenAIInstrumentationConfig {
  /**
   * Override the span name for chat completion calls.
   * Defaults to `"openai.chat.completions.create"`.
   */
  chatSpanName?: string;
}

export class OpenAIInstrumentation {
  private _config: OpenAIInstrumentationConfig;
  private _original: ((...args: any[]) => any) | null = null;
  private _proto: any = null;

  constructor(config: OpenAIInstrumentationConfig = {}) {
    this._config = config;
  }

  /**
   * Dynamically import the `openai` module and patch
   * `Completions.prototype.create`.
   *
   * Returns a Promise because `import()` is async. `TruSession.init()`
   * awaits this before returning, guaranteeing the patch is in place
   * before any user code runs.
   */
  async enable(): Promise<void> {
    if (this._original) {
      return;
    }

    let mod: any;
    try {
      mod = await import("openai");
    } catch {
      console.warn(
        `[${PKG_NAME}] Could not import("openai"). ` +
          "OpenAI auto-instrumentation is disabled."
      );
      return;
    }

    const CompletionsClass =
      mod?.OpenAI?.Chat?.Completions ??
      mod?.default?.Chat?.Completions ??
      mod?.Chat?.Completions;

    if (!CompletionsClass?.prototype) {
      console.warn(
        `[${PKG_NAME}] Could not find OpenAI.Chat.Completions class. ` +
          "OpenAI auto-instrumentation is disabled."
      );
      return;
    }

    const proto = CompletionsClass.prototype;
    if (this._original) {
      return;
    }

    this._original = proto.create;
    this._proto = proto;

    const instrumentation = this;
    proto.create = function patchedCreate(
      this: any,
      ...args: any[]
    ) {
      return instrumentation._traceCreate(
        instrumentation._original!,
        this,
        args
      );
    };
  }

  /** Restore the original `create` method. */
  disable(): void {
    if (this._proto && this._original) {
      this._proto.create = this._original;
      this._original = null;
      this._proto = null;
    }
  }

  /**
   * Accepts a TracerProvider for API compatibility with TruSession, but
   * this instrumentation uses the global `trace.getTracer()` so the
   * provider set via `NodeTracerProvider.register()` is used automatically.
   */
  setTracerProvider(_provider: any): void {
    // no-op — we use the globally registered provider.
  }

  private _traceCreate(
    original: (...args: any[]) => any,
    thisArg: any,
    args: any[]
  ): any {
    const spanName =
      this._config.chatSpanName ?? "openai.chat.completions.create";

    const tracer = trace.getTracer(PKG_NAME);
    const span = tracer.startSpan(spanName);

    span.setAttribute(SpanAttributes.SPAN_TYPE, SpanType.GENERATION);
    span.setAttribute(SpanAttributes.CALL.FUNCTION, spanName);

    const recordId = propagation
      .getBaggage(context.active())
      ?.getEntry(SpanAttributes.RECORD_ID)?.value;
    if (recordId) {
      span.setAttribute(SpanAttributes.RECORD_ID, recordId);
    }

    const body = args[0] as Record<string, unknown> | undefined;
    if (body?.model) {
      span.setAttribute(SpanAttributes.COST.MODEL, String(body.model));
    }

    const ctx = trace.setSpan(context.active(), span);

    let result: any;
    try {
      result = context.with(ctx, () => original.apply(thisArg, args));
    } catch (err) {
      this._finalizeSpan(span, undefined, err);
      throw err;
    }

    if (result instanceof Promise) {
      return result.then(
        (response: any) => {
          this._finalizeSpan(span, response, undefined);
          return response;
        },
        (err: unknown) => {
          this._finalizeSpan(span, undefined, err);
          throw err;
        }
      );
    }

    this._finalizeSpan(span, result, undefined);
    return result;
  }

  private _finalizeSpan(
    span: Span,
    response: any,
    error: unknown
  ): void {
    if (error !== undefined) {
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: String(error),
      });
      span.setAttribute(SpanAttributes.CALL.ERROR, String(error));
      span.end();
      return;
    }

    if (response) {
      if (response.model) {
        span.setAttribute(SpanAttributes.COST.MODEL, response.model);
      }
      const usage = response.usage;
      if (usage) {
        if (usage.total_tokens != null) {
          span.setAttribute(
            SpanAttributes.COST.NUM_TOKENS,
            usage.total_tokens
          );
        }
        if (usage.prompt_tokens != null) {
          span.setAttribute(
            SpanAttributes.COST.NUM_PROMPT_TOKENS,
            usage.prompt_tokens
          );
        }
        if (usage.completion_tokens != null) {
          span.setAttribute(
            SpanAttributes.COST.NUM_COMPLETION_TOKENS,
            usage.completion_tokens
          );
        }
      }
    }

    span.setStatus({ code: SpanStatusCode.OK });
    span.end();
  }
}
