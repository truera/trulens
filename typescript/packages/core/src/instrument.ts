/**
 * instrument() — wrap a function (or method) to emit an OTEL span on each call
 * with TruLens semantic convention attributes.
 *
 * Mirrors the Python @instrument decorator from trulens.core.otel.instrument.
 */

import {
  context,
  propagation,
  SpanStatusCode,
  trace,
} from "@opentelemetry/api";
import { randomUUID } from "node:crypto";
import { SpanAttributes, SpanType } from "@trulens/semconv";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * A plain attribute map, or a lazy resolver that receives call context.
 * For async functions, `ret` is the unwrapped (awaited) return value.
 */
export type AttributeResolver<TArgs extends unknown[], TReturn> =
  | Record<string, string>
  | ((
      ret: Awaited<TReturn> | undefined,
      error: unknown,
      ...args: TArgs
    ) => Record<string, unknown>);

export interface InstrumentOptions<TArgs extends unknown[], TReturn> {
  /**
   * Semantic span type — sets `ai.observability.span_type`.
   * Defaults to `SpanType.UNKNOWN`.
   */
  spanType?: SpanType;
  /**
   * Attribute mappings for span attributes.
   *
   * As a plain object:
   *   keys   — attribute names (e.g. `SpanAttributes.RETRIEVAL.QUERY_TEXT`)
   *   values — arg name string (e.g. `"query"`) or `"return"` for the return value
   *
   * As a function:
   *   `(ret, error, ...args) => Record<string, unknown>`
   *   Mirrors the Python lambda attribute resolver.
   */
  attributes?: AttributeResolver<TArgs, TReturn>;
  /**
   * Override the span name. Defaults to the function's name.
   */
  spanName?: string;
}

// ---------------------------------------------------------------------------
// Core wrapper
// ---------------------------------------------------------------------------

/**
 * Wraps `fn` so that every call creates an OTEL span carrying TruLens semconv
 * attributes. Works with both sync and async functions.
 *
 * @example
 * ```ts
 * const retrieve = instrument(
 *   async (query: string): Promise<string[]> => { ... },
 *   {
 *     spanType: SpanType.RETRIEVAL,
 *     attributes: {
 *       [SpanAttributes.RETRIEVAL.QUERY_TEXT]: "query",
 *       [SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS]: "return",
 *     },
 *   }
 * );
 * ```
 */
export function instrument<TArgs extends unknown[], TReturn>(
  fn: (...args: TArgs) => TReturn,
  options: InstrumentOptions<TArgs, TReturn> = {}
): (...args: TArgs) => TReturn {
  const { spanType = SpanType.UNKNOWN, attributes, spanName } = options;

  const wrappedName = spanName ?? fn.name ?? "anonymous";

  return function instrumentedFn(this: unknown, ...args: TArgs): TReturn {
    const tracer = trace.getTracer("@trulens/core");
    const span = tracer.startSpan(wrappedName);
    span.setAttribute(SpanAttributes.SPAN_TYPE, spanType);
    span.setAttribute(SpanAttributes.CALL.FUNCTION, wrappedName);

    // Propagate RECORD_ID from baggage (set by withRecord) to every span.
    const recordId = propagation.getBaggage(context.active())
      ?.getEntry(SpanAttributes.RECORD_ID)?.value;
    if (recordId) {
      span.setAttribute(SpanAttributes.RECORD_ID, recordId);
    }

    const ctx = trace.setSpan(context.active(), span);

    const finalize = (
      ret: Awaited<TReturn> | undefined,
      error: unknown
    ): void => {
      if (attributes) {
        const resolved = resolveAttributes(attributes, args, ret, error);
        for (const [k, v] of Object.entries(resolved)) {
          if (v !== undefined && v !== null) {
            span.setAttribute(k, serializeAttrValue(v));
          }
        }
      }

      if (error !== undefined) {
        span.setStatus({
          code: SpanStatusCode.ERROR,
          message: String(error),
        });
        span.setAttribute(SpanAttributes.CALL.ERROR, String(error));
      } else {
        span.setStatus({ code: SpanStatusCode.OK });
      }
      span.end();
    };

    let result: TReturn;
    try {
      result = context.with(ctx, () => fn.apply(this, args));
    } catch (err) {
      finalize(undefined, err);
      throw err;
    }

    if (result instanceof Promise) {
      return result.then(
        (ret) => {
          finalize(ret as Awaited<TReturn>, undefined);
          return ret;
        },
        (err: unknown) => {
          finalize(undefined, err);
          throw err;
        }
      ) as TReturn;
    }

    finalize(result as Awaited<TReturn>, undefined);
    return result;
  };
}

// ---------------------------------------------------------------------------
// Method decorator (TypeScript 5 Stage 3)
// ---------------------------------------------------------------------------

/**
 * Method decorator equivalent of `instrument()`.
 *
 * Automatically infers the span name from the method name when no
 * explicit `spanName` is provided.
 *
 * @example
 * ```ts
 * class MyRAG {
 *   @instrumentDecorator({ spanType: SpanType.RETRIEVAL })
 *   async retrieve(query: string) { ... }
 * }
 * ```
 */
export function instrumentDecorator<TArgs extends unknown[], TReturn>(
  options: InstrumentOptions<TArgs, TReturn> = {}
) {
  return function (
    _target: unknown,
    _context: ClassMethodDecoratorContext
  ) {
    const inferredName =
      options.spanName ?? String(_context.name);

    return function (this: unknown, ...args: TArgs): TReturn {
      const fn = _target as (...args: TArgs) => TReturn;
      return instrument(fn.bind(this), {
        ...options,
        spanName: inferredName,
      })(...args);
    };
  };
}

// ---------------------------------------------------------------------------
// withRecord — RECORD_ROOT span
// ---------------------------------------------------------------------------

export interface WithRecordOptions {
  /** The main input to the app run (serialised to a string if not already). */
  input?: unknown;
  /** Ground truth output, if known. */
  groundTruthOutput?: unknown;
  /** Optional run name, written to `ai.observability.run.name`. */
  runName?: string;
}

/**
 * Wraps `fn` in a RECORD_ROOT span — the top-level span that the TruLens
 * dashboard uses as the entry point for a complete app record.
 *
 * @example
 * ```ts
 * const output = await withRecord(
 *   () => myApp.query("What is RAG?"),
 *   { input: "What is RAG?" }
 * );
 * ```
 */
export async function withRecord<T>(
  fn: () => T | Promise<T>,
  options: WithRecordOptions = {}
): Promise<T> {
  const { input, groundTruthOutput, runName } = options;
  const tracer = trace.getTracer("@trulens/core");
  const span = tracer.startSpan("record_root");

  // Generate a unique RECORD_ID and propagate it via baggage so that
  // all child spans (created by instrument()) share the same record ID.
  const recordId = randomUUID();
  span.setAttribute(SpanAttributes.SPAN_TYPE, SpanType.RECORD_ROOT);
  span.setAttribute(SpanAttributes.RECORD_ID, recordId);
  span.setAttribute(SpanAttributes.INPUT_ID, "");

  if (input !== undefined) {
    span.setAttribute(
      SpanAttributes.RECORD_ROOT.INPUT,
      serializeAttrValue(input)
    );
  }
  if (groundTruthOutput !== undefined) {
    span.setAttribute(
      SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT,
      serializeAttrValue(groundTruthOutput)
    );
  }
  if (runName !== undefined) {
    span.setAttribute(SpanAttributes.RUN_NAME, runName);
  }

  // Attach RECORD_ID to baggage so children can read it.
  let baggage =
    propagation.getBaggage(context.active()) ??
    propagation.createBaggage();
  baggage = baggage.setEntry(SpanAttributes.RECORD_ID, { value: recordId });
  let ctx = propagation.setBaggage(context.active(), baggage);
  ctx = trace.setSpan(ctx, span);

  try {
    const result = await context.with(ctx, () => fn());
    span.setAttribute(
      SpanAttributes.RECORD_ROOT.OUTPUT,
      serializeAttrValue(result)
    );
    span.setStatus({ code: SpanStatusCode.OK });
    return result;
  } catch (err) {
    span.setAttribute(SpanAttributes.RECORD_ROOT.ERROR, String(err));
    span.setStatus({ code: SpanStatusCode.ERROR, message: String(err) });
    throw err;
  } finally {
    span.end();
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function resolveAttributes<TArgs extends unknown[], TReturn>(
  attributes: AttributeResolver<TArgs, TReturn>,
  args: TArgs,
  ret: Awaited<TReturn> | undefined,
  error: unknown
): Record<string, unknown> {
  if (typeof attributes === "function") {
    return attributes(ret, error, ...args);
  }

  // Build a lookup: arg-name → value (positional args use index as fallback)
  const resolved: Record<string, unknown> = {};
  for (const [attrKey, sourceKey] of Object.entries(attributes)) {
    if (sourceKey === "return") {
      resolved[attrKey] = ret;
    } else {
      // Try to find the arg by its positional index (not ideal, but workable
      // without runtime parameter name reflection). Users should prefer the
      // lambda form when they need named-arg mapping.
      const idx = parseInt(sourceKey, 10);
      resolved[attrKey] = isNaN(idx) ? undefined : args[idx];
    }
  }
  return resolved;
}

function serializeAttrValue(v: unknown): string | number | boolean {
  if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") {
    return v;
  }
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}
