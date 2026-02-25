/**
 * TruLensLangChainTracer — extends LangChain's BaseTracer to emit
 * OpenTelemetry spans with TruLens semantic conventions.
 *
 * Each LangChain "run" (chain, llm, retriever, tool) maps to a span
 * whose type, attributes, and parent-child relationships follow the
 * TruLens convention schema.
 */

import {
  context,
  propagation,
  SpanStatusCode,
  trace,
  type Span,
  type SpanContext,
} from "@opentelemetry/api";
import { SpanAttributes, SpanType } from "@trulens/semconv";
import type { Run } from "@langchain/core/tracers/base";
import { BaseTracer } from "@langchain/core/tracers/base";

const TRACER_NAME = "@trulens/instrumentation-langchain";

interface RunWithSpan {
  run: Run;
  span: Span;
}

function runTypeToSpanType(
  runType: string
): SpanType {
  switch (runType) {
    case "llm":
    case "chat_model":
      return SpanType.GENERATION;
    case "retriever":
      return SpanType.RETRIEVAL;
    case "tool":
      return SpanType.TOOL;
    case "chain":
      return SpanType.WORKFLOW_STEP;
    default:
      return SpanType.UNKNOWN;
  }
}

function safeStringify(v: unknown): string {
  if (typeof v === "string") return v;
  try {
    return JSON.stringify(v);
  } catch {
    return String(v);
  }
}

/**
 * Extract token usage from a LangChain run's outputs. LangChain stores
 * token counts in different locations depending on the provider and
 * version.
 */
function extractTokenUsage(
  run: Run
): {
  totalTokens?: number;
  promptTokens?: number;
  completionTokens?: number;
  model?: string;
} {
  const outputs = run.outputs as Record<string, any> | undefined;
  if (!outputs) return {};

  // ChatOpenAI / BaseChatModel path: outputs.llmOutput.tokenUsage
  const llmOutput = outputs.llmOutput ?? outputs.llm_output;
  if (llmOutput) {
    const usage =
      llmOutput.tokenUsage ?? llmOutput.token_usage ?? llmOutput.usage;
    if (usage) {
      return {
        totalTokens:
          usage.totalTokens ?? usage.total_tokens ?? undefined,
        promptTokens:
          usage.promptTokens ?? usage.prompt_tokens ?? undefined,
        completionTokens:
          usage.completionTokens ??
          usage.completion_tokens ??
          undefined,
        model:
          llmOutput.model ?? llmOutput.model_name ?? undefined,
      };
    }
  }

  // Some providers put usage directly on outputs
  if (outputs.usage) {
    const u = outputs.usage;
    return {
      totalTokens: u.totalTokens ?? u.total_tokens,
      promptTokens: u.promptTokens ?? u.prompt_tokens,
      completionTokens: u.completionTokens ?? u.completion_tokens,
      model: outputs.model ?? undefined,
    };
  }

  return {};
}

/**
 * Extract retrieved documents from a retriever run.
 * LangChain retriever runs store documents in `outputs.documents`.
 */
function extractRetrievalDocs(run: Run): string[] | undefined {
  const outputs = run.outputs as Record<string, any> | undefined;
  if (!outputs) return undefined;

  const docs: any[] | undefined = outputs.documents ?? outputs.docs;
  if (!Array.isArray(docs)) return undefined;

  return docs.map((d: any) => {
    if (typeof d === "string") return d;
    return d.pageContent ?? d.page_content ?? safeStringify(d);
  });
}

/**
 * Extract the query text that was fed to a retriever.
 */
function extractRetrieverQuery(run: Run): string | undefined {
  const inputs = run.inputs as Record<string, any> | undefined;
  if (!inputs) return undefined;
  if (typeof inputs.query === "string") return inputs.query;
  if (typeof inputs.input === "string") return inputs.input;
  if (typeof inputs.question === "string") return inputs.question;
  return undefined;
}

export class TruLensLangChainTracer extends BaseTracer {
  name = "TruLensLangChainTracer";

  private runs: Record<string, RunWithSpan> = {};

  protected persistRun(_run: Run): Promise<void> {
    return Promise.resolve();
  }

  // -----------------------------------------------------------------
  // Run creation — called on both LC 0.2+ and 0.3+
  // -----------------------------------------------------------------

  /**
   * LC >= 0.2 uses onRunCreate. We implement both paths.
   */
  async onRunCreate(run: Run): Promise<void> {
    if (typeof super.onRunCreate === "function") {
      await super.onRunCreate(run);
    }
    this._startSpan(run);
  }

  /**
   * LC 0.1 path — older versions call _startTrace instead.
   */
  protected async _startTrace(run: Run): Promise<void> {
    const proto = Object.getPrototypeOf(
      Object.getPrototypeOf(this)
    );
    if (proto && typeof proto._startTrace === "function") {
      await proto._startTrace.call(this, run);
    }
    this._startSpan(run);
  }

  private _startSpan(run: Run): void {
    // Avoid double-starting if both hooks fire
    if (this.runs[run.id]) return;

    const tracer = trace.getTracer(TRACER_NAME);
    const spanType = runTypeToSpanType(run.run_type);

    // Resolve parent context so child spans nest correctly
    let activeCtx = context.active();
    const parentSpanCtx = this._parentSpanContext(run);
    if (parentSpanCtx) {
      activeCtx = trace.setSpanContext(activeCtx, parentSpanCtx);
    }

    const span = tracer.startSpan(
      run.name,
      { attributes: { [SpanAttributes.SPAN_TYPE]: spanType } },
      activeCtx
    );

    // Propagate RECORD_ID from baggage (set by withRecord / createTruApp)
    const recordId = propagation
      .getBaggage(context.active())
      ?.getEntry(SpanAttributes.RECORD_ID)?.value;
    if (recordId) {
      span.setAttribute(SpanAttributes.RECORD_ID, recordId);
    }

    span.setAttribute(SpanAttributes.CALL.FUNCTION, run.name);

    // Set input attributes based on run type
    if (spanType === SpanType.RETRIEVAL) {
      const query = extractRetrieverQuery(run);
      if (query) {
        span.setAttribute(SpanAttributes.RETRIEVAL.QUERY_TEXT, query);
      }
    }

    // For LLM runs, record the model if available in extra
    if (spanType === SpanType.GENERATION) {
      const extra = run.extra as Record<string, any> | undefined;
      const invocationParams =
        extra?.invocation_params ?? extra?.invocationParams;
      if (invocationParams) {
        const model =
          invocationParams.model ??
          invocationParams.model_name ??
          invocationParams.modelName;
        if (model) {
          span.setAttribute(SpanAttributes.COST.MODEL, String(model));
        }
      }
    }

    this.runs[run.id] = { run, span };
  }

  // -----------------------------------------------------------------
  // Run end — called for all versions
  // -----------------------------------------------------------------

  protected async _endTrace(run: Run): Promise<void> {
    await super._endTrace(run);

    const entry = this.runs[run.id];
    if (!entry) return;

    const { span } = entry;
    const spanType = runTypeToSpanType(run.run_type);

    // Error handling
    if (run.error) {
      span.recordException(
        typeof run.error === "string"
          ? new Error(run.error)
          : (run.error as Error)
      );
      span.setAttribute(SpanAttributes.CALL.ERROR, String(run.error));
      span.setStatus({
        code: SpanStatusCode.ERROR,
        message: String(run.error),
      });
    } else {
      span.setStatus({ code: SpanStatusCode.OK });
    }

    // Set output attributes based on run type
    if (spanType === SpanType.GENERATION) {
      const { totalTokens, promptTokens, completionTokens, model } =
        extractTokenUsage(run);
      if (model) {
        span.setAttribute(SpanAttributes.COST.MODEL, model);
      }
      if (totalTokens != null) {
        span.setAttribute(SpanAttributes.COST.NUM_TOKENS, totalTokens);
      }
      if (promptTokens != null) {
        span.setAttribute(
          SpanAttributes.COST.NUM_PROMPT_TOKENS,
          promptTokens
        );
      }
      if (completionTokens != null) {
        span.setAttribute(
          SpanAttributes.COST.NUM_COMPLETION_TOKENS,
          completionTokens
        );
      }
    }

    if (spanType === SpanType.RETRIEVAL) {
      const docs = extractRetrievalDocs(run);
      if (docs) {
        span.setAttribute(
          SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
          safeStringify(docs)
        );
        span.setAttribute(
          SpanAttributes.RETRIEVAL.NUM_CONTEXTS,
          docs.length
        );
      }
    }

    // Generic input/output capture
    if (run.inputs) {
      span.setAttribute(
        SpanAttributes.CALL.KWARGS,
        safeStringify(run.inputs)
      );
    }
    if (run.outputs) {
      span.setAttribute(
        SpanAttributes.CALL.RETURN,
        safeStringify(run.outputs)
      );
    }

    span.end();
    delete this.runs[run.id];
  }

  // -----------------------------------------------------------------
  // Helpers
  // -----------------------------------------------------------------

  private _parentSpanContext(run: Run): SpanContext | undefined {
    if (!run.parent_run_id) return undefined;
    const parent = this.runs[run.parent_run_id];
    return parent?.span.spanContext();
  }
}
