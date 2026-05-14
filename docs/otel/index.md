# 🔭 OpenTelemetry

TruLens is built on [OpenTelemetry (OTEL)](https://opentelemetry.io/), the industry-standard
observability framework. Every function call, LLM generation, retrieval, and tool invocation
in your app is captured as an OTEL span, giving you a structured, queryable trace of your
application's execution.

## How TruLens uses OTEL

When you instrument an app with TruLens, the library emits OTEL spans for each tracked
operation. These spans carry semantic attributes (defined in TruLens'
[Semantic Conventions](./semantic_conventions.md)) that describe what happened — the query
sent to a retriever, the contexts retrieved, the LLM prompt and response, and so on.

TruLens works seamlessly with existing OTEL setups in two directions:

1. **TruLens consuming external spans** — TruLens can read spans emitted by non-TruLens
   code (e.g., from `opentelemetry-instrumentation-openai`) and use their attributes as
   inputs to feedback functions.
2. **Existing OTEL backends consuming TruLens spans** — TruLens spans can be exported to
   any OTEL-compatible backend (Jaeger, Grafana Tempo, Datadog, Honeycomb) alongside your
   existing traces.

## Enabling OTEL tracing

OTEL tracing is **enabled by default** in TruLens. To disable it, set:

```bash
export TRULENS_OTEL_TRACING=0
```

Or in Python before importing TruLens:

```python
import os
os.environ["TRULENS_OTEL_TRACING"] = "0"
```

## Span type taxonomy

TruLens defines a set of span types that describe the role of each span in a trace.
Each span type has a corresponding set of semantic attributes in the
`ai.observability.*` namespace.

| Span type | Description |
|-----------|-------------|
| `RECORD_ROOT` | Root span for a single app invocation (one "record"). Carries `input` and `output`. |
| `RETRIEVAL` | A context retrieval operation. Carries `query_text` and `retrieved_contexts`. |
| `GENERATION` | An LLM generation call. Carries model and token usage information. |
| `AGENT` | An agent execution step. |
| `TOOL` | A tool or function call made by an agent. |
| `MCP` | A Model Context Protocol tool call. Carries tool name, arguments, and output. |
| `GRAPH_TASK` | A task node in an agentic graph (e.g., LangGraph node). |
| `GRAPH_NODE` | A graph node execution. |
| `WORKFLOW` | A workflow step in an event-driven agent (e.g., LlamaIndex workflow). |
| `RERANKING` | A reranking operation applied to retrieved contexts. |
| `EVAL_ROOT` | Root span for a feedback evaluation. Carries metric name and score. |
| `EVAL` | A sub-step within a feedback evaluation. |
| `UNKNOWN` | Default span type when no type is specified. |

## Using span types with `@instrument`

Attach a span type to any instrumented method using the `span_type` parameter:

```python
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

class MyRAG:
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str) -> list:
        ...

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate(self, query: str, contexts: list) -> str:
        ...

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str) -> str:
        ...
```

The span type determines which feedback selectors can target the span and what attributes
are available for evaluation.

## Deeper guides

- [Semantic Conventions](./semantic_conventions.md) — full attribute reference
- [Instrumentation Overview](../component_guides/instrumentation/index.md) — how to use `@instrument`
- [Feedback Selectors](../component_guides/evaluation/feedback_selectors/index.md) — selecting span attributes for evaluation
