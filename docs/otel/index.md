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

## Disabling OTEL tracing

OTEL tracing is **enabled by default** in TruLens. To disable it, set:

```bash
export TRULENS_OTEL_TRACING=0
```

Or in Python before importing TruLens:

```python
import os

os.environ["TRULENS_OTEL_TRACING"] = "0"
```

## Exporting to an OTLP backend

Install the OTLP exporter extra before using this option:

```bash
pip install "trulens[otlp]"
```

For applications that depend directly on `trulens-core`, install
`trulens-core[otlp]` instead.

The current OTLP gRPC exporter release resolves against the matching
OpenTelemetry SDK release line. The lockfile currently resolves
`opentelemetry-exporter-otlp-proto-grpc` to `1.44.0`, which requires
`opentelemetry-sdk>=1.44.0,<1.45.0`. If your application pins a newer SDK,
install a compatible exporter/SDK pair or isolate the TruLens OTLP
environment.

The default exporter writes TruLens spans to the configured TruLens database.
To send traces and GenAI metrics to an OTLP-compatible collector, select the
OTLP gRPC exporter when creating the session:

```python
from trulens.core import TruSession

session = TruSession(
    otel_exporter="otlp",
    otlp_endpoint="http://localhost:4317",
)
```

This option uses gRPC transport. HTTP/protobuf export is not currently
supported; use a gRPC collector endpoint such as port `4317`, not the HTTP
default port `4318`.

!!! warning

    OTLP mode replaces the TruLens database/Snowflake span exporter. Spans are
    sent to the OTLP backend instead of being persisted to the TruLens
    database, so the TruLens dashboard and feedback evaluations that read
    spans from that database will not see them. This option is not a fan-out
    configuration.

When `otlp_endpoint` is omitted, the standard OpenTelemetry environment
variables are used, including `OTEL_EXPORTER_OTLP_ENDPOINT` and
`OTEL_EXPORTER_OTLP_HEADERS`. OTLP mode exports TruLens traces and the
following histogram metrics:

- `gen_ai.client.token.usage`, with `gen_ai.token.type` set to `input` or
  `output`.
- `gen_ai.client.operation.duration`, measured in seconds.

Metrics use the standard `gen_ai.operation.name` and `gen_ai.provider.name`
attributes. The existing `gen_ai.system` span attribute remains available for
backward compatibility.

Call `session.force_flush()` before process exit when the application needs to
wait for pending trace and metric exports.

### Metric results as `gen_ai.evaluation.result` events

Each metric result is also written as an OTEL GenAI
[`gen_ai.evaluation.result`](https://github.com/open-telemetry/semantic-conventions/blob/v1.39.0/docs/gen-ai/gen-ai-events.md)
span event, so tools that read the GenAI conventions can read _TruLens_ scores
without a _TruLens_-specific mapping. No configuration is needed. The event is
attached to the `EVAL_ROOT` span of the result, just before that span ends.
That span links to the evaluated span through `source_span_contexts`. The
existing `EVAL_ROOT` and `EVAL` spans and their `ai.observability.*` attributes
are unchanged.

| Event attribute | Value |
|---|---|
| `gen_ai.evaluation.name` | The metric name (`ai.observability.eval_root.metric_name`). |
| `gen_ai.evaluation.score.value` | The score written to `ai.observability.eval_root.score`. Not set when the evaluation fails or the score is not a number. |
| `gen_ai.evaluation.explanation` | The judge's reason (an `explanation`, `explanations`, `reason`, or `reasons` metadata key), converted the same way as `ai.observability.eval.explanation`. Not set when the score is aggregated over several sub-evaluations. |
| `error.type` | The exception class name, when the evaluation fails. `non_numeric_score` when a custom aggregator returns a score that is not a number. |

`gen_ai.evaluation.score.label` is not set, because _TruLens_ scores are
numeric. Like the rest of `gen_ai.*`, the event is at Development stability in
the OTEL specification.

The event is stored with its span in every destination: the _TruLens_ database
(under `record["events"]`), the _Snowflake_ span proto, and OTLP exports.

Metrics read the app's spans from the _TruLens_ database, so to see metric
results in an OTLP backend such as _Jaeger_, keep the default database exporter
and add an OTLP span processor to the tracer provider. For example, start
_Jaeger_, which accepts OTLP gRPC on port `4317`:

```shell
docker run --rm -p 16686:16686 -p 4317:4317 jaegertracing/jaeger:latest
```

Then add the processor after the session is created:

```python
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from trulens.core import TruSession

session = TruSession()
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(OTLPSpanExporter(endpoint="http://localhost:4317"))
)

# ... record the app with `tru_app` and compute its metrics as usual ...
tru_app.compute_feedbacks()
session.force_flush()
```

In the _Jaeger_ UI at `http://localhost:16686`, open the trace and select an
`EVAL_ROOT` span. Its span events (listed as logs in some _Jaeger_ versions)
include one `gen_ai.evaluation.result` event:

```text
EVAL_ROOT
  ai.observability.span_type = "eval_root"
  ai.observability.eval_root.metric_name = "Groundedness"
  ai.observability.eval_root.score = 0.99

  EVENT gen_ai.evaluation.result
    ATTRIBUTE gen_ai.evaluation.name = "Groundedness"
    ATTRIBUTE gen_ai.evaluation.score.value = 0.99
    ATTRIBUTE gen_ai.evaluation.explanation = "<judge's reason>"
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
    def retrieve(self, query: str) -> list: ...

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate(self, query: str, contexts: list) -> str: ...

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str) -> str: ...
```

The span type determines which feedback selectors can target the span and what attributes
are available for evaluation.

## Deeper guides

- [Semantic Conventions](./semantic_conventions.md) — full attribute reference
- [Instrumentation Overview](../component_guides/instrumentation/index.md) — how to use `@instrument`
- [Feedback Selectors](../component_guides/evaluation/feedback_selectors/index.md) — selecting span attributes for evaluation
