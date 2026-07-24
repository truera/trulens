# Span Groups — Per-Segment Metric Localization

When one metric matches multiple spans of the same type in a single trace — for
example, groundedness over the 3 retrieval steps of a multi-hop RAG — the default
behaviour pools all matched spans into one score or splits per item.  Neither
gives a score per logical step.

Span groups fix this.  Tag each step's spans with a group label, and
`compute_feedback_by_span_group` runs the metric **once per group**, localizing
a same-metric score to a logical segment of the trace.

## Basic usage

Use the `span_group()` context manager to tag every span created inside the
block:

```python
from trulens.core.otel.instrument import instrument, span_group
from trulens.otel.semconv.trace import SpanAttributes

@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes={
        SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
    },
)
def retrieve(query):
    # your retrieval logic
    ...

with span_group("hop1"):
    ctx1 = retrieve("query 1")
with span_group("hop2"):
    ctx2 = retrieve("query 2")
with span_group("hop3"):
    ctx3 = retrieve("query 3")
```

Each retrieval span will carry:

```
ai.observability.span_groups = ["hop1"]   # or ["hop2"], ["hop3"]
```

A single groundedness metric then produces three scores — one per hop —
pinpointing which retrieval step was ungrounded.

## Nesting

Nesting `span_group()` calls merges the labels:

```python
with span_group("hop1"):
    with span_group("retry"):
        retrieve("q")    # SPAN_GROUPS = ["hop1", "retry"]
```

## How it works

`span_group()` stores the current group name(s) in
[OpenTelemetry Baggage](https://opentelemetry.io/docs/concepts/signals/baggage/).
The `@instrument` decorator reads the baggage when building a span and sets
`SpanAttributes.SPAN_GROUPS` automatically.  No manual argument threading is
required — every instrumented call inside the block inherits the label.

## Patterns this unlocks

| Pattern | How to tag |
|---------|-----------|
| Per-hop quality in multi-hop RAG | `span_group("hop1")`, `span_group("hop2")`, … |
| Per-source quality (web vs knowledge base) | `span_group("web")`, `span_group("kb")` |
| Per-agent-turn quality in a multi-turn loop | `span_group("turn_1")`, `span_group("turn_2")`, … |
| A/B testing two strategies in one trace | `span_group("strategy_a")`, `span_group("strategy_b")` |

## Notes

* The group name must be a string.
* Spans created outside any `span_group()` block will not have
  `SPAN_GROUPS` set, which is fully backwards-compatible.
* The value is stored in OTEL Baggage during the context-manager lifetime and
  does not leak across concurrent invocations that use different groups.
* You can also set `SPAN_GROUPS` explicitly via the `attributes` parameter of
  `@instrument` if you need a span's group to differ from the surrounding block.
