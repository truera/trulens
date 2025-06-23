# Selecting Spans for Evaluation

LLM applications come in all shapes and sizes and with a variety of different
control flows. As a result it’s a challenge to consistently evaluate parts of an
LLM application trace.

Therefore, we’ve adapted the use of [OpenTelemetry spans](https://opentelemetry.io/docs/specs/otel/overview/#spans)
to refer to parts of an execution flow when defining evaluations.

## Selecting Span Attributes for Evaluation

When defining evaluations, we want to evaluate particular span attributes, such as retrieved context, or an agent's plan.

This happens in two phases:

1. Instrumentation is used to annotate span attributes. This is covered in detail in the [instrumentation guide](../../instrumentation/index.md).
2. Then when defining the evaluation, you can refer to those span attributes using the `Selector`.

Let's walk through an example. Take this example where a method named `query` is instrumented. In this example, we annotate both the span type, and set span attributes to refer to the `query` argument to the function and the `return` of the function.

```python
@instrument(
    attributes={
        SpanAttributes.RECORD_ROOT.INPUT: "query",
        SpanAttributes.RECORD_ROOT.OUTPUT: "return",
    },
)
def query(self, query: str) -> str:
    context_str = self.retrieve(query=query)
    completion = self.generate_completion(query=query, context_str=context_str)
    return completion
```

Once we've done this, now we can map the inputs to a feedback function to these span attributes:

```python
from trulens.core.feedback.selector import Selector

f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on({
        "prompt": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
        ),
    })
    .on({
        "response": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
        ),
    })
)
```

In the example above, you can see how a dictionary is passed to `on()` that maps the feedback function argument to a span attribute, accessed via a `Selector()`.

## Using Shortcuts to Evaluate Pre-defined Span Attributes

Span attributes can pre-defined to refer to particular parts of an execution flow via the [TruLens semantic conventions](../../../otel/semantic_conventions.md). To ease the evaluation of particular span attributes, TruLens creates shortcuts to evaluate commonly used semantic conventions. These shortcuts are supported via `TruApp`, as well as `TruLlama` and `TruChain` when using `LangChain` and `LlamaIndex` frameworks.

!!! note

    Use of selector shortcuts respects the order of arguments passed to the feedback function, rather than requiring the use of named arguments.

### Evaluating App Input

To evaluate the application input, you can use the selector shortcut `on_input()` to refer to the span attribute `RECORD_ROOT.INPUT`.

This means that the following feedback function using the `Selector`:

```python
f_answer_relevance = (
    Feedback(provider.coherence, name="Coherence")
    .on({
        "text": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
        ),
    })
)
```

...is equivalent to using the shortcut `on_input()`.

```python
f_answer_relevance = (
    Feedback(provider.coherence, name="Coherence")
    .on_input()
)
```

### Evaluating App Output

Likewise, to evaluate the application output, you can use the selector shortcut `on_output()` to refer to the span attribute `RECORD_ROOT.OUTPUT`.

This means that the following feedback function using the `Selector`:

```python
f_coherence = (
    Feedback(provider.coherence, name="Coherence")
    .on({
        "text": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
        ),
    })
)
```

...is equivalent to using the shortcut `on_output()`.

```python
f_coherence = (
    Feedback(provider.coherence, name="Coherence")
    .on_output()
)
```

### Evaluating Retrieved Context

To evaluate the retrieved context, you can use the selector shortcut `on_context()` to refer to the span attribute `RETRIEVAL.RETRIEVED_CONTEXTS`.

This means that the following feedback function using the `Selector`:

```python
f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on({
        "context": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
        ),
    })
    .on_output()
)
```

...is equivalent to using the shortcut `on_context()`.

```python
f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on_context(collect_list=True)
    .on_output()
)
```

Additionally, in the above you see we set the `collect_list` argument in `on_context()`. This allows us to concatenate the selected span attributes into a single blob for evaluation when set to `True`.

When set to `False`, each piece of context will be evaluated individually.

The `on_context()` shortcut can also be used for `LangChain` and `LlamaIndex` apps to refer to the retrieved context. Doing so does not require annotating your app with the `RETRIEVAL.RETRIEVED_CONTEXT` span attribute, as that is done for you.
