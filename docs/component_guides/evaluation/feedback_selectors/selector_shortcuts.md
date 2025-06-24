# Using Shortcuts to Evaluate Pre-defined Span Attributes

Span attributes can pre-defined to refer to particular parts of an execution flow via the [TruLens semantic conventions](../../../otel/semantic_conventions.md). To ease the evaluation of particular span attributes, TruLens creates shortcuts to evaluate commonly used semantic conventions. These shortcuts are supported via `TruApp`, as well as `TruChain` and `TruLlama` when using `LangChain` and `LlamaIndex` frameworks, respectively.

!!! note

    Use of selector shortcuts respects the order of arguments passed to the feedback function, rather than requiring the use of named arguments.

## Evaluating App Input

To evaluate the application input, you can use the selector shortcut `on_input()` to refer to the span attribute `RECORD_ROOT.INPUT`.

This means that the following feedback function using the `Selector`:

```python
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector

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
from trulens.core import Feedback

f_answer_relevance = (
    Feedback(provider.coherence, name="Coherence")
    .on_input()
)
```

## Evaluating App Output

Likewise, to evaluate the application output, you can use the selector shortcut `on_output()` to refer to the span attribute `RECORD_ROOT.OUTPUT`.

This means that the following feedback function using the `Selector`:

```python
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector

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
from trulens.core import Feedback

f_coherence = (
    Feedback(provider.coherence, name="Coherence")
    .on_output()
)
```

## Evaluating Retrieved Context

To evaluate the retrieved context, you can use the selector shortcut `on_context()` to refer to the span attribute `RETRIEVAL.RETRIEVED_CONTEXTS`.

This means that the following feedback function using the `Selector`:

```python
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector

f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on({
        "context": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=True
        ),
    })
    .on_output()
)
```

...is equivalent to using the shortcut `on_context()`.

```python
from trulens.core import Feedback

f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on_context(collect_list=True)
    .on_output()
)
```

!!! note

    `collect_list` can also be passed as an argument to `on_context` to achieve the same effect as when passed to `Selector`.
