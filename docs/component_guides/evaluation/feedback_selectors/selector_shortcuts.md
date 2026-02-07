# Using Selector Shortcuts

The `Selector` class provides static methods that serve as shortcuts for commonly used span selections. These shortcuts are supported via `TruApp`, as well as `TruChain` and `TruLlama` when using `LangChain` and `LlamaIndex` frameworks, respectively.

## Evaluating App Input

To evaluate the application input, use `Selector.select_record_input()`:

```python
from trulens.core import Metric, Selector

f_coherence = Metric(
    implementation=provider.coherence,
    name="Coherence",
    selectors={
        "text": Selector.select_record_input(),
    },
)
```

This is equivalent to the explicit selector:

```python
from trulens.core import Metric, Selector
from trulens.otel.semconv.trace import SpanAttributes

f_coherence = Metric(
    implementation=provider.coherence,
    name="Coherence",
    selectors={
        "text": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
        ),
    },
)
```

## Evaluating App Output

To evaluate the application output, use `Selector.select_record_output()`:

```python
from trulens.core import Metric, Selector

f_coherence = Metric(
    implementation=provider.coherence,
    name="Coherence",
    selectors={
        "text": Selector.select_record_output(),
    },
)
```

This is equivalent to the explicit selector:

```python
from trulens.core import Metric, Selector
from trulens.otel.semconv.trace import SpanAttributes

f_coherence = Metric(
    implementation=provider.coherence,
    name="Coherence",
    selectors={
        "text": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
        ),
    },
)
```

## Evaluating Retrieved Context

To evaluate retrieved context, use `Selector.select_context()`:

```python
from trulens.core import Metric, Selector

f_groundedness = Metric(
    implementation=provider.groundedness_measure_with_cot_reasons,
    name="Groundedness",
    selectors={
        "source": Selector.select_context(collect_list=True),
        "statement": Selector.select_record_output(),
    },
)
```

This is equivalent to the explicit selector:

```python
from trulens.core import Metric, Selector
from trulens.otel.semconv.trace import SpanAttributes

f_groundedness = Metric(
    implementation=provider.groundedness_measure_with_cot_reasons,
    name="Groundedness",
    selectors={
        "source": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=True,
        ),
        "statement": Selector.select_record_output(),
    },
)
```

## Using `collect_list`

The `collect_list` parameter controls how multiple matching spans are handled:

- `collect_list=True`: Concatenates all matching span attributes into a single value for evaluation
- `collect_list=False`: Evaluates each span attribute individually and aggregates results

This is particularly useful for context relevance where you want to evaluate each context chunk separately:

```python
from trulens.core import Metric, Selector
import numpy as np

f_context_relevance = Metric(
    implementation=provider.context_relevance_with_cot_reasons,
    name="Context Relevance",
    selectors={
        "question": Selector.select_record_input(),
        "context": Selector.select_context(collect_list=False),
    },
    agg=np.mean,
)
```

## Summary of Selector Shortcuts

| Shortcut | Span Type | Span Attribute |
|----------|-----------|----------------|
| `Selector.select_record_input()` | `RECORD_ROOT` | `RECORD_ROOT.INPUT` |
| `Selector.select_record_output()` | `RECORD_ROOT` | `RECORD_ROOT.OUTPUT` |
| `Selector.select_context(collect_list=...)` | `RETRIEVAL` | `RETRIEVAL.RETRIEVED_CONTEXTS` |
