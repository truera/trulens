# Migrating from Feedback to Metric

The `Metric` class is the new unified API for evaluation metrics in TruLens. It replaces both `Feedback` and `MetricConfig` with a cleaner, more explicit interface.

## Why Migrate?

- **Clearer API**: Explicit `selectors` dict instead of chained `.on_*()` methods
- **Better OTel integration**: Native support for OpenTelemetry span selection
- **Unified interface**: One class instead of separate `Feedback` and `MetricConfig`
- **Same functionality**: All existing features are preserved

## Quick Comparison

### Before (Feedback)

```python
from trulens.core import Feedback
from trulens.providers.openai import OpenAI

provider = OpenAI()

f_relevance = (
    Feedback(provider.relevance, name="Answer Relevance")
    .on_input()
    .on_output()
)

f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on_context(collect_list=True)
    .on_output()
    .on_input()
)

f_context_relevance = (
    Feedback(provider.context_relevance, name="Context Relevance")
    .on_input()
    .on_context(collect_list=False)
    .aggregate(np.mean)
)
```

### After (Metric)

```python
from trulens.core import Metric, Selector
from trulens.providers.openai import OpenAI
import numpy as np

provider = OpenAI()

f_relevance = Metric(
    implementation=provider.relevance,
    name="Answer Relevance",
    selectors={
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output(),
    },
)

f_groundedness = Metric(
    implementation=provider.groundedness_measure_with_cot_reasons,
    name="Groundedness",
    selectors={
        "source": Selector.select_context(collect_list=True),
        "statement": Selector.select_record_output(),
        "question": Selector.select_record_input(),
    },
)

f_context_relevance = Metric(
    implementation=provider.context_relevance,
    name="Context Relevance",
    selectors={
        "question": Selector.select_record_input(),
        "context": Selector.select_context(collect_list=False),
    },
    agg=np.mean,
)
```

## Selector Reference

The `Selector` class provides static methods for common selections:

| Method | Description |
|--------|-------------|
| `Selector.select_record_input()` | The input to the traced application |
| `Selector.select_record_output()` | The output from the traced application |
| `Selector.select_context(collect_list=True)` | Retrieved contexts from RAG |

### Custom Selectors

For advanced use cases, create custom selectors by matching span attributes:

```python
from trulens.core import Selector
from trulens.otel.semconv.trace import SpanAttributes

# Select by span type
retrieval_query = Selector(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    function_attribute="query",
)

# Select by function name
custom_output = Selector(
    function_name="my_module.my_function",
    function_attribute="return",
)

# Select specific span attribute
generation_output = Selector(
    span_type=SpanAttributes.SpanType.GENERATION,
    span_attribute=SpanAttributes.GENERATION.OUTPUT,
)
```

## Parameter Mapping

| Feedback | Metric | Notes |
|----------|--------|-------|
| `imp=fn` | `implementation=fn` | Renamed for clarity |
| `.on_input()` | `selectors={"param": Selector.select_record_input()}` | Explicit selector |
| `.on_output()` | `selectors={"param": Selector.select_record_output()}` | Explicit selector |
| `.on_context()` | `selectors={"param": Selector.select_context()}` | Explicit selector |
| `.aggregate(fn)` | `agg=fn` | Same functionality |
| `name="..."` | `name="..."` | Unchanged |
| `examples=[...]` | `examples=[...]` | Unchanged |
| `criteria="..."` | `criteria="..."` | Unchanged |

## New Fields

The `Metric` class adds optional metadata fields:

```python
Metric(
    implementation=my_fn,
    name="My Metric",
    metric_type="custom",  # NEW: Implementation identifier
    description="Measures X quality",  # NEW: Human-readable description
    selectors={...},
)
```

## Migrating from MetricConfig

If you were using `MetricConfig` for custom client-side metrics:

### Before (MetricConfig)

```python
from trulens.core.feedback.custom_metric import MetricConfig

config = MetricConfig(
    metric_name="word_count",
    metric_implementation=lambda text: len(text.split()),
    selectors={"text": Selector.select_record_output()},
)
feedback = config.create_feedback_definition()
```

### After (Metric)

```python
from trulens.core import Metric, Selector

metric = Metric(
    name="word_count",
    implementation=lambda text: len(text.split()),
    selectors={"text": Selector.select_record_output()},
)
```

## Backward Compatibility

The old `Feedback` class still works and will emit a deprecation warning:

```python
# This still works but shows a deprecation warning
from trulens.core import Feedback
f = Feedback(provider.relevance).on_input().on_output()
```

The chained `.on_*()` methods also work on `Metric` for gradual migration, but the `selectors={}` style is recommended for new code.

### Type Checking Notes

The `Feedback` class is implemented as a subclass of `Metric`:

```python
class Feedback(Metric):
    ...
```

This means:

| Check | Result |
|-------|--------|
| `isinstance(metric_obj, Metric)` | `True` |
| `isinstance(feedback_obj, Metric)` | `True` |
| `isinstance(feedback_obj, Feedback)` | `True` |
| `isinstance(metric_obj, Feedback)` | `False` |

**For downstream integrators**: If your code performs type checks with `isinstance(obj, Feedback)`, you should migrate to `isinstance(obj, Metric)` before the `Feedback` alias is removed. After removal:

- `isinstance(obj, Metric)` will continue to work ✅
- `isinstance(obj, Feedback)` will raise `NameError` ❌

```python
# Before (will break when Feedback is removed)
if isinstance(obj, Feedback):
    ...

# After (recommended)
if isinstance(obj, Metric):
    ...
```

## Timeline

- **Future release**: `Feedback` and `MetricConfig` will be removed

We recommend migrating to `Metric` for all new code.
