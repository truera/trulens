---
skill_spec_version: 0.1.0
name: trulens-evaluation-setup
version: 1.0.0
description: Configure feedback functions and selectors for TruLens evaluations
tags: [trulens, llm, evaluation, feedback, selectors]
---

# TruLens Evaluation Setup

Configure feedback functions to evaluate your LLM app's quality.

## Overview

Feedback functions evaluate specific aspects of your app by:

1. Selecting data from instrumented spans (inputs, outputs, retrieved contexts)
2. Applying evaluation logic (LLM-as-judge, similarity metrics, etc.)
3. Returning scores between 0.0 and 1.0

## Prerequisites

```bash
pip install trulens trulens-providers-openai
```

## Instructions

### Step 1: Initialize a Feedback Provider

```python
from trulens.providers.openai import OpenAI

provider = OpenAI(model_engine="gpt-4o")
```

### Step 2: Create Feedback Functions with Selector Shortcuts

TruLens provides shortcuts for common selection patterns:

```python
from trulens.core import Feedback

# Answer relevance: input → output
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)

# Context relevance: input → each context chunk
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on_context(collect_list=False)  # Evaluates each context individually
)

# Groundedness: all contexts → output
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on_context(collect_list=True)  # Concatenates all contexts
    .on_output()
)
```

**Shortcut Reference:**

| Shortcut | Selects | Span Attribute |
|----------|---------|----------------|
| `on_input()` | App input | `RECORD_ROOT.INPUT` |
| `on_output()` | App output | `RECORD_ROOT.OUTPUT` |
| `on_context()` | Retrieved contexts | `RETRIEVAL.RETRIEVED_CONTEXTS` |

### Step 3: Using Explicit Selectors

For more control, use `Selector` to target specific span attributes:

```python
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector
from trulens.otel.semconv.trace import SpanAttributes

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

### Step 4: Understanding collect_list

The `collect_list` parameter controls how multiple values are handled:

| Setting | Behavior | Use Case |
|---------|----------|----------|
| `collect_list=False` | Evaluate each value individually | Context relevance (score each chunk) |
| `collect_list=True` | Concatenate all values | Groundedness (check against all context) |

```python
# Evaluate each retrieved context individually (returns multiple scores)
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on({
        "context": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=False  # Each context evaluated separately
        ),
    })
)

# Evaluate against all contexts combined (returns single score)
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on({
        "context": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=True  # All contexts concatenated
        ),
    })
    .on_output()
)
```

### Step 5: Aggregating Multiple Scores

When `collect_list=False` produces multiple scores, aggregate them:

```python
import numpy as np

f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on_context(collect_list=False)
    .aggregate(np.mean)  # Average all context relevance scores
)
```

Common aggregation functions:
- `np.mean` - Average score
- `np.min` - Worst score (conservative)
- `np.max` - Best score (optimistic)

## Common Patterns

### RAG Triad Setup

```python
import numpy as np
from trulens.core import Feedback
from trulens.providers.openai import OpenAI

provider = OpenAI()

# Context Relevance: Is each retrieved chunk relevant to the query?
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on_context(collect_list=False)
    .aggregate(np.mean)
)

# Groundedness: Is the response grounded in the retrieved context?
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on_context(collect_list=True)
    .on_output()
)

# Answer Relevance: Does the response answer the original question?
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)

rag_feedbacks = [f_context_relevance, f_groundedness, f_answer_relevance]
```

### Agent GPA Setup

```python
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector
from trulens.providers.openai import OpenAI

provider = OpenAI()

# All Agent GPA metrics use trace-level selection
trace_selector = {"trace": Selector(trace_level=True)}

# Logical Consistency
f_logical_consistency = (
    Feedback(provider.logical_consistency_with_cot_reasons, name="Logical Consistency")
    .on(trace_selector)
)

# Plan Quality (exclude if agent doesn't do explicit planning)
f_plan_quality = (
    Feedback(provider.plan_quality_with_cot_reasons, name="Plan Quality")
    .on(trace_selector)
)

# Plan Adherence (exclude if agent doesn't do explicit planning)
f_plan_adherence = (
    Feedback(provider.plan_adherence_with_cot_reasons, name="Plan Adherence")
    .on(trace_selector)
)

# Execution Efficiency
f_execution_efficiency = (
    Feedback(provider.execution_efficiency_with_cot_reasons, name="Execution Efficiency")
    .on(trace_selector)
)

# Tool Selection
f_tool_selection = (
    Feedback(provider.tool_selection_with_cot_reasons, name="Tool Selection")
    .on(trace_selector)
)

# Tool Calling
f_tool_calling = (
    Feedback(provider.tool_calling_with_cot_reasons, name="Tool Calling")
    .on(trace_selector)
)

# Tool Quality
f_tool_quality = (
    Feedback(provider.tool_quality_with_cot_reasons, name="Tool Quality")
    .on(trace_selector)
)

# Use all for agents with planning
agent_feedbacks_with_planning = [
    f_logical_consistency,
    f_plan_quality,
    f_plan_adherence,
    f_execution_efficiency,
    f_tool_selection,
    f_tool_calling,
    f_tool_quality,
]

# For agents without explicit planning, exclude plan metrics
agent_feedbacks_no_planning = [
    f_logical_consistency,
    f_execution_efficiency,
    f_tool_selection,
    f_tool_calling,
    f_tool_quality,
]
```

### Custom Feedback Function

```python
def my_custom_metric(input_text: str, output_text: str) -> float:
    """Custom evaluation returning score between 0.0 and 1.0."""
    # Your evaluation logic here
    score = len(output_text) / (len(input_text) + len(output_text))
    return min(max(score, 0.0), 1.0)

f_custom = (
    Feedback(my_custom_metric, name="Custom Metric")
    .on_input()
    .on_output()
)
```

## Troubleshooting

- **Selector not finding data**: Ensure the span attribute was set during instrumentation
- **Empty context**: Verify `RETRIEVAL.RETRIEVED_CONTEXTS` is mapped in your `@instrument()` decorator
- **Aggregation errors**: Check that `collect_list=False` is set when using `.aggregate()`
