---
skill_spec_version: 0.1.0
name: trulens-evaluation-setup
version: 1.0.0
description: Configure feedback functions and selectors for TruLens evaluations
tags: [trulens, llm, evaluation, feedback, selectors]
---

# TruLens Evaluation Setup

Configure feedback functions to evaluate your LLM app's quality.

## Interactive Evaluation Selection

**Before proceeding, let's determine the right evaluations for your app.**

### Question 1: What type of app are you building?

**Option A: RAG (Retrieval-Augmented Generation)**
- Your app retrieves documents/chunks from a knowledge base
- It generates responses grounded in the retrieved context
- Examples: Q&A systems, document search, knowledge assistants

→ **Recommended: RAG Triad metrics**
- Context Relevance
- Groundedness
- Answer Relevance

**Option B: Agent**
- Your app uses tools to accomplish tasks
- It may involve multi-step reasoning or planning
- Examples: research agents, coding assistants, task automation

→ **Recommended: Agent GPA metrics** (continue to Question 2)

---

### Question 2 (Agents only): Does your agent do explicit planning?

**Yes, my agent creates plans before executing:**
- Agent outputs a plan/strategy before taking actions
- Agent references its plan during execution

→ **Use all Agent GPA metrics:**
- Logical Consistency
- Plan Quality
- Plan Adherence
- Execution Efficiency
- Tool Selection
- Tool Calling
- Tool Quality

**No, my agent acts without explicit planning:**
- Agent takes actions directly without stating a plan
- Agent uses reactive decision-making

→ **Use Agent GPA metrics (excluding plan metrics):**
- Logical Consistency
- Execution Efficiency
- Tool Selection
- Tool Calling
- Tool Quality

---

### Question 3: Do you want to add any additional evaluations?

Consider adding these based on your needs:

| Evaluation | Use Case |
|------------|----------|
| **Coherence** | Check if output is well-structured and readable |
| **Conciseness** | Ensure responses aren't unnecessarily verbose |
| **Harmlessness** | Detect potentially harmful content |
| **Sentiment** | Analyze emotional tone of responses |
| **Custom metrics** | Domain-specific evaluations (see below) |

#### Creating Custom Metrics

If you need domain-specific evaluations, describe what you want to measure:

**What aspect of your app do you want to evaluate?**

Examples:
- "Check if the response follows our brand voice guidelines"
- "Verify the output contains required legal disclaimers"
- "Measure technical accuracy for code generation"
- "Evaluate if customer support responses show empathy"

**Template for custom metrics:**

```python
def my_custom_metric(input_text: str, output_text: str) -> float:
    """
    Describe what this metric evaluates.

    Returns:
        float: Score between 0.0 (worst) and 1.0 (best)
    """
    # Option 1: Rule-based logic
    # score = 1.0 if "required phrase" in output_text else 0.0

    # Option 2: Use LLM-as-judge
    # provider = OpenAI()
    # response = provider.client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{
    #         "role": "user",
    #         "content": f"Rate this response on [YOUR CRITERIA]. Input: {input_text} Output: {output_text}. Return only a number 0-10."
    #     }]
    # )
    # score = float(response.choices[0].message.content) / 10.0

    return score

f_custom = (
    Feedback(my_custom_metric, name="My Custom Metric")
    .on_input()
    .on_output()
)
```

**Custom metric with context:**

```python
def custom_with_context(query: str, context: str, response: str) -> float:
    """Evaluate using query, retrieved context, and response."""
    # Your evaluation logic
    return score

f_custom_context = (
    Feedback(custom_with_context, name="Custom Context Metric")
    .on_input()
    .on_context(collect_list=True)
    .on_output()
)
```

**Tell me what you want to evaluate and I'll help you create the metric!**

---

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

| Shortcut | Selects | Required Span Type |
|----------|---------|-------------------|
| `on_input()` | App input | `RECORD_ROOT` |
| `on_output()` | App output | `RECORD_ROOT` |
| `on_context()` | Retrieved contexts | `RETRIEVAL` |

**⚠️ IMPORTANT: `.on_input()` and `.on_output()` require `RECORD_ROOT` spans!**

These shortcuts look for spans with `span_type=SpanAttributes.SpanType.RECORD_ROOT`. If you use manual instrumentation with a different span type (like `AGENT`), the shortcuts will not find any data.

**Solutions:**
- Use framework wrappers (`TruGraph`, `TruChain`, `TruLlama`) which create `RECORD_ROOT` automatically
- Use explicit `@instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT, ...)` on your entry point
- Use explicit `Selector` objects instead of shortcuts (see Step 3)

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
- **`.on_input()/.on_output()` returning no data**: These shortcuts require `RECORD_ROOT` span type. Use framework wrappers or explicit `@instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT, ...)`. See the instrumentation skill for details.
- **Feedback columns show empty/null**: Verify your instrumentation creates `RECORD_ROOT` spans with `INPUT` and `OUTPUT` attributes
