---
skill_spec_version: 0.1.0
name: trulens-running-evaluations
version: 1.0.0
description: Execute TruLens evaluations and view results
tags: [trulens, llm, evaluation, rag, agents]
---

# TruLens Running Evaluations

Execute your configured evaluations and analyze results.

## Prerequisites

Before running evaluations, ensure you have:

1. **Instrumented your app** (see `instrumentation` skill)
2. **Configured your feedback functions** (see `evaluation-setup` skill)

## Instructions

### Step 1: Wrap Your App with Feedbacks

Pass your configured feedbacks to the appropriate wrapper:

```python
from trulens.core import TruSession

session = TruSession()

# Use the wrapper that matches your framework
tru_app = YourWrapper(
    your_app,
    app_name="MyApp",
    app_version="v1",
    feedbacks=your_feedbacks,  # From evaluation-setup
)
```

| Framework | Wrapper |
|-----------|---------|
| LangChain | `TruChain` |
| LangGraph | `TruGraph` |
| LlamaIndex | `TruLlama` / `TruLlamaWorkflow` |
| Custom | `TruApp` |

### Step 2: Run Your App with Recording

Use the context manager to record traces and run evaluations:

```python
# Single query
with tru_app as recording:
    result = your_app.query("What is TruLens?")

# Multiple queries
test_queries = [
    "What is machine learning?",
    "How does RAG work?",
    "Explain transformers.",
]

with tru_app as recording:
    for query in test_queries:
        your_app.query(query)
```

### Step 3: Wait for and View Results

Evaluations run asynchronously. Use `retrieve_feedback_results()` to wait for them to complete:

```python
# Wait for evaluations to complete and get results as a DataFrame
# The timeout parameter controls how long to wait (default: 180 seconds)
feedback_results = recording.retrieve_feedback_results(timeout=300)
print(feedback_results)

# For a single record:
single_record_results = recording[0].retrieve_feedback_results(timeout=300)

# View leaderboard summary across all records
print(session.get_leaderboard())

# Launch interactive dashboard
from trulens.dashboard import run_dashboard

run_dashboard(session)
```

**Important**: Do NOT use `time.sleep()` to wait for evaluations. The `retrieve_feedback_results()` method properly waits for:
1. Records to be written to the database
2. Feedback evaluations to complete
3. Results to be available

## Common Patterns

### Comparing App Versions

```python
# Version A
tru_v1 = TruLlama(query_engine_v1, app_name="MyRAG", app_version="v1", feedbacks=feedbacks)
with tru_v1 as recording:
    for q in test_queries:
        query_engine_v1.query(q)

# Version B
tru_v2 = TruLlama(query_engine_v2, app_name="MyRAG", app_version="v2", feedbacks=feedbacks)
with tru_v2 as recording:
    for q in test_queries:
        query_engine_v2.query(q)

# Compare on leaderboard (same app_name, different app_version)
print(session.get_leaderboard())
```

### Batch Evaluation with Test Dataset

```python
import pandas as pd

# Load test dataset
test_df = pd.read_csv("test_queries.csv")

with tru_app as recording:
    for _, row in test_df.iterrows():
        result = your_app.query(row["query"])
        # Optionally store results
        # results.append({"query": row["query"], "response": result})
```

### Evaluating with Ground Truth

```python
from trulens.feedback import GroundTruthAgreement

# Load ground truth dataset (see dataset-curation skill)
ground_truth_df = session.get_ground_truth("my_dataset")

# Add ground truth feedback
ground_truth = GroundTruthAgreement(ground_truth_df, provider=provider)
f_agreement = Feedback(ground_truth.agreement_measure, name="Ground Truth Agreement").on_input_output()

# Include with other feedbacks
all_feedbacks = your_feedbacks + [f_agreement]
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No evaluation results | Ensure `feedbacks` list is passed to wrapper |
| Missing context scores | Verify `RETRIEVAL.RETRIEVED_CONTEXTS` is instrumented |
| Agent metrics empty | Check that trace contains tool calls and reasoning |
| Dashboard not loading | Run `pip install trulens-dashboard`, check port 8501 |
| Feedback columns empty | Your root span must use `SpanType.RECORD_ROOT` for `.on_input()/.on_output()` to work. Use framework wrappers (TruGraph, TruChain) which handle this automatically |
| `PydanticForbiddenQualifier` error | Update to latest TruLens version - this error occurs with Deep Agents/LangGraph apps that use `NotRequired` type annotations |
| Results not appearing | Use `recording.retrieve_feedback_results()` instead of `time.sleep()` - it properly waits for evaluations to complete |

### Deep Agents / LangGraph Specific Issues

If evaluating a Deep Agent or LangGraph app:

1. **Use `TruGraph`** instead of `TruApp` + manual instrumentation:
   ```python
   from trulens.apps.langgraph import TruGraph

   tru_agent = TruGraph(agent, app_name="DeepAgent", feedbacks=[...])
   ```

2. **Why?** TruGraph automatically:
   - Creates `RECORD_ROOT` spans (required for `.on_input()/.on_output()`)
   - Captures all graph nodes and transitions
   - Handles LangGraph-specific data structures

3. **Common mistake**: Using `@instrument(span_type=SpanType.AGENT)` instead of `RECORD_ROOT` will cause feedback selector shortcuts to fail silently
