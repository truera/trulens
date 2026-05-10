---
skill_spec_version: 0.1.0
name: trulens-diagnosis
version: 1.0.0
description: Diagnose low evaluation scores and generate actionable improvement recommendations
tags: [trulens, llm, evaluation, diagnosis, improvement, rag, agents]
---

# TruLens Diagnosis and Improvement

After running evaluations, use this skill to diagnose *why* scores are low
and *what to change* — closing the loop from eval results to concrete fixes.

## Prerequisites

Before diagnosing, ensure you have:

1. **Run evaluations** with at least one feedback function (see `running-evaluations` skill)
2. **Access to the session** to retrieve records and feedback results

## Instructions

### Step 1: Triage — identify failing metrics and patterns

Pull records with feedback scores and filter to failing cases:

```python
import pandas as pd
from trulens.core import TruSession

session = TruSession()

# Get all records with feedback columns
records_df, feedback_cols = session.get_records_and_feedback()

# Identify feedback functions present
print("Feedback functions:", list(feedback_cols))

# Filter to low-scoring records (score < 0.7)
FAIL_THRESHOLD = 0.7
failing = {}
for col in feedback_cols:
    if col in records_df.columns:
        low = records_df[records_df[col] < FAIL_THRESHOLD]
        if not low.empty:
            failing[col] = low
            print(f"\n{col}: {len(low)} failing records out of {len(records_df)}")
            print(low[["input", "output", col]].head(5))
```

Look for patterns:
- Which feedback functions fail most often?
- Are failures clustered around specific queries or query types?
- Do failures correlate with specific app versions?

### Step 2: Root cause analysis — trace failures to their span

For each failing metric, inspect the OTEL trace to find the problematic span:

```python
import json

# Get the record_id of a failing record
failing_record_id = failing["Context Relevance"].iloc[0]["record_id"]

# Fetch the full trace for that record
records = session.get_records_and_feedback()[0]
record = records[records["record_id"] == failing_record_id].iloc[0]

# Inspect the raw trace JSON
trace = json.loads(record["record_json"])
print(json.dumps(trace, indent=2))
```

Use the dashboard to inspect individual span attributes:

```python
from trulens.dashboard import run_dashboard
run_dashboard(session)
# Navigate to Records → click a failing record → expand spans
```

**Span-to-metric mapping:**

| Failing metric | Look at this span type |
|---------------|----------------------|
| Context Relevance | `RETRIEVAL` span — check `query_text` vs `retrieved_contexts` |
| Groundedness | `GENERATION` span — check if output is grounded in retrieved contexts |
| Answer Relevance | `RECORD_ROOT` span — check if output answers the input |
| Tool Selection | `TOOL` / `MCP` span — check tool chosen vs task goal |
| Tool Calling | `TOOL` / `MCP` span — check argument values and output interpretation |
| Plan Adherence | `AGENT` spans — check if execution matches plan steps |
| Logical Consistency | Full trace — check for unsupported assertions or contradictions |

### Step 3: Apply targeted fixes

Use the failure pattern and root-cause span to select the right fix:

#### Low Context Relevance (retrieved contexts don't match query)

```python
# Symptom: RETRIEVAL span query_text doesn't match retrieved_contexts

# Fix A: Increase retrieval k
vector_store.query(query_texts=query, n_results=5)  # was 2

# Fix B: Improve chunking — reduce chunk size, add overlap
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=50)

# Fix C: Switch embedding model
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")  # was small

# Fix D: Add query rewriting before retrieval
@instrument(span_type=SpanAttributes.SpanType.RETRIEVAL)
def retrieve(self, query: str) -> list:
    rewritten = rewrite_query(query)  # expand/clarify query
    return vector_store.query(rewritten)
```

#### Low Groundedness (response not grounded in retrieved context)

```python
# Symptom: GENERATION span output contains facts not in retrieved contexts

# Fix A: Strengthen system prompt to cite sources
SYSTEM_PROMPT = """Answer ONLY using the provided context.
If the context doesn't contain the answer, say 'I don't know.'
Do not add information from your training data."""

# Fix B: Filter low-relevance chunks before passing to LLM
def filter_contexts(contexts: list, query: str, threshold: float = 0.7) -> list:
    # Re-rank and filter by relevance score
    return [c for c, score in zip(contexts, scores) if score >= threshold]

# Fix C: Reduce temperature to minimize hallucination
response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0.0,  # was 0.7
    messages=[...]
)
```

#### Low Answer Relevance (response doesn't address the question)

```python
# Symptom: RECORD_ROOT output doesn't align with input question

# Fix A: Add chain-of-thought prompting
SYSTEM_PROMPT = """First, identify what the user is asking.
Then answer that specific question directly.
Question: {query}"""

# Fix B: Add output format instructions
SYSTEM_PROMPT += "\nStart your response by restating the question, then answer."

# Fix C: Use a more capable model for complex questions
model = "gpt-4o"  # upgrade from gpt-4o-mini for complex reasoning
```

#### Low Tool Selection (agent picks wrong tools)

```python
# Symptom: TOOL span tool_name doesn't match the subtask goal

# Fix A: Improve tool descriptions — be explicit about when to use each tool
tools = [
    {
        "name": "search_documents",
        "description": (
            "Search the knowledge base for relevant documents. "
            "Use this for factual questions about stored content. "
            "Do NOT use for real-time data or calculations."
        ),
    },
    ...
]

# Fix B: Add few-shot examples to the agent prompt showing correct tool use
AGENT_PROMPT = """You have these tools: {tools}

Examples of correct tool selection:
- For 'What does the policy say about X?' → use search_documents
- For 'Calculate the total cost' → use calculator
- For 'Get today's weather' → use web_search
"""
```

#### Low Tool Calling (agent forms bad tool arguments)

```python
# Symptom: TOOL span input_arguments are invalid or semantically off

# Fix A: Add argument validation examples to the tool description
# Fix B: Use structured output for tool argument generation
response = client.chat.completions.create(
    model="gpt-4o",
    tools=tools,
    tool_choice="required",
    # Add a system message explaining each argument
)

# Fix C: Add error handling and retry logic
try:
    result = call_tool(tool_name, args)
except ToolArgumentError as e:
    # Re-prompt with the error message
    corrected_args = correct_arguments(args, error=str(e))
    result = call_tool(tool_name, corrected_args)
```

#### Low Plan Adherence (agent doesn't follow its own plan)

```python
# Symptom: AGENT spans don't correspond to planned steps

# Fix A: Simplify the plan — fewer, clearer steps
# Fix B: Add step validation checkpoints
# Fix C: Use structured plan format the model follows more reliably
PLAN_PROMPT = """Create a plan with numbered steps.
After each step, verify it was completed before proceeding to the next."""
```

#### Low Logical Consistency (reasoning contains contradictions)

```python
# Symptom: AGENT/GENERATION spans contain unsupported assertions

# Fix A: Reduce temperature
# Fix B: Add explicit reasoning instructions
PROMPT += "\nThink step by step. Each claim must be traceable to the context."

# Fix C: Add a self-review step
PROMPT += "\nAfter your answer, review it for logical consistency."
```

### Step 4: Re-evaluate — compare before and after

Run the same test set under a new app version and compare:

```python
# New version with fixes applied
tru_v2 = YourWrapper(
    fixed_app,
    app_name="MyApp",
    app_version="v2",  # bump version
    feedbacks=feedbacks,
)

with tru_v2 as recording:
    for query in test_queries:
        fixed_app.query(query)

recording.retrieve_feedback_results(timeout=300)

# Compare on leaderboard
session.get_leaderboard()
```

### Step 5: Regression check — verify fixes didn't hurt other metrics

```python
records_v2, _ = session.get_records_and_feedback(app_versions=["v2"])

for col in feedback_cols:
    if col in records_v2.columns:
        v1_mean = records_df[col].mean()
        v2_mean = records_v2[col].mean()
        delta = v2_mean - v1_mean
        status = "✅" if delta >= 0 else "⚠️"
        print(f"{status} {col}: {v1_mean:.3f} → {v2_mean:.3f} ({delta:+.3f})")
```

Flag any metric that regressed by more than 0.05 — investigate whether the
fix for metric A introduced a failure in metric B.

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No failing records found | Lower the threshold (e.g., `< 0.8`) or check if evaluations completed |
| Trace JSON empty | Ensure `OTEL_TRACING` is enabled and spans are captured |
| Same metric still failing after fix | Check if the fix is actually being applied — add logging to confirm |
| Regression in another metric | The fix may be too aggressive; tune parameters (temperature, k, prompt) |
| Can't identify root cause from trace | Enable debug logging: `import logging; logging.basicConfig(level=logging.DEBUG)` |
