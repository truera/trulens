---
skill_spec_version: 0.1.0
name: trulens-dataset-curation
version: 1.0.0
description: Create and curate evaluation datasets with ground truth for TruLens
tags: [trulens, llm, evaluation, dataset, ground-truth]
---

# TruLens Dataset Curation

Create evaluation datasets with ground truth to measure your LLM app's performance.

## Overview

Ground truth datasets allow you to:

- Compare LLM outputs against expected responses
- Evaluate retrieval quality against expected chunks
- Track performance across app versions
- Share evaluation data across your team

## Prerequisites

```bash
pip install trulens pandas
```

## Instructions

### Step 1: Initialize TruSession

```python
from trulens.core import TruSession

session = TruSession()
```

### Step 2: Create Ground Truth Data

Structure your data as a pandas DataFrame with these columns:

| Column | Required | Description |
|--------|----------|-------------|
| `query` | Yes | The input query/question |
| `query_id` | No | Unique identifier for the query |
| `expected_response` | No | The expected/ideal response |
| `expected_chunks` | No | Expected retrieved contexts (list or string) |

```python
import pandas as pd

data = {
    "query": [
        "What is TruLens?",
        "How do I instrument a LangChain app?",
        "What is the RAG triad?",
    ],
    "query_id": ["q1", "q2", "q3"],
    "expected_response": [
        "TruLens is an open source library for evaluating and tracing AI agents.",
        "Use TruChain to wrap your LangChain app for automatic instrumentation.",
        "The RAG triad consists of context relevance, groundedness, and answer relevance.",
    ],
    "expected_chunks": [
        ["TruLens is an open source library for evaluating and tracing AI agents, including RAG systems."],
        ["from trulens.apps.langchain import TruChain", "tru_recorder = TruChain(chain, app_name='MyApp')"],
        ["Context relevance evaluates retrieved chunks", "Groundedness checks if response is supported by context", "Answer relevance measures if the response answers the question"],
    ],
}

ground_truth_df = pd.DataFrame(data)
```

### Step 3: Persist Dataset to TruLens

```python
session.add_ground_truth_to_dataset(
    dataset_name="my_evaluation_dataset",
    ground_truth_df=ground_truth_df,
    dataset_metadata={"domain": "TruLens QA", "version": "1.0"},
)
```

### Step 4: Load Dataset for Evaluation

```python
# Load the persisted ground truth
ground_truth_df = session.get_ground_truth("my_evaluation_dataset")

print(f"Loaded {len(ground_truth_df)} ground truth examples")
```

### Step 5: Use Ground Truth in Evaluations

```python
from trulens.core import Feedback
from trulens.feedback import GroundTruthAgreement
from trulens.providers.openai import OpenAI

provider = OpenAI()

# Create ground truth agreement feedback
ground_truth_agreement = GroundTruthAgreement(
    ground_truth_df,
    provider=provider
)

f_groundtruth = Feedback(
    ground_truth_agreement.agreement_measure,
    name="Ground Truth Agreement",
).on_input_output()
```

## Common Patterns

### Creating Dataset from Production Logs

If you have existing logs, convert them to the ground truth format:

```python
# From a list of dictionaries
logs = [
    {"input": "What is X?", "output": "X is...", "retrieved": ["doc1", "doc2"]},
    {"input": "How does Y work?", "output": "Y works by...", "retrieved": ["doc3"]},
]

ground_truth_df = pd.DataFrame({
    "query": [log["input"] for log in logs],
    "expected_response": [log["output"] for log in logs],
    "expected_chunks": [log["retrieved"] for log in logs],
})
```

### Ingesting External Logs with VirtualRecord

For apps logged outside TruLens, use VirtualRecord to ingest data:

```python
from trulens.apps.virtual import VirtualApp, VirtualRecord, TruVirtual
from trulens.core import Select

# Define virtual app structure
virtual_app = VirtualApp()
retriever_component = Select.RecordCalls.retriever
virtual_app[retriever_component] = "retriever"

# Create virtual records from your data
records = []
for row in ground_truth_df.itertuples():
    rec = VirtualRecord(
        main_input=row.query,
        main_output=row.expected_response,
        calls={
            retriever_component.get_context: dict(
                args=[row.query],
                rets=row.expected_chunks if isinstance(row.expected_chunks, list) else [row.expected_chunks]
            )
        }
    )
    records.append(rec)

# Create recorder and ingest
virtual_recorder = TruVirtual(
    app_name="ingested_data",
    app=virtual_app,
    feedbacks=[f_context_relevance, f_groundedness]
)

for record in records:
    virtual_recorder.add_record(record)
```

### Updating Existing Datasets

Add new examples to an existing dataset:

```python
# Load existing
existing_df = session.get_ground_truth("my_evaluation_dataset")

# Add new examples
new_examples = pd.DataFrame({
    "query": ["New question?"],
    "expected_response": ["New answer."],
})

updated_df = pd.concat([existing_df, new_examples], ignore_index=True)

# Re-persist (overwrites)
session.add_ground_truth_to_dataset(
    dataset_name="my_evaluation_dataset",
    ground_truth_df=updated_df,
)
```

## Troubleshooting

- **Dataset not found**: Verify the dataset name matches exactly when loading
- **Missing columns**: Ground truth DataFrames need at minimum a `query` column
- **Type errors**: Ensure `expected_chunks` is a list of strings, not a nested list
