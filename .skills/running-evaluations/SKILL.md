---
skill_spec_version: 0.1.0
name: trulens-running-evaluations
version: 1.0.0
description: Execute TruLens evaluations for RAG and Agent applications
tags: [trulens, llm, evaluation, rag, agents]
---

# TruLens Running Evaluations

Execute evaluations appropriate to your app type (RAG vs Agent).

## Overview

TruLens evaluations differ based on app type:

- **RAG Apps**: Prioritize the RAG Triad (context relevance, groundedness, answer relevance)
- **Agent Apps**: Prioritize Agent GPA metrics (logical consistency, plan quality, execution efficiency, tool metrics)

## Prerequisites

```bash
pip install trulens trulens-providers-openai
```

## Instructions

### Step 1: Identify Your App Type

| App Type | Characteristics | Primary Metrics |
|----------|-----------------|-----------------|
| **RAG** | Retriever + generator pattern, returns grounded responses | RAG Triad |
| **Agent** | Tool-calling, multi-step reasoning, planning | Agent GPA |

### Step 2: Set Up Evaluations Based on App Type

#### For RAG Applications

```python
import numpy as np
from trulens.core import TruSession, Feedback
from trulens.providers.openai import OpenAI

session = TruSession()
provider = OpenAI()

# RAG Triad Feedbacks
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on_context(collect_list=False)
    .aggregate(np.mean)
)

f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on_context(collect_list=True)
    .on_output()
)

f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)

rag_feedbacks = [f_context_relevance, f_groundedness, f_answer_relevance]
```

#### For Agent Applications

```python
from trulens.core import TruSession, Feedback
from trulens.core.feedback.selector import Selector
from trulens.providers.openai import OpenAI

session = TruSession()
provider = OpenAI()

# All GPA metrics use trace-level selection
trace_selector = {"trace": Selector(trace_level=True)}

# Core Agent GPA Feedbacks (always include)
f_logical_consistency = (
    Feedback(provider.logical_consistency_with_cot_reasons, name="Logical Consistency")
    .on(trace_selector)
)

f_execution_efficiency = (
    Feedback(provider.execution_efficiency_with_cot_reasons, name="Execution Efficiency")
    .on(trace_selector)
)

f_tool_selection = (
    Feedback(provider.tool_selection_with_cot_reasons, name="Tool Selection")
    .on(trace_selector)
)

f_tool_calling = (
    Feedback(provider.tool_calling_with_cot_reasons, name="Tool Calling")
    .on(trace_selector)
)

f_tool_quality = (
    Feedback(provider.tool_quality_with_cot_reasons, name="Tool Quality")
    .on(trace_selector)
)

# Planning metrics (include only if agent does explicit planning)
f_plan_quality = (
    Feedback(provider.plan_quality_with_cot_reasons, name="Plan Quality")
    .on(trace_selector)
)

f_plan_adherence = (
    Feedback(provider.plan_adherence_with_cot_reasons, name="Plan Adherence")
    .on(trace_selector)
)

# Choose feedbacks based on whether agent does planning
agent_feedbacks = [
    f_logical_consistency,
    f_plan_quality,        # Include if agent plans
    f_plan_adherence,      # Include if agent plans
    f_execution_efficiency,
    f_tool_selection,
    f_tool_calling,
    f_tool_quality,
]

# If agent does NOT do explicit planning, use this instead:
# agent_feedbacks = [
#     f_logical_consistency,
#     f_execution_efficiency,
#     f_tool_selection,
#     f_tool_calling,
#     f_tool_quality,
# ]
```

### Step 3: Wrap Your App with the Appropriate Wrapper

#### RAG with TruLlama

```python
from trulens.apps.llamaindex import TruLlama

query_engine = index.as_query_engine()

tru_rag = TruLlama(
    query_engine,
    app_name="MyRAG",
    app_version="v1",
    feedbacks=rag_feedbacks,
)
```

#### RAG with TruChain

```python
from trulens.apps.langchain import TruChain

tru_rag = TruChain(
    rag_chain,
    app_name="MyRAG",
    app_version="v1",
    feedbacks=rag_feedbacks,
)
```

#### Agent with TruGraph

```python
from trulens.apps.langgraph import TruGraph

tru_agent = TruGraph(
    agent_graph,
    app_name="MyAgent",
    app_version="v1",
    feedbacks=agent_feedbacks,
)
```

#### Custom App with TruApp

```python
from trulens.apps.app import TruApp

tru_app = TruApp(
    my_custom_app,
    app_name="MyApp",
    app_version="v1",
    feedbacks=rag_feedbacks,  # or agent_feedbacks
)
```

### Step 4: Run Your App and Record Traces

```python
# Run with recording
with tru_rag as recording:
    response = query_engine.query("What is TruLens?")

# Or for agents
with tru_agent as recording:
    result = agent_graph.invoke({"messages": [HumanMessage(content="Research AI trends")]})
```

### Step 5: View Results

```python
# View leaderboard
leaderboard = session.get_leaderboard()
print(leaderboard)

# Launch dashboard for detailed analysis
from trulens.dashboard import run_dashboard

run_dashboard(session)
```

## Common Patterns

### Complete RAG Evaluation Flow

```python
import numpy as np
from trulens.core import TruSession, Feedback
from trulens.apps.llamaindex import TruLlama
from trulens.providers.openai import OpenAI

# Setup
session = TruSession()
provider = OpenAI()

# RAG Triad
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on_context(collect_list=False)
    .aggregate(np.mean)
)

f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons, name="Groundedness")
    .on_context(collect_list=True)
    .on_output()
)

f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)

# Wrap and run
tru_rag = TruLlama(
    query_engine,
    app_name="ProductionRAG",
    app_version="v1",
    feedbacks=[f_context_relevance, f_groundedness, f_answer_relevance],
)

test_queries = [
    "What is machine learning?",
    "How does RAG work?",
    "Explain transformers.",
]

with tru_rag as recording:
    for query in test_queries:
        query_engine.query(query)

# Review
print(session.get_leaderboard())
```

### Complete Agent Evaluation Flow

```python
from trulens.core import TruSession, Feedback
from trulens.core.feedback.selector import Selector
from trulens.apps.langgraph import TruGraph
from trulens.providers.openai import OpenAI

# Setup
session = TruSession()
provider = OpenAI()
trace_selector = {"trace": Selector(trace_level=True)}

# Agent GPA (with planning)
feedbacks = [
    Feedback(provider.logical_consistency_with_cot_reasons, name="Logical Consistency").on(trace_selector),
    Feedback(provider.plan_quality_with_cot_reasons, name="Plan Quality").on(trace_selector),
    Feedback(provider.plan_adherence_with_cot_reasons, name="Plan Adherence").on(trace_selector),
    Feedback(provider.execution_efficiency_with_cot_reasons, name="Execution Efficiency").on(trace_selector),
    Feedback(provider.tool_selection_with_cot_reasons, name="Tool Selection").on(trace_selector),
    Feedback(provider.tool_calling_with_cot_reasons, name="Tool Calling").on(trace_selector),
    Feedback(provider.tool_quality_with_cot_reasons, name="Tool Quality").on(trace_selector),
]

# Wrap and run
tru_agent = TruGraph(
    agent_graph,
    app_name="ResearchAgent",
    app_version="v1",
    feedbacks=feedbacks,
)

test_tasks = [
    "Research the latest AI trends and summarize",
    "Find information about climate change policies",
    "Analyze stock market performance this quarter",
]

with tru_agent as recording:
    for task in test_tasks:
        agent_graph.invoke({"messages": [HumanMessage(content=task)]})

# Review
print(session.get_leaderboard())
```

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

# Compare on leaderboard
print(session.get_leaderboard())
```

## Troubleshooting

- **No evaluation results**: Ensure feedbacks list is passed to the wrapper
- **Missing context scores**: Verify `RETRIEVAL.RETRIEVED_CONTEXTS` is instrumented
- **Agent metrics empty**: Check that trace contains tool calls and reasoning steps
- **Dashboard not loading**: Run `pip install trulens-dashboard` and ensure port 8501 is available
