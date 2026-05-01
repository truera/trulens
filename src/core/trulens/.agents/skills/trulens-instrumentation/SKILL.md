---
skill_spec_version: 0.1.0
name: trulens-instrumentation
version: 2.0.0
description: Instrument LLM apps with TruLens OTEL-based tracing - from setup to debugging and optimization
tags: [trulens, llm, instrumentation, opentelemetry, tracing, debugging]
---

# TruLens Instrumentation

Instrument your LLM application to capture traces for evaluation and debugging. This skill covers everything from initial setup to iterative improvement of trace quality.

## When to Use This Skill

- Setting up instrumentation for a new app
- Adding custom spans to framework-wrapped apps
- Improving trace readability (unclear span names, missing context)
- Debugging why evaluations aren't working (missing attributes)
- Optimizing what gets captured for visualization

---

## Part 1: Setup
Instrument your LLM application to capture traces for evaluation and debugging.

## Interactive Instrumentation Setup

**Let's identify what you need to instrument for visualization and/or evaluation.**

### Question 1: What framework are you using?

| Framework | Wrapper | Auto-instrumented |
|-----------|---------|-------------------|
| **LangChain** | `TruChain` | Chain components, LLM calls |
| **LangGraph** | `TruGraph` | Graph nodes, `@task` decorators |
| **LlamaIndex** | `TruLlama` / `TruLlamaWorkflow` | Query engines, retrievers, workflows |
| **Custom/Other** | `TruApp` | Only what you explicitly `@instrument()` |

→ If using a framework, the wrapper handles basic instrumentation automatically. Continue to Question 2 to add custom attributes.

---

### Question 2: What data do you want to capture?

**Tell me what's important to track in your app.** This could be for:
- **Visualization**: Understanding execution flow in the dashboard
- **Evaluation**: Feeding data into feedback functions

Common attributes to instrument:

| What to Capture | Span Type | Attributes |
|-----------------|-----------|------------|
| **User query/input** | `RECORD_ROOT` | `INPUT` |
| **Final response** | `RECORD_ROOT` | `OUTPUT` |
| **Retrieved documents/chunks** | `RETRIEVAL` | `QUERY_TEXT`, `RETRIEVED_CONTEXTS` |
| **LLM prompts/completions** | `GENERATION` | (auto-captured by wrappers) |
| **Tool calls** | `TOOL` | Tool name, arguments, results |
| **Agent reasoning** | `AGENT` | Plans, decisions |
| **Reranking results** | `RERANKING` | `QUERY_TEXT`, `INPUT_CONTEXT_TEXTS`, `TOP_N` |

**What specific data do you want to capture that isn't listed above?**

Examples:
- "I want to capture the similarity scores from my retriever"
- "I need to track which documents were filtered out"
- "I want to see the intermediate chain-of-thought reasoning"
- "I need to capture metadata about each retrieved chunk (source, page number)"

---

### Question 3: Do you have custom functions that need instrumentation?

If you have functions that aren't automatically instrumented, list them:

**Example response:**
- `retrieve_documents(query)` - returns list of documents
- `rerank_results(query, docs)` - reranks and filters documents
- `generate_response(query, context)` - calls LLM to generate answer

**For each function, I'll help you add the right `@instrument()` decorator with appropriate span types and attributes.**

---

### Template: Instrumenting Your Custom Function

Tell me about your function and I'll generate the instrumentation:

```
Function name: _______________
What it does: _______________
Input parameters: _______________
What it returns: _______________
What data should be captured for eval/visualization: _______________
```

**Example:**

```
Function name: retrieve_documents
What it does: Searches vector store for relevant documents
Input parameters: query (str), top_k (int)
What it returns: List of document dicts with 'text', 'source', 'score' keys
What data should be captured: The query text and the document texts (not scores/sources)
```

→ Generated instrumentation:

```python
@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes={
        SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
    }
)
def retrieve_documents(query: str, top_k: int = 5) -> list:
    # If you need to extract just the text from complex returns:
    pass

# Or with lambda for complex extraction:
@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: kwargs.get("query", args[0]),
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [doc["text"] for doc in ret],
    }
)
def retrieve_documents(query: str, top_k: int = 5) -> list:
    pass
```

---

## Overview

TruLens provides two approaches to instrumentation:

1. **Framework Wrappers**: Auto-instrument apps built with LangChain, LangGraph, or LlamaIndex
2. **Custom Instrumentation**: Use `@instrument()` decorator for custom apps or to add additional spans to framework apps

## Prerequisites

```bash
pip install trulens
# For framework-specific support:
pip install trulens-apps-langchain  # LangChain/LangGraph
pip install trulens-apps-llamaindex  # LlamaIndex
```

## Instructions

### Step 1: Initialize TruSession

```python
from trulens.core import TruSession

session = TruSession()
```

### Step 2: Choose Your Instrumentation Approach

#### Option A: Framework Wrappers (Recommended for Framework Apps)

**For LangChain apps:**

```python
from trulens.apps.langchain import TruChain

tru_recorder = TruChain(
    chain,
    app_name="MyLangChainApp",
    app_version="v1"
)

with tru_recorder as recording:
    result = chain.invoke("your query")
```

**For LangGraph apps:**

```python
from trulens.apps.langgraph import TruGraph

# TruGraph auto-detects graph nodes and @task decorators
tru_recorder = TruGraph(
    graph,
    app_name="MyLangGraphAgent",
    app_version="v1"
)

with tru_recorder as recording:
    result = graph.invoke({"messages": [HumanMessage(content="your query")]})
```

 ## Deep Agents / LangGraph Instrumentation

LangChain's **Deep Agents** framework is built on LangGraph. Use `TruGraph` for full instrumentation:

```python
from deepagents import create_deep_agent
from trulens.apps.langgraph import TruGraph
from trulens.core import TruSession

# Create the Deep Agent
agent = create_deep_agent(
    model=model,
    tools=[your_tools],
    system_prompt="Your prompt"
)

# Wrap with TruGraph - captures all internal nodes, tool calls, planning steps
tru_agent = TruGraph(
    agent,
    app_name="DeepAgent",
    app_version="v1",
    feedbacks=[f_answer_relevance]
)

with tru_agent as recording:
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})
```

**For LlamaIndex apps**

```python
from trulens.apps.llamaindex import TruLlama

tru_recorder = TruLlama(query_engine, app_name="MyRAG", app_version="v1")

with tru_recorder as recording:
    result = query_engine.query("your query")
**For LlamaIndex query engines:**

```python
from trulens.apps.llamaindex import TruLlama

query_engine = index.as_query_engine()

tru_recorder = TruLlama(
    query_engine,
    app_name="MyLlamaIndexApp",
    app_version="v1"
)

with tru_recorder as recording:
    result = query_engine.query("your query")
```

**For LlamaIndex workflows:**

```python
from trulens.apps.llamaindex import TruLlamaWorkflow

tru_recorder = TruLlamaWorkflow(
    workflow,
    app_name="MyLlamaWorkflow",
    app_version="v1"
)

with tru_recorder as recording:
    result = await workflow.run(query="your query")
```

#### Option B: Custom Instrumentation with @instrument()

For custom apps or to add spans to framework apps:

```python
from trulens.apps.app import TruApp
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes


class MyRAG:
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str) -> list:
        # Your retrieval logic
        return contexts

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate(self, query: str, contexts: list) -> str:
        # Your generation logic
        return response

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def query(self, query: str) -> str:
        contexts = self.retrieve(query)
        return self.generate(query, contexts)


rag = MyRAG()
tru_app = TruApp(rag, app_name="MyCustomRAG", app_version="v1")

with tru_app as recording:
    result = rag.query("your query")
```

### Step 3: Combining Wrappers with Custom Instrumentation

Use `@instrument()` alongside framework wrappers to add custom span attributes for evaluation:

```python
from trulens.apps.langgraph import TruGraph
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

@instrument()
def preprocess_input(topic: str) -> str:
    """Custom preprocessing - will appear in traces."""
    return f"Preprocessed: {topic}"

@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes={
        SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
    },
)
def custom_retrieve(query: str) -> list:
    """Custom retrieval with semantic attributes for evaluation."""
    return ["context1", "context2"]

# TruGraph will capture both auto-instrumented spans and your @instrument spans
tru_recorder = TruGraph(graph, app_name="EnhancedAgent", app_version="v1")
```

### Step 4: Lambda-Based Attribute Extraction

For complex data structures, use a lambda to extract attributes:

```python
@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [doc["text"] for doc in ret],
        SpanAttributes.RETRIEVAL.QUERY_TEXT: kwargs.get("query", args[0] if args else ""),
    }
)
def retrieve_documents(query: str) -> list:
    return [{"text": "doc1", "score": 0.9}, {"text": "doc2", "score": 0.8}]
```

### Step 5: Instrumenting Third-Party Classes

When you can't modify source code, use `instrument_method()`:

```python
from trulens.core.otel.instrument import instrument_method
from some_library import ExternalRetriever

instrument_method(
    cls=ExternalRetriever,
    method_name="search",
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes={
        SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
    }
)
```

## Common Patterns

### RAG Application

```python
@instrument(span_type=SpanAttributes.SpanType.RETRIEVAL, attributes={...})
def retrieve(self, query): ...

@instrument(span_type=SpanAttributes.SpanType.GENERATION)
def generate(self, query, context): ...

@instrument(span_type=SpanAttributes.SpanType.RECORD_ROOT, attributes={...})
def query(self, query): ...
```

### Agent Application

```python
@instrument(span_type=SpanAttributes.SpanType.AGENT)
def run_agent(self, task): ...

@instrument(span_type=SpanAttributes.SpanType.TOOL)
def call_tool(self, tool_name, args): ...

@instrument(span_type=SpanAttributes.SpanType.WORKFLOW)
def execute_workflow(self, steps): ...
```

**Why TruGraph instead of TruApp + @instrument?**

- TruGraph automatically captures all LangGraph nodes and transitions
- TruGraph creates `RECORD_ROOT` spans required for `.on_input()/.on_output()` shortcuts
- Manual `@instrument(span_type=SpanType.AGENT)` will NOT work with feedback selector shortcuts

## Critical: Span Types and Feedback Selectors

The `.on_input()` and `.on_output()` feedback selector shortcuts look for spans with type `RECORD_ROOT`:

```python
# This WORKS - TruGraph creates RECORD_ROOT spans automatically
tru_agent = TruGraph(agent, feedbacks=[f_answer_relevance])

# This also WORKS - explicit RECORD_ROOT
@instrument(
    span_type=SpanAttributes.SpanType.RECORD_ROOT,
    attributes={
        SpanAttributes.RECORD_ROOT.INPUT: "query",
        SpanAttributes.RECORD_ROOT.OUTPUT: "return",
    }
)
def query(self, query: str) -> str:
    ...

# This WILL NOT WORK with .on_input()/.on_output() shortcuts!
@instrument(span_type=SpanAttributes.SpanType.AGENT)  # Wrong span type
def run_agent(self, task):
    ...
```

**If your evaluations show empty feedback columns**, check that your root span uses `RECORD_ROOT` span type.

## Troubleshooting

- **Spans not appearing**: Ensure you're using `@instrument()` with parentheses (not `@instrument`)
- **Missing context in evaluations**: Add semantic attributes to map function args/returns
- **Framework not detected**: Verify the correct wrapper is imported (TruChain vs TruGraph vs TruLlama)
- **Feedback columns empty/evaluations not running**: Your root span must use `SpanType.RECORD_ROOT` for `.on_input()/.on_output()` shortcuts to work. Use framework wrappers (TruGraph, TruChain) which handle this automatically.
