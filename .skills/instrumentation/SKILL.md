---
skill_spec_version: 0.1.0
name: trulens-instrumentation
version: 1.0.0
description: Instrument LLM apps with TruLens OTEL-based tracing using framework wrappers and custom instrumentation
tags: [trulens, llm, instrumentation, opentelemetry, tracing]
---

# TruLens Instrumentation

Instrument your LLM application to capture traces for evaluation and debugging.

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

# TruGraph auto-detects @task decorators
tru_recorder = TruGraph(
    graph,
    app_name="MyLangGraphAgent",
    app_version="v1"
)

with tru_recorder as recording:
    result = graph.invoke({"messages": [HumanMessage(content="your query")]})
```

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

**For simple input/output apps:**

```python
from trulens.apps.basic import TruBasicApp

def my_llm_app(prompt):
    # Your LLM logic here
    return response

tru_recorder = TruBasicApp(
    my_llm_app,
    app_name="MyBasicApp",
    app_version="v1"
)

with tru_recorder as recording:
    result = my_llm_app("your prompt")
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

# TruGraph will capture both @task spans and your @instrument spans
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

## Troubleshooting

- **Spans not appearing**: Ensure you're using `@instrument()` with parentheses (not `@instrument`)
- **Missing context in evaluations**: Add semantic attributes to map function args/returns
- **Framework not detected**: Verify the correct wrapper is imported (TruChain vs TruGraph vs TruLlama)
