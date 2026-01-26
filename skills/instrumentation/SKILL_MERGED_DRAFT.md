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

### Step 1: Initialize TruSession

```python
from trulens.core import TruSession

session = TruSession()
```

### Step 2: Choose Your Instrumentation Approach

#### Option A: Framework Wrappers (Recommended)

Framework wrappers auto-instrument common patterns. Choose based on your framework:

| Framework | Wrapper | Auto-instrumented |
|-----------|---------|-------------------|
| **LangChain** | `TruChain` | Chain components, LLM calls |
| **LangGraph** | `TruGraph` | Graph nodes, `@task` decorators |
| **LlamaIndex** | `TruLlama` / `TruLlamaWorkflow` | Query engines, retrievers, workflows |
| **Custom/Other** | `TruApp` | Only what you explicitly `@instrument()` |

**LangGraph example:**

```python
from trulens.apps.langgraph import TruGraph

tru_recorder = TruGraph(
    graph,
    app_name="MyAgent",
    app_version="v1"
)

with tru_recorder as recording:
    result = graph.invoke({"messages": [HumanMessage(content="your query")]})
```

**LangChain example:**

```python
from trulens.apps.langchain import TruChain

tru_recorder = TruChain(chain, app_name="MyChain", app_version="v1")

with tru_recorder as recording:
    result = chain.invoke("your query")
```

**LlamaIndex example:**

```python
from trulens.apps.llamaindex import TruLlama

tru_recorder = TruLlama(query_engine, app_name="MyRAG", app_version="v1")

with tru_recorder as recording:
    result = query_engine.query("your query")
```

#### Option B: Custom Instrumentation with @instrument()

For custom apps or adding spans to framework apps:

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
        return contexts

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate(self, query: str, contexts: list) -> str:
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

---

## Part 2: Clean Instrumentation

Use these techniques to produce readable, well-structured traces from the start.

### Clean Span Names with `name=`

By default, span names include the module prefix (e.g., `__main__.call_llm`). Use `name=` for clean names:

```python
# Before: Span named "__main__.call_llm" or "langchain_core.tools.structured.StructuredTool.invoke"
@instrument()
def call_llm(messages):
    ...

# After: Span named "call_llm"
@instrument(name="call_llm")
def call_llm(messages):
    ...
```

### Tool Instrumentation with `instrument_tools()`

For agent apps with tools, use `instrument_tools()` to get spans named after each tool:

```python
from trulens.core.otel.instrument import instrument, instrument_tools, generation_attributes
from trulens.otel.semconv.trace import SpanAttributes

# Define tools
tools = [add, multiply, divide]
tools_by_name = {tool.name: tool for tool in tools}

# One line - instruments all tools for clean span names
instrument_tools(tools_by_name)

# Now tool.invoke() creates spans named "add", "multiply", etc.
def call_tool(tool_call: dict):
    tool = tools_by_name[tool_call["name"]]
    result = tool.invoke(tool_call["args"])  # Creates "add" span, not "StructuredTool.invoke"
    return ToolMessage(content=str(result), tool_call_id=tool_call["id"])
```

### Generation Spans with Content Extraction

Use `generation_attributes()` to extract readable content from LLM calls:

```python
@instrument(
    name="call_llm",
    span_type=SpanAttributes.SpanType.GENERATION,
    attributes=generation_attributes()
)
def call_llm(messages):
    return model.invoke(messages)
```

This extracts:
- `input_content`: The user's message text
- `output_content`: The LLM's response text
- `tool_calls`: Formatted tool calls like `"add(a=3, b=4)"`

### Complete Agent Example

```python
from langchain.messages import SystemMessage, ToolMessage
from langchain_core.messages import BaseMessage
from langgraph.func import entrypoint
from langgraph.graph import add_messages

from trulens.core.otel.instrument import instrument, instrument_tools, generation_attributes
from trulens.otel.semconv.trace import SpanAttributes

# Instrument tools for clean span names
instrument_tools(tools_by_name)


@instrument(
    name="call_llm",
    span_type=SpanAttributes.SpanType.GENERATION,
    attributes=generation_attributes()
)
def call_llm(messages):
    return model_with_tools.invoke(
        [SystemMessage(content="You are a helpful assistant.")] + messages
    )


def call_tool(tool_call: dict):
    tool = tools_by_name[tool_call["name"]]
    result = tool.invoke(tool_call["args"])
    return ToolMessage(content=str(result), tool_call_id=tool_call["id"])


@entrypoint()
def agent(messages: list[BaseMessage]):
    messages = add_messages([], messages)
    while True:
        response = call_llm(messages)
        messages = add_messages(messages, [response])
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            messages = add_messages(messages, [call_tool(tc)])
    return messages
```

**Result:**
```
├─ call_llm [generation]
│     input_content: "Add 3 and 4"
│     tool_calls: "add(a=3, b=4)"
├─ add [tool]
├─ call_llm [generation]
│     input_content: "Add 3 and 4"
│     output_content: "The result is 7."
```

---

## Part 3: Debugging & Improving Traces

If traces aren't clear or evaluations aren't working, use this iterative process.

### The Debug Loop

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Run a single trace                                          │
│  2. Query the events table to see raw span data                 │
│  3. Analyze: What's good? What's confusing? What's missing?     │
│  4. Improve instrumentation                                     │
│  5. Repeat until satisfied                                      │
└─────────────────────────────────────────────────────────────────┘
```

### Step 1: Run a Trace

```python
from trulens.core import TruSession
from trulens.apps.langgraph import TruGraph

session = TruSession()
# DO NOT reset_database() - preserve history for comparison

tru_app = TruGraph(app, app_name="MyApp", app_version="v1")

with tru_app as recording:
    result = app.invoke(test_input)

record_id = recording[0].record_id
print(f"Captured trace: {record_id}")
```

### Step 2: Query the Events Table

```python
# Force flush to ensure all spans are written
session.force_flush()

# Get events for this specific record
events_df = session.connector.get_events(record_ids=[record_id])

# Display key columns
print(events_df[['record', 'record_attributes']].to_string())
```

**Events table columns:**

| Column | Description |
|--------|-------------|
| `record` | Contains `name` (span name), `kind`, `parent_id` |
| `record_attributes` | Span type, semantic attributes, custom data |
| `trace` | Trace context (trace_id, span_id) |
| `start_timestamp` / `timestamp` | Timing |

### Step 3: Visualize Span Hierarchy

Use the `print_span_tree` helper from this skill's debug utilities:

```python
# Copy debug_utils.py from this skill directory, then:
from debug_utils import print_span_tree

print_span_tree(events_df)

# Output:
# ├─ calculator_agent [record_root]
#   ├─ call_llm [generation]
#   ├─ add [tool]
#   ├─ call_llm [generation]
```

### Step 4: Analyze Quality

**Clarity Checklist:**
- [ ] Are span names descriptive? (`call_llm` not `invoke`)
- [ ] No module prefixes? (`call_llm` not `__main__.call_llm`)
- [ ] Tool spans named after the tool? (`add` not `call_tool`)

**Completeness Checklist:**
- [ ] Root input/output captured?
- [ ] Tool calls visible with arguments?
- [ ] Retrieval showing query and contexts?
- [ ] Generation showing prompts/completions?

**Hierarchy Checklist:**
- [ ] Parent-child relationships make sense?
- [ ] Not too deep or too shallow?

**Semantic Attributes Checklist:**
- [ ] Right span types? (RETRIEVAL, GENERATION, TOOL)
- [ ] Attributes populated for feedback functions?

### Step 5: Common Fixes

| Issue | Solution |
|-------|----------|
| `__main__.func_name` | Add `name="func_name"` parameter |
| `StructuredTool.invoke` | Use `instrument_tools(tools_by_name)` |
| Raw message objects in generation | Use `attributes=generation_attributes()` |
| Missing root input/output | Add `RECORD_ROOT` span type with attributes |
| Can't see tool arguments | Add `SpanType.TOOL` to tool functions |
| Retrieval missing query | Add `RETRIEVAL.QUERY_TEXT` attribute |
| Flat trace (no hierarchy) | Ensure parent functions are also instrumented |

### Step 6: Iterate

After making improvements, increment version and run again:

```python
tru_app = TruGraph(app, app_name="MyApp", app_version="v2")

with tru_app as recording:
    result = app.invoke(test_input)

# Compare in dashboard - both v1 and v2 visible
```

### Step 7: Validate with Dashboard

```python
from trulens.dashboard import run_dashboard

run_dashboard(session)
```

---

## Part 4: Reference

### Span Types

| Span Type | Use For |
|-----------|---------|
| `RECORD_ROOT` | Entry point - captures main input/output |
| `RETRIEVAL` | Vector search, document fetching |
| `GENERATION` | LLM calls |
| `TOOL` | Tool/function calls |
| `AGENT` | Agent reasoning/planning |
| `RERANKING` | Result reranking |
| `WORKFLOW` | Multi-step workflows |

### Semantic Attributes

**RECORD_ROOT:**
```python
@instrument(
    span_type=SpanAttributes.SpanType.RECORD_ROOT,
    attributes={
        SpanAttributes.RECORD_ROOT.INPUT: "query",
        SpanAttributes.RECORD_ROOT.OUTPUT: "return",
    }
)
```

**RETRIEVAL:**
```python
@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes={
        SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
    }
)
```

**RERANKING:**
```python
@instrument(
    span_type=SpanAttributes.SpanType.RERANKING,
    attributes={
        SpanAttributes.RERANKING.QUERY_TEXT: "query",
        SpanAttributes.RERANKING.INPUT_CONTEXT_TEXTS: "contexts",
        SpanAttributes.RERANKING.TOP_N: "top_n",
    }
)
```

### Lambda-Based Attribute Extraction

For complex return values:

```python
@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: kwargs.get("query", args[0]),
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [doc["text"] for doc in ret],
    }
)
def retrieve_documents(query: str) -> list:
    return [{"text": "doc1", "score": 0.9}, {"text": "doc2", "score": 0.8}]
```

### Instrumenting Third-Party Classes

When you can't modify source code:

```python
from trulens.core.otel.instrument import instrument_method

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

### Critical: RECORD_ROOT for Feedback Selectors

The `.on_input()` and `.on_output()` shortcuts require `RECORD_ROOT` spans:

```python
# WORKS - TruGraph creates RECORD_ROOT automatically
tru_agent = TruGraph(agent, feedbacks=[f_relevance])

# WORKS - explicit RECORD_ROOT
@instrument(
    span_type=SpanAttributes.SpanType.RECORD_ROOT,
    attributes={
        SpanAttributes.RECORD_ROOT.INPUT: "query",
        SpanAttributes.RECORD_ROOT.OUTPUT: "return",
    }
)
def query(self, query: str) -> str:
    ...

# WILL NOT WORK with .on_input()/.on_output()!
@instrument(span_type=SpanAttributes.SpanType.AGENT)  # Wrong type
def run_agent(self, task):
    ...
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Spans not appearing | Use `@instrument()` with parentheses |
| Feedback columns empty | Root span needs `RECORD_ROOT` type |
| Module prefix in span name | Add `name=` parameter |
| Tool spans show `StructuredTool.invoke` | Use `instrument_tools()` |
| Raw objects in generation spans | Use `generation_attributes()` |
| Missing context in evaluations | Add semantic attributes |
| Framework not detected | Verify correct wrapper imported |

---

## Style Guide

### 1. Clean Span Names

**Bad:**
```
├─ __main__.call_llm
├─ langchain_core.tools.structured.StructuredTool.invoke
```

**Good:**
```
├─ call_llm
├─ add
```

### 2. Descriptive Tool Names

**Bad:**
```
├─ call_tool [tool]
├─ call_tool [tool]
```

**Good:**
```
├─ add [tool]
├─ multiply [tool]
```

### 3. Readable Generation Content

**Bad:**
```
├─ call_llm [generation]
    input: [HumanMessage(content="..."), AIMessage(...)]
    output: AIMessage(content="", tool_calls=[...])
```

**Good:**
```
├─ call_llm [generation]
    input_content: "Add 3 and 4"
    tool_calls: "add(a=3, b=4)"
```
