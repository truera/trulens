---
skill_spec_version: 0.1.0
name: trulens-beautify-trace
version: 1.0.0
description: Iteratively improve instrumentation by analyzing trace events and beautifying span information
tags: [trulens, tracing, instrumentation, otel, spans, debugging]
---

# TruLens Beautify Trace

Iteratively run traces, analyze their events, and improve instrumentation so that trace and span information best captures the agent's execution flow.

## When to Use This Skill

Use this skill when:
- User wants to improve the readability of their traces
- Traces are too noisy or missing important context
- Span names are unclear or don't reflect what's happening
- The dashboard trace view is hard to understand
- User wants to optimize what gets captured for visualization/debugging

## The Beautify Loop (Up to 3 Iterations)

```
┌─────────────────────────────────────────────────────────────────┐
│  ITERATION 1-3                                                   │
│                                                                  │
│  1. Run a single trace                                           │
│  2. Query the events table to see raw span data                  │
│  3. Analyze: What's good? What's confusing? What's missing?      │
│  4. Improve instrumentation (add attributes, rename spans, etc.) │
│  5. Repeat until satisfied or 3 iterations complete              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  VALIDATION                                                      │
│                                                                  │
│  1. Launch dashboard from the correct directory                  │
│  2. Ask user to validate the trace visualization                 │
│  3. Offer to continue improving if needed                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Step 1: Run a Single Trace

Execute the app with instrumentation to capture one trace. **Never reset the database** - keep all versions for comparison:

```python
from trulens.core import TruSession
from trulens.apps.langgraph import TruGraph  # or TruChain, TruLlama, TruApp

session = TruSession()
# DO NOT reset_database() - preserve history for comparison

# Get existing app version to build on (if any)
existing_version = "v1"  # or query from existing app

# Version naming: {existing_version}-beautified-v{iteration}
# Examples: v1-beautified-v1, v1-beautified-v2, base-beautified-v1
iteration = 1
app_version = f"{existing_version}-beautified-v{iteration}"

tru_app = TruGraph(app, app_name="MyApp", app_version=app_version)

with tru_app as recording:
    result = app.invoke({"messages": [{"role": "user", "content": "test query"}]})

# Get the record ID for this trace
record_id = recording[0].record_id
print(f"Captured trace: {record_id} (version: {app_version})")
```

### Version Naming Convention

Keep the original version as a prefix and append beautified iteration:

| Original Version | Iteration 1 | Iteration 2 | Iteration 3 |
|-----------------|-------------|-------------|-------------|
| `v1` | `v1-beautified-v1` | `v1-beautified-v2` | `v1-beautified-v3` |
| `base` | `base-beautified-v1` | `base-beautified-v2` | `base-beautified-v3` |
| `prod-2024` | `prod-2024-beautified-v1` | `prod-2024-beautified-v2` | `prod-2024-beautified-v3` |

This allows you to compare traces across iterations in the dashboard to see improvements.

## Step 2: Query the Events Table

Use the connector's `get_events()` method to retrieve raw span data:

```python
import pandas as pd

# Force flush to ensure all spans are written
session.force_flush()

# Get events for this specific record
events_df = session.connector.get_events(record_ids=[record_id])

# Display key columns for analysis
print(events_df[['record', 'record_attributes']].to_string())
```

### Understanding the Events DataFrame

The events table contains these key columns:

| Column | Description |
|--------|-------------|
| `event_id` | Unique identifier for the span |
| `record` | Contains `name` (span name), `kind` (span kind), `parent_id` |
| `record_attributes` | Dict with span type, semantic attributes, custom data |
| `resource_attributes` | Service info, SDK version |
| `start_timestamp` | When the span started |
| `timestamp` | When the span ended |
| `trace` | Trace context (trace_id, span_id) |

### Extracting Useful Information

```python
# Extract span names and types
for idx, row in events_df.iterrows():
    span_name = row['record'].get('name', 'unnamed')
    span_type = row['record_attributes'].get('trulens.span_type', 'unknown')

    # Get semantic attributes if present
    attrs = row['record_attributes']
    input_val = attrs.get('trulens.record_root.input', attrs.get('input'))
    output_val = attrs.get('trulens.record_root.output', attrs.get('output'))

    print(f"Span: {span_name}")
    print(f"  Type: {span_type}")
    if input_val:
        print(f"  Input: {str(input_val)[:100]}...")
    if output_val:
        print(f"  Output: {str(output_val)[:100]}...")
    print()
```

### Visualizing Span Hierarchy

```python
def print_span_tree(events_df):
    """Print spans as a tree structure."""
    # Build parent-child relationships
    spans = {}
    for _, row in events_df.iterrows():
        span_id = row['trace'].get('span_id')
        parent_id = row['record'].get('parent_id')
        name = row['record'].get('name', 'unnamed')
        span_type = row['record_attributes'].get('trulens.span_type', '')
        spans[span_id] = {
            'name': name,
            'type': span_type,
            'parent_id': parent_id,
            'children': []
        }

    # Link children to parents
    roots = []
    for span_id, span in spans.items():
        parent_id = span['parent_id']
        if parent_id and parent_id in spans:
            spans[parent_id]['children'].append(span_id)
        else:
            roots.append(span_id)

    # Print tree
    def print_node(span_id, depth=0):
        span = spans[span_id]
        indent = "  " * depth
        type_str = f" [{span['type']}]" if span['type'] else ""
        print(f"{indent}├─ {span['name']}{type_str}")
        for child_id in span['children']:
            print_node(child_id, depth + 1)

    for root_id in roots:
        print_node(root_id)

print_span_tree(events_df)
```

## Step 3: Analyze the Trace Quality

Ask these questions when reviewing the events:

### Clarity
- [ ] Are span names descriptive? (`call_llm` is better than `invoke`)
- [ ] Can you understand the flow from span names alone?
- [ ] Are there meaningless spans cluttering the trace?

### Completeness
- [ ] Is the root input/output captured?
- [ ] Are tool calls visible with their arguments and results?
- [ ] Are retrieval operations showing the query and contexts?
- [ ] Is LLM generation capturing prompts and completions?

### Hierarchy
- [ ] Does the parent-child relationship make sense?
- [ ] Are related operations grouped under the same parent?
- [ ] Is nesting too deep or too shallow?

### Semantic Attributes
- [ ] Are the right span types assigned (RETRIEVAL, GENERATION, TOOL, AGENT)?
- [ ] Are semantic attributes populated for feedback functions to use?

## Step 4: Improve Instrumentation

Based on your analysis, make targeted improvements:

### A. Rename Unclear Spans

```python
# Before: Generic name
@instrument()
def process(data):
    ...

# After: Descriptive name
@instrument(name="parse_user_query")
def process(data):
    ...
```

### B. Add Missing Semantic Attributes

```python
from trulens.otel.semconv.trace import SpanAttributes

# Before: No attributes
@instrument()
def search(query):
    results = vector_store.search(query)
    return results

# After: With semantic attributes for evaluation
@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes={
        SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
    }
)
def search(query):
    results = vector_store.search(query)
    return [r.text for r in results]  # Return just the text for attributes
```

### C. Add Root Input/Output

```python
# Ensure the entry point has RECORD_ROOT span type
@instrument(
    span_type=SpanAttributes.SpanType.RECORD_ROOT,
    attributes={
        SpanAttributes.RECORD_ROOT.INPUT: "query",
        SpanAttributes.RECORD_ROOT.OUTPUT: "return",
    }
)
def main(query: str) -> str:
    ...
```

### D. Use Lambda for Complex Attribute Extraction

```python
@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes=lambda ret, exception, *args, **kwargs: {
        SpanAttributes.RETRIEVAL.QUERY_TEXT: kwargs.get("query", args[0]),
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: [
            doc["content"] for doc in ret["documents"]
        ],
    }
)
def search_documents(query: str) -> dict:
    ...
```

### E. Suppress Noisy Spans

If certain spans add noise without value, consider:
- Not instrumenting internal helper functions
- Using a wrapper function that is instrumented, calling uninstrumented helpers

## Step 5: Repeat the Loop

After making improvements:

1. **DO NOT reset the database** - keep all versions for comparison
2. Increment the version number (e.g., `v1-beautified-v1` → `v1-beautified-v2`)
3. Run another trace with the new version
4. Query events and compare to previous iteration
5. Continue improving or stop after 3 iterations

```python
# Quick iteration pattern - increment version, don't reset
iteration += 1
app_version = f"{existing_version}-beautified-v{iteration}"

tru_app = TruGraph(app, app_name="MyApp", app_version=app_version)

with tru_app as recording:
    result = app.invoke(test_input)

record_id = recording[0].record_id
session.force_flush()
events_df = session.connector.get_events(record_ids=[record_id])

# Analyze and compare to previous iteration
print(f"Version: {app_version}")
print_span_tree(events_df)
```

## Step 6: Validate with Dashboard

After iterations are complete, launch the dashboard for visual validation:

```bash
# IMPORTANT: Run from the directory where the database was created
cd /path/to/working/directory && \
python3 << 'EOF'
from trulens.core import TruSession
from trulens.dashboard import run_dashboard

session = TruSession()
run_dashboard(session)
EOF
```

Use `run_in_background=true` to keep the dashboard alive.

### Ask User to Validate

After launching the dashboard, ask the user:

1. "Does the trace visualization clearly show the agent's execution flow?"
2. "Are there any spans that are confusing or unnecessary?"
3. "Is any important information missing from the trace?"
4. "Would you like me to continue improving the instrumentation?"

## Common Improvements Checklist

| Issue | Solution |
|-------|----------|
| Span names like "invoke" or "call" | Use descriptive names: `generate_response`, `search_documents` |
| Missing input in trace | Add `SpanAttributes.RECORD_ROOT.INPUT` to root span |
| Missing output in trace | Add `SpanAttributes.RECORD_ROOT.OUTPUT` to root span |
| Can't see what tools were called | Add `SpanAttributes.SpanType.TOOL` to tool functions |
| Retrieval doesn't show query | Add `SpanAttributes.RETRIEVAL.QUERY_TEXT` attribute |
| Retrieval doesn't show contexts | Add `SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS` attribute |
| Too many low-level spans | Remove `@instrument()` from internal helper functions |
| Flat trace (no hierarchy) | Ensure parent functions are also instrumented |
| Span type is wrong | Change `span_type` parameter in `@instrument()` |

## Example Iteration Session

```
=== ITERATION 1 (v1-beautified-v1) ===
Running trace...
Captured: record_abc123

Span Tree:
├─ invoke
  ├─ _call
    ├─ process
    ├─ generate

Issues found:
- Root span named "invoke" (unclear)
- No input/output on root
- "process" doesn't explain what it does

=== ITERATION 2 (v1-beautified-v2) ===
Applied fixes:
- Renamed root to "answer_question"
- Added RECORD_ROOT attributes
- Renamed "process" to "retrieve_context"

Running trace...
Captured: record_def456

Span Tree:
├─ answer_question [RECORD_ROOT]
  ├─ retrieve_context [RETRIEVAL]
    ├─ generate_response [GENERATION]

Much better! Input/output now visible.

=== ITERATION 3 (v1-beautified-v3) ===
Final polish:
- Added query text to retrieval span
- Added retrieved contexts attribute

Running trace...
Captured: record_ghi789

Span Tree:
├─ answer_question [RECORD_ROOT] input="What is TruLens?" output="TruLens is..."
  ├─ retrieve_context [RETRIEVAL] query="What is TruLens?" contexts=["TruLens is...", ...]
    ├─ generate_response [GENERATION]

Excellent! Ready for dashboard validation.

=== LAUNCHING DASHBOARD ===
Dashboard running at: http://localhost:55872

All versions visible in dashboard for comparison:
- v1 (original)
- v1-beautified-v1
- v1-beautified-v2
- v1-beautified-v3

Please check the trace visualization and let me know:
- Does it clearly show the execution flow?
- Any improvements you'd like?
```

## Style Guide

### 1. Never Include Module Prefix in Span Names

**Bad:** Span names with `__main__.` or other module prefixes
```
├─ __main__.call_llm [generation]
├─ __main__.process_query [unknown]
├─ mymodule.helper.do_stuff [unknown]
```

**Good:** Clean span names without any prefix
```
├─ call_llm [generation]
├─ process_query [unknown]
├─ do_stuff [unknown]
```

### 2. Name Tool Spans After the Actual Tool

**Bad:** Generic names like `call_tool`, `execute_tool`, `tool`
```
├─ call_tool [tool]
├─ call_tool [tool]
├─ call_tool [tool]
```

**Good:** Use the actual tool name: `add`, `multiply`, `web_search`
```
├─ add [tool]
├─ multiply [tool]
├─ web_search [tool]
```

### 3. Extract Content and Tool Calls from Generation Spans

For GENERATION span types, extract the actual text content AND tool calls as span attributes.

**Bad:** Generation span showing raw message objects
```
├─ call_llm [generation]
    input: [HumanMessage(content="..."), AIMessage(...), ...]
    output: AIMessage(content="", tool_calls=[{'name': 'multiply', 'args': {...}}])
```

**Good:** Generation span with extracted, readable attributes
```
├─ call_llm [generation]
    input_content: "Add 3 and 4, then multiply the result by 2."
    tool_calls: "add(a=3, b=4), multiply(a=7, b=2)"
```

```
├─ call_llm [generation]
    input_content: "Add 3 and 4, then multiply the result by 2."
    output_content: "The result is 14."
```

## Using the Beautify Library

TruLens provides built-in utilities for clean traces:

```python
from trulens.core.otel.instrument import instrument, instrument_tools, generation_attributes
from trulens.otel.semconv.trace import SpanAttributes
```

### Clean Span Names with `@instrument(name=...)`

Use the `name` parameter to get clean span names without module prefixes:

```python
@instrument(name="call_llm", span_type=SpanAttributes.SpanType.GENERATION)
def my_llm_function(messages):
    return model.invoke(messages)

# Span will be named "call_llm" not "__main__.my_llm_function"
```

### Tool Spans with `instrument_tools`

Instrument a tools dictionary in place - no changes to app code needed:

```python
from trulens.core.otel.instrument import instrument_tools

instrument_tools(tools_by_name)  # One line setup
```

Now `tool.invoke()` creates spans named after the tool:
```python
# App code unchanged
def call_tool(tool_call):
    tool = tools_by_name[tool_call["name"]]
    result = tool.invoke(tool_call["args"])  # Creates "add" or "multiply" span
    return ToolMessage(content=str(result), ...)
```

### Generation Spans with Content Extraction

Use `generation_attributes()` to automatically extract `input_content`, `output_content`, and `tool_calls`:

```python
@instrument(
    name="call_llm",
    span_type=SpanAttributes.SpanType.GENERATION,
    attributes=generation_attributes()
)
def call_llm(messages):
    return model.invoke(messages)
```

This produces clean, readable generation spans:
- First call: `input_content` + `tool_calls` (when LLM decides to use tools)
- Final call: `input_content` + `output_content` (when LLM produces text response)

### Example: Complete Agent with Instrumentation

```python
from langchain.messages import SystemMessage, ToolMessage
from langgraph.func import entrypoint

from trulens.core.otel.instrument import instrument, instrument_tools, generation_attributes
from trulens.otel.semconv.trace import SpanAttributes

# Instrument tools for clean span names (e.g., "add" instead of "StructuredTool.invoke")
instrument_tools(tools_by_name)


@instrument(
    name="call_llm",
    span_type=SpanAttributes.SpanType.GENERATION,
    attributes=generation_attributes()
)
def call_llm(messages):
    return model.invoke(messages)


def call_tool(tool_call: dict):
    tool = tools_by_name[tool_call["name"]]
    result = tool.invoke(tool_call["args"])  # Creates "add" span automatically
    return ToolMessage(content=str(result), ...)


@entrypoint()
def my_agent(messages):
    ...
```

This produces clean, readable spans:
```
├─ call_llm [generation]
│     input_content: "Add 3 and 4"
│     tool_calls: "add(a=3, b=4)"
├─ add [tool]
├─ call_llm [generation]
│     input_content: "Add 3 and 4"
│     output_content: "The result is 7."
```

## Integration with Other Skills

This skill works alongside:
- `instrumentation/` - Reference for `@instrument()` patterns and span types
- `notebook-execution/` - For running notebooks and keeping dashboard alive
- `evaluation-setup/` - For ensuring semantic attributes work with feedback functions
