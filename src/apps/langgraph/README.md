# trulens-apps-langgraph

TruLens integration for LangGraph applications. This package provides comprehensive instrumentation and evaluation capabilities for LangGraph-based multi-agent workflows.

## Features

- **Automatic Detection**: TruGraph automatically detects LangGraph applications
- **Combined Instrumentation**: Inherits all LangChain instrumentation plus LangGraph-specific methods
- **Multi-Agent Evaluation**: Comprehensive evaluation capabilities for complex workflows
- **Automatic @task Instrumentation**: Automatically detects and instruments functions decorated with `@task`
- **Smart Attribute Extraction**: Intelligently extracts information from function arguments

## Installation

```bash
pip install trulens-apps-langgraph
```

## Quick Start

```python
from langgraph.graph import StateGraph, MessagesState, END
from langchain_core.messages import HumanMessage
from trulens.apps.langgraph import TruGraph

# Create your LangGraph application
workflow = StateGraph(MessagesState)
workflow.add_node("agent", your_agent_function)
workflow.add_edge("agent", END)
workflow.set_entry_point("agent")
graph = workflow.compile()

# Automatically instrument with TruGraph
tru_app = TruGraph(graph, app_name="MyLangGraphApp")

# Use normally - all interactions are automatically logged
with tru_app as recording:
    result = graph.invoke({"messages": [HumanMessage(content="Hello!")]})
```

## Automatic @task Instrumentation

TruGraph automatically instruments functions decorated with LangGraph's `@task` decorator by monkey-patching the decorator itself. This follows TruLens instrumentation patterns and ensures seamless integration:

```python
from langgraph.func import task

@task  # Automatically instrumented by TruGraph when TruGraph is imported
def my_agent_function(state, config):
    # Your agent logic here
    return updated_state
```

### How it works:

1. **Decorator Monkey-Patching**: TruGraph patches the `@task` decorator at import time
2. **Intelligent Attribute Extraction**: Automatically extracts information from function arguments:
   - Handles `BaseChatModel` and `BaseModel` objects
   - Extracts data from dataclasses and Pydantic models
   - Skips non-serializable objects like LLM pools
   - Captures return values and exceptions
3. **No Code Changes Required**: Works with existing `@task` decorated functions

This approach follows TruLens conventions and is more robust than scanning `sys.modules`.

## Usage

See the [TruLens documentation](https://trulens.org/getting_started/) for complete usage instructions.
