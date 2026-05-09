![PyPI - Version](https://img.shields.io/pypi/v/trulens?label=trulens&link=https%3A%2F%2Fpypi.org%2Fproject%2Ftrulens%2F)
[![Azure Build Status](https://dev.azure.com/truera/trulens/_apis/build/status%2FTruLens%20E2E%20Tests?branchName=main)](https://dev.azure.com/truera/trulens/_build/latest?definitionId=8&branchName=main)
![GitHub](https://img.shields.io/github/license/truera/trulens)
![PyPI - Downloads](https://img.shields.io/pypi/dm/trulens-core)
[![Discourse](https://img.shields.io/discourse/users?server=https://snowflake.discourse.group/)](https://snowflake.discourse.group/c/ai-research-and-development-community/trulens/97)
[![Docs](https://img.shields.io/badge/docs-trulens.org-blue)](https://www.trulens.org/getting_started/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/examples/quickstart/langchain_quickstart.ipynb)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/truera/trulens)

# 🦑 Welcome to TruLens!

![TruLens](https://www.trulens.org/assets/images/Neural_Network_Explainability.png)

**Don't just vibe-check your LLM app!** Systematically evaluate and track your
LLM experiments with TruLens. As you develop your app including prompts, models,
retrievers, knowledge sources and more, *TruLens* is the tool you need to
understand its performance.

Fine-grained, stack-agnostic instrumentation and comprehensive evaluations help
you to identify failure modes & systematically iterate to improve your
application.

Read more about the core concepts behind TruLens including [Feedback Functions](https://www.trulens.org/getting_started/core_concepts/feedback_functions/),
[The RAG Triad](https://www.trulens.org/getting_started/core_concepts/rag_triad/),
and [Honest, Harmless and Helpful Evals](https://www.trulens.org/getting_started/core_concepts/honest_harmless_helpful_evals/).

## TruLens in the development workflow

Build your first prototype then connect instrumentation and logging with
TruLens. Decide what feedbacks you need, and specify them with TruLens to run
alongside your app. Then iterate and compare versions of your app in an
easy-to-use user interface 👇

![Architecture
Diagram](https://www.trulens.org/assets/images/TruLens_Architecture.png)

## Installation and Setup

Install the trulens pip package from PyPI.

```bash
pip install trulens
```

Install with a specific LLM provider for feedback evaluation:

```bash
pip install trulens trulens-providers-openai   # OpenAI / Azure OpenAI
pip install trulens trulens-providers-litellm  # LiteLLM (Anthropic, Cohere, Mistral, …)
pip install trulens trulens-providers-google   # Google Gemini
pip install trulens trulens-providers-bedrock  # AWS Bedrock
pip install trulens trulens-providers-cortex   # Snowflake Cortex
pip install trulens trulens-providers-huggingface  # HuggingFace
pip install trulens trulens-providers-langchain    # LangChain models
```

Install with a specific app framework integration:

```bash
pip install trulens trulens-apps-langchain    # LangChain / LangGraph
pip install trulens trulens-apps-llamaindex  # LlamaIndex
```

## Quick Usage

Walk through how to instrument and evaluate a RAG built from scratch with
TruLens.

[![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/examples/quickstart/quickstart.ipynb)

## Key Features

### 🔭 OpenTelemetry-based tracing

TruLens instrumentation is built on [OpenTelemetry](https://opentelemetry.io/).
Every function call, LLM generation, retrieval, and tool invocation is captured
as a structured OTEL span. This makes TruLens interoperable with existing
observability infrastructure — export traces to Jaeger, Grafana Tempo, Datadog,
or any OTLP-compatible backend.

```python
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
        ...
```

### 🤖 Agentic evaluations

Seven purpose-built evaluators for agentic systems — each measuring a distinct
aspect of agent behavior:

| Evaluator | What it measures |
|-----------|-----------------|
| LogicalConsistency | Reasoning coherence; flags hallucinations and unsupported assertions |
| ExecutionEfficiency | Redundant steps, unnecessary retries, wasted computation |
| PlanAdherence | Whether execution followed the stated plan |
| PlanQuality | Intrinsic plan quality — strategy, not outcome |
| ToolSelection | Right tool chosen for each subtask |
| ToolCalling | Argument validity and output interpretation |
| ToolQuality | External tool/service reliability |

### 📊 Batch and inline evaluation

Run evaluations alongside your app, on existing data, or in offline batch mode:

```python
# Inline — evaluate as the app runs
with tru_recorder as recording:
    response = my_app.query("What is TruLens?")

# Batch — evaluate a pre-collected dataset without a live app
results = session.evaluate(dataset=df, metrics=[relevance, groundedness])
```

### 🔌 MCP support

Instrument [Model Context Protocol](https://modelcontextprotocol.io/) tool calls
with the `MCP` span type to capture tool name, arguments, output, and latency:

```python
@instrument(span_type=SpanAttributes.SpanType.MCP)
def call_mcp_tool(self, tool_name: str, arguments: dict) -> str:
    ...
```

### 🎯 Selector API

Target any span attribute for evaluation using the flexible Selector API:

```python
from trulens.core.schema.select import Select

f_context_relevance = (
    Feedback(provider.context_relevance)
    .on_input()
    .on(Select.RecordCalls.retrieve.rets[:])
)
```

## Supported LLM Providers

| Provider | Package |
|----------|---------|
| OpenAI / Azure OpenAI | `trulens-providers-openai` |
| LiteLLM (Anthropic, Cohere, Mistral, and more) | `trulens-providers-litellm` |
| Google Gemini | `trulens-providers-google` |
| AWS Bedrock | `trulens-providers-bedrock` |
| Snowflake Cortex | `trulens-providers-cortex` |
| HuggingFace | `trulens-providers-huggingface` |
| LangChain models | `trulens-providers-langchain` |

## 💡 Contributing & Community

Interested in contributing? See our [contributing
guide](https://www.trulens.org/contributing/) for more details.

The best way to support TruLens is to give us a ⭐ on
[GitHub](https://www.github.com/truera/trulens) and join our [discourse
community](https://snowflake.discourse.group/c/ai-research-and-development-community/trulens/97)!
