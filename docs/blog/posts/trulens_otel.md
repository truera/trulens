---
categories:
  - General
date: 2025-06-02
---

# Telemetry for the Agentic World: TruLens + OpenTelemetry

Agents are rapidly gaining traction across AI applications. With this growth comes a new set of challenges: how do we trace, observe, and evaluate these dynamic, distributed systems?
Today, we’re excited to share that **TruLens now supports [OpenTelemetry](https://opentelemetry.io/) (OTel)**, unlocking powerful, interoperable observability for the agentic world.

---

## Challenge for Tracing Agents

Tracing agentic applications is fundamentally different from tracing traditional software systems:

- **Language-agnostic**: Agents can be written in Python, Go, Java, or more, requiring tracing that transcends language boundaries.
- **Distributed by nature**: Multi-agent systems often span multiple machines or processes.
- **Existing telemetry stacks**: Many developers and enterprises already use OpenTelemetry, so tracing compatibility is essential.
- **Dynamic execution**: Unlike traditional apps, agents often make decisions on the fly, with branching workflows that can’t be fully known in advance.
- **Interoperability standards**: As frameworks like Model Context Protocol (MCP) and Agent2Agent Protocol (A2A) emerge, tracing must support agents working across different systems.
- **Repeated tool usage**: Agents may call the same function or tool multiple times in a single execution trace, requiring fine-grained visibility into span grouping to understand what’s happening and why.

---

## What is TruLens

**TruLens** is an open source library for evaluating and tracing AI agents, including RAG systems and other LLM applications. It combines OpenTelemetry-based tracing with trustworthy evaluations, including both ground truth metrics and reference-free (LLM-as-a-Judge) feedback.

TruLens pioneered the **RAG Triad**—a structured evaluation of:

- Context relevance
- Groundedness
- Answer relevance

These evaluations provide a foundation for understanding the performance of RAGs and agentic RAGs, supported by [benchmarks](https://www.snowflake.com/en/engineering-blog/benchmarking-LLM-as-a-judge-RAG-triad-metrics/) like **LLM-AggreFact**, **TREC-DL**, and **HotPotQA**.

This combination of trusted evaluators and open standard tracing gives you tools to both **improve your application offline** and **monitor once it reaches production**.

---

## How TruLens Augments OpenTelemetry

As AI applications become increasingly agentic, TruLens’ shift to OpenTelemetry enables observability that is:

- **Interoperable with existing telemetry stacks**
- **Compatible across languages and frameworks**
- **Capable of tracing dynamic agent workflows**

TruLens now accepts any span that adheres to the OTel standard.

---

## What is OpenTelemetry?

**OpenTelemetry (OTel)** is an open-source observability framework for generating, collecting, and exporting telemetry data such as traces, metrics, and logs.

In LLM and agentic contexts, OpenTelemetry enables language-agnostic, interoperable tracing for:

- Multi-agent systems
- Distributed environments
- Tooling interoperability

> **What is a span?**
> A span represents a single unit of work. In LLM apps, this might be: planning, routing, retrieval, tool usage, or generation.

---

## TruLens Defines Semantic Conventions for the Agentic World

TruLens maps **span attributes** to common definitions using semantic conventions to ensure:

- Cross-framework interoperability
- Shared instrumentation for MCP and A2A
- Consistent evaluation across implementations

Read more about [TruLens Semantic Conventions](https://www.trulens.org/otel/semantic_conventions/).

---

## Using Semantic Conventions to Compute Evaluation Metrics

TruLens allows evaluation of metrics based on span instrumentation.

```python
@instrument(
    span_type=SpanAttributes.SpanType.RETRIEVAL,
    attributes={
        SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
        SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
    },
)
def retrieve(self, query: str) -> list:
    results = vector_store.query(query_texts=query, n_results=4)
    return [doc for sublist in results["documents"] for doc in sublist]
```

```python
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons, name="Context Relevance")
    .on_input()
    .on_context(call_feedback_function_per_entry_in_list=True)
    .aggregate(np.mean)
)
```

---

## Computing Metrics on Complex Execution Flows

TruLens introduces **span groups** to handle repeated tool calls within a trace.

```python
class App:

    @instrument(attributes={SpanAttributes.SPAN_GROUPS: "idx"})
    def clean_up_question(question: str, idx: str) -> str:
        ...

    @instrument(attributes={SpanAttributes.SPAN_GROUPS: "idx"})
    def clean_up_response(response: str, idx: str) -> str:
        ...

    @instrument()
    def combine_responses(cleaned_responses: List[str]) -> str:
        ...

    @instrument()
    def query(complex_question: str) -> str:
        questions = break_question_down(complex_question)
        cleaned_responses = []
        for i, question in enumerate(questions):
            cleaned_question = clean_up_question(question, str(i))
            response = call_llm(cleaned_question)
            cleaned_response = clean_up_response(response, str(i))
            cleaned_responses.append(cleaned_response)
        return combine_responses(cleaned_responses)
```

---

## How to Examine Execution Flows in TruLens

Run:

```python
session.run_dashboard()
```

…and visually inspect execution traces. Span types are shown directly in the dashboard to help identify branching, errors, or performance issues.

![TruLens Trace](../assets/trulens_otel/langgraph_trace.png)

---

## How to Get Started

Ready to get started?

Today, we are launching a pre-release of TruLens on Otel. Below is a minimal walkthrough of using TruLens with OpenTelemetry. You can also find a curated list of examples of working with TruLens and Otel in this [folder](https://github.com/truera/trulens/tree/main/examples/experimental/otel), including a new LangGraph quickstart - showing how to trace and evaluate a multi-agent graph.

1. **Install TruLens**:

```bash
pip install trulens-core==1.5.0
```

2. **Enable OpenTelemetry**:

```python
os.environ["TRULENS_OTEL_TRACING"] = "1"
```

3. **Instrument Methods**:

```python
from trulens.core.otel.instrument import instrument

@instrument(
    attributes={
        SpanAttributes.RECORD_ROOT.INPUT: "query",
        SpanAttributes.RECORD_ROOT.OUTPUT: "return",
    },
)
def query(self, query: str) -> str:
    context_str = self.retrieve(query=query)
    completion = self.generate_completion(query=query, context_str=context_str)
    return completion
```

4. **Add Evaluations**:

```python
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on_input()
    .on_output()
)
```

Using selectors:

```python
from trulens.core.feedback.selector import Selector

f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance")
    .on({
        "prompt": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.INPUT,
        ),
    })
    .on({
        "response": Selector(
            span_type=SpanAttributes.SpanType.RECORD_ROOT,
            span_attribute=SpanAttributes.RECORD_ROOT.OUTPUT,
        ),
    })
)
```

5. **Register Your App**:

```python
from trulens.apps.app import TruApp

rag = RAG(model_name="gpt-4.1-mini")

tru_rag = TruApp(
    rag,
    app_name="OTEL-RAG",
    app_version="4.1-mini",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)
```

6. **Run the Dashboard**:

```python
from trulens.dashboard import run_dashboard

run_dashboard(session)
```

---

## Concluding Thoughts

By building on top of **OpenTelemetry**, TruLens delivers a **universal tracing and evaluation platform** for modern AI systems. Whether your agents are built in Python, composed via MCP, or distributed across systems—TruLens provides a common observability layer for telemetry and evaluation.

Try our new TruLens-OTel quickstarts for [custom python apps](https://github.com/truera/trulens/tree/main/examples/experimental/otel/quickstart_otel.ipynb), [LangGraph](https://github.com/truera/trulens/tree/main/examples/experimental/otel/langgraph_quickstart_otel.ipynb), and [Llama-Index](https://github.com/truera/trulens/tree/main/examples/experimental/otel/llama_index_quickstart_otel.ipynb).

**Let’s build the future of trustworthy agentic AI together.**
