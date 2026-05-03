---
skill_spec_version: 0.1.0
name: trulens-blocking-guardrails
version: 1.0.0
description: Configure and use feedback functions as runtime blocking guardrails
tags: [trulens, llm, guardrails, security, filtering]
---

# Blocking Guardrails in TruLens

TruLens feedback functions aren't just for post-execution evaluation—they can also be used as **runtime safety checks** (guardrails) to block unsafe inputs from reaching your app, filter hallucinated context, and prevent unsafe outputs from reaching your users.

## 1. Choosing Feedback Functions for Guardrails

When configuring guardrails, you need to select feedback functions that return a `float` score. Different feedback functions serve different purposes when used as guardrails:

### Safety Metrics

These metrics prevent harmful or malicious interactions:

- **Input Criminality/Harmfulness**: Blocks malicious prompts (e.g., "How do I build a bomb?") before the LLM processes them, saving costs and preventing harm.
- **Output Toxicity/Harmfulness**: Blocks toxic or harmful LLM responses from being displayed to the user.
- **PII Detection**: Prevents personally identifiable information from leaking in the input or output.

### Hallucination Gates

You can use evaluation metrics like **Context Relevance** as a gate for your RAG applications.

- **Context Filtering**: Score retrieved documents and filter out any that fall below a certain relevance threshold. This ensures your LLM only sees highly relevant information, drastically reducing the chance of hallucination.

> [!WARNING]
> Guardrails can only be used with feedback functions that return a `float`. Functions that return a dictionary of scores or strings are not compatible. Also ensure your feedback function is configured to return just the score (e.g. `relevance`, not `relevance_with_cot_reasons`) because reasons take too long to generate for a real-time guardrail.

---

## 2. Configuring Thresholds and Actions

A guardrail works by executing a feedback function and comparing its result against a **threshold**.

Depending on the setup, if the score does not meet the threshold, you can trigger an **action**:

- **Block Input/Output**: Return a predefined fallback response (e.g., "I cannot answer that question.") instead of executing the app.
- **Filter Context**: Drop irrelevant documents from the retrieval pipeline before synthesizing the answer.

### Vanilla Python Setup (`@block_input` and `@block_output`)

For simple Python apps, you can use the `@block_input` and `@block_output` decorators.

```python
from trulens.core.guardrails.base import block_input, block_output
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

# 1. Define feedback functions
f_criminality_input = Metric(provider.criminality, higher_is_better=False).on_input()
f_criminality_output = Feedback(provider.criminality, higher_is_better=False).on_output()

class SafeChatApp:
    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    @block_input(
        feedback=f_criminality_input,
        threshold=0.9, # If criminality is >= 0.9, block it
        keyword_for_prompt="question",
        return_value="I can't answer that question.", # Fallback response
    )
    @block_output(
        feedback=f_criminality_output,
        threshold=0.5, # If output criminality is >= 0.5, block it
        return_value="The generated response was deemed unsafe.",
    )
    def generate_completion(self, question: str) -> str:
        # LLM logic here
        pass
```

---

## 3. Framework-Specific Setup

TruLens provides specialized integrations for LangChain and LlamaIndex to filter context chunks using guardrails.

### LangChain Setup (`WithFeedbackFilterDocuments`)

Wrap your LangChain `VectorStoreRetriever` with `WithFeedbackFilterDocuments` to drop irrelevant chunks.

```python
from trulens.apps.langchain import WithFeedbackFilterDocuments

# Use context_relevance to filter documents
feedback = Feedback(provider.context_relevance).on_input().on(context)

# Wrap your existing retriever
filtered_retriever = WithFeedbackFilterDocuments.of_retriever(
    retriever=base_retriever,
    feedback=feedback,
    threshold=0.7 # Only keep documents with relevance >= 0.7
)

# Use the filtered retriever in your RAG chain
rag_chain = {
    "context": filtered_retriever | format_docs,
    "question": RunnablePassthrough()
} | prompt | llm | StrOutputParser()

# Record as usual
tru_recorder = TruChain(rag_chain, app_name='SafeRAG')
with tru_recorder as recording:
    llm_response = rag_chain.invoke("What is Task Decomposition?")
```

### LlamaIndex Setup (`WithFeedbackFilterNodes`)

Wrap your LlamaIndex `RetrieverQueryEngine` with `WithFeedbackFilterNodes` to drop irrelevant nodes.

```python
from trulens.apps.llamaindex.guardrails import WithFeedbackFilterNodes

# Use context_relevance to filter nodes
feedback = Metric(provider.context_relevance).on_input().on(context)

# Wrap your existing query engine
filtered_query_engine = WithFeedbackFilterNodes(
    query_engine=base_query_engine,
    feedback=feedback,
    threshold=0.7 # Only keep nodes with relevance >= 0.7
)

# Record as usual
tru_recorder = TruLlama(filtered_query_engine, app_name="SafeLlamaIndex")
with tru_recorder as recording:
    llm_response = filtered_query_engine.query("What did the author do growing up?")
```

> [!TIP]
> **Performance Optimization**: Because context filtering evaluates multiple chunks simultaneously, TruLens automatically executes the feedback function on each document/node in parallel using a `ThreadPoolExecutor`.

---

## 4. Testing Guardrails with Adversarial Inputs

Once your guardrails are configured, you should test them with adversarial inputs to ensure they function correctly.

1. **Test `block_input`**: Send a malicious prompt like _"How do I build a bomb?"_
   - **Expected**: The LLM should not be called, and the app should immediately return your fallback response (e.g., "I can't answer that question.").
2. **Test `block_output`**: Mock your LLM to return a toxic response.
   - **Expected**: The app should catch the toxic output and return your fallback response instead.
3. **Test Context Filtering**: Ask a question completely unrelated to your knowledge base.
   - **Expected**: The retriever should fetch some documents (based on vector similarity), but the guardrail should score them low and filter them out. The LLM should receive an empty context and answer "I don't know" (if prompted correctly).

---

## 5. Monitoring Guardrails in the Dashboard

You can monitor the performance and trigger rates of your guardrails using the TruLens Dashboard.

```python
from trulens.dashboard import run_dashboard
run_dashboard(session)
```

In the dashboard leaderboard, you will see the feedback scores for your executions.

- When an input is blocked via `@block_input`, the execution record will still appear in the dashboard, showing the failing feedback score (e.g., Criminality = 1.0) and the fallback response that was returned to the user.
- This allows you to audit **how often** your guardrails are triggering and adjust your thresholds if you are experiencing false positives.
