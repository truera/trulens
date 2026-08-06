---
categories:
  - General
date: 2026-08-06
---

# Track and Evaluate Multi-Turn Conversations with TruLens

A response can look good by itself while the conversation around it fails. TruLens
2.12 adds first-class conversation tracking and evaluation so you can measure context
retention, consistency, and progress across turns without losing per-turn metrics and
traces.

<!-- more -->

---

## A Conversation Is More Than a Collection of Good Turns

Answer relevance asks whether one response addresses one prompt. That remains useful,
but averaging relevance across a thread cannot tell you whether the assistant:

- Contradicted an earlier answer
- Remembered a constraint from several turns ago
- Repeated itself without moving the task forward
- Recovered after a correction
- Kept a coherent plan across tool calls and follow-up questions

Those are properties of the ordered interaction. Evaluating them requires the complete
transcript, not a set of independent rows.

TruLens 2.12 supports both scopes:

| Scope | Evaluator input | Result |
|---|---|---|
| **Turn level** | One record's input and output | One score per turn |
| **Conversation level** | All ordered `{"input", "output"}` records | One score for the conversation |

## Group Turns with `conversation_id`

Your application owns the conversation identifier. Pass it into the TruLens recording
context around the turns that belong together:

```python
conversation_id = "conv-climate-001"

with tru_chatbot(conversation_id=conversation_id) as recording:
    for user_input in turns:
        chatbot.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": conversation_id}},
        )
```

TruLens propagates the value through
[OpenTelemetry Baggage](https://opentelemetry.io/docs/concepts/signals/baggage/).
Every child span recorded inside the context inherits
`ai.observability.conversation_id`, including spans from instrumented third-party
libraries.

The application framework's session identifier and the TruLens conversation identifier
have different jobs. In the example above, LangChain uses `session_id` to select message
history. TruLens uses `conversation_id` to group records for evaluation, retrieval, and
display. Reusing the same value keeps application memory and observability aligned.

You can also reuse an ID across recording contexts when each turn is recorded
separately:

```python
conversation_id = "support-case-42"

with tru_app(conversation_id=conversation_id):
    app.respond("I cannot sign in.")

with tru_app(conversation_id=conversation_id):
    app.respond("I already reset my password.")
```

## Run Turn and Conversation Metrics Together

Define turn-level metrics with the existing record selectors:

```python
from trulens.core import Metric
from trulens.core import Selector
from trulens.providers.openai import OpenAI

provider = OpenAI(model_engine="gpt-4o-mini")

answer_relevance = Metric(
    implementation=provider.relevance_with_cot_reasons,
    name="Answer Relevance",
    selectors={
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output(),
    },
)
```

Use `.on_conversation()` when the evaluator needs the complete ordered transcript:

```python
conversation_coherence = Metric(
    implementation=provider.coherence_across_turns,
    name="Coherence Across Turns",
).on_conversation()
```

Attach both metrics to the same app:

```python
from trulens.apps.langchain import TruChain

tru_chatbot = TruChain(
    chatbot,
    app_name="Support Assistant",
    app_version="v1",
    feedbacks=[answer_relevance, conversation_coherence],
)
```

When the recording context exits successfully, turn-level metrics are queued once per
record. Conversation-level metrics are queued once over the exact ordered records
created in that context.

Use the returned recording when your next step needs to wait for the results:

```python
feedback_results = recording.retrieve_feedback_results()
```

For a custom evaluator, TruLens also exposes selectors for the full transcript or only
one side of it:

```python
Selector.select_conversation()
Selector.select_conversation_input()
Selector.select_conversation_output()
```

Conversation selectors cannot be mixed with record or span selectors in one `Metric`.
Keep metrics at one scope and attach separate metrics when you need both views.

## Retrieve Conversations as First-Class Objects

TruLens reconstructs conversations from their ordered records. You can list
conversations for an app or retrieve the records from one thread:

```python
conversations = session.get_conversations(app_id=tru_chatbot.app_id)

records = session.get_records_by_conversation(
    conversation_id="conv-climate-001",
    app_id=tru_chatbot.app_id,
)
```

Conversation-level scores belong to the latest record in the evaluated batch. Earlier
rows intentionally remain empty for that metric. Do not average a conversation metric;
it is already one evaluation over the ordered transcript. You can still group and
average turn-level metrics when that is useful for reporting.

## Inspect Threads Without Losing Turn-Level Detail

The **Records** page now groups records that share a `conversation_id` into one thread.
The conversation view shows:

- The first and last turns, with long middle sections collapsed
- One result for each conversation-level metric
- Separate per-turn metric results
- Total latency, tokens, and cost across the recorded turns
- A link from every turn to its ordinary record trace

Standalone records still use the existing record detail view. Adding conversation
tracking does not remove the per-record traces you use to investigate retrievals,
generations, tool calls, or latency.

## Try the Complete Workflow

The framework-neutral quickstart records a 12-turn conversation and a single-turn
conversation, waits for their evaluations, and opens the thread-aware dashboard:

- [Conversation Evaluation Quickstart](https://github.com/truera/trulens/blob/main/examples/quickstart/conversation_evaluation.ipynb)
- [LangChain Conversation Evaluation](https://github.com/truera/trulens/blob/main/examples/expositional/frameworks/langchain/conversation_evaluation.ipynb)
- [Conversation Evaluation Guide](https://www.trulens.org/component_guides/instrumentation/conversation_evaluation/)
- [Conversation ID Guide](https://www.trulens.org/component_guides/instrumentation/conversation_id/)
