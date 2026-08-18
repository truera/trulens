# Conversation Evaluation

TruLens can evaluate both individual turns and the complete ordered transcript of a
multi-turn conversation. This guide covers the full loop: recording turns with a
shared `conversation_id`, defining metrics at the correct scope, computing one
conversation-level result, and exploring the results in the dashboard.

!!! tip "Quickstart notebook"
    Run the
    [Conversation Evaluation Quickstart](../../getting_started/quickstarts/conversation_evaluation.ipynb)
    for a framework-neutral end-to-end example using an instrumented Python app.

## How `conversation_id` flows into spans

Each top-level invocation of your application produces a separate TruLens record.
Wrap all turns that should be evaluated together in one recording context:

```python
conversation_id = "conv-climate-001"

with tru_chatbot(conversation_id=conversation_id) as recording:
    for user_input in turns:
        chatbot.invoke(
            {"input": user_input},
            config={
                "configurable": {
                    "session_id": conversation_id,
                }
            },
        )
```

TruLens propagates `conversation_id` through
[OpenTelemetry Baggage](https://opentelemetry.io/docs/concepts/signals/baggage/),
so every child span created during that invocation carries
`ai.observability.conversation_id`.

The context defines the online evaluation batch. When it exits successfully,
turn-level metrics remain queued per record and conversation-level metrics are queued
once over the exact ordered records created in that context. Context exit does not
wait for evaluation to finish.

The two identifiers in the example have different responsibilities:

- LangChain's `session_id` selects the message history used by
  `RunnableWithMessageHistory`.
- TruLens' `conversation_id` groups independently recorded turns for conversation
  evaluation and display.

Reusing the same string keeps application memory and observability aligned, but
TruLens does not derive one identifier from the other.

See [Conversation ID — Grouping Multi-Turn Traces](conversation_id.md) for the
recording API and propagation details.

## Turn-level and conversation-level metrics

Metric scope determines what the evaluator receives and how many scores it produces.

| Scope | Example | Selected input | Result |
|---|---|---|---|
| **Turn level** | Answer Relevance | One record's input and output | One score for every turn |
| **Conversation level** | Coherence Across Turns | All ordered `{"input", "output"}` records with the same conversation ID | One score for the conversation |

Averaging turn-level scores can summarize a conversation for reporting, but it cannot
measure contradictions, context retention, or flow across turns. Those require a
conversation-level evaluator.

## Define a turn-level metric

Answer Relevance evaluates each user input and assistant response independently:

```python
from trulens.core import Metric
from trulens.core import Selector
from trulens.providers.openai import OpenAI

provider = OpenAI(model_engine="gpt-4o-mini")

f_answer_relevance = Metric(
    implementation=provider.relevance_with_cot_reasons,
    name="Answer Relevance",
    selectors={
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output(),
    },
)
```

## Define a conversation-level metric

Use `.on_conversation()` when the evaluator needs the complete ordered transcript:

```python
f_conversation_coherence = Metric(
    implementation=provider.coherence_across_turns,
    name="Coherence Across Turns",
).on_conversation()
```

The selected value has this shape:

```python
[
    {"input": "First user message", "output": "First assistant response"},
    {"input": "Follow-up question", "output": "Follow-up response"},
]
```

TruLens also provides explicit selector factories when a custom evaluator needs only
part of the transcript:

```python
Selector.select_conversation()         # Ordered input/output records
Selector.select_conversation_input()   # Ordered record inputs
Selector.select_conversation_output()  # Ordered record outputs
```

Conversation selectors cannot be mixed with record or span selectors in the same
`Metric`. Attach separate metrics at each scope instead.

## Attach both metrics to an app

Turn-level and conversation-level metrics can run on the same recorded application:

```python
from trulens.apps.langchain import TruChain

tru_chatbot = TruChain(
    chatbot,
    app_name="Conversation Evaluation Quickstart",
    app_version="v1",
    feedbacks=[
        f_answer_relevance,
        f_conversation_coherence,
    ],
)
```

Record all turns that belong to one evaluation batch in the same context. Conversation
evaluation is queued automatically on successful context exit:

```python
with tru_chatbot(conversation_id=conversation_id) as recording:
    for user_input in turns:
        chatbot.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": conversation_id}},
        )
```

Use the returned recording when the next step must wait for persisted results:

```python
feedback_results = recording.retrieve_feedback_results()
```

The evaluator waits until every exact record is visible, orders their roots by start
time, and runs the conversation metric once. Records without a `conversation_id` are
not included in conversation-level evaluation.

Reusing the same `conversation_id` in another context creates another independent
evaluation batch. Earlier contexts are not reloaded or re-evaluated. Ordered
contributing record-root span IDs provide provenance and deduplication, and the
conversation evaluation root links to those record roots using OpenTelemetry links.

For historical analysis or backfills, explicit event-batch APIs remain available:

```python
tru_chatbot.compute_feedbacks()
session.compute_feedbacks_on_events(events, feedbacks)
```

Those APIs evaluate the conversation groups present in the supplied event batch; they
are separate from context-bounded online evaluation.

## Retrieve and interpret scores

Retrieve records as usual:

```python
records, feedback_columns = session.get_records_and_feedback(
    app_ids=[tru_chatbot.app_id]
)

records[
    [
        "conversation_id",
        "input",
        "output",
        "Answer Relevance",
        "Coherence Across Turns",
    ]
]
```

`Answer Relevance` is populated on every evaluated turn. The conversation-level
`Coherence Across Turns` score is owned by the latest record in the evaluated
conversation, so the tabular column is intentionally empty on earlier turns.

### Aggregate a turn-level metric for reporting

Use `groupby` only when you want to summarize a per-turn metric:

```python
answer_relevance_by_conversation = (
    records.dropna(subset=["conversation_id", "Answer Relevance"])
    .groupby("conversation_id", as_index=False)["Answer Relevance"]
    .mean()
    .rename(columns={"Answer Relevance": "Average Answer Relevance"})
)
```

Do not aggregate `Coherence Across Turns`; it is already one evaluation over the
ordered conversation.

## Dashboard conversation view

Launch the dashboard with the same TruLens session:

```python
from trulens.dashboard import run_dashboard

run_dashboard(session)
```

On the **Records** page, records sharing a `conversation_id` appear as one thread:

- The first and last turns remain visible; middle turns in long conversations are
  collapsed under **See N more turns**.
- **Conversation metrics** render once for the thread.
- **Turn metrics** remain separate and can be inspected per record.
- Conversation metric details show the transcript, score, and explanation.
- Aggregate latency, tokens, and cost summarize the recorded turns.

## Requirements and scope

- Conversation metrics require OpenTelemetry tracing.
- `conversation_id` must be a string and is managed by your application.
- Online conversation metrics are queued once when a conversation recording context
  exits successfully.
- Direct metric execution does not support conversation selectors because it does not
  receive a multi-record OpenTelemetry event batch.
- Explicit event-batch computation remains supported through `App.compute_feedbacks()`,
  `TruSession.compute_feedbacks_on_events()`, and client-side `Run.compute_metrics()`.

## See also

- [Conversation ID — Grouping Multi-Turn Traces](conversation_id.md) — propagation
  mechanics and API reference
- [Span Groups](span_groups.md) — localize metrics to specific segments within a
  single turn
- [Feedback Selectors](../evaluation/feedback_selectors/index.md) — choosing what
  to evaluate in each turn
- [Conversation Evaluation Quickstart notebook](../../getting_started/quickstarts/conversation_evaluation.ipynb)
- [LangChain Conversation Evaluation](https://github.com/truera/trulens/blob/main/examples/expositional/frameworks/langchain/conversation_evaluation.ipynb)
