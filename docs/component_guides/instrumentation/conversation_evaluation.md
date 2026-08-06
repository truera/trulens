# Conversation Evaluation

This guide explains how to evaluate multi-turn conversations end-to-end with TruLens:
how `conversation_id` flows from your app into TruLens, how to wire per-turn metrics,
the difference between record-level and conversation-level scores, and how to explore
results in the dashboard's conversation view.

!!! tip "Quickstart notebook"
    The concepts on this page are demonstrated end-to-end in the
    [Conversation Evaluation Quickstart](../../getting_started/quickstarts/conversation_evaluation.ipynb)
    notebook.

## How `conversation_id` flows

When you pass `conversation_id` to the recording context manager, TruLens propagates
the value through [OpenTelemetry Baggage](https://opentelemetry.io/docs/concepts/signals/baggage/)
so every span produced during that invocation carries the attribute
`ai.observability.conversation_id`.  Each invocation (a single turn) becomes one TruLens
**record**.  Records that share a `conversation_id` are logically grouped into a
**conversation**.

```
Turn 1  →  with tru_app(conversation_id="conv-42") as recording:  →  Record A  ┐
Turn 2  →  with tru_app(conversation_id="conv-42") as recording:  →  Record B  ├─ Conversation "conv-42"
Turn 3  →  with tru_app(conversation_id="conv-42") as recording:  →  Record C  ┘
```

See [Conversation ID — Grouping Multi-Turn Traces](conversation_id.md) for the
full API reference and propagation details.

## Wiring conversation metrics

Feedback metrics are defined and attached to the `TruApp` wrapper exactly as for
single-turn apps.  Each metric is evaluated **per turn** (per record) by default.

```python
from trulens.core import Metric, Selector, TruSession
from trulens.apps.langchain import TruChain
from trulens.providers.openai import OpenAI

session = TruSession()
provider = OpenAI(model_engine="gpt-4o-mini")

# Per-turn answer relevance
f_answer_relevance = Metric(
    implementation=provider.relevance_with_cot_reasons,
    name="Answer Relevance",
    selectors={
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output(),
    },
)

# Per-turn coherence
f_coherence = Metric(
    implementation=provider.coherence_with_cot_reasons,
    name="Coherence",
    selectors={
        "text": Selector.select_record_output(),
    },
)

tru_chatbot = TruChain(
    chatbot,
    app_name="Chatbot",
    app_version="v1",
    feedbacks=[f_answer_relevance, f_coherence],
)
```

## Recording multi-turn conversations

Pass the same `conversation_id` string to every turn of the same conversation.
Use a different ID for each independent conversation.

```python
CONV_ID = "conv-climate-001"

turns = [
    "What is the greenhouse effect?",
    "How does it relate to global warming?",
    "What are the most effective ways to reduce carbon emissions?",
]

for turn in turns:
    with tru_chatbot(conversation_id=CONV_ID) as recording:
        chatbot.invoke(
            {"input": turn},
            config={"configurable": {"session_id": CONV_ID}},
        )
```

Each `with` block produces one record.  All three records share the same
`conversation_id`, so TruLens groups them as a single conversation thread.

## Per-record vs conversation-level scores

| Level | What it measures | How to get it |
|---|---|---|
| **Per-record (turn-level)** | Quality of a single response — relevance, coherence, groundedness | `Metric` evaluated inside each recording context; visible per row in `get_records_and_feedback()` |
| **Conversation-level** | Quality across the whole thread — e.g., average score, worst-turn detection | Aggregate per-turn scores by `conversation_id` in post-processing (see below) |

### Retrieving turn-level scores

```python
records, feedback = session.get_records_and_feedback(app_ids=["Chatbot"])

# All turns for one conversation
conv_records = records[records["conversation_id"] == "conv-climate-001"]
conv_records[["input", "output", "Answer Relevance", "Coherence"]]
```

### Aggregating to conversation-level scores

```python
summary = (
    records
    .groupby("conversation_id")[["Answer Relevance", "Coherence"]]
    .mean()
    .round(3)
)
```

This gives one row per conversation with the mean score across all its turns —
useful for comparing conversations or detecting which thread had the most trouble.

## Dashboard conversation view

Launch the dashboard to explore conversations visually:

```python
from trulens.dashboard import run_dashboard

run_dashboard(session)
```

In the dashboard:

- The **Records** tab shows all turns. Filter by `conversation_id` to see only
  the turns belonging to one conversation.
- Each turn's feedback scores are displayed inline so you can spot the exact
  turn where quality dropped.
- The **Leaderboard** tab summarises aggregate metrics across all conversations
  for quick comparison.

## See also

- [Conversation ID — Grouping Multi-Turn Traces](conversation_id.md) — propagation
  mechanics and API reference
- [Span Groups](span_groups.md) — localize metrics to specific segments within a
  single turn
- [Feedback Selectors](../evaluation/feedback_selectors/index.md) — choosing what
  to evaluate in each turn
- [Conversation Evaluation Quickstart notebook](../../getting_started/quickstarts/conversation_evaluation.ipynb)
