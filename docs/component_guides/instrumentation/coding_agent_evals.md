# Evaluating Coding-Agent Hook Traces

This guide covers running TruLens metrics on traces already captured by client hooks. The hooks export completed turns into a TruLens database; evaluation is a separate pass over those events.

## Evaluate exported traces

Use the same database URL configured for the hook worker:

```python
from trulens.core import Metric, Selector, TruSession
from trulens.providers.openai import OpenAI

session = TruSession(
    database_url="postgresql+psycopg://trulens@localhost/traces"
)
provider = OpenAI(model_engine="gpt-4o")

metrics = [
    Metric(
        implementation=provider.tool_selection_with_cot_reasons,
        name="Tool Selection",
        selectors={"trace": Selector(trace_level=True)},
    ),
    Metric(
        implementation=provider.execution_efficiency_with_cot_reasons,
        name="Execution Efficiency",
        selectors={"trace": Selector(trace_level=True)},
    ),
    Metric(
        implementation=provider.coherence_across_turns_with_cot_reasons,
        name="Session Coherence",
    ).on_conversation(),
]

events = session.get_events(app_name="cursor", app_version=None)
session.compute_feedbacks_on_events(events, metrics)
```

`Tool Selection` and `Execution Efficiency` score individual turns against their full trace. `Session Coherence` scores the ordered turns in each native coding-agent conversation.

## Capture content first

Lifecycle metadata is captured by default, but prompts, responses, tool payloads, and diffs are independently privacy-controlled. Enable the fields required by the metrics before the coding session starts:

```bash
export TRULENS_CAPTURE_CONTENT=true
export TRULENS_CAPTURE_TOOL_PAYLOADS=true
export TRULENS_CAPTURE_DIFFS=true
```

The defaults are intentionally metadata-only. Content cannot be reconstructed after a turn has been exported without it.

## Related resources

- [Client Hooks](client_hooks.md) - installation, destinations, privacy, and trace structure
- [Conversation Evaluation](conversation_evaluation.md) - session-scoped metrics
- [Running Metrics on Existing Data](../evaluation/running_feedback_functions/existing_data.md) - offline evaluation patterns
