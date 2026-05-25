# Conversation ID — Grouping Multi-Turn Traces

The `conversation_id` attribute (`ai.observability.conversation_id`) lets you group
multiple app invocations that belong to the same logical conversation or thread.
Every span recorded inside a context manager that carries a `conversation_id` will
have the attribute set, making it easy to filter, visualise, or evaluate all turns
of a multi-turn interaction together.

The value is propagated through
[OpenTelemetry Baggage](https://opentelemetry.io/docs/concepts/signals/baggage/)
so it is automatically inherited by child spans, including those created by
instrumented third-party libraries.

## Basic usage

Pass `conversation_id` when entering the recording context manager:

```python
from trulens.core import TruSession
from trulens.apps.app import TruApp

session = TruSession()

tru_app = TruApp(my_app, app_name="MyApp", app_version="v1",
                 main_method=my_app.query)

with tru_app(conversation_id="conv-abc-123") as recording:
    response = my_app.query("What is TruLens?")
```

All spans produced during `my_app.query(...)` will carry:

```
ai.observability.conversation_id = "conv-abc-123"
```

## Multi-turn conversations

Use the same `conversation_id` across multiple context managers to stitch several
turns into one conversation:

```python
CONV_ID = "session-42"

with tru_app(conversation_id=CONV_ID) as recording:
    my_app.query("Hello, who are you?")

with tru_app(conversation_id=CONV_ID) as recording:
    my_app.query("What can you help me with?")

with tru_app(conversation_id=CONV_ID) as recording:
    my_app.query("Thanks, goodbye!")
```

All three invocations will share `ai.observability.conversation_id = "session-42"`,
so you can retrieve all their records together:

```python
import pandas as pd
from trulens.core.session import TruSession

session = TruSession()
records, _ = session.get_records_and_feedback(app_ids=["MyApp"])
conv_records = records[records["conversation_id"] == "session-42"]
```

## Single-turn (default behaviour)

Omitting `conversation_id` (or using `with tru_app as recording:`) leaves the
attribute unset.  This is the default and is fully backwards-compatible:

```python
# No conversation_id — identical to pre-existing behaviour
with tru_app as recording:
    my_app.query("Standalone question")
```

## Notes

* `conversation_id` must be a string.  Use any format you like — UUIDs, slugs,
  integer strings, etc.
* The attribute is **not** automatically generated; you are responsible for
  creating and managing conversation identifiers in your application.
* The value is stored in OTEL Baggage during the context-manager lifetime and
  therefore does not leak across concurrent invocations that use different IDs.
