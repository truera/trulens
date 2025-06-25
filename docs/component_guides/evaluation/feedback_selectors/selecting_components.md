# Selecting Spans for Evaluation

LLM applications come in all shapes and sizes and with a variety of different
control flows. As a result, it’s a challenge to consistently evaluate parts of an
LLM application trace.

Therefore, we’ve adapted the use of [OpenTelemetry spans](https://opentelemetry.io/docs/specs/otel/overview/#spans)
to refer to parts of an execution flow when defining evaluations.

## Selecting Span Attributes for Evaluation

When defining evaluations, we want to evaluate particular span attributes, such as retrieved context, or an agent's plan.

This happens in two phases:

1. Instrumentation is used to annotate span attributes. This is covered in detail in the [instrumentation guide](../../instrumentation/index.md).
2. Then when defining the evaluation, you can refer to those span attributes using the `Selector`.

Let's walk through an example. Take this example where a method named `query` is instrumented. In this example, we annotate both the span type, and set span attributes to refer to the `query` argument to the function and the `return` argument of the function.

```python
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes

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

Once we've done this, now we can map the inputs to a feedback function to these span attributes:

```python
from trulens.core import Feedback
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

In the example above, you can see how a dictionary is passed to `on()` that maps the feedback function argument to a span attribute, accessed via a `Selector`.

### Using `collect_list`

In the above examples you see we set the `collect_list` argument in the `Selector` and in `on_context`. Setting `collect_list` to `True` concatenates the selected span attributes into a single blob for evaluation. Alternatively, when set to `False` each span attribute selected will be evaluated individually.

Using `collect_list` is particularly advantageous when working with retrieved context. When evaluating context relevance, we evaluate each context individually (by setting `collect_list=False`).

```python
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector

f_context_relevance = (
    Feedback(
        provider.context_relevance_with_cot_reasons, name="Context Relevance"
    )
    .on_input()
    .on({
        "context": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=False
        ),
    })
)
```

Alternatively, when evaluating groundedness we assess if each LLM claim can be attributed to any evidence from the entire set of retrieved contexts (by setting `collect_list=True`).

```python
from trulens.core import Feedback
from trulens.core.feedback.selector import Selector

f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    )
    .on({
        "context": Selector(
            span_type=SpanAttributes.SpanType.RETRIEVAL,
            span_attribute=SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS,
            collect_list=True
        ),
    })
    .on_output()
)
```

### Evaluating retrieved context from other frameworks

The `on_context()` shortcut can also be used for `LangChain` and `LlamaIndex` apps to refer to the retrieved contexts. Doing so does not require annotating your app with the `RETRIEVAL.RETRIEVED_CONTEXTS` span attribute, as that is done for you.
