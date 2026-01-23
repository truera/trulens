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

!!! example "Setting Span Attributes in Instrumentation"

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

!!! info "Connection to Instrumentation"

    The span attributes used in evaluation (`RECORD_ROOT.INPUT`, `RETRIEVAL.RETRIEVED_CONTEXTS`, etc.) must first be set during instrumentation. If you're using custom attributes, make sure they are properly instrumented using the techniques described in [Instrumenting Custom Attributes](../../instrumentation/index.md#instrumenting-custom-attributes) and [Manipulating Custom Attributes](../../instrumentation/index.md#manipulating-custom-attributes).

!!! example "Selecting Instrumented Span Attributes for Evaluation"

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

!!! example "Using Collect List to Evaluate Individual Contexts"

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

!!! example "Using Collect List to Evaluate All Contexts At Once"

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

## Selecting at the Trace Level

In addition to selecting individual spans or span attributes, you can also select and evaluate at the trace level. This is useful when you want to apply feedback functions to an entire trace or to all spans matching certain criteria within a trace.

### Trace-Level Selection with Selector

The Selector class now supports a trace_level argument. When `trace_level=True`, the selector will match all spans in a trace, optionally filtered by `function_name`, `span_name`, or `span_type`. This allows you to evaluate feedback across multiple spans in a single trace.

Each filter field (e.g., function_name) accepts a single value (not a list). Filters across fields are combined with AND logic (i.e., a span must match all specified criteria).

!!! example "Evaluating All Spans in a Trace"

    ```python

    from trulens.core import Feedback
    from trulens.core.feedback.selector import Selector

    f_trace_level = (
        Feedback(provider.some_trace_level_metric, name="Trace Level Metric")
        .on({
            "trace": Selector(
                trace_level=True
            ),
        })
    )
    ```

### Example: Filtering Spans by Function Name

You can filter spans at the trace level by specifying a function name. This is useful if you want to evaluate only those spans in a trace that correspond to a particular function.

!!! example "Filtering Spans by Function Name at the Trace Level"

    ```python
    from trulens.core import Feedback
    from trulens.core.feedback.selector import Selector

    # Example feedback function that counts the number of selected spans
    def count_spans(trace):
        # trace is a ProcessedContentNode representing the filtered trace
        def count_nodes(node):
            return 1 + sum(count_nodes(child) for child in getattr(node, 'children', []))
        return count_nodes(trace)

    f_filtered_trace = (
        Feedback(count_spans, name="Count Query Spans")
        .on({
            "trace": Selector(
                trace_level=True,
                function_name="query"
            ),
        })
    )
    ```

In this example, the feedback function `count_spans` will receive a tree of spans (as a `ProcessedContentNode`) filtered to only those with `function_name="query"`, and will return the total count of such spans in the trace.

### When to Use Trace-Level Selection

Use trace-level selection when your feedback metric needs to consider the relationships between multiple spans, or when you want to aggregate information across an entire trace, such as holistic trace quality.
