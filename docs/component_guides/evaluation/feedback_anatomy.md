# 🦴 Anatomy of Metrics

The [Metric][trulens.core.Metric] class is the
starting point for metric specification and evaluation.

!!! example

    ```python
    from trulens.core import Metric, Selector
    import numpy as np

    # Context relevance between question and each context chunk.
    f_context_relevance = Metric(
        implementation=provider.context_relevance_with_cot_reasons,
        name="Context Relevance",
        selectors={
            "question": Selector.select_record_input(),
            "context": Selector.select_context(collect_list=False),
        },
        agg=np.mean,
    )
    ```

The components of this specification are:

## Providers

The provider is the back-end on which a given metric is run.
Multiple underlying models are available through each provider, such as GPT-4 or
Llama-2. In many, but not all cases, the metric implementation is shared
across providers (such as with LLM-based evaluations).

Read more about [providers][trulens.core.feedback.provider.Provider].

## Metric Implementations

[OpenAI.context_relevance][trulens.providers.openai.provider.OpenAI.context_relevance]
is an example of a metric implementation.

Metric implementations are simple callables that can be run
on any arguments matching their signatures. In the example, the implementation
has the following signature:

!!! example

    ```python
    def context_relevance(self, prompt: str, context: str) -> float:
    ```

That is,
[context_relevance][trulens.providers.openai.provider.OpenAI.context_relevance]
is a plain Python method that accepts the prompt and context, both strings, and
produces a float (assumed to be between 0.0 and 1.0).

Read more about [metric implementations](./feedback_implementations/index.md)

## Metric Constructor

The `Metric(implementation=provider.relevance)` constructs a
Metric object with a metric implementation.

## Selectors

The `selectors` parameter specifies how the metric implementation's
arguments are determined from an app record or app definition. Selectors
map parameter names to span data using the `Selector` class.

Common selector methods:

- `Selector.select_record_input()` - The main app input
- `Selector.select_record_output()` - The main app output
- `Selector.select_context(collect_list=True/False)` - Retrieved contexts

Read more about [selectors](./feedback_selectors/selecting_components.md).

## Aggregation Specification

The `agg=np.mean` parameter specifies how metric outputs are to be
aggregated. This only applies to cases where the selector names
more than one value for an input (e.g., when `collect_list=False` returns
multiple context chunks). The function is called on the `float` results of
metric evaluations to produce a single float. The default is
[numpy.mean][numpy.mean].

Read more about [aggregation](feedback_aggregation.md).
