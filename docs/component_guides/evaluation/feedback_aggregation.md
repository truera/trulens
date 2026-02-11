# Metric Aggregation

For cases where a selector names more than one value as an input,
aggregation can be used.

!!! example

    ```python
    import numpy as np
    from trulens.core import Metric, Selector

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

The `agg` parameter specifies how metric outputs are to be aggregated.
This only applies to cases where the selector names more than one value
for an input. The `context` selector with `collect_list=False` is of this type,
meaning the metric will be evaluated for each context individually.

The input to `agg` must be a method which can be imported globally. This function
is called on the `float` results of metric evaluations to produce a single float.

The default is `numpy.mean`.
