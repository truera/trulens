# Cross-Model Alignment

When you switch the model behind a judge (`gpt-4o-mini` to `gpt-4.1-nano`, or one
provider to another) the scores can shift in ways a single accuracy number hides.

`CrossModelAlignment` runs the same feedback method with several judges over one
golden set and reports how much they agree: pairwise Spearman correlation, mean
absolute difference and score-shift bias between every pair, plus each judge's
agreement with the ground truth, and a plain recommendation of which judges are
interchangeable and which are outliers.

!!! Initial Setup

    ```bash
    pip install trulens-benchmark
    pip install trulens-providers-openai
    pip install matplotlib  # optional, for report.plot_heatmap()
    ```

!!! example

    ```python
    import os
    from trulens.benchmark.cross_model_alignment import CrossModelAlignment
    from trulens.providers.openai import OpenAI

    os.environ["OPENAI_API_KEY"] = "<replace-with-your-api-key>"

    alignment = CrossModelAlignment(
        judges=[
            {"provider": OpenAI(model_engine="gpt-4o-mini"), "name": "4o-mini"},
            {"provider": OpenAI(model_engine="gpt-4.1-mini"), "name": "4.1-mini"},
        ],
        feedback_method="relevance",
        golden_set=my_golden_set,
    )
    report = alignment.run()
    report.print_matrix()
    report.plot_heatmap()  # matplotlib heatmap of pairwise agreement
    ```

`print_matrix()` produces:

```
Cross-Model Alignment
========================================

Spearman matrix:
                 4o-mini 4.1-mini
   4o-mini          1.00     0.92
  4.1-mini          0.92     1.00

Score-shift bias:
  4.1-mini scores 0.11 higher than 4o-mini on average

Agreement with ground truth:
      judge   mae  spearman  kendall
    4o-mini  0.256     0.87     1.00
   4.1-mini  0.367     0.87     1.00

Recommendations:
  - 4o-mini and 4.1-mini are interchangeable (Spearman 0.92).
```

Use this before swapping the model behind a production judge, to confirm the new
model ranks examples the same way and to quantify any systematic score shift.
