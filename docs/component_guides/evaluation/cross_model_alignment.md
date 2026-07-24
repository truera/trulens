# Cross-Model Alignment

When you switch the model behind a judge (`llama-3.3-70b` to `llama-3.1-8b`, or
one provider to another) the scores can shift in ways a single accuracy number
hides.

`CrossModelAlignment` runs the same feedback method with several judges over one
golden set and reports how much they agree: pairwise Spearman correlation, mean
absolute difference and score-shift bias between every pair, plus each judge's
agreement with the ground truth, and a plain recommendation of which judges are
interchangeable and which are outliers.

!!! Initial Setup

    ```bash
    pip install trulens-benchmark
    pip install trulens-providers-litellm
    pip install matplotlib  # optional, for report.plot_heatmap()
    ```

!!! example

    ```python
    import os
    from trulens.benchmark.cross_model_alignment import CrossModelAlignment
    from trulens.providers.litellm import LiteLLM

    os.environ["GROQ_API_KEY"] = "<replace-with-your-api-key>"

    judge_a = LiteLLM(model_engine="groq/llama-3.3-70b-versatile")
    judge_b = LiteLLM(model_engine="groq/llama-3.1-8b-instant")
    alignment = CrossModelAlignment(
        judges=[
            {"provider": judge_a, "name": "70b"},
            {"provider": judge_b, "name": "8b"},
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
            70b      8b
  70b      1.00    0.92
   8b      0.92    1.00

Score-shift bias:
  70b scores 0.11 higher than 8b on average

Agreement with ground truth:
    judge   mae  spearman  kendall
      70b  0.256     0.87     1.00
       8b  0.367     0.87     1.00

Recommendations:
  - 70b and 8b are interchangeable (Spearman 0.92).
```

Use this before swapping the model behind a production judge, to confirm the new
model ranks examples the same way and to quantify any systematic score shift.
