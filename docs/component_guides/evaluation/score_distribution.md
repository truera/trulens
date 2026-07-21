# Score Distribution Analysis

A feedback function can post a low mean error yet still be useless if it returns
nearly the same score for every input. A judge that scores everything 0.4-0.6
cannot rank examples no matter how good its average looks.

`ScoreDistributionAnalyzer` runs a feedback function over a golden set and reports
how well the resulting scores *discriminate*: a score histogram, a calibration
curve against the expected scores, discrimination metrics (standard deviation,
unique score count, normalized entropy) and the predicted spread per
expected-score bucket. It also flags common judge pathologies such as poor
discrimination, leniency bias and binary (bimodal) scoring. It complements the
scalar metrics in `GroundTruthAggregator` (mae, brier, ece) by diagnosing the
*shape* of a judge's output rather than a single error number.

!!! Initial Setup

    ```bash
    pip install trulens-benchmark
    pip install trulens-providers-openai
    pip install matplotlib  # optional, for report.plot()
    ```

!!! example

    ```python
    import os
    from trulens.benchmark.score_distribution import ScoreDistributionAnalyzer
    from trulens.providers.openai import OpenAI

    os.environ["OPENAI_API_KEY"] = "<replace-with-your-api-key>"

    golden_set = [
        {"query": "...", "expected_response": "...", "expected_score": 0.2},
        # ... more labelled examples ...
    ]

    analyzer = ScoreDistributionAnalyzer(
        feedback_fn=OpenAI().relevance,
        golden_set=golden_set,
    )
    report = analyzer.run()
    report.print_summary()
    report.plot()  # matplotlib histogram + calibration curve
    ```

`print_summary()` produces:

```
Score Distribution Report
========================================
n=8  mean=0.250  std=0.363  min=0.000  max=1.000
unique(2dp)=4  entropy=0.466 (0=constant, 1=uniform)

Histogram:
  [0.0-0.1)    5  #####
  [0.3-0.4)    1  #
  [0.6-0.7)    1  #
  [0.9-1.0)    1  #

Predicted score by expected bucket:
     low: mean=0.000 std=0.000 (n=1)
  medium: mean=0.250 std=0.433 (n=4)
    high: mean=0.333 std=0.272 (n=3)

No distribution pathologies detected.
```

Use this when tuning a feedback function's `criteria` or `additional_instructions`
to confirm the judge actually separates good and bad examples instead of
clustering every score into a narrow band.
