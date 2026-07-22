# Criteria A/B Testing

TruLens feedback functions accept `criteria` and `additional_instructions` to
tune judge behaviour, but there is no easy way to tell whether a new criteria
string actually agrees with human judgement better than the old one.

`CriteriaABTest` runs two configurations of a feedback function over the same
golden set and reports which one aligns better: side-by-side MAE / Spearman /
Kendall / Brier against the ground truth, the examples where the two disagree
most, a paired significance test (a sign-flip permutation test, no scipy
dependency) on the score differences, and a winner.

!!! Initial Setup

    ```bash
    pip install trulens-benchmark
    pip install trulens-providers-openai
    ```

!!! example

    ```python
    import os
    from trulens.benchmark.criteria_ab_test import CriteriaABTest
    from trulens.providers.openai import OpenAI

    os.environ["OPENAI_API_KEY"] = "<replace-with-your-api-key>"
    provider = OpenAI()

    test = CriteriaABTest(
        golden_set=my_golden_set,
        variant_a={"fn": provider.relevance, "name": "default"},
        variant_b={
            "fn": provider.relevance,
            "kwargs": {"criteria": "Be strict; a high score only if fully relevant."},
            "name": "strict",
        },
    )
    report = test.run()
    report.print_comparison()
    ```

`print_comparison()` produces:

```
Criteria A/B Test
========================================
A = default    B = strict    n = 8

  metric           default        strict
  mae                0.383         0.425
  spearman           0.438         0.319
  kendall            0.500         0.455
  brier              0.200         0.244

Mean difference (A - B): +0.042  (permutation p = 1.000)

Largest disagreements:
  +0.33  A=0.33 B=0.00  How does the social structure of a lion pride im...

Winner: default (lower MAE vs ground truth; difference not significant at alpha=0.05).
```

Use this when iterating on a judge's `criteria` or `additional_instructions`, to
confirm a change actually improves alignment with human labels rather than just
shifting scores around.
