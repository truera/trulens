# LLM Jury: Ensemble Evaluation with Multiple Judges

## Why a jury instead of a single judge?

A single LLM judge is both noisy and subject to *intra-model bias* — it
systematically favours its own style, vocabulary, and reasoning patterns.
Research shows that ensembling a **panel of diverse judges** (a "jury")
substantially improves evaluation reliability:

- [Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models](https://arxiv.org/abs/2404.18796)
- [SE-Jury: An LLM-as-Ensemble-Judge Metric](https://arxiv.org/abs/2512.01786)
- [From Many Voices to One: A Statistically Principled Aggregation of LLM Judges](https://openreview.net/forum?id=Ou53DNvjx7)
- [Auto-Prompt Ensemble for LLM Judge](https://arxiv.org/abs/2510.06538)
- [Debate, Deliberate, Decide (D3)](https://aclanthology.org/2026.eacl-long.392/)

`Jury` is not tied to one paper's algorithm. It provides the common
infrastructure for running several LLM judges through the existing Metric API
and combining their scores.

## The `Jury` class

`Jury` wraps N `LLMProvider` instances, calls the same named feedback method on
each in parallel, and aggregates their scores using a configurable strategy.
Because `Jury.__call__` has the same parameter names as the underlying provider
method, it is a drop-in `implementation` for `Metric` — **no changes to
`Metric`, `Selector`, or the evaluation pipeline are needed**.

`Jury` always returns `(score, {"reason": ...})`, matching the
`_with_cot_reasons` convention. Per-juror scores and any chain-of-thought
explanations are embedded in the reason string, so they flow into
`FeedbackCall.meta["reason"]` → `ai.observability.eval.metadata.reason` in
OTEL spans and appear in the dashboard automatically — no UI changes needed.

```python
from trulens.feedback.jury import Jury
```

### API reference

```python
Jury(
    jurors,       # list[LLMProvider] — the panel of judges
    method,       # str — method name, e.g. "relevance"
    aggregation,  # str | Callable — how to combine scores (default "mean")
    *,
    weights,      # list[float] — per-juror weights (weighted_mean only)
    threshold,    # float — binarisation threshold (majority_vote, default 0.5)
    max_workers,  # int — parallel threads (default len(jurors))
)
```

## Aggregation strategies

| Strategy | Description | When to use |
|---|---|---|
| `"mean"` *(default)* | Simple average | General purpose |
| `"median"` | Middle value — robust to outlier judges | When one judge may be erratic |
| `"trimmed_mean"` | Drop highest and lowest, average the rest | 3+ judges; outlier resistance with less bias than median |
| `"majority_vote"` | Binarise at `threshold`, majority wins; ties fall back to median | Pass/fail guardrails |
| `"weighted_mean"` | Weight each judge by alignment quality | When you have benchmark-derived judge quality scores |
| `Callable[[list[float]], float]` | Any custom function | Specialist use cases |

## Usage examples

### Basic jury with median aggregation

```python
from trulens.core import Metric
from trulens.feedback.jury import Jury
from trulens.providers.openai import OpenAI
from trulens.providers.litellm import LiteLLM

jury = Jury(
    jurors=[
        OpenAI(model_engine="gpt-4o-mini"),
        OpenAI(model_engine="gpt-4.1-mini"),
        LiteLLM(model_engine="anthropic/claude-3-haiku-20240307"),
    ],
    method="relevance",
    aggregation="median",
)

m_relevance = (
    Metric(implementation=jury, name="Jury Relevance")
    .on_input()
    .on_output()
)
```

### Per-juror score breakdown

`Jury` always returns `(score, meta)`. The `meta["reason"]` string contains
the aggregated score, each juror's individual score, and any chain-of-thought
explanations from `_with_cot_reasons` methods — all visible in the dashboard.

```python
jury = Jury(
    jurors=[openai_provider, litellm_provider],
    method="relevance_with_cot_reasons",
    aggregation="median",
)

score, meta = jury(prompt="What is TruLens?", response="An eval library.")
print(meta["reason"])
# Aggregation: median → 0.825
#   gpt-4o-mini: 0.750
#     Criteria: relevance
#     Supporting Evidence: the response directly addresses...
#   claude-3-haiku: 0.900
#     Criteria: relevance
#     Supporting Evidence: mostly relevant but...
```

### Weighted jury from benchmark results

```python
jury = Jury(
    jurors=[openai_provider, litellm_provider, cortex_provider],
    method="groundedness_measure_with_cot_reasons",
    aggregation="weighted_mean",
    weights=[0.5, 0.3, 0.2],  # from prior benchmark alignment scores
)
```

### Majority-vote guardrail

```python
jury = Jury(
    jurors=[provider_a, provider_b, provider_c],
    method="moderation_hate",
    aggregation="majority_vote",
    threshold=0.5,  # any score ≥ 0.5 counts as a positive (flagged) vote
)
```

On an exact tie, `majority_vote` falls back to the median of scores rather
than returning 0.0.

### Error handling

If one juror fails, `Jury` logs a warning and aggregates the rest. For
`weighted_mean`, the failed juror's weight is redistributed proportionally
among the survivors. Only when *all* jurors fail does `Jury` raise a
`RuntimeError`.

## Choosing juror models

**Diversity beats size.** The key driver of jury quality is *disagreement
diversity* among models — a mix of model families (OpenAI, Anthropic, Google,
Snowflake Cortex) provides more independent signal than three instances of the
same model. Smaller, cheaper models are often sufficient.

Practical starting point:

- 3 judges from different families
- `"median"` or `"trimmed_mean"` aggregation
- Use `_with_cot_reasons` methods to capture per-juror explanations in the dashboard

## Cost and latency

`Jury` runs all jurors in parallel (via `ThreadPoolExecutor`), so wall-clock
latency is approximately that of the *slowest* juror. Cost scales linearly with
the number of jurors. Using 3 small models (e.g. GPT-4o-mini, Claude Haiku,
Gemini Flash) is typically cheaper than a single GPT-4o call.

## Configuration note

`Jury` stores provider instances, including their model configuration. Build the
jury in the same environment where those providers and credentials are
available, especially when using deferred or remote evaluation workflows.
