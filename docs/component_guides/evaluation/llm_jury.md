# LLM Jury: Ensemble Evaluation with Multiple Judges

## Why a jury instead of a single judge?

A single LLM judge is both noisy and subject to *intra-model bias* — it
systematically favours its own style, vocabulary, and reasoning patterns.
Research shows that ensembling a **panel of diverse judges** (a "jury")
substantially improves evaluation reliability:

| Paper | Key finding |
|---|---|
| [Verga et al. 2024 — PoLL](https://arxiv.org/abs/2404.18796) | A panel of diverse smaller models outperforms a single large judge and is 7× cheaper. |
| [Zhou et al. 2025 — SE-Jury](https://arxiv.org/abs/2501.16676) | Ensemble judges narrow the gap with human evaluation on software engineering tasks. |
| [Zhao et al. 2025 — Statistically Principled Aggregation](https://arxiv.org/abs/2503.07977) | PoLL-style juries with 3 diverse small models achieve higher human-correlation than a single large judge. |
| [Li et al. 2025 — Auto-Prompt Ensemble](https://arxiv.org/abs/2502.07853) | Treating each evaluation dimension as an independent juror with auto-generated prompts further improves reliability. |

## The `Jury` class

`Jury` wraps N `LLMProvider` instances, calls the same named feedback method on
each in parallel, and aggregates their scores using a configurable strategy.
Because `Jury.__call__` has the same parameter names as the underlying provider
method, it is a drop-in `implementation` for `Metric` — **no changes to
`Metric`, `Selector`, or the evaluation pipeline are needed**.

```python
from trulens.feedback.jury import Jury
```

### API reference

```python
Jury(
    jurors,          # List[LLMProvider] — the panel of judges
    method,          # str — method name, e.g. "relevance"
    aggregation,     # str | Callable — how to combine scores (default "mean")
    *,
    weights,         # List[float] — per-juror weights (weighted_mean only)
    threshold,       # float — binarisation threshold (majority_vote, default 0.5)
    return_details,  # bool — include per-juror scores in metadata (default False)
    max_workers,     # int — parallel threads (default len(jurors))
)
```

## Aggregation strategies

| Strategy | Description | When to use |
|---|---|---|
| `"mean"` *(default)* | Simple average | General purpose |
| `"median"` | Middle value — robust to outlier judges | When one judge may be erratic |
| `"trimmed_mean"` | Drop highest and lowest, average the rest | 3+ judges; outlier resistance with less bias than median |
| `"majority_vote"` | Binarise at `threshold`, majority wins | Pass/fail guardrails |
| `"weighted_mean"` | Weight each judge by alignment quality | When you have benchmark-derived judge quality scores |
| `Callable[[List[float]], float]` | Any custom function | Specialist use cases |

## Usage examples

### Basic jury with mean aggregation

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
    aggregation="mean",
)

m_relevance = (
    Metric(implementation=jury, name="Jury Relevance")
    .on_input()
    .on_output()
)
```

### Median jury (robust to outliers)

```python
jury = Jury(
    jurors=[openai_provider, litellm_provider, cortex_provider],
    method="context_relevance",
    aggregation="median",
)
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

### Per-juror score breakdown

```python
jury = Jury(
    jurors=[openai_provider, litellm_provider],
    method="relevance",
    aggregation="median",
    return_details=True,
)

# Returns: (0.82, {"gpt-4o-mini": 0.75, "claude-3-haiku": 0.90})
score, per_juror = jury(prompt="What is TruLens?", response="An eval library.")
```

### Error handling

If one juror fails, `Jury` logs a warning and aggregates the rest. Only when
*all* jurors fail does `Jury` raise a `RuntimeError`.

## Choosing juror models

**Diversity beats size.** The key driver of jury quality is *disagreement
diversity* among models — a mix of model families (OpenAI, Anthropic, Google,
Snowflake Cortex) provides more independent signal than three instances of the
same model. Smaller, cheaper models are often sufficient.

Practical starting point:

- 3 judges from different families
- `"median"` or `"trimmed_mean"` aggregation
- `return_details=True` during development to inspect per-juror scores

## Cost and latency

`Jury` runs all jurors in parallel (via `ThreadPoolExecutor`), so wall-clock
latency is approximately that of the *slowest* juror. Cost scales linearly with
the number of jurors. Using 3 small models (e.g. GPT-4o-mini, Claude Haiku,
Gemini Flash) is typically cheaper than a single GPT-4o call.
