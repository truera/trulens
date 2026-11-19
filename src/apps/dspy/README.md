# trulens-apps-dspy

TruLens adapter for [DSPy](https://dspy.ai) metrics.

DSPy optimizes a program's instructions and few-shot demos against a metric.
This package provides `TruDSPyMetric`, a thin adapter that wraps a TruLens
`Metric`, such as groundedness or answer relevance, as a DSPy metric for
`dspy.Evaluate`, the bootstrap optimizers, `dspy.MIPROv2` and `dspy.GEPA`.

## Installation

```bash
pip install trulens-apps-dspy trulens-providers-openai
```

The examples below use the OpenAI provider from `trulens-providers-openai`;
any TruLens provider works. DSPy itself is installed separately:

```bash
pip install dspy
```

`dspy.MIPROv2` also needs DSPy's `optuna` extra: `pip install "dspy[optuna]"`.

## Quick start

A TruLens implementation names its own arguments, such as `source` and
`statement`, while DSPy passes a metric an `example` and a `prediction`.
`fields` maps each implementation argument to the DSPy field it reads. Nothing
is inferred from argument names.

```python
from trulens.apps.dspy import TruDSPyMetric
from trulens.core import Metric
from trulens.providers.openai import OpenAI

provider = OpenAI()

groundedness = TruDSPyMetric(
    Metric(
        implementation=provider.groundedness_measure_with_cot_reasons,
        name="Groundedness",
    ),
    fields={"source": "example.context", "statement": "prediction.answer"},
)
answer_relevance = TruDSPyMetric(
    Metric(
        implementation=provider.relevance_with_cot_reasons,
        name="Answer Relevance",
    ),
    fields={"prompt": "example.question", "response": "prediction.answer"},
)

score = groundedness(example, prediction)  # float
```

A string value is a dotted path that starts with `example.` or `prediction.`
and may reach nested attributes. A callable value receives
`(example, prediction)` and returns the argument, for example to join retrieved
passages:

```python
fields = {
    "source": lambda example, prediction: "\n\n".join(prediction.context),
    "statement": "prediction.answer",
}
```

The mapping is checked when the metric is built. An unknown argument name, an
argument already bound with `Metric.with_arguments` or set as a metric setting
such as `criteria`, a malformed path, or a required argument left unmapped
raises a `ValueError`. A field missing from the example or prediction at call
time raises an error that names the argument and its path.

`metric` may also be a plain callable, which is wrapped as
`Metric(implementation=callable)`.

## Call shapes

One `TruDSPyMetric` serves each way DSPy calls a metric:

```python
score = groundedness(example, prediction)
ok = groundedness(example, prediction, trace)
fb = groundedness(example, prediction, trace, pred_name, pred_trace)
```

### Evaluation and MIPROv2

`dspy.Evaluate` and MIPROv2 trials call `metric(example, prediction)` and get
the TruLens score as a float.

```python
import dspy

evaluate = dspy.Evaluate(devset=devset, metric=groundedness)
result = evaluate(program)

optimizer = dspy.MIPROv2(metric=groundedness, auto="light")
optimized = optimizer.compile(program, trainset=trainset)
```

To optimize for several TruLens metrics at once, combine them in a plain
function:

```python
def rag_quality(example, prediction, trace=None):
    return (
        groundedness(example, prediction)
        + answer_relevance(example, prediction)
    ) / 2
```

### Bootstrapping

Bootstrap optimizers such as `dspy.BootstrapFewShot`, and the bootstrapping
step of MIPROv2, call `metric(example, prediction, trace)` and keep a demo when
the result is truthy. With `threshold` set, those calls return
`score >= threshold` as a bool. Calls without a trace still return the float.

```python
strict_groundedness = TruDSPyMetric(
    Metric(
        implementation=provider.groundedness_measure_with_cot_reasons,
        name="Groundedness",
    ),
    fields={"source": "example.context", "statement": "prediction.answer"},
    threshold=0.7,
)

optimizer = dspy.BootstrapFewShot(metric=strict_groundedness)
compiled = optimizer.compile(program, trainset=trainset)
```

### GEPA

`dspy.GEPA` scores candidates with `metric(example, prediction)` or
`metric(example, prediction, None)` and asks for feedback with
`metric(example, prediction, trace, pred_name, pred_trace)`. On that call the
metric returns `dspy.Prediction(score=..., feedback=...)` when the TruLens
result carries reasons, as the `*_with_cot_reasons` implementations do, so
GEPA's reflection sees why a candidate scored as it did. Otherwise it returns
the float. If the judge returns no usable score on that call, the metric
returns feedback saying so instead of raising, so the reflection step still
runs.

```python
optimizer = dspy.GEPA(
    metric=groundedness,
    auto="light",
    reflection_lm=dspy.LM("openai/gpt-5", temperature=1.0, max_tokens=32000),
)
optimized = optimizer.compile(program, trainset=trainset)
```

`threshold` has no effect on GEPA. GEPA scores candidates with `trace=None` and
asks for feedback with a predictor name, so neither call returns a bool.

## Tracing optimizer runs

`TruDSPyMetric` does not record optimizer candidates in TruLens. The metric
callback cannot tell which candidate program produced a prediction, and DSPy
may score examples concurrently. Instead, record the baseline and the optimized
program as two versions of the same TruLens app, for example with `TruApp`, and
compare them in the leaderboard. The
[cookbook](https://github.com/truera/trulens/blob/main/examples/expositional/frameworks/dspy/dspy_trulens_metrics.ipynb)
walks through this end to end.
