# trulens-apps-gepa

TruLens adapter for GEPA (Genetic/Evolutionary Prompt Adaptation).

GEPA optimizes prompts using evolutionary algorithms. This package provides
`TruLensFitness`, a thin adapter that wraps any TruLens feedback callable as a
GEPA-compatible fitness function, plus a simple `run_evolution` helper that
implements the evolutionary loop.

## Installation

```bash
pip install trulens-apps-gepa
```

## Quick start

```python
from trulens.apps.gepa import TruLensFitness, run_evolution

def my_relevance(prompt: str) -> float:
    return len(prompt) / 200

fitness = TruLensFitness(my_relevance, input_key="prompt")

best_prompt, best_score, history = run_evolution(
    base_prompt="Summarize the document.",
    fitness_fn=fitness,
    mutate_fn=lambda p: p + " Be concise.",
    n_generations=5,
    population_size=4,
)
print(f"Best prompt ({best_score:.3f}): {best_prompt}")
```

## Logging to TruLens

Pass a `TruVirtual` recorder to log every evaluation to the TruLens dashboard:

```python
from trulens.apps.virtual import TruVirtual
from trulens.core import TruSession

session = TruSession()

recorder = TruVirtual(
    app_name="gepa_optimizer",
    app_version="v1",
)

fitness = TruLensFitness(
    my_relevance,
    input_key="prompt",
    recorder=recorder,
)
```

Each evaluation call is recorded as a virtual TruLens record, giving you a
full audit trail and generation-over-generation trend in the dashboard.
