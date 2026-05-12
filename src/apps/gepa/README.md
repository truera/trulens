# trulens-apps-gepa

TruLens adapter for GEPA (Genetic/Evolutionary Prompt Adaptation).

GEPA optimizes prompts using evolutionary algorithms. This package provides
`TruGEPA`, a thin adapter that wraps any TruLens feedback callable as a
GEPA-compatible fitness function and automatically logs every evaluation as a
`TruVirtual` record for dashboard visibility, plus a simple `run_evolution`
helper that implements the evolutionary loop.

## Installation

```bash
pip install trulens-apps-gepa
```

## Quick start

```python
from trulens.apps.gepa import TruGEPA, run_evolution

def my_relevance(prompt: str) -> float:
    return len(prompt) / 200  # replace with a real TruLens provider method

# Without logging:
fitness = TruGEPA(my_relevance, input_key="prompt")

# With logging — supply both app_name and app_version (omit both to disable;
# supplying only one raises a ValueError immediately):
from trulens.core import TruSession
session = TruSession()

fitness = TruGEPA(
    my_relevance,
    input_key="prompt",
    app_name="my_optimizer",
    app_version="v1",
)

best_prompt, best_score, history = run_evolution(
    base_prompt="Summarize the document.",
    fitness_fn=fitness,
    mutate_fn=lambda p: p + " Be concise.",
    n_generations=5,
    population_size=4,
)
print(f"Best prompt ({best_score:.3f}): {best_prompt}")
```

When both `app_name` and `app_version` are provided, a `TruVirtual` recorder
is created automatically and every evaluation is logged. Omit both to run
without any logging.
