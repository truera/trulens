# Running Metrics on Existing Data

In many cases, developers have already logged runs of an LLM app they wish to evaluate, or have existing datasets of questions, contexts, and answers. Metrics can be run on this existing data using a few different approaches.

## Direct Provider Calls

At the most basic level, metric implementations are simple callables that can be run on any arguments matching their signatures.

!!! example

    ```python
    from trulens.providers.openai import OpenAI

    provider = OpenAI()
    score = provider.relevance("What is the capital of France?", "The capital of France is Paris.")
    print(score)  # Returns a float between 0 and 1
    ```

!!! note
    Running the metric implementation in isolation will not log the evaluation results in TruLens.

## Evaluating DataFrames with Data Replay

To evaluate existing data (e.g., from a CSV or DataFrame) while logging results to TruLens, you can create a simple "replay" app that passes your data through instrumented functions. This creates the proper spans that metrics can evaluate.

### Step 1: Prepare Your Data

```python
import pandas as pd

# Your existing data - could come from CSV, database, etc.
df = pd.DataFrame({
    'question': [
        'Where is Germany?',
        'What is the capital of France?'
    ],
    'context': [
        'Germany is a country located in Europe.',
        'France is a country in Europe and its capital is Paris.'
    ],
    'answer': [
        'Germany is in Europe',
        'The capital of France is Paris'
    ],
})
```

### Step 2: Create a Replay App

Create a simple class with instrumented methods that pass through your existing data:

```python
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes


class DataReplay:
    """A simple app that replays existing data through instrumented functions."""

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str, contexts: list) -> list:
        """Pass through existing contexts."""
        return contexts if isinstance(contexts, list) else [contexts]

    @instrument()
    def query(self, question: str, context: list, answer: str) -> str:
        """Replay a single record through the instrumented pipeline."""
        # Create the retrieval span with the existing context
        self.retrieve(query=question, contexts=context)
        # Return the existing answer
        return answer
```

### Step 3: Define Your Metrics

```python
import numpy as np
from trulens.core import Metric, Selector
from trulens.providers.openai import OpenAI

provider = OpenAI()

f_groundedness = Metric(
    implementation=provider.groundedness_measure_with_cot_reasons,
    name="Groundedness",
    selectors={
        "source": Selector.select_context(collect_list=True),
        "statement": Selector.select_record_output(),
    },
)

f_answer_relevance = Metric(
    implementation=provider.relevance_with_cot_reasons,
    name="Answer Relevance",
    selectors={
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output(),
    },
)

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

### Step 4: Replay Data and Evaluate

```python
from trulens.core import TruSession
from trulens.apps.app import TruApp

session = TruSession()

# Create and wrap the replay app
replay = DataReplay()
tru_replay = TruApp(
    replay,
    app_name="ExistingDataEval",
    app_version="v1",
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance],
)

# Replay each row of existing data
with tru_replay:
    for _, row in df.iterrows():
        replay.query(
            question=row['question'],
            context=[row['context']],  # Wrap in list if single context
            answer=row['answer']
        )
```

### Step 5: View Results

```python
# View results in the leaderboard
session.get_leaderboard()

# Or launch the dashboard
from trulens.dashboard import run_dashboard
run_dashboard(session)
```

## Handling Multiple Contexts

If your data has multiple contexts per question, you can pass them as a list:

```python
df = pd.DataFrame({
    'question': ['What is coffee culture?'],
    'contexts': [['Coffee has three waves...', 'Seattle is the birthplace...']],
    'answer': ['Coffee culture evolved through three waves...'],
})

with tru_replay:
    for _, row in df.iterrows():
        replay.query(
            question=row['question'],
            context=row['contexts'],  # Already a list
            answer=row['answer']
        )
```

## Including Ground Truth

If you have ground truth answers for comparison, you can include them in your replay app:

```python
class DataReplayWithGroundTruth:
    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str, contexts: list) -> list:
        return contexts if isinstance(contexts, list) else [contexts]

    @instrument(
        attributes={
            SpanAttributes.RECORD_ROOT.GROUND_TRUTH_OUTPUT: "expected_answer",
        }
    )
    def query(self, question: str, context: list, answer: str, expected_answer: str = None) -> str:
        self.retrieve(query=question, contexts=context)
        return answer
```

Then use ground truth metrics:

```python
from trulens.feedback import GroundTruthAgreement

ground_truth_metric = Metric(
    implementation=GroundTruthAgreement(...).agreement_measure,
    name="Ground Truth Agreement",
    selectors={
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output(),
    },
)
```
