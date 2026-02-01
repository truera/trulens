# MLflow Integration

TruLens feedback functions are available as first-class scorers in MLflow's GenAI evaluation framework starting with MLflow 3.10.0. This integration was contributed by [Debu Sinha](https://github.com/debu-sinha) in [MLflow PR #19492](https://github.com/mlflow/mlflow/pull/19492).

## Installation

Install MLflow with TruLens support:

```bash
pip install 'mlflow>=3.10.0' trulens
```

## Available Scorers

TruLens provides two categories of scorers in MLflow:

### RAG Evaluation Scorers

| Scorer | Description |
|--------|-------------|
| `Groundedness` | Evaluates whether the response is grounded in the provided context |
| `ContextRelevance` | Evaluates whether the retrieved context is relevant to the query |
| `AnswerRelevance` | Evaluates whether the response is relevant to the input query |
| `Coherence` | Evaluates the coherence and logical flow of the response |

### Agent Trace Scorers

For evaluating agentic workflows and tool usage:

| Scorer | Description |
|--------|-------------|
| `LogicalConsistency` | Evaluates logical consistency of agent decisions |
| `ExecutionEfficiency` | Evaluates efficiency of agent execution |
| `PlanAdherence` | Evaluates whether the agent followed its plan |
| `PlanQuality` | Evaluates the quality of agent planning |
| `ToolSelection` | Evaluates appropriateness of tool selection |
| `ToolCalling` | Evaluates correctness of tool calls |

## Basic Usage

### Direct Scorer Calls

```python
from mlflow.genai.scorers.trulens import Groundedness

scorer = Groundedness(model="openai:/gpt-4o")

feedback = scorer(
    outputs="Paris is the capital of France.",
    expectations={"context": "France is a country in Europe. Its capital is Paris."},
)

print(feedback.value)  # "yes" or "no"
print(feedback.metadata["score"])  # 0.0 to 1.0
```

### Batch Evaluation with mlflow.genai.evaluate

```python
import mlflow
from mlflow.genai.scorers.trulens import Groundedness, ContextRelevance, AnswerRelevance

eval_dataset = [
    {
        "inputs": {"question": "What is MLflow?"},
        "outputs": "MLflow is an open-source platform for ML lifecycle management.",
        "expectations": {
            "context": "MLflow is an open-source platform for managing the end-to-end machine learning lifecycle."
        },
    },
]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[
        Groundedness(model="openai:/gpt-4o"),
        ContextRelevance(model="openai:/gpt-4o"),
        AnswerRelevance(model="openai:/gpt-4o"),
    ],
)

print(results.tables["eval_results"])
```

## Model Configuration

TruLens scorers in MLflow support multiple LLM providers:

### OpenAI

```python
from mlflow.genai.scorers.trulens import Groundedness

scorer = Groundedness(model="openai:/gpt-4o")
```

### Anthropic

```python
scorer = Groundedness(model="anthropic:/claude-3-5-sonnet")
```

### Azure OpenAI

```python
scorer = Groundedness(model="azure:/my-deployment-name")
```

### Other LiteLLM Providers

```python
# AWS Bedrock
scorer = Groundedness(model="bedrock:/anthropic.claude-3-sonnet")

# Google Vertex AI
scorer = Groundedness(model="vertex_ai:/gemini-pro")
```

## Threshold Configuration

TruLens scorers return a score between 0 and 1. You can configure the threshold for pass/fail:

```python
from mlflow.genai.scorers.trulens import Groundedness

# Default threshold is 0.5
scorer = Groundedness(model="openai:/gpt-4o", threshold=0.7)

feedback = scorer(outputs="...", expectations={"context": "..."})
print(feedback.value)  # "yes" if score >= 0.7, else "no"
print(feedback.metadata["score"])  # Actual score (0.0 to 1.0)
print(feedback.metadata["threshold"])  # 0.7
```

## Dynamic Scorer Creation

Use `get_scorer` to create scorers dynamically:

```python
from mlflow.genai.scorers.trulens import get_scorer

groundedness = get_scorer("Groundedness", model="openai:/gpt-4o")
context_relevance = get_scorer("ContextRelevance", model="openai:/gpt-4o")

results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=[groundedness, context_relevance],
)
```

## Using with MLflow Tracing

TruLens scorers integrate with MLflow's tracing infrastructure:

```python
import mlflow
from mlflow.genai.scorers.trulens import Groundedness

# Enable tracing
mlflow.openai.autolog()

@mlflow.trace
def my_rag_app(question: str) -> str:
    # Your RAG logic here
    return response

# Evaluate using trace
scorer = Groundedness(model="openai:/gpt-4o")
feedback = scorer(trace=trace)
```

## Viewing Results

Results are automatically logged to MLflow:

```python
# Access detailed results
df = results.tables["eval_results"]
print(df[["inputs", "outputs", "Groundedness", "ContextRelevance"]])

# Access aggregate metrics
print(results.metrics)
# Example: {'Groundedness/mean': 0.85, 'ContextRelevance/mean': 0.92}
```

## Best Practices

### Choose the Right Scorer

| Goal | Recommended Scorer |
|------|-------------------|
| Detect hallucinations | `Groundedness` |
| Evaluate retrieval quality | `ContextRelevance` |
| Check answer relevance | `AnswerRelevance` |
| Assess response quality | `Coherence` |
| Evaluate agent behavior | Agent trace scorers |

### Provide Context

For RAG evaluation scorers, always provide context:

```python
{
    "expectations": {
        "context": "The retrieved documents or ground truth...",
    }
}
```

## Troubleshooting

### Missing Dependencies

```
ModuleNotFoundError: No module named 'trulens'
```

Install the TruLens package:

```bash
pip install trulens
```

### API Key Issues

Ensure your API key is set:

```bash
export OPENAI_API_KEY="your-key"
# or
export ANTHROPIC_API_KEY="your-key"
```

## Related Resources

- [MLflow GenAI Evaluation Docs](https://mlflow.org/docs/latest/llms/llm-evaluate/index.html)
- [TruLens Feedback Functions](../evaluation/index.md)
- [MLflow PR #19492](https://github.com/mlflow/mlflow/pull/19492) - Original integration PR
