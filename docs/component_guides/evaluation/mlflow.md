# MLflow Integration

TruLens feedback functions are available as first-class scorers in MLflow's GenAI evaluation framework starting with MLflow 3.10.0.

## Installation

Install MLflow with TruLens support:

```bash
pip install 'mlflow>=3.10.0' trulens trulens-providers-litellm
```

## Available Scorers

TruLens provides three categories of scorers in MLflow:

### RAG Evaluation Scorers

| Scorer | Description |
|--------|-------------|
| `Groundedness` | Evaluates whether the response is grounded in the provided context |
| `ContextRelevance` | Evaluates whether the retrieved context is relevant to the query |
| `AnswerRelevance` | Evaluates whether the response is relevant to the input query |

### Output Scorers

| Scorer | Description |
|--------|-------------|
| `Coherence` | Evaluates the coherence and logical flow of any LLM output |

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

TruLens scorers in MLflow support multiple LLM providers through LiteLLM:

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

Use `get_scorer` to create scorers dynamically by name. This is useful when you need to configure scorers from external configuration files, environment variables, or user input rather than hardcoding scorer classes in your code:

```python
from mlflow.genai.scorers.trulens import get_scorer

# Load scorer names from config or user input
scorer_names = ["Groundedness", "ContextRelevance"]

scorers = [get_scorer(name, model="openai:/gpt-4o") for name in scorer_names]

results = mlflow.genai.evaluate(
    data=eval_dataset,
    scorers=scorers,
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

## Agent Evaluation

Agent GPA scorers evaluate tool selection and execution in agentic workflows. These scorers require traces since they inspect tool call spans.

### Batch Agent Evaluation

Use `predict_fn` with `mlflow.genai.evaluate` to trace and evaluate agent runs:

```python
import mlflow
from mlflow.genai.scorers.trulens import (
    Groundedness,
    ToolSelection,
    ToolCalling,
    Coherence,
)

mlflow.openai.autolog()


def run_agent(inputs: dict) -> str:
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": inputs["user_query"]}],
        tools=[...],  # your tool definitions
    )
    # ... handle tool calls and return result


agent_queries = [
    "What's the weather in Paris?",
    "Book a flight to Tokyo for next Monday",
    "Send an email to my team about the meeting",
]

agent_eval_results = mlflow.genai.evaluate(
    data=[{"inputs": {"user_query": q}} for q in agent_queries],
    predict_fn=run_agent,
    scorers=[
        Groundedness(model="openai:/gpt-4o-mini"),
        ToolSelection(model="openai:/gpt-4o-mini"),
        ToolCalling(model="openai:/gpt-4o-mini"),
        Coherence(model="openai:/gpt-4o-mini"),
    ],
)

print(agent_eval_results.tables["eval_results"])
```

### Evaluating Individual Agent Traces

You can also evaluate agent traces individually:

```python
import mlflow
from mlflow.genai.scorers.trulens import ToolSelection, ToolCalling

mlflow.openai.autolog()

# Run your agent
result = run_agent({"user_query": "What's the weather in Paris?"})

# Get the trace
trace = mlflow.get_last_active_trace()

# Evaluate tool usage
tool_selection = ToolSelection(model="openai:/gpt-4o-mini")
tool_calling = ToolCalling(model="openai:/gpt-4o-mini")

selection_feedback = tool_selection(trace=trace)
calling_feedback = tool_calling(trace=trace)

print(f"Tool Selection: {selection_feedback.value}")
print(f"Tool Calling: {calling_feedback.value}")
print(f"Rationale: {selection_feedback.rationale}")
```

!!! note "Agent vs RAG Scorers"
    RAG and output scorers (`Groundedness`, `Coherence`, etc.) can be called directly with data or on traces. Agent GPA scorers (`ToolSelection`, `ToolCalling`, etc.) require a `trace` parameter since they evaluate tool usage patterns within trace spans.

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

Install the TruLens packages:

```bash
pip install trulens trulens-providers-litellm
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
- [TruLens Feedback Functions](./index.md)
