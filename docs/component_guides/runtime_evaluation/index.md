# Runtime Evaluation

Evaluations play a crucial role in improving LLM app outputs by altering execution flow at runtime.

TruLens supports runtime evaluation via two different mechanisms:

1. [In-line Evaluations](./inline_evals.md) - evaluations that are executed during an agent's execution flow and passed back to agent to assist in orchestration.
2. [Guardrails](./guardrails.md) - evaluations that can be used to block input, output and intermediate results produced by an application such as a RAG or agent.
