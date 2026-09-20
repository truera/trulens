# Evaluation Dataset & TruLens Results

The evaluation workflow uses **TruLens** to evaluate RAG execution traces, feedback scores, and reasons.

## Golden Dataset (`seed.jsonl`)

The ground truth dataset lives in `evals/seed.jsonl`. Each line is a validated JSON object representing a question grounded in `data/aircraft_systems.pdf` with its expected answer directly referenced from the source document:

```json
{"input": "What is the primary function of the auxiliary power unit?", "expected_output": "The auxiliary power unit provides electrical power and pneumatic pressure for aircraft systems...", "metadata": {"source": "aircraft_systems.pdf", "page": 11}, "label": 1, "critique": "Grounding verified on page 11."}
```

### Golden Schema
- `input`: The test prompt/question asked to the RAG system.
- `expected_output`: Ground truth answer lifted from the source PDF (never invented).
- `metadata.source`: Name of the source document (`aircraft_systems.pdf`).
- `metadata.page`: 1-indexed PDF page the answer comes from, or `null` for clean negatives.
- `label`: `1` if answerable from the document, `0` for known unanswerable questions (clean negatives).
- `critique`: Explanation of grounding or why the question is unanswerable.

## TruLens RAG Triad Metrics

Each test case is evaluated against the classic TruLens RAG Triad:
1. **Context Relevance** (`context_relevance`): Measures how well the retrieved chunks match the query.
2. **Groundedness** (`groundedness`): Measures whether the generated answer is strictly supported by the retrieved context (detects hallucinations).
3. **Answer Relevance** (`answer_relevance`): Measures how directly and completely the response answers the question.
4. **Clean Negative Recall** (`clean_negative_recall`): Verifies that held-out unanswerable questions (`label == 0`) trigger honest refusals rather than fabricated answers.

All metrics are on a [0.0, 1.0] scale where **higher is better**.

## Persisted Results (`evals/results/`)

`rag-loop run <name>` saves a local snapshot of the run's evaluation results to `evals/results/<name>.json`:
- `aggregated_scores`: Mean, min, max, and std for each TruLens feedback metric.
- `experiment_scores`: Run-level metrics like `clean_negative_recall`.
- `cases`: Detailed per-query breakdown with user input, expected output, generated answer, and TruLens Chain-of-Thought (COT) feedback reasons.

These cached results allow `rag-loop run` and `rag-loop report` to immediately diff performance against the baseline without re-running evaluations.
