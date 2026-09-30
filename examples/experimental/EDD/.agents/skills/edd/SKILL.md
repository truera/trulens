---
name: edd
description: Eval-driven development loop for this repo (rag-loop, an aircraft-systems RAG app on LangGraph, OpenAI, Qdrant Cloud, and TruLens). Use when the user wants to run an experiment, inspect feedback metrics, or decide keep-or-reject on a config/prompt change.
---

# EDD loop for rag-loop with TruLens (aircraft systems)

Orchestration for this repo's eval loop: which command to run, in what order, and what's automated
vs. what needs judgment. For TruLens evaluation setup, instrumentation, or feedback functions, reference
TruLens core patterns (`Feedback`, `Selector`, `SpanAttributes`).

Stack: LangGraph orchestrates retrieve → generate; retrieval embeddings are OpenAI
(`text-embedding-3-small`, requires `OPENAI_API_KEY`); generation uses OpenAI
(`gpt-5.6-luna`); TruLens feedback judges use the **RAG Triad** with OpenAI
(`gpt-4.1-nano`);
vector database is Qdrant (`QDRANT_URL`/`QDRANT_API_KEY`). Traces and evaluations are recorded via
TruLens OpenTelemetry instrumentation and saved in TruLens database.

Two fundamental rules:

1. **Never invent an `expected_output`.** Every golden's expected answer must be lifted from the
   actual PDF text, not written from general knowledge of aircraft systems.
2. **Grade outcomes, not paths.** A judge scores whether the final answer is relevant, grounded in
   the retrieved context, and directly answers the question.

## Preconditions

Everything except reading/editing files needs `.env` populated:
- `OPENAI_API_KEY`
- `QDRANT_URL`
- `QDRANT_API_KEY`

`rag-loop run` also requires git tracking to commit accepted changes and revert rejected ones.

## What's automated vs. what's your job

`rag-loop run <name>` (`src/rag_loop/loop.py: run_iteration`) automates the mechanical steps:
- Runs the experiment through the RAG pipeline
- Evaluates with TruLens feedback functions (`answer_relevance`, `groundedness`, `context_relevance`)
- Computes `clean_negative_recall` for unanswerable held-out items
- Persists scores and COT reasons to `evals/results/<name>.json`
- Compares against `acceptance.compare_against` on `acceptance.required_metrics`
- Writes the decision back to `evals/experiments.json`
- Commits `src/` (keep) or reverts via git (reject)

Your job as an agent: **read the failures, form a hypothesis, and write the next single-variable change.**
The iterative cycle:

1. `rag-loop report <name>` — inspect worst-scoring cases (input, expected, generated, TruLens COT reasons).
2. Form a hypothesis about *why* retrieval failed (low context relevance) or generation drifted (low groundedness/relevance).
3. Write one new entry in `evals/experiments.json` changing exactly one variable (`chat_model`, `retrieval_k`, prompt in `src/rag_loop/rag.py`), with `compare_against` set to the last kept experiment.
4. `rag-loop run <name>` and review the printed verdict and metric deltas.
5. Repeat on reject. If `Plateaued` (consecutive kept runs with flat scores), stop tweaking wording and propose a structural change (chunking strategy, reranking, hybrid search).

## The loop, in order

### 1. Index the source PDF into Qdrant

```sh
rag-loop index
```

Rebuild any time `chunk_size`/`chunk_overlap` in `src/rag_loop/config.py` changes.

### 2. Validate golden dataset

```sh
rag-loop seed
```

Validates `evals/seed.jsonl` structure and ensures all required fields and page references are sound.

### 3. Run one experiment at a time

```sh
rag-loop run <name>
```

`<name>` must exist in `evals/experiments.json`. Always run `baseline` first:

```sh
rag-loop run baseline
# Then add "topk-2" to experiments.json:
rag-loop run topk-2
```

For quick sweeps without git commit/revert bookkeeping:
```sh
rag-loop loop --config evals/configs.json
```

### 4. On reject, analyze before writing the next change

```sh
rag-loop report <name>
```

Review the low-scoring questions and the reasons provided by TruLens evaluators before formulating your next hypothesis.

### 5. Launch TruLens Dashboard (Optional)

```sh
rag-loop dashboard
```
Opens the interactive Streamlit TruLens dashboard to inspect execution traces and score leaderboards.
