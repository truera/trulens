# TruLens CI/CD Eval Gate (GitHub Actions)

A minimal, copy-pasteable example showing how to run TruLens evaluations in CI
and **fail the build when quality drops**.

## What's here

| File | Purpose |
|------|---------|
| `eval_gate.py` | Standalone script: runs a tiny instrumented RAG app over a fixed test set, evaluates it with the RAG triad, and exits non-zero if any score is below the threshold. |
| `trulens-eval-gate.yml` | A GitHub Actions workflow template. Copy it to `.github/workflows/` in your repo. |

> The workflow file lives here as a template rather than under this repo's own
> `.github/workflows/` so it doesn't run as part of TruLens CI. Copy it into your
> project to activate it.

## Quickstart (local)

```bash
pip install trulens trulens-providers-openai
export OPENAI_API_KEY=sk-...
export TRULENS_MIN_SCORE=0.7   # optional, defaults to 0.7
python eval_gate.py
```

The script exits with code `0` if all metrics meet the threshold and `1`
otherwise, so it works directly as a CI gate.

## Use in GitHub Actions

1. Copy `trulens-eval-gate.yml` to `.github/workflows/trulens-eval-gate.yml` in
   your repository.
2. Add your OpenAI key as a repository secret named `OPENAI_API_KEY`
   (*Settings → Secrets and variables → Actions*).
3. Adjust `TRULENS_MIN_SCORE` in the workflow (or via env) to set your quality
   bar.

The job runs on every pull request and blocks merge if evaluation scores fall
below the threshold.

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `OPENAI_API_KEY` | — | Required. Key for the OpenAI LLM judge. |
| `TRULENS_MIN_SCORE` | `0.7` | Minimum acceptable mean score per metric (0–1). |
| `TRULENS_EVAL_MODEL` | `gpt-4o-mini` | Judge model used by the OpenAI provider. |

## Adapting to your app

Replace `TinyRAG` and `TEST_SET` in `eval_gate.py` with your own application and
evaluation dataset. Keep the dataset small so the gate stays fast (a few seconds
to under a minute) and cheap. The same pattern works for any framework — swap
`TruApp` for `TruChain`, `TruLlama`, or `TruGraph` as appropriate.
