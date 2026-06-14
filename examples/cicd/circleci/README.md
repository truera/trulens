# TruLens CI/CD Eval Gate for CircleCI

Run a small TruLens evaluation suite in CircleCI and fail the workflow when
quality drops below a threshold.

## Files

| File | Purpose |
|------|---------|
| `eval_gate.py` | Self-contained TruLens RAG eval gate. |
| `config.yml` | CircleCI config with a Python Docker executor and JUnit output. |

## Setup

1. Copy `config.yml` to `.circleci/config.yml` in your repository.
2. Copy `eval_gate.py` to `examples/cicd/circleci/eval_gate.py` or update the
   script path in `config.yml`.
3. Create a CircleCI context named `trulens-eval-secrets` and add `OPENAI_API_KEY`:
   **Organization Settings → Contexts → Create Context**.
4. Adjust `TRULENS_MIN_SCORE` in `config.yml` to set your quality bar.

The job writes a JUnit XML report to `test-results/trulens/` and stores both test
results and artifacts so score failures are visible in the CircleCI UI.

## Local run

```bash
pip install trulens trulens-providers-openai
export OPENAI_API_KEY=sk-...
export TRULENS_MIN_SCORE=0.7
python eval_gate.py
```
