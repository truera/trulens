# TruLens CI/CD Eval Gate for GitLab CI

Run a small TruLens evaluation suite in GitLab CI and fail the pipeline when
quality drops below a threshold.

## Files

| File | Purpose |
|------|---------|
| `eval_gate.py` | Self-contained TruLens RAG eval gate. |
| `.gitlab-ci.yml` | GitLab CI pipeline example with a JUnit report artifact. |

## Setup

1. Copy `.gitlab-ci.yml` to the root of your GitLab repository.
2. Copy `eval_gate.py` to `examples/cicd/gitlab_ci/eval_gate.py` or update the
   script path in `.gitlab-ci.yml`.
3. Add `OPENAI_API_KEY` as a masked and protected CI/CD variable:
   **Settings → CI/CD → Variables → Add variable**.
4. Adjust `TRULENS_MIN_SCORE` in `.gitlab-ci.yml` to set your quality bar.

The job runs for merge requests and the default branch. It writes a JUnit report
to `reports/trulens-eval-gate.xml` so failures show up in GitLab's test report UI.

## How the gate decides

The gate checks each metric the script configures, by name, using the per-record
results. A metric fails if it has no column, if any record has no score (the
judge failed on it), or if its mean is below `TRULENS_MIN_SCORE`. Other numeric
leaderboard columns, such as tool-call counts, are not treated as metrics.

## Local run

```bash
pip install trulens trulens-providers-openai
export OPENAI_API_KEY=sk-...
export TRULENS_MIN_SCORE=0.7
python eval_gate.py
```

## CI variables and cost

A protected variable is not available to unprotected branches or to merge
requests from forks. Keep it that way: do not expose a paid judge key to
untrusted code. Each run calls the judge model, which costs money. If the key is
missing the script exits with an error before any evaluation, and no JUnit
report is written.
