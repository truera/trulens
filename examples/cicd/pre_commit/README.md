# TruLens Pre-Commit Eval Smoke Test

Run a lightweight TruLens evaluation before code is committed. This catches
obvious quality regressions locally before they reach CI.

## Files

| File | Purpose |
|------|---------|
| `eval_smoke_test.py` | Fast one-case TruLens eval using a single relevance metric. |
| `.pre-commit-config.yaml` | Local hook snippet that runs the smoke test. |

## Setup

1. Install dependencies:

```bash
pip install pre-commit trulens trulens-providers-openai
```

2. Copy `.pre-commit-config.yaml` into your repository, or merge the local hook
   into your existing config.
3. Copy `eval_smoke_test.py` to `examples/cicd/pre_commit/eval_smoke_test.py` or
   update the hook entry path.
4. Export your key locally:

```bash
export OPENAI_API_KEY=sk-...
export TRULENS_MIN_SCORE=0.7
pre-commit install
```

Now the eval smoke test runs before every commit.

## Tradeoffs

Pre-commit hooks should be fast. This example intentionally uses:

- 1 test case
- 1 feedback function (`relevance`, no chain-of-thought reasons)
- a small judge model (`gpt-4o-mini` by default)

Use this hook for quick local guardrails, then run a fuller suite in CI (GitHub
Actions, GitLab CI, CircleCI, Azure Pipelines) before merge.

## Skipping intentionally

When you need to bypass the hook:

```bash
git commit --no-verify
```

Or skip only this hook while keeping other pre-commit checks:

```bash
SKIP_TRULENS_PRECOMMIT=1 git commit
```
