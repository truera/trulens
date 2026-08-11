# TruLens CI/CD Eval Gate for Azure Pipelines

Run a small TruLens evaluation suite in Azure Pipelines and fail the build when
quality drops below a threshold.

## Files

| File | Purpose |
|------|---------|
| `eval_gate.py` | Self-contained TruLens RAG eval gate. |
| `azure-pipelines.yml` | Azure Pipelines config with Python setup and JUnit publishing. |

## Setup

1. Copy `azure-pipelines.yml` to the root of your repository.
2. Copy `eval_gate.py` to `examples/cicd/azure_pipelines/eval_gate.py` or update
   the script path in the pipeline.
3. Create a variable group named `trulens-eval-secrets` in Azure DevOps:
   **Pipelines → Library → Variable groups → + Variable group**.
4. Add a secret variable named `OPENAI_API_KEY`.
5. Adjust `TRULENS_MIN_SCORE` in `azure-pipelines.yml` to set your quality bar.

The pipeline writes a JUnit XML report and publishes it with `PublishTestResults@2`
so metric failures appear in Azure DevOps test results.

## Optional: Azure Key Vault

For production teams, store `OPENAI_API_KEY` in Azure Key Vault and link the key
vault to your variable group. The pipeline does not change — it still reads
`$(OPENAI_API_KEY)` from the variable group.

## Local run

```bash
pip install trulens trulens-providers-openai
export OPENAI_API_KEY=sk-...
export TRULENS_MIN_SCORE=0.7
python eval_gate.py
```
