# Pipelines

Note that the pipeline definitions in this folder are for azure pipelines, not
github pipelines. There are differences between these systems.

- `ci-eval-pr.yaml` is run for all PRs to _TruLens-Eval_.
- `ci-eval.yaml` for _TruLens-Eval_ releases. This includes database migration
  tests as well as running notebooks.
- `ci.yaml` is run for all PRs to _TruLens-Explain_.

## More information.

- [Triggers](https://learn.microsoft.com/en-us/azure/devops/pipelines/build/triggers?view=azure-devops).
