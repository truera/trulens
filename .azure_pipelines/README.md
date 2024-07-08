# Pipelines

Note that the pipeline definitions in this folder are for azure pipelines, not
github pipelines. There are differences between these systems.

- `ci-eval-pr.yaml` is run for all PRs to _TruLens-Eval_. Success is needed for
  merging into `main`.
- `ci-eval.yaml` for _TruLens-Eval_ releases. This includes database migration
  tests as well as running notebooks. Success is needed for merging into
  `releases/*`. Also, any branch named `releases/*` needs to pass the pipeline
  tests before merging into `main`.
- `ci.yaml` is run for all PRs to _TruLens-Explain_. Success is needed for
  merging into `main`.

## More information

- Branch protection rules. These specify what pipelines must succeed before a PR
  can be merged. These are configured from the [Branches
  settings](https://github.com/truera/trulens/settings/branches) panel.

- Pipelines. Pipelines are described by the various `.yaml` files in this folder
  and pointed to by the [Azure
  Pipelines](https://dev.azure.com/truera/trulens/_build) definitions.

- [Triggers
  documentation](https://learn.microsoft.com/en-us/azure/devops/pipelines/build/triggers?view=azure-devops)
  describes how to setup trigers (when a pipeline needs to run).
