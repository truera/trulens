# Pipelines

Note that the pipeline definitions in this folder are for azure pipelines, not
github pipelines. There are differences between these systems.

- `ci-eval-pr.yaml` is run for all PRs to _TruLens_. Success is needed for
  merging into `main`. It is the single required status check ("PR Validation
  Eval") for branch protection, so it must **always run and report success**.
  A `DetectChanges` gate job inspects the changed paths: code-affecting PRs run
  the full conda build + test matrix, while docs/examples-only PRs skip those
  heavy jobs and the pipeline still succeeds. This lets docs/examples PRs merge
  normally instead of getting stuck on a "skipped" required check.

  > Do **not** add a `paths:` filter to this pipeline (in the YAML `pr:` trigger
  > or via the Azure DevOps UI "Override the YAML trigger from here" option).
  > A path-filtered required check reports **skipped** rather than **success**
  > for excluded PRs, which GitHub branch protection treats as unmet — forcing
  > an admin override to merge. Keep triggering in YAML and let `DetectChanges`
  > decide what work to run.
- `ci-eval.yaml` for _TruLens_ releases. This includes database migration
  tests as well as running notebooks. Success is needed for merging into
  `releases/*`. Also, any branch named `releases/*` needs to pass the pipeline
  tests before merging into `main`.

## More information

- Branch protection rules. These specify what pipelines must succeed before a PR
  can be merged. These are configured from the [Branches
  settings](https://github.com/truera/trulens/settings/branches) panel.

- Pipelines. Pipelines are described by the various `.yaml` files in this folder
  and pointed to by the [Azure
  Pipelines](https://dev.azure.com/truera/trulens/_build) definitions.

- [Triggers
  documentation](https://learn.microsoft.com/en-us/azure/devops/pipelines/build/triggers?view=azure-devops)
  describes how to setup triggers (when a pipeline needs to run).
