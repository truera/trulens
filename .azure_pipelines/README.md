# Pipelines

Note that the pipeline definitions in this folder are for azure pipelines, not
github pipelines. There are differences between these systems.

- `ci-eval-pr.yaml` is run for all PRs to _TruLens_. Success is needed for
  merging into `main`. It is the single required status check ("PR Validation
  Eval") for branch protection, so it must **always run and report success**.
  It has two tiers of validation:
    - **Lightweight (every PR):** the `PreCommit` job runs `make run-precommit`
      (ruff, ruff-format, yaml/whitespace, nb-clean, ...) on all PRs — including
      docs/examples-only ones — so formatting/lint issues are always caught.
    - **Expensive (code PRs only):** the conda build + full unit test matrix run
      only when the `DetectChanges` gate finds code-affecting changes.
  Docs/examples-only PRs skip the expensive jobs but still run `PreCommit`, and
  the pipeline succeeds when lint passes — so they merge normally instead of
  getting stuck on a "skipped" required check, while still being lint-gated.

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
- `cd-docs-main.yaml` publishes the documentation site to the `gh-pages` branch.
  It triggers on every merge to `main` but only publishes when a **release**
  lands, so trulens.org reflects the latest released version rather than the tip
  of `main`.

  Release branches are squash-merged, so the commit on `main` records only a PR
  number and never the branch name, and the release tags sit on the rc branch
  rather than on `main`. What a release does leave behind is a version bump, so
  the cheap `Gate` job compares one line of `src/core/pyproject.toml` against the
  previous commit and the `DocsDeploy` job depends on its output. Ordinary merges
  stop at the gate, having paid only for a two-commit shallow checkout.

  Two things about this pipeline are easy to break:

  > Its checkout must set `lfs: true` (via the `env-setup.yaml` parameter). Every
  > image in the repository is LFS-tracked, and a checkout without LFS leaves
  > them as ~130 byte pointer stubs which the build copies into the site without
  > any warning. That is not hypothetical: it is how every image on trulens.org
  > came to be broken at once. `make check-no-lfs-pointers` guards against it,
  > and the pipeline runs that check both before building and afterwards against
  > the published branch.

  > Do **not** add it to GitHub branch protection as a required check. It runs on
  > `main` after merge, so it can never report on a PR.

  A **manual run** bypasses the release gate, which is the intended way to
  publish a docs-only fix between releases.

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
