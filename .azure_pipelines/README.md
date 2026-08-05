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
- `ci-docs-pr.yaml` runs for PRs into `main` that touch `docs`, `examples`,
  `mkdocs.yml`, `tools/check_llms_txt.py` or the pipeline itself. It runs the
  pre-commit hooks, builds the documentation with `make docs-linkcheck-strict`, and
  validates `llms.txt` against the generated sitemap.

  Unlike `ci-eval-pr.yaml` it is safe for this one to carry a `paths:` filter,
  because it is not a required status check, so reporting **skipped** for unrelated
  PRs costs nothing. It is also where `--strict` is enforced: the release pipeline
  deliberately builds without it, so this is the gate that catches a broken
  reference while a human can still retry it.

- `cd-release-prep.yaml` prepares a release. Run it **manually**, with the
  `releases/rc-trulens-<version>` branch selected. The exact version comes from
  the branch name: selecting `releases/rc-trulens-2.11.0` sets every package to
  `2.11.0`. It also updates each conda recipe's version so local-source recipe
  validation matches the package being built. The existing recipe hashes remain
  unchanged until the packages exist on PyPI. It refreshes the lockfile, commits,
  and pushes to that branch. You then open the release PR from the "Compare &
  pull request" banner GitHub shows after the push.

  It exists so nobody has to sit and watch a dependency resolve, which was the
  tedious part of cutting a release. Validation remains on the release PR into
  `main` rather than in this preparation job.

  > Direct CI pushes to `releases/*` require the Azure Pipelines GitHub App on the
  > pull-request bypass list and required status checks disabled for that branch
  > protection rule. Keep pull requests required: the app bypasses that rule only
  > for release preparation, while the release PR into `main` remains the human
  > and validation gate.

  > It refuses to run unless the selected branch matches
  > `releases/rc-trulens-X.Y.Z`, or if that target is not greater than the current
  > version. The branch picker in the Run dialog makes running against `main` an
  > easy mistake, so these checks happen before any files are changed.

- `cd-release-main.yaml` publishes the release. It triggers on every merge to
  `main`, which is the point at which a release is real — nothing can be published
  that failed to land on `main`. Four stages, each depending on the last:

    - **Gate** decides whether a release actually landed. Release branches are
      squash-merged, so the commit on `main` records only a PR number and never the
      branch name; the subjects are inconsistent (`TruLens 2.10.0` against
      `TruLens 2.9`); and the tags sit on the rc branch and are pushed before the
      merge. What a release does leave behind is a version bump, so this compares
      one line of `src/core/pyproject.toml` against the previous commit. Ordinary
      merges stop here, having paid only for a two-commit shallow checkout.
    - **PublishPyPI** builds the wheels and uploads them. No approval step: the
      human checkpoint is the review on the release pull request, and once that is
      merged the release is meant to happen.
    - **DeployDocs** publishes trulens.org, so the site never documents a version
      that is not yet installable.
    - **UpdateMetaYaml** refreshes the conda recipes and pushes them to a
      `chore/meta-yaml-<version>` branch for review.

  > Merging a release PR uploads to PyPI unattended, and that cannot be undone --
  > a version number can never be reused. The two things standing between a merge
  > and an upload are the PR review and the `Gate` stage, which will not let an
  > ordinary merge reach the publish.

  > The PyPI token is read from a `pypiToken` secret variable into
  > `TWINE_PASSWORD`, never passed on a command line.

  > Do **not** add it to GitHub branch protection as a required check. It runs on
  > `main` after merge, so it can never report on a PR.

- `cd-docs-main.yaml` publishes the documentation site to the `gh-pages` branch
  **on demand only** (`trigger: none`). Deciding that a release landed is
  `cd-release-main.yaml`'s job, and that pipeline publishes the docs as a stage
  once the release has reached PyPI. What is left here is the escape hatch: a
  docs-only fix between releases.

  The publish steps themselves live in `templates/deploy-docs.yaml`, shared with
  the release pipeline so the two cannot drift.

  > That template's checkout must keep `lfs: true`. Every image in the repository
  > is LFS-tracked, and a checkout without LFS leaves them as ~130 byte pointer
  > stubs which the build copies into the site without any warning. That is not
  > hypothetical: it is how every image on trulens.org came to be broken at once.
  > `make check-no-lfs-pointers` guards against it, and the template runs that
  > check both before building and afterwards against the published branch.

  > Do **not** add it to GitHub branch protection as a required check. It only ever
  > runs manually.


## More information

Nothing in this folder validates a pull request **into** a `releases/*` branch.
`ci-eval-pr.yaml` triggers only on PRs into `main`, and the `ci-eval.yaml` that
this README described until recently — database migration tests and notebook runs,
gating merges into `releases/*` — was removed from the repository in #1813 while
its entry here was left behind. If that validation is meant to still exist, it is
not defined here; check the [pipeline
definitions](https://dev.azure.com/truera/trulens/_build) for one pointing
somewhere else. Otherwise the release gate is the Snowflake end-to-end suite, run
by hand as part of the release procedure.

- Branch protection rules. These specify what pipelines must succeed before a PR
  can be merged. These are configured from the [Branches
  settings](https://github.com/truera/trulens/settings/branches) panel.

- Pipelines. Pipelines are described by the various `.yaml` files in this folder
  and pointed to by the [Azure
  Pipelines](https://dev.azure.com/truera/trulens/_build) definitions.

- [Triggers
  documentation](https://learn.microsoft.com/en-us/azure/devops/pipelines/build/triggers?view=azure-devops)
  describes how to setup triggers (when a pipeline needs to run).
