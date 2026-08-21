# 🤝 Contributing to TruLens

Interested in contributing to TruLens? Here's how to get started!

**Step 1:** Join the [community](https://snowflake.discourse.group/c/ai-research-and-development-community/89).

**Step 2:** Find something to work on below, or browse [open issues](https://github.com/truera/trulens/issues).

---

## Getting Started

New to TruLens? These are great entry points:

### Good First Issues

Issues tagged [`good first issue`](https://github.com/truera/trulens/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
are curated for new contributors. They're well-scoped and often include guidance
on where to start.

### Help Wanted

Issues tagged [`help wanted`](https://github.com/truera/trulens/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22)
are meatier tasks where we'd love community help. These may require more context
but have significant impact.

### Add Usage Examples

Applied TruLens to an interesting use case? Share it as a cookbook example
notebook—like [Evaluating Weaviate Query Agents](https://www.trulens.org/cookbook/vector_stores/weaviate/weaviate_query_agent/).

Examples live in `examples/` and are organized into folders:

| Folder | Purpose |
| ------ | ------- |
| `quickstart/` | Minimal, focused notebooks for getting started fast. These should be simple and demonstrate core TruLens concepts with minimal dependencies. |
| `expositional/` | In-depth tutorials organized by topic (`frameworks/`, `models/`, `use_cases/`, `vector_stores/`). These can be longer and cover advanced integrations. |
| `experimental/` | Work-in-progress examples, internal testing notebooks, or demos of experimental features. Not published to docs. |

Example notebooks should:

- Start with a clear title and description
- Include versioned dependencies: `# !pip install trulens trulens-apps-langchain==1.2.0`
- Be self-contained and runnable
- Go in the appropriate folder based on scope and audience

### Improve Documentation

Found something confusing? If it confused you, it's confusing others too. Documentation improvements are always welcome—from fixing typos to clarifying concepts.

---

## Core Contributions

Ready to dive deeper? These areas have significant impact:

### Feedback Functions

Feedback functions are the backbone of TruLens evaluations. Extend the library
with new evaluation methods:

- Add to an existing [provider module](https://github.com/truera/trulens/tree/main/src/providers/)
- See the [custom feedback functions guide](https://www.trulens.org/component_guides/evaluation/feedback_implementations/custom_feedback_functions/)

**Requirements:** Functions should accept text input(s) and return a `float` (0.0–1.0) or `dict[str, float]`.

### Provider Integrations

Need a model provider we don't support? Add a new `trulens-providers-*` package:

- Browse existing providers in [`src/providers/`](https://github.com/truera/trulens/tree/main/src/providers/)
- Each provider is a separate installable package (see [Package Architecture](docs/contributing/optional.md))

Or [open an issue](https://github.com/truera/trulens/issues/new) requesting a provider—we track demand.

### App Integrations

Instrument a new LLM framework by adding a `trulens-apps-*` package:

- See existing integrations: LangChain, LlamaIndex, NeMo Guardrails
- App packages live in [`src/apps/`](https://github.com/truera/trulens/tree/main/src/apps/)

### Connector Integrations

Connectors define where TruLens stores trace and evaluation logs. Add a new
`trulens-connectors-*` package to support additional databases:

- See the existing Snowflake connector in [`src/connectors/`](https://github.com/truera/trulens/tree/main/src/connectors/)
- Connectors implement the storage interface for traces, records, and feedback results

### Bug Fixes

Bugs are tracked in [GitHub Issues](https://github.com/truera/trulens/issues?q=is%3Aissue+is%3Aopen+label%3Abug).
Feel free to claim an issue by commenting or assigning yourself.

---

## Advanced Contributions

For contributors familiar with the codebase:

### Dashboard & Frontend

The TruLens dashboard (`src/dashboard/`) uses React + TypeScript. Contributions welcome for:

- UI/UX improvements
- New visualizations
- Performance optimizations

### Instrumentation & OTEL

TruLens uses OpenTelemetry for tracing. Work in this area includes:

- Span and attribute improvements in `trulens.core.otel`
- New exporters and integrations
- Performance and reliability enhancements

See [Design Principles](docs/contributing/design.md) for architecture context.

### Experimental Features

Have an idea that pushes TruLens in a new direction? Experimental features use
the `experimental_` prefix and can be toggled via `TruSession.experimental_enable_feature()`.

Past community contributions include the SQLAlchemy connector and LiteLLM provider.

---

## Signing Off Your Commits

TruLens uses the [Developer Certificate of Origin](DCO), the same mechanism the
Linux kernel and CNCF projects use. There is no agreement to sign, no form, and no
account to create. You certify each contribution by adding one line to the commit
message:

```
Signed-off-by: Your Name <your.email@example.com>
```

By adding it you state that you wrote the change, or that you have the right to
submit it under this project's license. The full text is in [DCO](DCO); it is
short and worth reading once.

### What this means day to day

Commit with `-s` and git writes the line for you, using your configured name and
email:

```bash
git commit -s -m "fix: handle empty context list"
```

Set your identity once and it works from then on:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

If you want it always on for this repository, alias it or use
`git config --global format.signOff true` with `git format-patch` workflows.

Two cases need a little care:

- **Editing files in the GitHub web UI**, including accepting a reviewer's
  suggestion, produces a commit without a sign-off. Add the line yourself in the
  extended description box before committing.
- **Forgetting on the first few commits** is the common one. The check tells you
  the fix:

  ```bash
  git rebase --signoff origin/main
  git push --force-with-lease
  ```

  That rewrites your branch's commit messages and nothing else.

### What the check does

A CI job reads the commits your pull request adds and looks for a sign-off whose
email matches the commit author. It only looks at commits your PR introduces, so
the project's existing history is unaffected. Bot commits are exempt. Several
sign-offs on one commit are fine, which is what co-authored work produces.

A bare `Signed-off-by` mismatch is the only thing it fails on. It does not check
the content of your change, and it is not a code-quality gate.

### Does this add friction?

A little, and only once. After `git config` and learning `-s`, the cost is a flag
you stop noticing. The real cost lands on people who do not know about it yet,
which is why the check prints the recovery command rather than just failing, and
why the pull request template mentions it.

If you get stuck on it, say so in the pull request. Nobody's contribution gets
turned away over a missing trailer.

---

## Reference

Before contributing, familiarize yourself with:

| Guide | Description |
| ----- | ----------- |
| [Development Setup](docs/contributing/development.md) | Environment setup, running tests, local development |
| [Standards](docs/contributing/standards.md) | Code style, testing, and documentation conventions |
| [Design Principles](docs/contributing/design.md) | Architecture goals and API design rationale |
| [Package Architecture](docs/contributing/optional.md) | Modular package structure since TruLens 1.0 |
| [Release Policies](docs/contributing/policies.md) | Versioning, deprecation, and experimental features |
| [Tech Debt](docs/contributing/techdebt.md) | Known issues and areas needing refactoring |
| [Database Schema](docs/contributing/database.md) | OTEL events table, legacy schema, and migrations |

---

## Taking on More Responsibility

Contributing regularly and want a larger role? TruLens has a documented path from
contributor to maintainer, with published criteria at each step:

| Rung | Write access | Scope |
| ---- | ------------ | ----- |
| Contributor | None | — |
| Area Triager | None (GitHub Triage role) | Issues and PRs in one area |
| Area Reviewer | Merge, scoped to paths | One named area |
| Maintainer | Merge, project-wide | Whole project |

Area Triager needs about a month of contribution history and one sponsor. Most active
contributors are ready for it sooner than they expect. See the [Contributor
Ladder](CONTRIBUTOR_LADDER.md) for requirements and the nomination process, and
[Governance](GOVERNANCE.md) for how decisions get made.

If you're already contributing and interested, say so in an issue or to a
maintainer. You don't need to wait to be noticed.

---

## Contributors

{%
   include-markdown "CONTRIBUTORS.md"
   heading-offset=2
%}

## Maintainers

{%
   include-markdown "MAINTAINERS.md"
   heading-offset=2
%}
