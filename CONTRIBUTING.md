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
