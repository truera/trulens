# 🤝 Contributing to TruLens

Interested in contributing to TruLens? Here's how to get started!

Step 1: Join the [community](https://snowflake.discourse.group/c/ai-research-and-development-community/89).

## What can you work on?

1. 💪 Add new [feedback
   functions][trulens.core.feedback.provider.Provider]
2. 🤝 Add new feedback function providers.
3. 🐛 Fix bugs
4. 🎉 Add usage examples
5. 🧪 Add experimental features
6. 📄 Improve code quality & documentation
7. ⛅ Address open issues.

## 💪 Add new [feedback functions][trulens.core.feedback.provider.Provider]

Feedback functions are the backbone of TruLens, and evaluating unique LLM apps
may require new evaluations. We'd love your contribution to extend the feedback
functions library so others can benefit!

- To add a feedback function for an existing model provider, you can add it to
  an existing provider module. You can read more about the structure of a
  feedback function in this
  [guide](https://www.trulens.org/component_guides/evaluation/feedback_implementations/custom_feedback_functions/).
- New methods can either take a single text (str) as a parameter or two
  different texts (str), such as prompt and retrieved context. It should return
  a float, or a dict of multiple floats. Each output value should be a float on
  the scale of 0 (worst) to 1 (best).

## 🤝 Add new feedback function providers

Feedback functions often rely on a model provider, such as OpenAI or
HuggingFace. If you need a new model provider to utilize feedback functions for
your use case, we'd love it if you added a new provider class, e.g. Ollama.

You can do so by creating a new provider module in this
[folder](https://github.com/truera/trulens/blob/main/src/providers/).

Alternatively, we also appreciate if you open a GitHub Issue if there's a model
provider you need!

## 🐛 Fix Bugs

Most bugs are reported and tracked in the [GitHub Issues](https://github.com/truera/trulens/issues) page. We try our best in
triaging and tagging these issues:

Issues tagged as "bug" are confirmed bugs. New contributors may want to start with
issues tagged with "good first issue". Please feel free to open an issue and/or
assign an issue to yourself.

## 🎉 Add Usage Examples

If you have applied TruLens to track and evaluate a unique use case, we would
love your contribution to the cookbook in the form of an example notebook: e.g. [Evaluating Weaviate Query Agents](https://www.trulens.org/cookbook/vector_stores/weaviate/weaviate_query_agent/)

All example notebooks are expected to:

- Start with a title and description of the example
- Include a commented-out list of dependencies and their versions, e.g. `# !pip
  install trulens==0.10.0 langchain==0.0.268`
- Include a linked button to a Google Colab version of the notebook
- Add any additional requirements

## 🧪 Add Experimental Features

If you have a crazy idea, make a PR for it! Whether it's the latest research,
or what you thought of in the shower, we'd love to see creative ways to improve
TruLens.

Community contributions that have been accepted in the past include the SQLAlchemy logging connection and the LiteLLM provider.

## 📄 Improve Code Quality & Documentation

We would love your help in making the project cleaner, more robust, and more
understandable. If you find something confusing, it most likely is for other
people as well. Help us be better!

Large portions of the codebase currently do not follow the code standards outlined
in the [Standards index](standards.md). Many good contributions can be made in
adapting us to the standards.

## ⛅ Address Open Issues

See [🍼 good first
issue](https://github.com/truera/trulens/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
or [🧙 all open issues](https://github.com/truera/trulens/issues).

## 👀 Things to be Aware Of

### Development guide

See [Development guide](development.md).

### 🧭 Design Goals and Principles

The design of the API is governed by the principles outlined in the
[Design](design.md) doc.

### 📦 Release Policies

Versioning and deprecation guidelines are included. [Release policies](policies.md).

### ✅ Standards

We try to respect various code, testing, and documentation standards outlined in
the [Standards index](standards.md).

### 💣 Tech Debt

Parts of the code are nuanced in ways that should be avoided by new contributors.
Discussions of these points are welcome to help the project rid itself of these
problematic designs. See [Tech debt index](techdebt.md).

### ⛅ Optional Packages

Limit the packages installed by default when installing _TruLens_. For
optional functionality, additional packages can be requested for the user to
install and their usage is aided by an optional imports scheme. See [Optional
Packages](optional.md) for details.

### ✨ Database Migration

[Database migration](migration.md).

## 👋👋🏻👋🏼👋🏽👋🏾👋🏿 Contributors

{%
   include-markdown "../../CONTRIBUTORS.md"
   heading-offset=2
%}

## 🧰 Maintainers

{%
   include-markdown "../../MAINTAINERS.md"
   heading-offset=2
%}
