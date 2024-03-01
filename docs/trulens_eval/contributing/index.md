# 🤝 Contributing to TruLens

Interested in contributing to TruLens? Here's how to get started!

## What can you work on?

1. 💪 Add new [feedback
functions](https://www.trulens.org/trulens_eval/api/providers)
2. 🤝 Add new feedback function providers.
3. 🐛 Fix bugs
4. 🎉 Add usage examples
5. 🧪 Add experimental features
6. 📄 Improve code quality & documentation

Also, join the [AI Quality Slack
community](https://communityinviter.com/apps/aiqualityforum/josh) for ideas and
discussions.

## 💪 Add new [feedback functions](https://www.trulens.org/trulens_eval/api/providers)

Feedback functions are the backbone of TruLens, and evaluating unique LLM apps
may require new evaluations. We'd love your contribution to extend the feedback
functions library so others can benefit!

- To add a feedback function for an existing model provider, you can add it to
an existing provider module. You can read more about the structure of a
feedback function in this
[guide](https://www.trulens.org/trulens_eval/custom_feedback_functions/).
- New methods can either take a single text (str) as a parameter or two
different texts (str), such as prompt and retrieved context. It should return
a float, or a dict of multiple floats. Each output value should be a float on
the scale of 0 (worst) to 1 (best).
- Make sure to add its definition to this
[list](https://github.com/truera/trulens/blob/main/docs/trulens_eval/api/providers.md).

## 🤝 Add new feedback function providers.

Feedback functions often rely on a model provider, such as OpenAI or
HuggingFace. If you need a new model provider to utilize feedback functions for
your use case, we'd love if you added a new provider class, e.g. Ollama.

You can do so by creating a new provider module in this
[folder](https://github.com/truera/trulens/blob/main/trulens_eval/trulens_eval/feedback/provider/).

Alternatively, we also appreciate if you open a GitHub Issue if there's a model
provider you need!

## 🐛 Fix Bugs

Most bugs are reported and tracked in the Github Issues Page. We try our best in
triaging and tagging these issues:

Issues tagged as bug are confirmed bugs. New contributors may want to start with
issues tagged with good first issue. Please feel free to open an issue and/or
assign an issue to yourself.

## 🎉 Add Usage Examples

If you have applied TruLens to track and evalaute a unique use-case, we would
love your contribution in the form of an example notebook: e.g. [Evaluating
Pinecone Configuration Choices on Downstream App
Performance](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/expositional/vector-dbs/pinecone/pinecone_evals_build_better_rags.ipynb)

All example notebooks are expected to:

- Start with a title and description of the example
- Include a commented out list of dependencies and their versions, e.g. `# ! pip
install trulens==0.10.0 langchain==0.0.268`
- Include a linked button to a Google colab version of the notebook
- Add any additional requirements

## 🧪 Add Experimental Features

If you have a crazy idea, make a PR for it! Whether if it's the latest research,
or what you thought of in the shower, we'd love to see creative ways to improve
TruLens.

## 📄 Improve Code Quality & Documentation

We would love your help in making the project cleaner, more robust, and more
understandable. If you find something confusing, it most likely is for other
people as well. Help us be better!

Big parts of the code base currently do not follow the code standards outlined
in [Standards index](/contributing/standards). Many good contributions can be made in adapting
us to the standards.

## 👀 Things to be Aware Of

### ✅ Standards

We try to respect various code, testing, and documentation
standards outlined in the [Standards index](/contributing/standards).

### 💣 Tech Debt

Parts of the code are nuanced in ways should be avoided by new contributors.
Discussions of these points are welcome to help the project rid itself of these
problematic designs. See [Tech debt index](techdebt).

## Contributors

{%
include-markdown "../../../trulens_eval/CONTRIBUTORS.md"
heading-offset=2
%}


{%
include-markdown "../../../trulens_explain/CONTRIBUTORS.md"
heading-offset=2
%}