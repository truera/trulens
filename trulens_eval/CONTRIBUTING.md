# Contributing to TruLens

Interested in contributing to TruLens? Here's how to get started!

## Contribution Guideline

What should I work on?

1. 🆕 Add new [feedback functions](https://www.trulens.org/trulens_eval/feedback_functions/)
2. 🤝 Add new feedback function providers.
3. 🐛 Fix bugs
4. 🎉 Add usage examples
5. 🧪 Add experimental features
6. 📄 Improve code quality & documentation

Also, join the AI Quality Slack community for ideas and discussions:

https://communityinviter.com/apps/aiqualityforum/josh

1. 🆕 Add new [feedback functions](https://www.trulens.org/trulens_eval/feedback_functions/)

Feedback functions are the backbone of TruLens, and evaluating unique LLM apps may require new evaluations. We'd love your contribution to extend the feedback functions library so others can benefit!

To add a feedback function, we'd love your contribution in the form of
- Adding a new feedback function to [feedback.py](https://github.com/truera/trulens/blob/main/trulens_eval/trulens_eval/feedback.py). You can find more information on how to do that here: https://www.trulens.org/trulens_eval/feedback_functions/

- Add it to the documentation! You can do so [here](https://github.com/truera/trulens/blob/main/trulens_eval/examples/feedback_functions.ipynb)

2. 🤝 Add new feedback function providers.

Feedback functions often rely on a model provider, such as OpenAI or HuggingFace. If you need a new model provider to utilize feedback functions for your use case, we'd love if you added a new provider class, e.g. AzureOpenAI.

You can do so by creating a new provider class that inherits the Provider base class in [feedback.py](https://github.com/truera/trulens/blob/main/trulens_eval/trulens_eval/feedback.py)

Alternatively, we also appreciate if you open a GitHub Issue if there's a model provider you need!

3. 🐛 Fix Bugs
Most bugs are reported and tracked in the Github Issues Page. We try our best in triaging and tagging these issues:

Issues tagged as bug are confirmed bugs.
New contributors may want to start with issues tagged with good first issue.
Please feel free to open an issue and/or assign an issue to yourself.

4. 🎉 Add Usage Examples
If you have applied TruLens to track and evalaute a unique use-case, we would love your contribution in the form of:

an example notebook: e.g. [Optimizing Llama-Index Parameters for Complex Questions](https://github.com/truera/trulens/blob/main/trulens_eval/examples/frameworks/llama_index/llamaindex-subquestion-query.ipynb)

5. 🧪 Add Experimental Features
If you have a crazy idea, make a PR for it! Whether if it's the latest research, or what you thought of in the shower, we'd love to see creative ways to improve TruLens.

6. 📄 Improve Code Quality & Documentation
We would love your help in making the project cleaner, more robust, and more understandable. If you find something confusing, it most likely is for other people as well. Help us be better!