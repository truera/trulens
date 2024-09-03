<!---
start of docs/trulens/intro.md
NOTE: This content is from docs/trulens/intro.md and is merged into
trulens/README.md . If you are editing README.md, your changes will be overwritten.
-->
# Welcome to TruLens!

![TruLens](https://www.trulens.org/assets/images/Neural_Network_Explainability.png)

**Don't just vibe-check your llm app!** Systematically evaluate and track your
LLM experiments with TruLens. As you develop your app including prompts, models,
retrievers, knowledge sources and more, *TruLens* is the tool you need to
understand its performance.

!!! info

    TruLens 1.0 is now available. [Read more](./release_blog_1dot.md) and check out the [migration guide](./guides/trulens_eval_migration.md)

Fine-grained, stack-agnostic instrumentation and comprehensive evaluations help
you to identify failure modes & systematically iterate to improve your
application.

Read more about the core concepts behind TruLens including [Feedback Functions](https://www.trulens.org/trulens/getting_started/core_concepts/feedback_functions/),
[The RAG Triad](https://www.trulens.org/trulens/getting_started/core_concepts/rag_triad/),
and [Honest, Harmless and Helpful Evals](https://www.trulens.org/trulens/getting_started/core_concepts/honest_harmless_helpful_evals/).

## TruLens in the development workflow

Build your first prototype then connect instrumentation and logging with
TruLens. Decide what feedbacks you need, and specify them with TruLens to run
alongside your app. Then iterate and compare versions of your app in an
easy-to-use user interface 👇

![Architecture
Diagram](https://www.trulens.org/assets/images/TruLens_Architecture.png)

## Installation and Setup

Install the trulens pip package from PyPI.

```bash
    pip install trulens
```

## Quick Usage

Walk through how to instrument and evaluate a RAG built from scratch with
TruLens.

[![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/examples/quickstart/quickstart.ipynb)

### 💡 Contributing

Interested in contributing? See our [contributing
guide](https://www.trulens.org/contributing/) for more details.
<!---
end of docs/trulens/intro.md
-->
