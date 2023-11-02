# Welcome to TruLens-Eval!

![TruLens](https://www.trulens.org/assets/images/Neural_Network_Explainability.png)

Evaluate and track your LLM experiments with TruLens. As you work on your models and prompts TruLens-Eval supports the iterative development and of a wide range of LLM applications by wrapping your application to log key metadata across the entire chain (or off chain if your project does not use chains) on your local machine.

Using feedback functions, you can objectively evaluate the quality of the responses provided by an LLM to your requests. This is completed with minimal latency, as this is achieved in a sequential call for your application, and evaluations are logged to your local machine. Finally, we provide an easy to use Streamlit dashboard run locally on your machine for you to better understand your LLM’s performance.

## Value Propositions

TruLens-Eval has two key value propositions:

1. Evaluation:
    * TruLens supports the the evaluation of inputs, outputs and internals of your LLM application using any model (including LLMs). 
    * A number of feedback functions for evaluation are implemented out-of-the-box such as groundedness, relevance and toxicity. The framework is also easily extensible for custom evaluation requirements.
2. Tracking:
    * TruLens contains instrumentation for any LLM application including question answering, retrieval-augmented generation, agent-based applications and more. This instrumentation allows for the tracking of a wide variety of usage metrics and metadata. Read more in the [instrumentation overview](https://www.trulens.org/trulens_eval/basic_instrumentation/).
    * TruLens' instrumentation can be applied to any LLM application without being tied down to a given framework. Additionally, deep integrations with [LangChain]() and [Llama-Index]() allow the capture of internal metadata and text.
    * Anything that is tracked by the instrumentation can be evaluated!

The process for building your evaluated and tracked LLM application with TruLens is below 👇

![Architecture Diagram](https://www.trulens.org/assets/images/TruLens_Architecture.png)

## Installation and Setup

Install the trulens-eval pip package from PyPI.

```bash
    pip install trulens-eval
```

## Setting Keys

In any of the quickstarts, you will need [OpenAI](https://platform.openai.com/account/api-keys) and [Huggingface](https://huggingface.co/settings/tokens) keys. You can add keys by setting the environmental variables:

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
os.environ["HUGGINGFACE_API_KEY"] = "..."
```

## Quick Usage

TruLens supports the evaluation of tracking for any LLM app framework. Choose a framework below to get started:

**Langchain**

[langchain_quickstart.ipynb](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.17.0b/trulens_eval/examples/quickstart/langchain_quickstart.ipynb).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/releases/rc-trulens-eval-0.17.0b/trulens_eval/examples/quickstart/colab/langchain_quickstart_colab.ipynb)

**Llama-Index**

[llama_index_quickstart.ipynb](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.17.0b/trulens_eval/examples/quickstart/llama_index_quickstart.ipynb).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/releases/rc-trulens-eval-0.17.0b/trulens_eval/examples/quickstart/colab/llama_index_quickstart_colab.ipynb)

**Custom Text to Text Apps**

[text2text_quickstart.ipynb](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.17.0b/trulens_eval/examples/quickstart/text2text_quickstart.ipynb).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/releases/rc-trulens-eval-0.17.0b/trulens_eval/examples/quickstart/colab/text2text_quickstart_colab.ipynb)

### 💡 Contributing

Interested in contributing? See our [contribution guide](https://github.com/truera/trulens/tree/main/trulens_eval/CONTRIBUTING.md) for more details.
