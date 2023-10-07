![PyPI - Version](https://img.shields.io/pypi/v/trulens_eval?label=trulens_eval&link=https%3A%2F%2Fpypi.org%2Fproject%2Ftrulens-eval%2F)
![Azure DevOps builds (job)](https://img.shields.io/azure-devops/build/truera/5a27f3d2-132d-40fc-9b0c-81abd1182f41/9)
![GitHub](https://img.shields.io/github/license/truera/trulens)
![PyPI - Downloads](https://img.shields.io/pypi/dm/trulens_eval)
[![Slack](https://img.shields.io/badge/slack-join-green?logo=slack)](https://communityinviter.com/apps/aiqualityforum/josh)
[![Docs](https://img.shields.io/badge/docs-trulens.org-blue)](https://www.trulens.org/welcome/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/releases/rc-trulens-eval-0.10.0/trulens_eval/examples/quickstart/colab/langchain_quickstart_colab.ipynb)

# 🦑 **Welcome to TruLens!**

TruLens provides a set of tools for developing and monitoring neural nets, including large language models. This includes both tools for evaluation of LLMs and LLM-based applications with *TruLens-Eval* and deep learning explainability with *TruLens-Explain*. *TruLens-Eval* and *TruLens-Explain* are housed in separate packages and can be used independently.

The best way to support TruLens is to give us a ⭐ and join our [slack community](https://communityinviter.com/apps/aiqualityforum/josh)!

## TruLens-Eval

**TruLens-Eval** contains instrumentation and evaluation tools for large language model (LLM) based applications. It supports the iterative development and monitoring of a wide range of LLM applications by wrapping your application to log key metadata across the entire chain (or off chain if your project does not use chains) on your local machine. Importantly, it also gives you the tools you need to evaluate the quality of your LLM-based applications.

TruLens-Eval has two key value propositions:

1. Evaluation:
    * TruLens supports the the evaluation of inputs, outputs and internals of your LLM application using any model (including LLMs). 
    * A number of feedback functions for evaluation are implemented out-of-the-box such as groundedness, relevance and toxicity. The framework is also easily extensible for custom evaluation requirements.
2. Tracking:
    * TruLens contains instrumentation for any LLM application including question answering, retrieval-augmented generation, agent-based applications and more. This instrumentation allows for the tracking of a wide variety of usage metrics and metadata. Read more in the [instrumentation overview](basic_instrumentation.ipynb).
    * TruLens' instrumentation can be applied to any LLM application without being tied down to a given framework. Additionally, deep integrations with [LangChain]() and [Llama-Index]() allow the capture of internal metadata and text.
    * Anything that is tracked by the instrumentation can be evaluated!

The process for building your evaluated and tracked LLM application with TruLens is shown below 👇
![Architecture Diagram](https://www.trulens.org/Assets/image/TruLens_Architecture.png)

### Installation and setup

Install trulens-eval from PyPI.

```bash
pip install trulens-eval
```

### Quick Usage

TruLens supports the evaluation of tracking for any LLM app framework. Choose a framework below to get started:

**Langchain**

[langchain_quickstart.ipynb](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.14.0/trulens_eval/examples/quickstart/langchain_quickstart.ipynb).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/releases/rc-trulens-eval-0.14.0/trulens_eval/examples/quickstart/colab/langchain_quickstart_colab.ipynb)

[langchain_quickstart.py](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.14.0/trulens_eval/examples/quickstart/py_script_quickstarts/langchain_quickstart.py).

**Llama-Index**

[llama_index_quickstart.ipynb](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.14.0/trulens_eval/examples/quickstart/llama_index_quickstart.ipynb).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/releases/rc-trulens-eval-0.14.0/trulens_eval/examples/quickstart/colab/llama_index_quickstart_colab.ipynb)

[llama_index_quickstart.py](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.14.0/trulens_eval/examples/quickstart/py_script_quickstarts/llama_index_quickstart.py)

**No Framework**

[text2text_quickstart.ipynb](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.14.0/trulens_eval/examples/quickstart/text2text_quickstart.ipynb).
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/releases/rc-trulens-eval-0.14.0/trulens_eval/examples/quickstart/colab/text2text_quickstart_colab.ipynb)

[text2text_quickstart.py](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.14.0/trulens_eval/examples/quickstart/py_script_quickstarts/text2text_quickstart.py)


## TruLens-Explain

**TruLens-Explain** is a cross-framework library for deep learning explainability. It provides a uniform abstraction over a number of different frameworks. It provides a uniform abstraction layer over TensorFlow, Pytorch, and Keras and allows input and internal explanations.

### Installation and Setup

These installation instructions assume that you have conda installed and added to your path.

0. Create a virtual environment (or modify an existing one).
```bash
conda create -n "<my_name>" python=3  # Skip if using existing environment.
conda activate <my_name>
```
 
1. Install dependencies.
```bash
conda install tensorflow-gpu=1  # Or whatever backend you're using.
conda install keras             # Or whatever backend you're using.
conda install matplotlib        # For visualizations.
```

2. [Pip installation] Install the trulens pip package from PyPI.
```bash
pip install trulens
```

### Quick Usage

To quickly play around with the TruLens library, check out the following Colab notebooks:

* PyTorch: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n77IGrPDO2XpeIVo_LQW0gY78enV-tY9?usp=sharing)
* TensorFlow 2 / Keras: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f-ETsdlppODJGQCdMXG-jmGmfyWyW2VD?usp=sharing)

For more information, see [TruLens-Explain Documentation](https://www.trulens.org/trulens_explain/quickstart/).
