<!---
start of trulens_eval/gh_top_intro.md
NOTE: This content is from trulens_eval/gh_top_intro.md and is merged into
README.md . If you are editing README.md, your changes will be overwritten.
-->

![PyPI - Version](https://img.shields.io/pypi/v/trulens_eval?label=trulens_eval&link=https%3A%2F%2Fpypi.org%2Fproject%2Ftrulens-eval%2F)
![Azure DevOps builds (job)](https://img.shields.io/azure-devops/build/truera/5a27f3d2-132d-40fc-9b0c-81abd1182f41/9)
![GitHub](https://img.shields.io/github/license/truera/trulens)
![PyPI - Downloads](https://img.shields.io/pypi/dm/trulens_eval)
[![Slack](https://img.shields.io/badge/slack-join-green?logo=slack)](https://communityinviter.com/apps/aiqualityforum/josh)
[![Docs](https://img.shields.io/badge/docs-trulens.org-blue)](https://www.trulens.org/install/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/releases/rc-trulens-eval-0.17.0/trulens_eval/examples/quickstart/colab/langchain_quickstart_colab.ipynb)

# 🦑 **Welcome to TruLens!**

TruLens provides a set of tools for developing and monitoring neural nets,
including large language models. This includes both tools for evaluation of LLMs
and LLM-based applications with *TruLens-Eval* and deep learning explainability
with *TruLens-Explain*. *TruLens-Eval* and *TruLens-Explain* are housed in
separate packages and can be used independently.

The best way to support TruLens is to give us a ⭐ on
[GitHub](https://www.github.com/truera/trulens) and join our [slack
community](https://communityinviter.com/apps/aiqualityforum/josh)!

![TruLens](https://www.trulens.org/assets/images/Neural_Network_Explainability.png)

## TruLens-Eval

**Don't just vibe-check your llm app!** Systematically evaluate and track your
LLM experiments with TruLens. As you develop your app including prompts, models,
retreivers, knowledge sources and more, TruLens-Eval is the tool you need to
understand its performance.

Fine-grained, stack-agnostic instrumentation and comprehensive evaluations help
you to identify failure modes & systematically iterate to improve your
application.

Read more about the core concepts behind TruLens including [Feedback
Functions](https://www.trulens.org/trulens_eval/core_concepts/feedback_functions/),
[The RAG Triad](https://www.trulens.org/trulens_eval/core_concepts/rag_triad/),
and [Honest, Harmless and Helpful
Evals](https://www.trulens.org/trulens_eval/core_concepts/honest_harmless_helpful_evals/).

## TruLens in the development workflow

Build your first prototype then connect instrumentation and logging with
TruLens. Decide what feedbacks you need, and specify them with TruLens to run
alongside your app. Then iterate and compare versions of your app in an
easy-to-use user interface 👇

![Architecture
Diagram](https://www.trulens.org/assets/images/TruLens_Architecture.png)

### Installation and Setup

Install the trulens-eval pip package from PyPI.

```bash
pip install trulens-eval
```

#### Installing from Github

To install the latest version from this repository, you can use pip in the following manner:

```bash
pip uninstall trulens_eval -y # to remove existing PyPI version
pip install git+https://github.com/truera/trulens#subdirectory=trulens_eval
```

To install a version from a branch BRANCH, instead use this:

```bash
pip uninstall trulens_eval -y # to remove existing PyPI version
pip install git+https://github.com/truera/trulens@BRANCH#subdirectory=trulens_eval
```


### Quick Usage

Walk through how to instrument and evaluate a RAG built from scratch with
TruLens.

[![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/quickstart.ipynb)

### 💡 Contributing

Interested in contributing? See our [contribution
guide](https://www.trulens.org/trulens_eval/CONTRIBUTING/) for more details.

<!---
end of trulens_eval/gh_top_intro.md
-->

<!---
start of trulens_explain/gh_top_intro.md
NOTE: This content is from trulens_explain/gh_top_intro.md and is merged into
README.md . If you are editing README.md, your changes will be overwritten.
-->

## TruLens-Explain

**TruLens-Explain** is a cross-framework library for deep learning
explainability. It provides a uniform abstraction over a number of different
frameworks. It provides a uniform abstraction layer over TensorFlow, Pytorch,
and Keras and allows input and internal explanations.

### Installation and Setup

These installation instructions assume that you have conda installed and added
to your path.

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

#### Installing from Github

To install the latest version from this repository, you can use pip in the following manner:

```bash
pip uninstall trulens -y # to remove existing PyPI version
pip install git+https://github.com/truera/trulens#subdirectory=trulens_explain
```

To install a version from a branch BRANCH, instead use this:

```bash
pip uninstall trulens -y # to remove existing PyPI version
pip install git+https://github.com/truera/trulens@BRANCH#subdirectory=trulens_explain
```

### Quick Usage

To quickly play around with the TruLens library, check out the following Colab
notebooks:

* PyTorch: [![Open In
  Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n77IGrPDO2XpeIVo_LQW0gY78enV-tY9?usp=sharing)
* TensorFlow 2 / Keras: [![Open In
  Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f-ETsdlppODJGQCdMXG-jmGmfyWyW2VD?usp=sharing)

For more information, see [TruLens-Explain
Documentation](https://www.trulens.org/trulens_explain/quickstart/).

<!---
end of trulens_explain/gh_top_intro.md
-->
