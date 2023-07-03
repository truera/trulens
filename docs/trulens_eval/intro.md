# Welcome to TruLens-Eval!

![TruLens](https://www.trulens.org/Assets/image/Neural_Network_Explainability.png)

Evaluate and track your LLM experiments with TruLens. As you work on your models and prompts TruLens-Eval supports the iterative development and of a wide range of LLM applications by wrapping your application to log key metadata across the entire chain (or off chain if your project does not use chains) on your local machine.

Using feedback functions, you can objectively evaluate the quality of the responses provided by an LLM to your requests. This is completed with minimal latency, as this is achieved in a sequential call for your application, and evaluations are logged to your local machine. Finally, we provide an easy to use Streamlit dashboard run locally on your machine for you to better understand your LLM’s performance.

![Architecture Diagram](https://www.trulens.org/Assets/image/TruLens_Architecture.png)

## Quick Usage

To quickly play around with the TruLens Eval library:

Langchain: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/google-colab/trulens_eval/examples/colab/quickstarts/langchain_quickstart_colab.ipynb)

[langchain_quickstart.ipynb](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.4.0/trulens_eval/examples/quickstart.ipynb).

[langchain_quickstart.py](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.4.0/trulens_eval/examples/quickstart.py).

Llama Index: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/google-colab/trulens_eval/examples/colab/quickstarts/llama_index_quickstart_colab.ipynb)

[llamaindex_quickstart.ipynb](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.4.0/trulens_eval/examples/frameworks/llama_index/llama_index_quickstart.ipynb).

[llamaindex_quickstart.py](https://github.com/truera/trulens/blob/releases/rc-trulens-eval-0.4.0/trulens_eval/examples/llama_index_quickstart.py)

## Installation and Setup

Install the trulens-eval pip package from PyPI.

```bash
    pip install trulens-eval
```

### API Keys

Our example chat app and feedback functions call external APIs such as OpenAI or HuggingFace. You can add keys by setting the environment variables. 

#### In Python

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```

#### In Terminal

```bash
export OPENAI_API_KEY = "..."
```
