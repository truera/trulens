# Welcome to TruLens-Eval!

![TruLens](https://www.trulens.org/Assets/image/Neural_Network_Explainability.png)

Evaluate and track your LLM experiments with TruLens. As you work on your models and prompts TruLens-Eval supports the iterative development and of a wide range of LLM applications by wrapping your application to log key metadata across the entire chain (or off chain if your project does not use chains) on your local machine.

Using feedback functions, you can objectively evaluate the quality of the responses provided by an LLM to your requests. This is completed with minimal latency, as this is achieved in a sequential call for your application, and evaluations are logged to your local machine. Finally, we provide an easy to use Streamlit dashboard run locally on your machine for you to better understand your LLM’s performance.

![Architecture Diagram](https://www.trulens.org/Assets/image/TruLens_Architecture.png)

## Quick Usage
To quickly play around with the TruLens Eval library, download this notebook: [quickstart.ipynb](https://github.com/truera/trulens/blob/releases/rc-0.1.2/docs/trulens_eval/quickstart.ipynb).



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
