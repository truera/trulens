# Welcome to TruLens!

![TruLens](https://www.trulens.org/Assets/image/Neural_Network_Explainability.png)

TruLens provides a set of tools for developing and monitoring neural nets, including large language models. This includes both tools for evaluation of LLMs and LLM-based applications with TruLens-Eval and deep learning explainability with TruLens-Explain. TruLens-Eval and TruLens-Explain are housed in separate packages and can be used independently.

## TruLens-Eval

**TruLens-Eval** contains instrumentation and evaluation tools for large language model (LLM) based applications. It supports the iterative development and monitoring of a wide range of LLM applications by wrapping your application to log key metadata across the entire chain (or off chain if your project does not use chains) on your local machine. Importantly, it also gives you the tools you need to evaluate the quality of your LLM-based applications.

![Architecture Diagram](docs/Assets/image/TruLens_Architecture.png)

### Get going with TruLens-Eval

Install trulens-eval from pypi.

```
pip install trulens-eval
```


```python
from trulens_eval import tru
from trulens_eval import tru_chain
tru = Tru()
```

This example uses langchain and OpenAI, but the same process can be followed with any framework and model provider.

```python
# imports from langchain to build app
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate

# Set your API key as an environmental variable
import os
os.environ["OPENAI_API_KEY"] = "..."

# create LLM chain
full_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="Provide a helpful response with relevant background information for the following: {prompt}",
            input_variables=["prompt"],
        )
    )
chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)

chain = LLMChain(llm=chat, prompt=chat_prompt_template)
```

Now that we created an LLM chain, we can set up our first feedback function. Here, we'll create a feedback function for language matching. After we've created the feedback function, we can include it in the TruChain wrapper. Now, whenever our wrapped chain is used we'll log both the metadata and feedback.

```python
# create a feedback function

from trulens_eval.tru_feedback import Feedback, Huggingface

os.environ["HUGGINGFACE_API_KEY"] = "..."

# Initialize Huggingface-based feedback function collection class:
hugs = Huggingface()

# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on(
    text1="prompt", text2="response"
)

# wrap your chain with TruChain
truchain = TruChain(
    chain,
    chain_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match],
    tru=tru
)
# Note: any `feedbacks` specified here will be evaluated and logged whenever the chain is used.
truchain("que hora es?")
```

For more information, see [TruLens-Eval Documentation](trulens_eval/quickstart.md).

## TruLens-Eval

**TruLens-Explain** is a cross-framework library for deep learning explainability. It provides a uniform abstraction over a number of different frameworks. It provides a uniform abstraction layer over TensorFlow, Pytorch, and Keras and allows input and internal explanations.

## Getting going with TruLens-Explain

These installation instructions assume that you have conda installed and added to your path.

0. Create a virtual environment (or modify an existing one).
```
conda create -n "<my_name>" python=3.7  # Skip if using existing environment.
conda activate <my_name>
```
 
1. Install dependencies.
```
conda install tensorflow-gpu=1  # Or whatever backend you're using.
conda install keras             # Or whatever backend you're using.
conda install matplotlib        # For visualizations.
```

2. [Pip installation] Install the trulens pip package.
```
pip install trulens
```

3. Get started!
To quickly play around with the TruLens library, check out the following CoLab notebooks:

* PyTorch: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1n77IGrPDO2XpeIVo_LQW0gY78enV-tY9?usp=sharing)
* Tensorflow 2 / Keras: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1f-ETsdlppODJGQCdMXG-jmGmfyWyW2VD?usp=sharing)
