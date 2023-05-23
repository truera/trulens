# Welcome to TruLens-Eval!

![TruLens](https://www.trulens.org/Assets/image/Neural_Network_Explainability.png)

TruLens-Eval is a library containing langchain instrumentation and evaluation tools for LLM-based applications. TruLens-Eval supports the iterative development and monitoring of a wide range of LLM applications by  wrapping your application to log key metadata across the entire chain (or off chain if your project does not use chains) on your local machine.

Using feedback functions, you can objectively evaluate the quality of the responses provided by an LLM to your requests. This is completed with minimal latency, as this is achieved in a sequential call for your application, and evaluations are logged to your local machine. Finally, we provide an easy to use streamlit dashboard run locally on your machine for you to better understand your LLM’s performance.

![Architecture Diagram](https://github.com/truera/trulens_private/assets/60949774/3efaba55-06cc-4a2b-b734-6030080bc4fb)

# Quick Usage
To quickly play around with the TruLens Eval library, check out the following CoLab notebooks:

* PyTorch: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rjXCVg4YlaFoVkLK2b3MLgbZbaJhi4dz?usp=share_link)



# Installation and Setup

Install trulens-eval from pypi.

```
pip install trulens-eval
```

Imports from langchain to build app, trulens for evaluation

```python
from IPython.display import JSON
# imports from langchain to build app
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
# imports from trulens to log and get feedback on chain
from trulens_eval import tru
from trulens_eval import tru_chain
tru = Tru()
```

## API Keys

Our example chat app and feedback functions call external APIs such as OpenAI or Huggingface. You can add keys by setting the environment variables. 

### In Python

```python
import os
os.environ["OPENAI_API_KEY"] = "..."
```
### In Terminal

```bash
export OPENAI_API_KEY = "..."
```

## Create a basic LLM chain to evaluate

This example uses langchain and OpenAI, but the same process can be followed with any framework and model provider. Once you've created your chain, just call TruChain to wrap it. Doing so allows you to capture the chain metadata for logging.

```python
full_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template="Provide a helpful response with relevant background information for the following: {prompt}",
            input_variables=["prompt"],
        )
    )
chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])

chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.9)

chain = LLMChain(llm=chat, prompt=chat_prompt_template)

# wrap with truchain to instrument your chain
tc = tru_chain.TruChain(chain)
```

## Set up logging and instrumentation

Make the first call to your LLM Application. The instrumented chain can operate like the original but can also produce a log or "record" of the chain execution.

```python
prompt_input = 'que hora es?'
gpt3_response, record = tc(prompt_input)
```

We can log the records but first we need to log the chain itself.

```python
tru.add_chain(chain_json=truchain.json)
```

Now we can log the record:
```python
tru.add_record(
    prompt=prompt_input, # prompt input
    response=gpt3_response['text'], # LLM response
    record_json=record # record is returned by the TruChain wrapper
)
```

# Evaluate Quality

Following the request to your app, you can then evaluate LLM quality using feedback functions. This is completed in a sequential call to minimize latency for your application, and evaluations will also be logged to your local machine.

To get feedback on the quality of your LLM, you can use any of the provided feedback functions or add your own.

To assess your LLM quality, you can provide the feedback functions to tru.run_feedback() in a list as shown below. Here we'll just add a simple language match checker.
```python
from trulens_eval.tru_feedback import Feedback, Huggingface

os.environ["HUGGINGFACE_API_KEY"] = "..."

# Initialize Huggingface-based feedback function collection class:
hugs = Huggingface()

# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on(
    text1="prompt", text2="response"
)

# Run feedack functions. This might take a moment if the public api needs to load the language model used by the feedback function.
feedback_result = f_lang_match.run_on_record(
    chain_json=truchain.json, record_json=record
)

JSON(feedback_result)

# We can also run a collection of feedback functions
feedback_results = tru.run_feedback_functions(
    record_json=record,
    feedback_functions=[f_lang_match]
)
display(feedback_results)
```

After capturing feedback, you can then log it to your local database
```python
tru.add_feedback(feedback_results)
```

## Automatic logging
The above logging and feedback function evaluation steps can be done by TruChain.
```python
truchain = TruChain(
    chain,
    chain_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match],
    tru=tru
)
# Note: providing `db: TruDB` causes the above constructor to log the wrapped chain in the database specified.
# Note: any `feedbacks` specified here will be evaluated and logged whenever the chain is used.

truchain("This will be automatically logged.")
```

## Out-of-band Feedback evaluation

In the above example, the feedback function evaluation is done in the same process as the chain evaluation. The alternative approach is the use the provided persistent evaluator started via `tru.start_deferred_feedback_evaluator`. Then specify the `feedback_mode` for `TruChain` as `deferred` to let the evaluator handle the feedback functions.

For demonstration purposes, we start the evaluator here but it can be started in another process.
```python
truchain: TruChain = TruChain(
    chain,
    chain_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match],
    tru=tru,
    feedback_mode="deferred"
)

tru.start_evaluator()
truchain("This will be logged by deferred evaluator.")
tru.stop_evaluator()
```

## Run the dashboard!
```python
tru.run_dashboard() # open a streamlit app to explore
# tru.stop_dashboard() # stop if needed
```

## Chain Leaderboard: Quickly identify quality issues.

Understand how your LLM application is performing at a glance. Once you've set up logging and evaluation in your application, you can view key performance statistics across all of your LLM apps using the chain leaderboard. As you iterate new versions of your LLM application, you can compare their performance.

## Understand chain performance with Evaluations
 
To learn more about the performance of a particular chain or LLM model, we can select it to view its evaluations at the record level. LLM quality is assessed through the use of feedback functions. Feedback functions are extensible methods for determining the quality of LLM responses and can be applied to any downstream LLM task. Out of the box we provide a number of feedback functions for assessing truthfulness, sentiment, relevance and more.

The evaluations tab provides record-level metadata and feedback on the quality of your LLM application. Click on a record to dive deep into all of the details of your chain stack and underlying LLM, captured by tru_chain.

## Out-of-the-box Feedback Functions
See: <https://www.trulens.org/trulens_eval/api/tru_feedback/>

### Relevance

This evaluates the *relevance* of the LLM response to the given text by LLM prompting.

Relevance is currently only available with OpenAI ChatCompletion API.

### Sentiment

This evaluates the *positive sentiment* of either the prompt or response.

Sentiment is currently available to use with OpenAI, HuggingFace or Cohere as the model provider.

* The OpenAI sentiment feedback function prompts a Chat Completion model to rate the sentiment from 1 to 10, and then scales the response down to 0-1.
* The HuggingFace sentiment feedback function returns a raw score from 0 to 1.
* The Cohere sentiment feedback function uses the classification endpoint and a small set of examples stored in feedback_prompts.py to return either a 0 or a 1.

### Model Agreement

Model agreement uses OpenAI to attempt an honest answer at your prompt with system prompts for correctness, and then evaluates the aggreement of your LLM response to this model on a scale from 1 to 10. The agreement with each honest bot is then averaged and scaled from 0 to 1.

### Language Match

This evaluates if the language of the prompt and response match.

Language match is currently only available to use with HuggingFace as the model provider. This feedback function returns a score in the range from 0 to 1, where 1 indicates match and 0 indicates mismatch.

### Toxicity

This evaluates the toxicity of the prompt or response.

Toxicity is currently only available to be used with HuggingFace, and uses a classification endpoint to return a score from 0 to 1. The feedback function is negated as not_toxicity, and returns a 1 if not toxic and a 0 if toxic.

### Moderation

The OpenAI Moderation API is made available for use as feedback functions. This includes hate, hate/threatening, self-harm, sexual, sexual/minors, violence, and violence/graphic. Each is negated (ex: not_hate) so that a 0 would indicate that the moderation rule is violated. These feedback functions return a score in the range 0 to 1.

# Contributing new feedback functions

Feedback functions are an extensible framework for evaluating LLMs. You can add your own feedback functions to evaluate the qualities required by your application using the process detailed below:


