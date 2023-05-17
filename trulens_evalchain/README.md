# Welcome to TruLens-EvalChain!

![TruLens](https://www.trulens.org/Assets/image/Neural_Network_Explainability.png)

TruLens-EvalChain is a library containing langchain instrumentation and evaluation tools for LLM-based applications. TruLens-EvalChain supports the iterative development and monitoring of a wide range of LLM applications by  wrapping your application to log key metadata across the entire chain (or off chain if your project does not use chains) on your local machine.

Using feedback functions, you can objectively evaluate the quality of the responses provided by an LLM to your requests. This is completed with minimal latency, as this is achieved in a sequential call for your application, and evaluations are logged to your local machine. Finally, we provide an easy to use streamlit dashboard run locally on your machine for you to better understand your LLM’s performance.

![Architecture Diagram](https://github.com/truera/llm-experiments/assets/60949774/482d68b9-5387-4f23-9d42-2dc0f0b67c6f)

# Installation

Install trulens-evalchain package from pypi:

```
pip install trulens-evalchain
```

In your application, you will need to import the following, dependent on your application requirements:

### If only using logging:
```python
from trulens_evalchain import tru
```

If your application is chain-based and you wish to capture chain metadata:
```python
from trulens_evalchain import tru
from trulens_evalchain import tru_chain
```

If also using evaluations:
```python
from trulens_evalchain import tru
from trulens_evalchain import tru_chain
from trulens_evalchain tru_feedback
```

## Set up logging and instrumentation

First, you need to create an empty database to store your logs. You can do this in the command line with the following:

```python
import trulens_evalchain
from trulens_evalchain import tru
tru.init_db('llm_quality') 
```

If your application is chain-based, you will need to wrap it using tru_chain. This is done like so:

```python
# create the chain
chain = LLMChain(llm=chat, prompt=chat_prompt_template)
# wrap it with tru_chain
tc = tru_chain.TruChain(chain)
```

Once you've done so, this metadata will be returned by your application along with the LLM response like so:

```python
llm_response, chain_details = generate_response(prompt_input, model_name)
```

You can then log information to a database on your using tru.add_data with each call to the application. Note: make sure to capture the record_id returned by add_data so that the LLM evaluation can be included later.

```python
record_id = tru.add_data(
        chain_id='Chain1_ChatApplication', # name of the chain/application version
        prompt=prompt_input, #LLM prompt
        response=llm_response, #LLM response
        details=chain_details
    )
```

## Capturing Tokens and Cost (OpenAI only)

If you are using OpenAI as your LLM provider, you can capture the total tokens and total cost of each LLM call and then pass them through to tru.add_data() to be logged.

To capture total_tokens and total_cost, simply wrap your LLM request with get_openai_callback() like shown below:

```python
with get_openai_callback() as cb:
        llm_response, record = generate_response(prompt_input, model_name)
        total_tokens = cb.total_tokens
        total_cost = cb.total_cost
```
## Evaluate LLM Quality

Following the request to your app, you can then evaluate LLM quality using feedback functions. This is completed in a sequential call to minimize latency for your application, and evaluations will also be logged to your local machine.

To get feedback on the quality of your LLM, you can use any of the provided feedback functions or add your own.

To assess your LLM quality, you can provide the feedback functions to tru.run_feedback() in a list as shown below:
```python
feedback = tru.run_feedback_function(
        prompt_input, llm_response, [
            tru_feedback.get_factagreement_function(
                evaluation_choice='both',
                provider='openai',
                model_engine='gpt-3.5-turbo'
            ),
            ... # add more feedback functions in a list
        ]
    )
```

After capturing feedback, you can then log it to your local database using tru.add_feedback()
```python
tru.add_feedback(record_id, feedback)
```

# Overview

## Chain Leaderboard

Understand how your LLM application is performing at a glance. Once you've set up logging and evaluation in your application, you can view key performance statistics across all of your LLM apps using the chain leaderboard. As you iterate new versions of your LLM application, you can compare their performance.

## Evaluations
 
To learn more about the performance of a particular chain or LLM model, we can select it to view its evaluations at the record level. LLM quality is assessed through the use of feedback functions. Feedback functions are extensible methods for determining the quality of LLM responses and can be applied to any downstream LLM task. Out of the box we provide a number of feedback functions for assessing truthfulness, sentiment, relevance and more.

The evaluations tab provides record-level metadata and feedback on the quality of your LLM application. Click on a record to dive deep into all of the details of your chain stack and underlying LLM, captured by tru_chain.

## Out-of-the-box Feedback Functions

### Relevance

This evaluates the *relevance* of the LLM response to the given text by LLM prompting.

Relevance is currently only available with OpenAI ChatCompletion API.

### Sentiment

This evaluates the *positive sentiment* of either the prompt or response.

Sentiment is currently available to use with OpenAI, HuggingFace or Cohere as the model provider.

* The OpenAI sentiment feedback function prompts a Chat Completion model to rate the sentiment from 1 to 10, and then scales the response down to 0-1.
* The HuggingFace sentiment feedback function returns a raw score from 0 to 1.
* The Cohere sentiment feedback function uses the classification endpoint and a small set of examples stored in feedback_prompts.py to return either a 0 or a 1.

### Fact Agreement

This evaluates the *truthfulness* of the response...

### Disinformation

This evaluates the prompt or response to determine if it is likely to be disinformation.

Disinformation is currently only available to be used with Cohere, and uses the classification endpoint and a small set of examples stored in feedback_prompts.py to return a 0 or 1.

### Toxicity

This evaluates the toxicity of the prompt or response.

Toxicity is currently only available to be used with HuggingFace, and uses a classification endpoint to return a score from 0 to 1. The feedback function is negated as not_toxicity, and returns a 1 if not toxic and a 0 if toxic.

### Moderation

The OpenAI Moderation API is made available for use as feedback functions. This includes hate, hate/threatening, self-harm, sexual, sexual/minors, violence, and violence/graphic. Each is negated (ex: not_hate) so that a 0 would indicate that the moderation rule is violated. These feedback functions return a score in the range 0 to 1.

# Contributing new feedback functions

Feedback functions are an extensible framework for evaluating LLMs. You can add your own feedback functions to evaluate the qualities required by your application using the process detailed below:

1. Add a new function to tru_feedback that takes in prompt, response, evaluation_choice and model_engine and returns a feedback value. The new function should be named using the convention: <model_descriptor>_<quality_being_evaluated>.
2. If the function falls under an existing quality, you can simply extend the applicable factory function named with the convention: get_<quality_being_evaluated>_function.
3. If the function falls under a new LLM quality you wish to evaluate, you will need to create a new factory function wrapper in which to place it. The factory function you create should take in the provider, model_engine, evaluation_choice and output the feedback function created in step 1.

Template for adding new feedback functions:

```python
def <provider>_<quality>(prompt, response, evaluation_choice):
    return # some function that takes text as input and returns a value, potentially dependent on parameters selected

def get_<quality>_function(provider, model_engine, evaluation_choice):

    def <provider>_<quality>_function(prompt, response):
        return <provider>_<quality>(prompt, response, model_engine, evaluation_choice)

    return <provider>_<quality>_function
```
