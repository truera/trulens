#!/usr/bin/env python
# coding: utf-8

# # Text to Text Quickstart
#
# In this quickstart you will create a simple text to text application and learn how to log it and get feedback.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/text2text_quickstart.ipynb)

# ! pip install trulens_eval==0.15.1

# ## Setup
# ### Add API keys
# For this quickstart you will need Open AI and Huggingface keys

import os

os.environ["OPENAI_API_KEY"] = "..."
os.environ["HUGGINGFACE_API_KEY"] = "..."

import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# ### Import from TruLens

# Imports main tools:
from trulens_eval import Feedback
from trulens_eval import Huggingface
from trulens_eval import Tru

tru = Tru()

# ### Create Simple Text to Text Application
#
# This example uses a bare bones OpenAI LLM, and a non-LLM just for demonstration purposes.


def llm_standalone(prompt):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role":
                    "system",
                "content":
                    "You are a question and answer bot, and you answer super upbeat."
            }, {
                "role": "user",
                "content": prompt
            }
        ]
    )["choices"][0]["message"]["content"]


import hashlib


def simple_hash_callable(prompt):
    h = hashlib.shake_256(prompt.encode('utf-8'))
    return str(h.hexdigest(20))


# ### Send your first request

prompt_input = "How good is language AI?"
prompt_output = llm_standalone(prompt_input)
prompt_output

simple_hash_callable(prompt_input)

# ## Initialize Feedback Function(s)

# Initialize Huggingface-based feedback function collection class:
hugs = Huggingface()

# Define a sentiment feedback function using HuggingFace.
f_sentiment = Feedback(hugs.positive_sentiment).on_output()

# ## Instrument the callable for logging with TruLens

from trulens_eval import TruBasicApp

tru_llm_standalone_recorder = TruBasicApp(
    llm_standalone, app_id="Happy Bot", feedbacks=[f_sentiment]
)
tru_simple_hash_callable_recorder = TruBasicApp(
    simple_hash_callable, app_id="Hasher", feedbacks=[f_sentiment]
)

with tru_llm_standalone_recorder as recording:
    tru_llm_standalone_recorder.app(prompt_input)

with tru_simple_hash_callable_recorder as recording:
    tru_simple_hash_callable_recorder.app(prompt_input)

# ## Explore in a Dashboard

tru.run_dashboard()  # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed

# Alternatively, you can run `trulens-eval` from a command line in the same folder to start the dashboard.

# ## Or view results directly in your notebook

tru.get_records_and_feedback(app_ids=[]
                            )[0]  # pass an empty list of app_ids to get all
