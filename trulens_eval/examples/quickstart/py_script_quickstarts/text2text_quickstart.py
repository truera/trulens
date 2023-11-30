#!/usr/bin/env python
# coding: utf-8

# # Text to Text Quickstart
#
# In this quickstart you will create a simple text to text application and learn how to log it and get feedback.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/text2text_quickstart.ipynb)

# ## Setup
# ### Add API keys
# For this quickstart you will need an OpenAI Key.

# In[ ]:

# ! pip install trulens_eval==0.18.1 openai==1.3.1

# In[ ]:

import os

os.environ["OPENAI_API_KEY"] = "..."

# ### Import from TruLens

# In[ ]:

from IPython.display import JSON
# Create openai client
from openai import OpenAI

client = OpenAI()

# Imports main tools:
from trulens_eval import Feedback
from trulens_eval import OpenAI as fOpenAI
from trulens_eval import Tru

tru = Tru()
tru.reset_database()

# ### Create Simple Text to Text Application
#
# This example uses a bare bones OpenAI LLM, and a non-LLM just for demonstration purposes.

# In[ ]:


def llm_standalone(prompt):
    return client.chat.completions.create(
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
    ).choices[0].message.content


# ### Send your first request

# In[ ]:

prompt_input = "How good is language AI?"
prompt_output = llm_standalone(prompt_input)
prompt_output

# ## Initialize Feedback Function(s)

# In[ ]:

# Initialize OpenAI-based feedback function collection class:
fopenai = fOpenAI()

# Define a relevance function from openai
f_relevance = Feedback(fopenai.relevance).on_input_output()

# ## Instrument the callable for logging with TruLens

# In[ ]:

from trulens_eval import TruBasicApp

tru_llm_standalone_recorder = TruBasicApp(
    llm_standalone, app_id="Happy Bot", feedbacks=[f_relevance]
)

# In[ ]:

with tru_llm_standalone_recorder as recording:
    tru_llm_standalone_recorder.app(prompt_input)

# ## Explore in a Dashboard

# In[ ]:

tru.run_dashboard()  # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed

# Alternatively, you can run `trulens-eval` from a command line in the same folder to start the dashboard.

# ## Or view results directly in your notebook

# In[ ]:

tru.get_records_and_feedback(app_ids=[]
                            )[0]  # pass an empty list of app_ids to get all
