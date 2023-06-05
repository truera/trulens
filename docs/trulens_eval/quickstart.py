#!/usr/bin/env python
# coding: utf-8

# # Quickstart
#
# In this quickstart you will create a simple LLM Chain and learn how to log it and get feedback on an LLM response.

# ## Setup
# ### Add API keys
# For this quickstart you will need Open AI and Huggingface keys

import os

os.environ["OPENAI_API_KEY"] = "..."
os.environ["HUGGINGFACE_API_KEY"] = "..."

# ### Import from LangChain and TruLens

# Imports main tools:
from trulens_eval import TruChain, Feedback, Huggingface, Tru

tru = Tru()

# imports from langchain to build app
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts.chat import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate

# ### Create Simple LLM Application
#
# This example uses a LangChain framework and OpenAI LLM

full_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        template=
        "Provide a helpful response with relevant background information for the following: {prompt}",
        input_variables=["prompt"],
    )
)

chat_prompt_template = ChatPromptTemplate.from_messages([full_prompt])

llm = OpenAI(temperature=0.9, max_tokens=128)

chain = LLMChain(llm=llm, prompt=chat_prompt_template, verbose=True)

# ### Send your first request

prompt_input = '¿que hora es?'

llm_response = chain(prompt_input)

print(llm_response)

# ## Initialize Feedback Function(s)

# Initialize Huggingface-based feedback function collection class:
hugs = Huggingface()

# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on(
    text1="prompt", text2="response"
)

# ## Instrument chain for logging with TruLens

truchain = TruChain(
    chain, chain_id='Chain3_ChatApplication', feedbacks=[f_lang_match], tru=tru
)

# Instrumented chain can operate like the original:
llm_response = truchain(prompt_input)

print(llm_response)

# ## Explore in a Dashboard

tru.run_dashboard()  # open a local streamlit app to explore

# tru.run_dashboard(_dev=True) # if running from repo
# tru.stop_dashboard() # stop if needed

# ### Chain Leaderboard
#
# Understand how your LLM application is performing at a glance. Once you've set up logging and evaluation in your application, you can view key performance statistics including cost and average feedback value across all of your LLM apps using the chain leaderboard. As you iterate new versions of your LLM application, you can compare their performance across all of the different quality metrics you've set up.
#
# Note: Average feedback values are returned and printed in a range from 0 (worst) to 1 (best).
#
# ![Chain Leaderboard](Assets/image/Leaderboard.png)
#
# To dive deeper on a particular chain, click "Select Chain".
#
# ### Understand chain performance with Evaluations
#
# To learn more about the performance of a particular chain or LLM model, we can select it to view its evaluations at the record level. LLM quality is assessed through the use of feedback functions. Feedback functions are extensible methods for determining the quality of LLM responses and can be applied to any downstream LLM task. Out of the box we provide a number of feedback functions for assessing model agreement, sentiment, relevance and more.
#
# The evaluations tab provides record-level metadata and feedback on the quality of your LLM application.
#
# ![Evaluations](Assets/image/Leaderboard.png)
#
# ### Deep dive into full chain metadata
#
# Click on a record to dive deep into all of the details of your chain stack and underlying LLM, captured by tru_chain.
#
# ![Explore a Chain](Assets/image/Chain_Explore.png)
#
# If you prefer the raw format, you can quickly get it using the "Display full chain json" or "Display full record json" buttons at the bottom of the page.

# Note: Feedback functions evaluated in the deferred manner can be seen in the "Progress" page of the TruLens dashboard.

# ## Or view results directly in your notebook

tru.get_records_and_feedback(chain_ids=[]
                            )[0]  # pass an empty list of chain_ids to get all
