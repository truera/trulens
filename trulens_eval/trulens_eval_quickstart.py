#!/usr/bin/env python
# coding: utf-8

# # TruLens for LLMs: Quickstart
# 
# In this quickstart you will create a simple LLM Chain and learn how to log it and get feedback on an LLM response.

# ## Add API keys
# For this quickstart you will need Open AI and Huggingface keys

import os
os.environ["OPENAI_API_KEY"] = "..."
os.environ["HUGGINGFACE_API_KEY"] = "..."

# ## Import from LangChain and TruLens

# Imports main tools:
from trulens_eval import TruChain, Feedback, Huggingface, Tru
tru = Tru()

# imports from langchain to build app
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts.chat import ChatPromptTemplate, PromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate

# ## Create Simple LLM Application
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

# ## Send your first request to your new app, asking what time it is in spanish

prompt_input = '¿que hora es?'

gpt3_response = chain(prompt_input)

print(gpt3_response)

# # Instrument chain for logging with TruLens

truchain: TruChain = TruChain(chain, chain_id='Chain1_ChatApplication')

# Instrumented chain can operate like the original:

gpt3_response = truchain(prompt_input)

print(gpt3_response)

# But can also produce a log or "record" of the execution of the chain:

gpt3_response, record = truchain.call_with_record(prompt_input)

# We can log the records but first we need to log the chain itself:

tru.add_chain(chain_json=truchain.json)

# Now the record:

tru.add_record(
    prompt=prompt_input, # prompt input
    response=gpt3_response['text'], # LLM response
    record_json=record # record is returned by the TruChain wrapper
)

# Note that the `add_record` call automatically sets the `record_id` field of the
# `record_json` to the returned record id. Retrieving it from the output of `add_record` is not 
# necessary.

# Initialize Huggingface-based feedback function collection class:
hugs = Huggingface()

# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on(
    text1="prompt", text2="response"
)

# This might take a moment if the public api needs to load the language model
# used in the feedback function:
feedback_result = f_lang_match.run_on_record(
    chain_json=truchain.json, record_json=record
)

# Alternatively, run a collection of feedback functions:

feedback_results = tru.run_feedback_functions(
    record_json=record,
    feedback_functions=[f_lang_match]
)

print(feedback_results)

# These can be logged:

tru.add_feedbacks(feedback_results)

# ## Run the TruLens dashboard to explore the quality of your LLM chain

tru.run_dashboard() # open a local streamlit app to explore

# tru.run_dashboard(_dev=True) # if running from repo
# tru.stop_dashboard() # stop if needed

# ## Automatic Logging
# 
# The above logging and feedback function evaluation steps can be done by TruChain.

truchain: TruChain = TruChain(
    chain,
    chain_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match],
    tru=tru
)
# or tru.Chain(...)

# Note: providing `db: TruDB` causes the above constructor to log the wrapped chain in the database specified.

# Note: any `feedbacks` specified here will be evaluated and logged whenever the chain is used.

truchain("This will be automatically logged.")

# ## Out-of-band Feedback evaluation
# 
# In the above example, the feedback function evaluation is done in the same process as the chain evaluation. The alternative approach is the use the provided persistent evaluator started via `tru.start_deferred_feedback_evaluator`. Then specify the `feedback_mode` for `TruChain` as `deferred` to let the evaluator handle the feedback functions.
# 
# For demonstration purposes, we start the evaluator here but it can be started in another process.

truchain: TruChain = TruChain(
    chain,
    chain_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match],
    tru=tru,
    feedback_mode="deferred"
)
# or tru.Chain(...)

tru.start_evaluator()

truchain("This will be logged by deferred evaluator.")

# Feedback functions evaluated in the deferred manner can be seen in the "Progress" page of the TruLens dashboard.
tru.stop_evaluator()

# Process will continue to for dashboard serving

