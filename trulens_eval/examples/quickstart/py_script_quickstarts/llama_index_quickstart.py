#!/usr/bin/env python
# coding: utf-8

# # 📓 Llama-Index Quickstart
#
# In this quickstart you will create a simple Llama Index app and learn how to log it and get feedback on an LLM response.
#
# For evaluation, we will leverage the "hallucination triad" of groundedness, context relevance and answer relevance.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/llama_index_quickstart.ipynb)

# ## Setup
#
# ### Install dependencies
# Let's install some of the dependencies for this notebook if we don't have them already

# pip install trulens_eval==0.24.0 llama_index

# ### Add API keys
# For this quickstart, you will need Open AI and Huggingface keys. The OpenAI key is used for embeddings and GPT, and the Huggingface key is used for evaluation.

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# ### Import from TruLens

from trulens_eval import Tru

tru = Tru()

# ### Download data
#
# This example uses the text of Paul Graham’s essay, [“What I Worked On”](https://paulgraham.com/worked.html), and is the canonical llama-index example.
#
# The easiest way to get it is to [download it via this link](https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt) and save it in a folder called data. You can do so with the following command:

get_ipython().system(
    'wget https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/paul_graham/paul_graham_essay.txt -P data/'
)

# ### Create Simple LLM Application
#
# This example uses LlamaIndex which internally uses an OpenAI LLM.

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

# ### Send your first request

response = query_engine.query("What did the author do growing up?")
print(response)

# ## Initialize Feedback Function(s)

import numpy as np

# Initialize provider class
from trulens_eval.feedback.provider.openai import OpenAI

openai = OpenAI()

# select context to be used in feedback. the location of context is app specific.
from trulens_eval.app import App

context = App.select_context(query_engine)

# imports for feedback
from trulens_eval import Feedback
# Define a groundedness feedback function
from trulens_eval.feedback import Groundedness

grounded = Groundedness(groundedness_provider=OpenAI())
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons
            ).on(context.collect())  # collect context chunks into a list
    .on_output().aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = (
    Feedback(openai.qs_relevance).on_input().on(context).aggregate(np.mean)
)

# ## Instrument app for logging with TruLens

from trulens_eval import TruLlama

tru_query_engine_recorder = TruLlama(
    query_engine,
    app_id='LlamaIndex_App1',
    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance]
)

# or as context manager
with tru_query_engine_recorder as recording:
    query_engine.query("What did the author do growing up?")

# ## Retrieve records and feedback

# The record of the app invocation can be retrieved from the `recording`:

rec = recording.get()  # use .get if only one record
# recs = recording.records # use .records if multiple

print(rec)

tru.run_dashboard()

# The results of the feedback functions can be rertireved from
# `Record.feedback_results` or using the `wait_for_feedback_result` method. The
# results if retrieved directly are `Future` instances (see
# `concurrent.futures`). You can use `as_completed` to wait until they have
# finished evaluating or use the utility method:

for feedback, feedback_result in rec.wait_for_feedback_results().items():
    print(feedback.name, feedback_result.result)

# See more about wait_for_feedback_results:
# help(rec.wait_for_feedback_results)

records, feedback = tru.get_records_and_feedback(app_ids=["LlamaIndex_App1"])

records.head()

tru.get_leaderboard(app_ids=["LlamaIndex_App1"])

# ## Explore in a Dashboard

tru.run_dashboard()  # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed

# Alternatively, you can run `trulens-eval` from a command line in the same folder to start the dashboard.
