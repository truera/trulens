#!/usr/bin/env python
# coding: utf-8

# # Llama-Index Quickstart
#
# In this quickstart you will create a simple Llama Index App and learn how to log it and get feedback on an LLM response.
#
# For evaluation, we will leverage the "hallucination triad" of groundedness, context relevance and answer relevance.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/llama_index_quickstart.ipynb)

# ## Setup
#
# ### Install dependencies
# Let's install some of the dependencies for this notebook if we don't have them already

# pip install trulens_eval==0.20.0 llama_index>=0.9.15post2 html2text>=2020.1.16

# ### Add API keys
# For this quickstart, you will need Open AI and Huggingface keys. The OpenAI key is used for embeddings and GPT, and the Huggingface key is used for evaluation.

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# ### Import from TruLens

from trulens_eval import Tru

tru = Tru()

# ### Create Simple LLM Application
#
# This example uses LlamaIndex which internally uses an OpenAI LLM.

from llama_index import VectorStoreIndex
from llama_index.readers.web import SimpleWebPageReader

documents = SimpleWebPageReader(html_to_text=True).load_data(
    ["http://paulgraham.com/worked.html"]
)
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

# The record of the ap invocation can be retrieved from the `recording`:

rec = recording.get()  # use .get if only one record
# recs = recording.records # use .records if multiple

print(rec)

# The results of the feedback functions can be rertireved from the record. These
# are `Future` instances (see `concurrent.futures`). You can use `as_completed`
# to wait until they have finished evaluating.

from concurrent.futures import as_completed

from trulens_eval.schema import FeedbackResult

for feedback_future in as_completed(rec.feedback_results):
    feedback, feedback_result = feedback_future.result()

    feedback: Feedback
    feedbac_result: FeedbackResult

    print(feedback.name, feedback_result.result)

records, feedback = tru.get_records_and_feedback(app_ids=["LlamaIndex_App1"])

records.head()

tru.get_leaderboard(app_ids=["LlamaIndex_App1"])

# ## Explore in a Dashboard

tru.run_dashboard()  # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed

# Alternatively, you can run `trulens-eval` from a command line in the same folder to start the dashboard.

# Note: Feedback functions evaluated in the deferred manner can be seen in the "Progress" page of the TruLens dashboard.
