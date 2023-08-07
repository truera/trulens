#!/usr/bin/env python
# coding: utf-8

# # Llama-Index Quickstart
#
# In this quickstart you will create a simple Llama Index App and learn how to log it and get feedback on an LLM response.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/frameworks/llama_index/llama_index_quickstart.ipynb)

# ## Setup
#
# ### Install dependencies
# Let's install some of the dependencies for this notebook if we don't have them already

get_ipython().system(
    ' pip install trulens-eval==0.8.0 llama_index==0.7.0 html2text==2020.1.16'
)

# ### Add API keys
# For this quickstart, you will need Open AI and Huggingface keys

import os

os.environ["OPENAI_API_KEY"] = "..."
os.environ["HUGGINGFACE_API_KEY"] = "..."

# ### Import from LlamaIndex and TruLens

# Imports main tools:
from trulens_eval import Feedback
from trulens_eval import feedback
from trulens_eval import Tru
from trulens_eval import TruLlama

tru = Tru()

# ### Create Simple LLM Application
#
# This example uses LlamaIndex which internally uses an OpenAI LLM.

from llama_index import SimpleWebPageReader
from llama_index import VectorStoreIndex

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

# Initialize Huggingface-based feedback function collection class:
hugs = feedback.Huggingface()
openai = feedback.OpenAI()

# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on_input_output()
# By default this will check language match on the main app input and main app
# output.

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai.qs_relevance).on_input().on(
    TruLlama.select_source_nodes().node.text
).aggregate(np.mean)

# ## Instrument chain for logging with TruLens

tru_query_engine = TruLlama(
    query_engine,
    app_id='LlamaIndex_App1',
    feedbacks=[f_lang_match, f_qa_relevance, f_qs_relevance]
)

# Instrumented query engine can operate like the original:
llm_response = tru_query_engine.query("What did the author do growing up?")

print(llm_response)

# ## Explore in a Dashboard

tru.run_dashboard()  # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed

# Alternatively, you can run `trulens-eval` from a command line in the same folder to start the dashboard.

# ### Leaderboard
#
# Understand how your LLM application is performing at a glance. Once you've set up logging and evaluation in your application, you can view key performance statistics including cost and average feedback value across all of your LLM apps using the chain leaderboard. As you iterate new versions of your LLM application, you can compare their performance across all of the different quality metrics you've set up.
#
# Note: Average feedback values are returned and printed in a range from 0 (worst) to 1 (best).
#
# ![Chain Leaderboard](https://www.trulens.org/Assets/image/Leaderboard.png)
#
# To dive deeper on a particular chain, click "Select Chain".
#
# ### Understand chain performance with Evaluations
#
# To learn more about the performance of a particular chain or LLM model, we can select it to view its evaluations at the record level. LLM quality is assessed through the use of feedback functions. Feedback functions are extensible methods for determining the quality of LLM responses and can be applied to any downstream LLM task. Out of the box we provide a number of feedback functions for assessing model agreement, sentiment, relevance and more.
#
# The evaluations tab provides record-level metadata and feedback on the quality of your LLM application.
#
# ![Evaluations](https://www.trulens.org/Assets/image/Leaderboard.png)
#
# ### Deep dive into full chain metadata
#
# Click on a record to dive deep into all of the details of your chain stack and underlying LLM, captured by tru_chain.
#
# ![Explore a Chain](https://www.trulens.org/Assets/image/Chain_Explore.png)
#
# If you prefer the raw format, you can quickly get it using the "Display full chain json" or "Display full record json" buttons at the bottom of the page.

# Note: Feedback functions evaluated in the deferred manner can be seen in the "Progress" page of the TruLens dashboard.

# ## Or view results directly in your notebook

tru.get_records_and_feedback(app_ids=[]
                            )[0]  # pass an empty list of app_ids to get all
