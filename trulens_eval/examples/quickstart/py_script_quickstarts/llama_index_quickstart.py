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

# In[ ]:


#! pip install trulens-eval==0.15.1 llama_index>=0.8.29post1 html2text>=2020.1.16


# ### Add API keys
# For this quickstart, you will need Open AI and Huggingface keys. The OpenAI key is used for embeddings and GPT, and the Huggingface key is used for evaluation.

# In[ ]:


import os
os.environ["OPENAI_API_KEY"] = "..."


# ### Import from LlamaIndex and TruLens

# In[ ]:


from trulens_eval import Feedback, Tru, TruLlama
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI

tru = Tru()


# ### Create Simple LLM Application
# 
# This example uses LlamaIndex which internally uses an OpenAI LLM.

# In[ ]:


from llama_index import VectorStoreIndex, SimpleWebPageReader

documents = SimpleWebPageReader(
    html_to_text=True
).load_data(["http://paulgraham.com/worked.html"])
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()


# ### Send your first request

# In[ ]:


response = query_engine.query("What did the author do growing up?")
print(response)


# ## Initialize Feedback Function(s)

# In[ ]:


import numpy as np

# Initialize provider class
openai = OpenAI()

grounded = Groundedness(groundedness_provider=OpenAI())

# Define a groundedness feedback function
f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons).on(
    TruLlama.select_source_nodes().node.text
    ).on_output(
    ).aggregate(grounded.grounded_statements_aggregator)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai.qs_relevance).on_input().on(
    TruLlama.select_source_nodes().node.text
    ).aggregate(np.mean)


# ## Instrument app for logging with TruLens

# In[ ]:


tru_query_engine_recorder = TruLlama(query_engine,
    app_id='LlamaIndex_App1',
    feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance])


# In[ ]:


# or as context manager
with tru_query_engine_recorder as recording:
    query_engine.query("What did the author do growing up?")


# ## Explore in a Dashboard

# In[ ]:


tru.run_dashboard() # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed


# Alternatively, you can run `trulens-eval` from a command line in the same folder to start the dashboard.

# Note: Feedback functions evaluated in the deferred manner can be seen in the "Progress" page of the TruLens dashboard.

# ## Or view results directly in your notebook

# In[ ]:


tru.get_records_and_feedback(app_ids=[])[0] # pass an empty list of app_ids to get all

