#!/usr/bin/env python
# coding: utf-8

# # Langchain Quickstart
#
# In this quickstart you will create a simple LLM Chain and learn how to log it and get feedback on an LLM response.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/langchain_quickstart.ipynb)

# ## Setup
# ### Add API keys
# For this quickstart you will need Open AI and Huggingface keys

# In[ ]:

# ! pip install trulens_eval==0.19.1 openai==1.3.7 langchain chromadb langchainhub bs4

# In[ ]:

import os

os.environ["OPENAI_API_KEY"] = "..."

# ### Import from LangChain and TruLens

# In[ ]:

from IPython.display import JSON

# Imports main tools:
from trulens_eval import Feedback
from trulens_eval import Huggingface
from trulens_eval import Tru
from trulens_eval import TruChain
from trulens_eval.schema import FeedbackResult

tru = Tru()
tru.reset_database()

# Imports from langchain to build app
import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

# ### Load documents

# In[ ]:

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# ### Create Vector Store

# In[ ]:

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)
splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)

# ### Create RAG

# In[ ]:

retriever = vectorstore.as_retriever()

prompt = hub.pull("rlm/rag-prompt")
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    } | prompt | llm | StrOutputParser()
)

# ### Send your first request

# In[ ]:

rag_chain.invoke("What is Task Decomposition?")

# ## Initialize Feedback Function(s)

# In[ ]:

import numpy as np

from trulens_eval import Select
from trulens_eval.feedback.provider import OpenAI

# Initialize provider class
openai = OpenAI()
from trulens_eval.feedback import Groundedness

grounded = Groundedness(groundedness_provider=OpenAI())
# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons).on(
        Select.RecordCalls.first.invoke.rets.context
    ).on_output().aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(openai.qs_relevance).on(
        Select.RecordCalls.first.invoke.args.input
    ).on(Select.RecordCalls.first.invoke.rets.context).aggregate(np.mean)
)

# ## Instrument chain for logging with TruLens

# In[ ]:

tru_recorder = TruChain(
    rag_chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_qa_relevance, f_context_relevance, f_groundedness]
)

# In[ ]:

with tru_recorder as recording:
    llm_response = rag_chain.invoke("What is Task Decomposition?")

display(llm_response)

# In[ ]:

tru.run_dashboard()

# ## Retrieve records and feedback

# In[ ]:

# The record of the ap invocation can be retrieved from the `recording`:

rec = recording.get()  # use .get if only one record
# recs = recording.records # use .records if multiple

display(rec)

# In[ ]:

# The results of the feedback functions can be rertireved from the record. These
# are `Future` instances (see `concurrent.futures`). You can use `as_completed`
# to wait until they have finished evaluating.

from concurrent.futures import as_completed

for feedback_future in as_completed(rec.feedback_results):
    feedback, feedback_result = feedback_future.result()

    feedback: Feedback
    feedbac_result: FeedbackResult

    display(feedback.name, feedback_result.result)

# ## Explore in a Dashboard

# In[ ]:

tru.run_dashboard()  # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed

# Alternatively, you can run `trulens-eval` from a command line in the same folder to start the dashboard.

# Note: Feedback functions evaluated in the deferred manner can be seen in the "Progress" page of the TruLens dashboard.

# ## Or view results directly in your notebook

# In[ ]:

tru.get_records_and_feedback(app_ids=[]
                            )[0]  # pass an empty list of app_ids to get all
