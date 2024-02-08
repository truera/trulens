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

# ! pip install trulens_eval==0.22.0 openai==1.3.7 langchain chromadb langchainhub bs4

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# ### Import from LangChain and TruLens

# Imports main tools:
from trulens_eval import Feedback
from trulens_eval import Tru
from trulens_eval import TruChain

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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
)

splits = text_splitter.split_documents(docs)

vectorstore = Chroma.from_documents(
    documents=splits, embedding=OpenAIEmbeddings()
)

# ### Create RAG

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

rag_chain.invoke("What is Task Decomposition?")

# ## Initialize Feedback Function(s)

import numpy as np

from trulens_eval.feedback.provider import OpenAI

# Initialize provider class
openai = OpenAI()

# select context to be used in feedback. the location of context is app specific.
from trulens_eval.app import App

context = App.select_context(rag_chain)

from trulens_eval.feedback import Groundedness

grounded = Groundedness(groundedness_provider=OpenAI())
# Define a groundedness feedback function
f_groundedness = (
    Feedback(grounded.groundedness_measure_with_cot_reasons
            ).on(context.collect())  # collect context chunks into a list
    .on_output().aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai.relevance).on_input_output()
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(openai.qs_relevance).on_input().on(context).aggregate(np.mean)
)

# ## Instrument chain for logging with TruLens

tru_recorder = TruChain(
    rag_chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_qa_relevance, f_context_relevance, f_groundedness]
)

with tru_recorder as recording:
    llm_response = rag_chain.invoke("What is Task Decomposition?")

print(llm_response)

# ## Retrieve records and feedback

# The record of the app invocation can be retrieved from the `recording`:

rec = recording.get()  # use .get if only one record
# recs = recording.records # use .records if multiple

print(rec)

# The results of the feedback functions can be rertireved from
# `Record.feedback_results` or using the `wait_for_feedback_result` method. The
# results if retrieved directly are `Future` instances (see
# `concurrent.futures`). You can use `as_completed` to wait until they have
# finished evaluating or use the utility method:

for feedback, feedback_result in rec.wait_for_feedback_results().items():
    print(feedback.name, feedback_result.result)

# See more about wait_for_feedback_results:
# help(rec.wait_for_feedback_results)

records, feedback = tru.get_records_and_feedback(
    app_ids=["Chain1_ChatApplication"]
)

records.head()

tru.get_leaderboard(app_ids=["Chain1_ChatApplication"])

# ## Explore in a Dashboard

tru.run_dashboard()  # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed

# Alternatively, you can run `trulens-eval` from a command line in the same folder to start the dashboard.

# Note: Feedback functions evaluated in the deferred manner can be seen in the "Progress" page of the TruLens dashboard.
