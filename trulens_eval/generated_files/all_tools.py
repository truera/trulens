#!/usr/bin/env python
# coding: utf-8

# # 📓 _LangChain_ Quickstart
#
# In this quickstart you will create a simple LLM Chain and learn how to log it and get feedback on an LLM response.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/langchain_quickstart.ipynb)

# ## Setup
# ### Add API keys
# For this quickstart you will need Open AI and Huggingface keys

# In[ ]:

# ! pip install trulens_eval openai langchain chromadb langchainhub bs4 tiktoken

# In[ ]:

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# ### Import from LangChain and TruLens

# In[ ]:

# Imports main tools:
from trulens_eval import Tru
from trulens_eval import TruChain

tru = Tru()
tru.reset_database()

# Imports from LangChain to build app
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

from trulens_eval import Feedback
from trulens_eval.feedback.provider import OpenAI

# Initialize provider class
provider = OpenAI()

# select context to be used in feedback. the location of context is app specific.
from trulens_eval.app import App

context = App.select_context(rag_chain)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons
            ).on(context.collect())  # collect context chunks into a list
    .on_output()
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (Feedback(provider.relevance).on_input_output())
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons
            ).on_input().on(context).aggregate(np.mean)
)

# ## Instrument chain for logging with TruLens

# In[ ]:

tru_recorder = TruChain(
    rag_chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_answer_relevance, f_context_relevance, f_groundedness]
)

# In[ ]:

response, tru_record = tru_recorder.with_record(
    rag_chain.invoke, "What is Task Decomposition?"
)

# In[ ]:

json_like = tru_record.layout_calls_as_app()

# In[ ]:

json_like

# In[ ]:

from ipytree import Node
from ipytree import Tree


def display_call_stack(data):
    tree = Tree()
    tree.add_node(Node('Record ID: {}'.format(data['record_id'])))
    tree.add_node(Node('App ID: {}'.format(data['app_id'])))
    tree.add_node(Node('Cost: {}'.format(data['cost'])))
    tree.add_node(Node('Performance: {}'.format(data['perf'])))
    tree.add_node(Node('Timestamp: {}'.format(data['ts'])))
    tree.add_node(Node('Tags: {}'.format(data['tags'])))
    tree.add_node(Node('Main Input: {}'.format(data['main_input'])))
    tree.add_node(Node('Main Output: {}'.format(data['main_output'])))
    tree.add_node(Node('Main Error: {}'.format(data['main_error'])))

    calls_node = Node('Calls')
    tree.add_node(calls_node)

    for call in data['calls']:
        call_node = Node('Call')
        calls_node.add_node(call_node)

        for step in call['stack']:
            step_node = Node('Step: {}'.format(step['path']))
            call_node.add_node(step_node)
            if 'expanded' in step:
                expanded_node = Node('Expanded')
                step_node.add_node(expanded_node)
                for expanded_step in step['expanded']:
                    expanded_step_node = Node(
                        'Step: {}'.format(expanded_step['path'])
                    )
                    expanded_node.add_node(expanded_step_node)

    return tree


# Usage
tree = display_call_stack(json_like)
tree

# In[ ]:

tree

# In[ ]:

with tru_recorder as recording:
    llm_response = rag_chain.invoke("What is Task Decomposition?")

display(llm_response)

# ## Retrieve records and feedback

# In[ ]:

# The record of the app invocation can be retrieved from the `recording`:

rec = recording.get()  # use .get if only one record
# recs = recording.records # use .records if multiple

display(rec)

# In[ ]:

# The results of the feedback functions can be rertireved from
# `Record.feedback_results` or using the `wait_for_feedback_result` method. The
# results if retrieved directly are `Future` instances (see
# `concurrent.futures`). You can use `as_completed` to wait until they have
# finished evaluating or use the utility method:

for feedback, feedback_result in rec.wait_for_feedback_results().items():
    print(feedback.name, feedback_result.result)

# See more about wait_for_feedback_results:
# help(rec.wait_for_feedback_results)

# In[ ]:

records, feedback = tru.get_records_and_feedback(
    app_ids=["Chain1_ChatApplication"]
)

records.head()

# In[ ]:

tru.get_leaderboard(app_ids=["Chain1_ChatApplication"])

# ## Explore in a Dashboard

# In[ ]:

tru.run_dashboard()  # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed

# Alternatively, you can run `trulens-eval` from a command line in the same folder to start the dashboard.

# Note: Feedback functions evaluated in the deferred manner can be seen in the "Progress" page of the TruLens dashboard.

# # 📓 LlamaIndex Quickstart
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

# In[ ]:

# pip install trulens_eval llama_index openai

# ### Add API keys
# For this quickstart, you will need Open AI and Huggingface keys. The OpenAI key is used for embeddings and GPT, and the Huggingface key is used for evaluation.

# In[ ]:

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# ### Import from TruLens

# In[ ]:

from trulens_eval import Tru

tru = Tru()

# ### Download data
#
# This example uses the text of Paul Graham’s essay, [“What I Worked On”](https://paulgraham.com/worked.html), and is the canonical llama-index example.
#
# The easiest way to get it is to [download it via this link](https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt) and save it in a folder called data. You can do so with the following command:

# In[ ]:

get_ipython().system(
    'wget https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt -P data/'
)

# ### Create Simple LLM Application
#
# This example uses LlamaIndex which internally uses an OpenAI LLM.

# In[ ]:

from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()

# ### Send your first request

# In[ ]:

response = query_engine.query("What did the author do growing up?")
print(response)

# ## Initialize Feedback Function(s)

# In[ ]:

import numpy as np

from trulens_eval import Feedback
from trulens_eval.feedback.provider import OpenAI

# Initialize provider class
provider = OpenAI()

# select context to be used in feedback. the location of context is app specific.
from trulens_eval.app import App

context = App.select_context(query_engine)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(provider.groundedness_measure_with_cot_reasons
            ).on(context.collect())  # collect context chunks into a list
    .on_output().aggregate(provider.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = (Feedback(provider.relevance).on_input_output())
# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(provider.context_relevance_with_cot_reasons
            ).on_input().on(context).aggregate(np.mean)
)

# ## Instrument app for logging with TruLens

# In[ ]:

from trulens_eval import TruLlama

tru_query_engine_recorder = TruLlama(
    query_engine,
    app_id='LlamaIndex_App1',
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
)

# In[ ]:

# or as context manager
with tru_query_engine_recorder as recording:
    query_engine.query("What did the author do growing up?")

# ## Retrieve records and feedback

# In[ ]:

# The record of the app invocation can be retrieved from the `recording`:

rec = recording.get()  # use .get if only one record
# recs = recording.records # use .records if multiple

display(rec)

# In[ ]:

tru.run_dashboard()

# In[ ]:

# The results of the feedback functions can be rertireved from
# `Record.feedback_results` or using the `wait_for_feedback_result` method. The
# results if retrieved directly are `Future` instances (see
# `concurrent.futures`). You can use `as_completed` to wait until they have
# finished evaluating or use the utility method:

for feedback, feedback_result in rec.wait_for_feedback_results().items():
    print(feedback.name, feedback_result.result)

# See more about wait_for_feedback_results:
# help(rec.wait_for_feedback_results)

# In[ ]:

records, feedback = tru.get_records_and_feedback(app_ids=["LlamaIndex_App1"])

records.head()

# In[ ]:

tru.get_leaderboard(app_ids=["LlamaIndex_App1"])

# ## Explore in a Dashboard

# In[ ]:

tru.run_dashboard()  # open a local streamlit app to explore

# tru.stop_dashboard() # stop if needed

# Alternatively, you can run `trulens-eval` from a command line in the same folder to start the dashboard.

# # 📓 TruLens Quickstart
#
# In this quickstart you will create a RAG from scratch and learn how to log it and get feedback on an LLM response.
#
# For evaluation, we will leverage the "hallucination triad" of groundedness, context relevance and answer relevance.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/quickstart.ipynb)

# In[ ]:

# ! pip install trulens_eval chromadb openai

# In[ ]:

import os

os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["HUGGINGFACE_API_KEY"] = "hf_..."

# ## Get Data
#
# In this case, we'll just initialize some simple text in the notebook.

# In[ ]:

university_info = """
The University of Washington, founded in 1861 in Seattle, is a public research university
with over 45,000 students across three campuses in Seattle, Tacoma, and Bothell.
As the flagship institution of the six public universities in Washington state,
UW encompasses over 500 buildings and 20 million square feet of space,
including one of the largest library systems in the world.
"""

# ## Create Vector Store
#
# Create a chromadb vector store in memory.

# In[ ]:

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ.get('OPENAI_API_KEY'),
    model_name="text-embedding-ada-002"
)

chroma_client = chromadb.Client()
vector_store = chroma_client.get_or_create_collection(
    name="Universities", embedding_function=embedding_function
)

# Add the university_info to the embedding database.

# In[ ]:

vector_store.add("uni_info", documents=university_info)

# ## Build RAG from scratch
#
# Build a custom RAG from scratch, and add TruLens custom instrumentation.

# In[ ]:

from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument

tru = Tru()

# In[ ]:

from openai import OpenAI

oai_client = OpenAI()


class RAG_from_scratch:

    @instrument
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = vector_store.query(query_texts=query, n_results=2)
        return results['documents']

    @instrument
    def generate_completion(self, query: str, context_str: list) -> str:
        """
        Generate answer from context.
        """
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content":
                        f"We have provided context information below. \n"
                        f"---------------------\n"
                        f"{context_str}"
                        f"\n---------------------\n"
                        f"Given this information, please answer the question: {query}"
                }
            ]
        ).choices[0].message.content
        return completion

    @instrument
    def query(self, query: str) -> str:
        context_str = self.retrieve(query)
        completion = self.generate_completion(query, context_str)
        return completion


rag = RAG_from_scratch()

# ## Set up feedback functions.
#
# Here we'll use groundedness, answer relevance and context relevance to detect hallucination.

# In[ ]:

import numpy as np

from trulens_eval import Feedback
from trulens_eval import Select
from trulens_eval.feedback.provider.openai import OpenAI

provider = OpenAI()

# Define a groundedness feedback function
f_groundedness = (
    Feedback(
        provider.groundedness_measure_with_cot_reasons, name="Groundedness"
    ).on(Select.RecordCalls.retrieve.rets.collect()).on_output()
)
# Question/answer relevance between overall question and answer.
f_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Answer Relevance").on(
        Select.RecordCalls.retrieve.args.query
    ).on_output()
)

# Context relevance between question and each context chunk.
f_context_relevance = (
    Feedback(
        provider.context_relevance_with_cot_reasons, name="Context Relevance"
    ).on(Select.RecordCalls.retrieve.args.query).on(
        Select.RecordCalls.retrieve.rets
    ).aggregate(np.mean)  # choose a different aggregation method if you wish
)

# ## Construct the app
# Wrap the custom RAG with TruCustomApp, add list of feedbacks for eval

# In[ ]:

from trulens_eval import TruCustomApp

tru_rag = TruCustomApp(
    rag,
    app_id='RAG v1',
    feedbacks=[f_groundedness, f_answer_relevance, f_context_relevance]
)

# ## Run the app
# Use `tru_rag` as a context manager for the custom RAG-from-scratch app.

# In[ ]:

with tru_rag as recording:
    rag.query("When was the University of Washington founded?")

# In[ ]:

tru.get_leaderboard(app_ids=["RAG v1"])

# In[ ]:

tru.run_dashboard()

# # Prototype Evals
# This notebook shows the use of the dummy feedback function provider which
# behaves like the huggingface provider except it does not actually perform any
# network calls and just produces constant results. It can be used to prototype
# feedback function wiring for your apps before invoking potentially slow (to
# run/to load) feedback functions.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/prototype_evals.ipynb)

# ## Import libraries

# In[ ]:

# ! pip install trulens_eval

# In[ ]:

from trulens_eval import Feedback
from trulens_eval import Tru

tru = Tru()

tru.run_dashboard()

# ## Set keys

# In[ ]:

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# ## Build the app

# In[ ]:

from openai import OpenAI

oai_client = OpenAI()

from trulens_eval.tru_custom_app import instrument


class APP:

    @instrument
    def completion(self, prompt):
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": f"Please answer the question: {prompt}"
                }
            ]
        ).choices[0].message.content
        return completion


llm_app = APP()

# ## Create dummy feedback
#
# By setting the provider as `Dummy()`, you can erect your evaluation suite and then easily substitute in a real model provider (e.g. OpenAI) later.

# In[ ]:

from trulens_eval.feedback.provider.hugs import Dummy

# hugs = Huggingface()
hugs = Dummy()

f_positive_sentiment = Feedback(hugs.positive_sentiment).on_output()

# ## Create the app

# In[ ]:

# add trulens as a context manager for llm_app with dummy feedback
from trulens_eval import TruCustomApp

tru_app = TruCustomApp(
    llm_app, app_id='LLM App v1', feedbacks=[f_positive_sentiment]
)

# ## Run the app

# In[ ]:

with tru_app as recording:
    llm_app.completion('give me a good name for a colorful sock company')

# In[ ]:

tru.get_leaderboard(app_ids=[tru_app.app_id])

# # 📓 Logging Human Feedback
#
# In many situations, it can be useful to log human feedback from your users about your LLM app's performance. Combining human feedback along with automated feedback can help you drill down on subsets of your app that underperform, and uncover new failure modes. This example will walk you through a simple example of recording human feedback with TruLens.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/human_feedback.ipynb)

# In[ ]:

# ! pip install trulens_eval openai

# In[ ]:

import os

from trulens_eval import Tru
from trulens_eval import TruCustomApp

tru = Tru()

# ## Set Keys
#
# For this example, you need an OpenAI key.

# In[ ]:

os.environ["OPENAI_API_KEY"] = "sk-..."

# ## Set up your app
#
# Here we set up a custom application using just an OpenAI chat completion. The process for logging human feedback is the same however you choose to set up your app.

# In[ ]:

from openai import OpenAI

oai_client = OpenAI()

from trulens_eval.tru_custom_app import instrument


class APP:

    @instrument
    def completion(self, prompt):
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": f"Please answer the question: {prompt}"
                }
            ]
        ).choices[0].message.content
        return completion


llm_app = APP()

# add trulens as a context manager for llm_app
tru_app = TruCustomApp(llm_app, app_id='LLM App v1')

# ## Run the app

# In[ ]:

with tru_app as recording:
    llm_app.completion("Give me 10 names for a colorful sock company")

# In[ ]:

# Get the record to add the feedback to.
record = recording.get()

# ## Create a mechamism for recording human feedback.
#
# Be sure to click an emoji in the record to record `human_feedback` to log.

# In[ ]:

from ipywidgets import Button
from ipywidgets import HBox
from ipywidgets import VBox

thumbs_up_button = Button(description='👍')
thumbs_down_button = Button(description='👎')

human_feedback = None


def on_thumbs_up_button_clicked(b):
    global human_feedback
    human_feedback = 1


def on_thumbs_down_button_clicked(b):
    global human_feedback
    human_feedback = 0


thumbs_up_button.on_click(on_thumbs_up_button_clicked)
thumbs_down_button.on_click(on_thumbs_down_button_clicked)

HBox([thumbs_up_button, thumbs_down_button])

# In[ ]:

# add the human feedback to a particular app and record
tru.add_feedback(
    name="Human Feedack",
    record_id=record.record_id,
    app_id=tru_app.app_id,
    result=human_feedback
)

# ## See the result logged with your app.

# In[ ]:

tru.get_leaderboard(app_ids=[tru_app.app_id])

# # 📓 Ground Truth Evaluations
#
# In this quickstart you will create a evaluate a _LangChain_ app using ground truth. Ground truth evaluation can be especially useful during early LLM experiments when you have a small set of example queries that are critical to get right.
#
# Ground truth evaluation works by comparing the similarity of an LLM response compared to its matching verified response.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/groundtruth_evals.ipynb)

# ### Add API keys
# For this quickstart, you will need Open AI keys.

# In[ ]:

# ! pip install trulens_eval openai

# In[2]:

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# In[3]:

from trulens_eval import Tru

tru = Tru()

# ### Create Simple LLM Application

# In[4]:

from openai import OpenAI

oai_client = OpenAI()

from trulens_eval.tru_custom_app import instrument


class APP:

    @instrument
    def completion(self, prompt):
        completion = oai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": f"Please answer the question: {prompt}"
                }
            ]
        ).choices[0].message.content
        return completion


llm_app = APP()

# ## Initialize Feedback Function(s)

# In[5]:

from trulens_eval import Feedback
from trulens_eval.feedback import GroundTruthAgreement

golden_set = [
    {
        "query": "who invented the lightbulb?",
        "response": "Thomas Edison"
    }, {
        "query": "¿quien invento la bombilla?",
        "response": "Thomas Edison"
    }
]

f_groundtruth = Feedback(
    GroundTruthAgreement(golden_set).agreement_measure, name="Ground Truth"
).on_input_output()

# ## Instrument chain for logging with TruLens

# In[6]:

# add trulens as a context manager for llm_app
from trulens_eval import TruCustomApp

tru_app = TruCustomApp(llm_app, app_id='LLM App v1', feedbacks=[f_groundtruth])

# In[7]:

# Instrumented query engine can operate as a context manager:
with tru_app as recording:
    llm_app.completion("¿quien invento la bombilla?")
    llm_app.completion("who invented the lightbulb?")

# ## See results

# In[8]:

tru.get_leaderboard(app_ids=[tru_app.app_id])

# # Logging Methods
#
# ## Automatic Logging
#
# The simplest method for logging with TruLens is by wrapping with TruChain and
# including the tru argument, as shown in the quickstart.
#
# This is done like so:

# In[ ]:

# Imports main tools:
from trulens_eval import Feedback
from trulens_eval import Huggingface
from trulens_eval import Tru
from trulens_eval import TruChain

tru = Tru()

Tru().migrate_database()

from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI

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

truchain = TruChain(chain, app_id='Chain1_ChatApplication', tru=tru)
with truchain:
    chain("This will be automatically logged.")

# Feedback functions can also be logged automatically by providing them in a list
# to the feedbacks arg.

# In[ ]:

# Initialize Huggingface-based feedback function collection class:
hugs = Huggingface()

# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on_input_output()
# By default this will check language match on the main app input and main app
# output.

# In[ ]:

truchain = TruChain(
    chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match],  # feedback functions
    tru=tru
)
with truchain:
    chain("This will be automatically logged.")

# ## Manual Logging
#
# ### Wrap with TruChain to instrument your chain

# In[ ]:

tc = TruChain(chain, app_id='Chain1_ChatApplication')

# ### Set up logging and instrumentation
#
# Making the first call to your wrapped LLM Application will now also produce a log or "record" of the chain execution.
#

# In[ ]:

prompt_input = 'que hora es?'
gpt3_response, record = tc.with_record(chain.__call__, prompt_input)

# We can log the records but first we need to log the chain itself.

# In[ ]:

tru.add_app(app=truchain)

# Then we can log the record:

# In[ ]:

tru.add_record(record)

# ### Log App Feedback
# Capturing app feedback such as user feedback of the responses can be added with
# one call.

# In[ ]:

thumb_result = True
tru.add_feedback(
    name="👍 (1) or 👎 (0)", record_id=record.record_id, result=thumb_result
)

# ### Evaluate Quality
#
# Following the request to your app, you can then evaluate LLM quality using
# feedback functions. This is completed in a sequential call to minimize latency
# for your application, and evaluations will also be logged to your local machine.
#
# To get feedback on the quality of your LLM, you can use any of the provided
# feedback functions or add your own.
#
# To assess your LLM quality, you can provide the feedback functions to
# `tru.run_feedback()` in a list provided to `feedback_functions`.
#

# In[ ]:

feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[f_lang_match]
)
for result in feedback_results:
    display(result)

# After capturing feedback, you can then log it to your local database.

# In[ ]:

tru.add_feedbacks(feedback_results)

# ### Out-of-band Feedback evaluation
#
# In the above example, the feedback function evaluation is done in the same
# process as the chain evaluation. The alternative approach is the use the
# provided persistent evaluator started via
# `tru.start_deferred_feedback_evaluator`. Then specify the `feedback_mode` for
# `TruChain` as `deferred` to let the evaluator handle the feedback functions.
#
# For demonstration purposes, we start the evaluator here but it can be started in
# another process.

# In[ ]:

truchain: TruChain = TruChain(
    chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match],
    tru=tru,
    feedback_mode="deferred"
)

with truchain:
    chain("This will be logged by deferred evaluator.")

tru.start_evaluator()
# tru.stop_evaluator()

# # 📓 Custom Feedback Functions
#
# Feedback functions are an extensible framework for evaluating LLMs. You can add your own feedback functions to evaluate the qualities required by your application by updating `trulens_eval/feedback.py`, or simply creating a new provider class and feedback function in youre notebook. If your contributions would be useful for others, we encourage you to contribute to TruLens!
#
# Feedback functions are organized by model provider into Provider classes.
#
# The process for adding new feedback functions is:
# 1. Create a new Provider class or locate an existing one that applies to your feedback function. If your feedback function does not rely on a model provider, you can create a standalone class. Add the new feedback function method to your selected class. Your new method can either take a single text (str) as a parameter or both prompt (str) and response (str). It should return a float between 0 (worst) and 1 (best).

# In[ ]:

from trulens_eval import Feedback
from trulens_eval import Provider
from trulens_eval import Select
from trulens_eval import Tru


class StandAlone(Provider):

    def custom_feedback(self, my_text_field: str) -> float:
        """
        A dummy function of text inputs to float outputs.

        Parameters:
            my_text_field (str): Text to evaluate.

        Returns:
            float: square length of the text
        """
        return 1.0 / (1.0 + len(my_text_field) * len(my_text_field))


# 2. Instantiate your provider and feedback functions. The feedback function is wrapped by the trulens-eval Feedback class which helps specify what will get sent to your function parameters (For example: Select.RecordInput or Select.RecordOutput)

# In[ ]:

standalone = StandAlone()
f_custom_function = Feedback(standalone.custom_feedback
                            ).on(my_text_field=Select.RecordOutput)

# 3. Your feedback function is now ready to use just like the out of the box feedback functions. Below is an example of it being used.

# In[ ]:

tru = Tru()
feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[f_custom_function]
)
tru.add_feedbacks(feedback_results)

# ## Extending existing providers.
#
# In addition to calling your own methods, you can also extend stock feedback providers (such as `OpenAI`, `AzureOpenAI`, `Bedrock`) to custom feedback implementations. This can be especially useful for tweaking stock feedback functions, or running custom feedback function prompts while letting TruLens handle the backend LLM provider.
#
# This is done by subclassing the provider you wish to extend, and using the `generate_score` method that runs the provided prompt with your specified provider, and extracts a float score from 0-1. Your prompt should request the LLM respond on the scale from 0 to 10, then the `generate_score` method will normalize to 0-1.
#
# See below for example usage:

# In[ ]:

from trulens_eval.feedback.provider import AzureOpenAI
from trulens_eval.utils.generated import re_0_10_rating


class Custom_AzureOpenAI(AzureOpenAI):

    def style_check_professional(self, response: str) -> float:
        """
        Custom feedback function to grade the professional style of the resposne, extending AzureOpenAI provider.

        Args:
            response (str): text to be graded for professional style.

        Returns:
            float: A value between 0 and 1. 0 being "not professional" and 1 being "professional".
        """
        professional_prompt = str.format(
            "Please rate the professionalism of the following text on a scale from 0 to 10, where 0 is not at all professional and 10 is extremely professional: \n\n{}",
            response
        )
        return self.generate_score(system_prompt=professional_prompt)


# Running "chain of thought evaluations" is another use case for extending providers. Doing so follows a similar process as above, where the base provider (such as `AzureOpenAI`) is subclassed.
#
# For this case, the method `generate_score_and_reasons` can be used to extract both the score and chain of thought reasons from the LLM response.
#
# To use this method, the prompt used should include the `COT_REASONS_TEMPLATE` available from the TruLens prompts library (`trulens_eval.feedback.prompts`).
#
# See below for example usage:

# In[ ]:

from typing import Dict, Tuple

from trulens_eval.feedback import prompts


class Custom_AzureOpenAI(AzureOpenAI):

    def context_relevance_with_cot_reasons_extreme(
        self, question: str, context: str
    ) -> Tuple[float, Dict]:
        """
        Tweaked version of context relevance, extending AzureOpenAI provider.
        A function that completes a template to check the relevance of the statement to the question.
        Scoring guidelines for scores 5-8 are removed to push the LLM to more extreme scores.
        Also uses chain of thought methodology and emits the reasons.

        Args:
            question (str): A question being asked. 
            context (str): A statement to the question.

        Returns:
            float: A value between 0 and 1. 0 being "not relevant" and 1 being "relevant".
        """

        # remove scoring guidelines around middle scores
        system_prompt = prompts.CONTEXT_RELEVANCE_SYSTEM.replace(
            "- STATEMENT that is RELEVANT to most of the QUESTION should get a score of 5, 6, 7 or 8. Higher score indicates more RELEVANCE.\n\n",
            ""
        )

        user_prompt = str.format(
            prompts.CONTEXT_RELEVANCE_USER, question=question, context=context
        )
        user_prompt = user_prompt.replace(
            "RELEVANCE:", prompts.COT_REASONS_TEMPLATE
        )

        return self.generate_score_and_reasons(system_prompt, user_prompt)


# ## Multi-Output Feedback functions
# Trulens also supports multi-output feedback functions. As a typical feedback function will output a float between 0 and 1, multi-output should output a dictionary of `output_key` to a float between 0 and 1. The feedbacks table will display the feedback with column `feedback_name:::outputkey`

# In[ ]:

multi_output_feedback = Feedback(
    lambda input_param: {
        'output_key1': 0.1,
        'output_key2': 0.9
    }, name="multi"
).on(input_param=Select.RecordOutput)
feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[multi_output_feedback]
)
tru.add_feedbacks(feedback_results)

# In[ ]:

# Aggregators will run on the same dict keys.
import numpy as np

multi_output_feedback = Feedback(
    lambda input_param: {
        'output_key1': 0.1,
        'output_key2': 0.9
    },
    name="multi-agg"
).on(input_param=Select.RecordOutput).aggregate(np.mean)
feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[multi_output_feedback]
)
tru.add_feedbacks(feedback_results)

# In[ ]:


# For multi-context chunking, an aggregator can operate on a list of multi output dictionaries.
def dict_aggregator(list_dict_input):
    agg = 0
    for dict_input in list_dict_input:
        agg += dict_input['output_key1']
    return agg


multi_output_feedback = Feedback(
    lambda input_param: {
        'output_key1': 0.1,
        'output_key2': 0.9
    },
    name="multi-agg-dict"
).on(input_param=Select.RecordOutput).aggregate(dict_aggregator)
feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[multi_output_feedback]
)
tru.add_feedbacks(feedback_results)
