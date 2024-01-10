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

# ! pip install trulens_eval==0.20.3 openai==1.3.7 langchain chromadb langchainhub bs4

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

# ### Import from LangChain and TruLens

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

# The results of the feedback functions can be rertireved from the record. These
# are `Future` instances (see `concurrent.futures`). You can use `as_completed`
# to wait until they have finished evaluating.

from concurrent.futures import as_completed

for feedback_future in as_completed(rec.feedback_results):
    feedback, feedback_result = feedback_future.result()

    feedback: Feedback
    feedbac_result: FeedbackResult

    print(feedback.name, feedback_result.result)

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

# pip install trulens_eval==0.20.2 llama_index>=0.9.15post2 html2text>=2020.1.16

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

# The record of the app invocation can be retrieved from the `recording`:

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

# # TruLens Quickstart
#
# In this quickstart you will create a RAG from scratch and learn how to log it and get feedback on an LLM response.
#
# For evaluation, we will leverage the "hallucination triad" of groundedness, context relevance and answer relevance.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/quickstart.ipynb)

# ! pip install trulens_eval==0.20.2 chromadb==0.4.18 openai==1.3.7

import os

os.environ["OPENAI_API_KEY"] = "..."

# ## Get Data
#
# In this case, we'll just initialize some simple text in the notebook.

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

from openai import OpenAI

oai_client = OpenAI()

oai_client.embeddings.create(
    model="text-embedding-ada-002", input=university_info
)

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

vector_store.add("uni_info", documents=university_info)

# ## Build RAG from scratch
#
# Build a custom RAG from scratch, and add TruLens custom instrumentation.

from trulens_eval import Tru
from trulens_eval.tru_custom_app import instrument

tru = Tru()


class RAG_from_scratch:

    @instrument
    def retrieve(self, query: str) -> list:
        """
        Retrieve relevant text from vector store.
        """
        results = vector_store.query(query_texts=query, n_results=2)
        return results['documents'][0]

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

import numpy as np

from trulens_eval import Feedback
from trulens_eval import Select
from trulens_eval.feedback import Groundedness
from trulens_eval.feedback.provider.openai import OpenAI as fOpenAI

# Initialize provider class
fopenai = fOpenAI()

grounded = Groundedness(groundedness_provider=fopenai)

# Define a groundedness feedback function
f_groundedness = (
    Feedback(
        grounded.groundedness_measure_with_cot_reasons, name="Groundedness"
    ).on(Select.RecordCalls.retrieve.rets.collect()
        ).on_output().aggregate(grounded.grounded_statements_aggregator)
)

# Question/answer relevance between overall question and answer.
f_qa_relevance = (
    Feedback(fopenai.relevance_with_cot_reasons, name="Answer Relevance").on(
        Select.RecordCalls.retrieve.args.query
    ).on_output()
)

# Question/statement relevance between question and each context chunk.
f_context_relevance = (
    Feedback(fopenai.qs_relevance_with_cot_reasons,
             name="Context Relevance").on(
                 Select.RecordCalls.retrieve.args.query
             ).on(Select.RecordCalls.retrieve.rets.collect()
                 ).aggregate(np.mean)
)

# ## Construct the app
# Wrap the custom RAG with TruCustomApp, add list of feedbacks for eval

from trulens_eval import TruCustomApp

tru_rag = TruCustomApp(
    rag,
    app_id='RAG v1',
    feedbacks=[f_groundedness, f_qa_relevance, f_context_relevance]
)

# ## Run the app
# Use `tru_rag` as a context manager for the custom RAG-from-scratch app.

with tru_rag as recording:
    rag.query("When was the University of Washington founded?")

tru.get_leaderboard(app_ids=["RAG v1"])

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

# ! pip install trulens_eval==0.20.2

from trulens_eval import Feedback
from trulens_eval import Tru

tru = Tru()

tru.run_dashboard()

# ## Set keys

import os

os.environ["OPENAI_API_KEY"] = "..."

# ## Build the app

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

from trulens_eval.feedback.provider.hugs import Dummy

# hugs = Huggingface()
hugs = Dummy()

f_positive_sentiment = Feedback(hugs.positive_sentiment).on_output()

# ## Create the app

# add trulens as a context manager for llm_app with dummy feedback
from trulens_eval import TruCustomApp

tru_app = TruCustomApp(
    llm_app, app_id='LLM App v1', feedbacks=[f_positive_sentiment]
)

# ## Run the app

with tru_app as recording:
    llm_app.completion('give me a good name for a colorful sock company')

tru.get_leaderboard(app_ids=[tru_app.app_id])

# ## Logging Human Feedback
#
# In many situations, it can be useful to log human feedback from your users about your LLM app's performance. Combining human feedback along with automated feedback can help you drill down on subsets of your app that underperform, and uncover new failure modes. This example will walk you through a simple example of recording human feedback with TruLens.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/human_feedback.ipynb)

# ! pip install trulens_eval==0.20.2 openai==1.3.7

import os

from trulens_eval import Tru
from trulens_eval import TruCustomApp

tru = Tru()

# ## Set Keys
#
# For this example, you need an OpenAI key.

os.environ["OPENAI_API_KEY"] = "..."

# ## Set up your app
#
# Here we set up a custom application using just an OpenAI chat completion. The process for logging human feedback is the same however you choose to set up your app.

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

with tru_app as recording:
    llm_app.completion("Give me 10 names for a colorful sock company")

# Get the record to add the feedback to.
record = recording.get()

# ## Create a mechamism for recording human feedback.
#
# Be sure to click an emoji in the record to record `human_feedback` to log.

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

# add the human feedback to a particular app and record
tru.add_feedback(
    name="Human Feedack",
    record_id=record.record_id,
    app_id=tru_app.app_id,
    result=human_feedback
)

# ## See the result logged with your app.

tru.get_leaderboard(app_ids=[tru_app.app_id])

# # Ground Truth Evaluations
#
# In this quickstart you will create a evaluate a LangChain app using ground truth. Ground truth evaluation can be especially useful during early LLM experiments when you have a small set of example queries that are critical to get right.
#
# Ground truth evaluation works by comparing the similarity of an LLM response compared to its matching verified response.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/groundtruth_evals.ipynb)

# ### Add API keys
# For this quickstart, you will need Open AI keys.

# ! pip install trulens_eval==0.20.2 openai==1.3.7

import os

os.environ["OPENAI_API_KEY"] = "sk-..."

from trulens_eval import Tru

tru = Tru()

# ### Create Simple LLM Application

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

# add trulens as a context manager for llm_app
from trulens_eval import TruCustomApp

tru_app = TruCustomApp(llm_app, app_id='LLM App v1', feedbacks=[f_groundtruth])

# Instrumented query engine can operate as a context manager:
with tru_app as recording:
    llm_app.completion("¿quien invento la bombilla?")
    llm_app.completion("who invented the lightbulb?")

# ## See results

tru.get_leaderboard(app_ids=[tru_app.app_id])

# # Logging Methods
#
# ## Automatic Logging
#
# The simplest method for logging with TruLens is by wrapping with TruChain and
# including the tru argument, as shown in the quickstart.
#
# This is done like so:

# Imports main tools:
from trulens_eval import Feedback
from trulens_eval import Huggingface
from trulens_eval import Tru
from trulens_eval import TruChain

tru = Tru()

Tru().migrate_database()

from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import PromptTemplate

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
truchain("This will be automatically logged.")

# Feedback functions can also be logged automatically by providing them in a list
# to the feedbacks arg.

# Initialize Huggingface-based feedback function collection class:
hugs = Huggingface()

# Define a language match feedback function using HuggingFace.
f_lang_match = Feedback(hugs.language_match).on_input_output()
# By default this will check language match on the main app input and main app
# output.

truchain = TruChain(
    chain,
    app_id='Chain1_ChatApplication',
    feedbacks=[f_lang_match],  # feedback functions
    tru=tru
)
truchain("This will be automatically logged.")

# ## Manual Logging
#
# ### Wrap with TruChain to instrument your chain

tc = TruChain(chain, app_id='Chain1_ChatApplication')

# ### Set up logging and instrumentation
#
# Making the first call to your wrapped LLM Application will now also produce a log or "record" of the chain execution.
#

prompt_input = 'que hora es?'
gpt3_response, record = tc.call_with_record(prompt_input)

# We can log the records but first we need to log the chain itself.

tru.add_app(app=truchain)

# Then we can log the record:

tru.add_record(record)

# ### Log App Feedback
# Capturing app feedback such as user feedback of the responses can be added with
# one call.

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

feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[f_lang_match]
)
for result in feedback_results:
    print(result)

# After capturing feedback, you can then log it to your local database.

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

# # Custom Functions
#
# Feedback functions are an extensible framework for evaluating LLMs. You can add your own feedback functions to evaluate the qualities required by your application by updating `trulens_eval/feedback.py`, or simply creating a new provider class and feedback function in youre notebook. If your contributions would be useful for others, we encourage you to contribute to TruLens!
#
# Feedback functions are organized by model provider into Provider classes.
#
# The process for adding new feedback functions is:
# 1. Create a new Provider class or locate an existing one that applies to your feedback function. If your feedback function does not rely on a model provider, you can create a standalone class. Add the new feedback function method to your selected class. Your new method can either take a single text (str) as a parameter or both prompt (str) and response (str). It should return a float between 0 (worst) and 1 (best).

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

standalone = StandAlone()
f_custom_function = Feedback(standalone.custom_feedback
                            ).on(my_text_field=Select.RecordOutput)

# 3. Your feedback function is now ready to use just like the out of the box feedback functions. Below is an example of it being used.

tru = Tru()
feedback_results = tru.run_feedback_functions(
    record=record, feedback_functions=[f_custom_function]
)
tru.add_feedbacks(feedback_results)

# ## Multi-Output Feedback functions
# Trulens also supports multi-output feedback functions. As a typical feedback function will output a float between 0 and 1, multi-output should output a dictionary of `output_key` to a float between 0 and 1. The feedbacks table will print the feedback with column `feedback_name:::outputkey`

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
