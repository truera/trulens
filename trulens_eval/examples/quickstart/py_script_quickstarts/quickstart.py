#!/usr/bin/env python
# coding: utf-8

# # TruLens Quickstart
#
# In this quickstart you will create a RAG from scratch and learn how to log it and get feedback on an LLM response.
#
# For evaluation, we will leverage the "hallucination triad" of groundedness, context relevance and answer relevance.
#
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/truera/trulens/blob/main/trulens_eval/examples/quickstart/quickstart.ipynb)

# ! pip install trulens_eval==0.20.1 chromadb==0.4.18 openai==1.3.7

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
