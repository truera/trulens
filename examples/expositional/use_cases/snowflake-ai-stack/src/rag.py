from src.generation import ChatModel
from langchain_core.vectorstores import VectorStore
from trulens.apps.app import instrument
from trulens.core import Feedback
from trulens.core.guardrails.base import context_filter
from trulens.providers.openai import OpenAI
import os
import streamlit as st

# Set up TruLens observability (similar to notebook example)
provider = OpenAI(
    model_engine=os.environ.get("EVAL_MODEL_NAME"),
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url="https://api.openai.com/v1"
)

context_relevance_custom_criteria = """
    When the question requires multiple different sources to answer, score based on the following criteria:
    - 0: The SEARCH RESULT is not relevant to the any part of the question.
    - 1: The SEARCH RESULT is somewhat relevant but not sufficient for answering a portion of the question.
    - 2: The SEARCH RESULT is sufficient for answering a portion of the question.
    - 3: The SEARCH RESULT is highly relevant and sufficient for answering the complete question.
    """

# note: feedback function used for guardrail must only return a score, not also reasons
f_context_relevance_score = Feedback(
    provider.context_relevance, name="Context Relevance",
    criteria = context_relevance_custom_criteria,
)

class Rag:
    def __init__(self, chat_model: ChatModel, vector_store: VectorStore, use_context_filter: int):
        self.chat_model = chat_model
        self.vector_store = vector_store
        self.use_context_filter = use_context_filter

    @context_filter(
        feedback=f_context_relevance_score,
        threshold=0.7,
        keyword_for_prompt="query",
    )
    def _retrieve_filtered(self, query: str):
        # Retrieve similar document chunks based on the query with context filtering
        results = self.vector_store.search(query)
        return [result.page_content for result in results]

    def _retrieve_plain(self, query: str):
        # Retrieve similar document chunks based on the query without context filtering
        results = self.vector_store.search(query)
        return [result.page_content for result in results]

    @instrument
    def retrieve(self, query: str):
        if int(self.use_context_filter)==1:
            return self._retrieve_filtered(query)
        else:
            return self._retrieve_plain(query)

    @instrument
    def generate(self, query: str, retrieved_chunks, message_history: list = None):
        # Construct prompt messages with query and retrieved chunks
        messages = self.chat_model.construct_prompt(query, retrieved_chunks, message_history)
        # Generate and return the answer from the chat model
        return self.chat_model.generate_answer(messages)

    @instrument
    def retrieve_and_generate(self, query: str, message_history: list = None):
        # Retrieve similar document chunks based on the query
        retrieved_chunks = self.retrieve(query)
        if len(retrieved_chunks) == 0:
            return "I'm sorry, I don't have the information I need to answer that question."
        # Generate and return the answer from the chat model
        return self.generate(query, retrieved_chunks, message_history)

    @instrument
    def retrieve_and_generate_stream(self, query: str, message_history: list = None):
        # Retrieve similar document chunks
        retrieved_chunks = self.retrieve(query)
        if len(retrieved_chunks) == 0:
            return "I'm sorry, I don't have the information I need to answer that question."
        messages = self.chat_model.construct_prompt(query, retrieved_chunks, message_history)
        # Stream tokens from the chat model
        thread = self.chat_model.generate_stream(messages)
        return thread

    @instrument
    def decompose_query(self, query: str) -> list:
        """
        Decomposes the input query into multiple sub-queries.
        """
        prompt = (
            "You are a helpful assistant that generates no more than 3 sub-questions needed to answer the input question. \n"
            "The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n"
            "Generate the needed sub-questions for: {question} \n"
        ).format(question=query)
        system_message = {"role": "system", "content": prompt}
        subqueries_response = self.chat_model.generate_answer([system_message])
        return [sq.strip() for sq in subqueries_response.split("\n") if sq.strip()]

    @instrument
    def retrieve_and_generate_decomposed(self, query: str, message_history: list = None):
        # Use the separate function to decompose the query into sub-questions
        subqueries = self.decompose_query(query)
        all_retrieved_chunks = []
        for subquery in subqueries:
            retrieved_chunks = self.retrieve(subquery)
            if retrieved_chunks:
                all_retrieved_chunks.extend(retrieved_chunks)

        # deduplicate
        deduplicated_chunks = list(set(all_retrieved_chunks))
        all_retrieved_chunks = sorted(deduplicated_chunks, key=lambda x: all_retrieved_chunks.index(x))

        if len(all_retrieved_chunks) == 0:
            return "I'm sorry, I don't have the information I need to answer that question."

        # Generate the final answer using the original query and all aggregated retrieved chunks
        return self.generate(query, all_retrieved_chunks, message_history)
