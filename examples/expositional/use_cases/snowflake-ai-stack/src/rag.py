from src.generation import ChatModel
from langchain_core.vectorstores import VectorStore
from trulens.core.otel.instrument import instrument
from trulens.otel.semconv.trace import SpanAttributes
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

    @instrument(
        span_type=SpanAttributes.SpanType.RETRIEVAL,
        attributes={
            SpanAttributes.RETRIEVAL.QUERY_TEXT: "query",
            SpanAttributes.RETRIEVAL.RETRIEVED_CONTEXTS: "return",
        },
    )
    def retrieve(self, query: str):
        if int(self.use_context_filter)==1:
            return self._retrieve_filtered(query)
        else:
            return self._retrieve_plain(query)

    @instrument(span_type=SpanAttributes.SpanType.GENERATION)
    def generate(self, query: str, retrieved_chunks, message_history: list = None):
        # Construct prompt messages with query and retrieved chunks
        messages = self.chat_model.construct_prompt(query, retrieved_chunks, message_history)
        # Generate and return the answer from the chat model
        return self.chat_model.generate_answer(messages)

    # @instrument(
    #     span_type=SpanAttributes.SpanType.RECORD_ROOT,
    #     attributes={
    #         SpanAttributes.RECORD_ROOT.INPUT: "query",
    #         SpanAttributes.RECORD_ROOT.OUTPUT: "return",
    #     },
    # )
    # def retrieve_and_generate(self, query: str, message_history: list = None):
    #     # Retrieve similar document chunks based on the query
    #     retrieved_chunks = self.retrieve(query)
    #     if len(retrieved_chunks) == 0:
    #         return "I'm sorry, I don't have the information I need to answer that question."
    #     # Generate and return the answer from the chat model
    #     return self.generate(query, retrieved_chunks, message_history)

    @instrument(
        span_type=SpanAttributes.SpanType.RECORD_ROOT,
        attributes={
            SpanAttributes.RECORD_ROOT.INPUT: "query",
            SpanAttributes.RECORD_ROOT.OUTPUT: "return",
        },
    )
    def retrieve_and_generate_stream(self, query: str, message_history: list = None):
        # Retrieve similar document chunks
        retrieved_chunks = self.retrieve(query)
        if len(retrieved_chunks) == 0:
            return "I'm sorry, I don't have the information I need to answer that question."
        messages = self.chat_model.construct_prompt(query, retrieved_chunks, message_history)
        # Stream tokens from the chat model
        thread = self.chat_model.generate_stream(messages)
        return thread
