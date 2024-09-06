"""Tests for TruLlama."""

import os
from pathlib import Path
from typing import Set
from unittest import main
from unittest import skip
import weakref

from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.base.response.schema import AsyncStreamingResponse
from llama_index.core.base.response.schema import Response
from llama_index.core.base.response.schema import StreamingResponse
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.llms.openai import OpenAI
from trulens.apps.llamaindex import TruLlama
from trulens.core.schema import base as base_schema
from trulens.core.utils.keys import check_keys

from tests.test import TruTestCase
from tests.test import async_test
from tests.test import optional_test


# All tests require optional packages.
@optional_test
class TestLlamaIndex(TruTestCase):
    DATA_PATH = Path("data") / "paul_graham_essay.txt"
    DATA_URL = "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt"

    ANSWERS = {
        "What did the author do growing up?": set([
            "The author worked on writing short stories and programming, starting with the IBM 1401 in 9th grade and later transitioning to microcomputers like the TRS-80.",
            "The author worked on writing short stories and programming while growing up. They started with the IBM 1401 in 9th grade and later transitioned to microcomputers like the TRS-80.",
            "The author worked on writing short stories and programming, starting with early attempts on an IBM 1401 using Fortran in 9th grade. Later, the author transitioned to microcomputers, building a Heathkit kit and eventually getting a TRS-80 to write simple games and programs. Despite enjoying programming, the author initially planned to study philosophy in college but eventually switched to AI due to a lack of interest in philosophy courses.",
            "I couldn't find specific information about what the author did growing up. Would you like me to try a different approach or provide information on a different topic related to the author?",
        ])
    }
    """Common answers to a simple question.

    Multiple answers are needed despite using temperature=0 for some reason.
    Only checking the prefix of these as they are too varied deeper in the text.
    """

    @staticmethod
    def _get_question_and_answers(index: int):
        return list(TestLlamaIndex.ANSWERS.items())[index]

    def setUp(self):
        check_keys("OPENAI_API_KEY")

        if not TestLlamaIndex.DATA_PATH.exists():
            os.system(f"wget {TestLlamaIndex.DATA_URL} -P data/")

        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)
        Settings.num_output = 64

        documents = SimpleDirectoryReader("data").load_data()
        self.index = VectorStoreIndex.from_documents(documents)

    def _create_query_engine(self, streaming: bool = False):
        engine = self.index.as_query_engine(streaming=streaming)
        recorder = TruLlama(engine)

        return engine, recorder

    def _create_chat_engine(self, streaming: bool = False):
        engine = self.index.as_chat_engine(streaming=streaming)
        recorder = TruLlama(engine)

        return engine, recorder

    def _is_reasonable_response(self, response, expected_answers: Set[str]):
        # Response is probabilistic, but these checks should hold:
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 32)

        # Check that at least the prefix of the response is in the expected answers.
        self.assertIn(
            response[:32], set(map(lambda s: s[:32], expected_answers))
        )

    def _check_generation_costs(self, cost: base_schema.Cost):
        # Check that all cost fields that should be filled in for a successful
        # generations are non-zero.

        for field in [
            "n_requests",
            "n_successful_requests",
            "n_tokens",
            "n_prompt_tokens",
            "n_completion_tokens",
            "cost",
        ]:
            with self.subTest(cost_field=field):
                self.assertGreater(getattr(cost, field), 0)

    def _check_stream_generation_costs(self, cost: base_schema.Cost):
        # Check that all cost fields that should be filled in for a successful
        # generations are non-zero.

        # NOTE(piotrm): tokens and cost are not tracked in streams:
        for field in [
            "n_requests",
            "n_successful_requests",
            # "n_tokens",
            # "n_prompt_tokens",
            # "n_completion_tokens",
            # "cost",
            "n_stream_chunks",  # but chunks are
        ]:
            with self.subTest(cost_field=field):
                self.assertGreater(getattr(cost, field), 0)

    def _sync_test(
        self, create_engine, engine_method_name: str, streaming: bool = False
    ):
        question, expected_answers = self._get_question_and_answers(0)

        query_engine, recorder = create_engine(streaming=streaming)

        with recorder as recording:
            response = getattr(query_engine, engine_method_name)(question)

        if isinstance(response, Response):
            response = response.response
        elif isinstance(response, AgentChatResponse):
            response = response.response
        elif isinstance(response, StreamingResponse):
            response = response.get_response().response
        else:
            assert isinstance(response, str)

        record = recording.get()

        self._is_reasonable_response(response, expected_answers)
        if streaming:
            self._check_stream_generation_costs(record.cost)
        else:
            self._check_generation_costs(record.cost)

        # Check that recorder is garbage collected.
        recorder_ref = weakref.ref(recorder)
        del recorder, recording, record
        self.assertCollected(recorder_ref)

    async def _async_test(
        self, create_engine, engine_method_name: str, streaming: bool = False
    ):
        question, expected_answers = self._get_question_and_answers(0)

        query_engine, recorder = create_engine(streaming=streaming)

        async with recorder as recording:
            method = getattr(query_engine, engine_method_name)
            response_or_stream = await method(question)

        if isinstance(response_or_stream, Response):
            response = response_or_stream.response
        elif isinstance(response_or_stream, AgentChatResponse):
            response = response_or_stream.response
        elif isinstance(response_or_stream, AsyncStreamingResponse):
            response = await response_or_stream.get_response()
            response = response.response
        elif isinstance(response_or_stream, StreamingResponse):
            response = response_or_stream.get_response().response
        else:
            assert isinstance(response_or_stream, str)
            response = response_or_stream

        record = recording.get()

        self._is_reasonable_response(response, expected_answers)

        if streaming:
            self._check_stream_generation_costs(record.cost)
        else:
            self._check_generation_costs(record.cost)

        # Check that recorder is garbage collected.
        recorder_ref = weakref.ref(recorder)
        del recorder, recording, record
        self.assertCollected(recorder_ref)

    # Query engine tests

    def test_query_engine_sync(self):
        """Synchronous query engine test."""

        self._sync_test(self._create_query_engine, "query")

    @async_test
    async def test_query_engine_async(self):
        """Asynchronous query engine test."""

        await self._async_test(self._create_query_engine, "aquery")

    def test_query_engine_sync_stream(self):
        """Synchronous streaming query engine test."""

        self._sync_test(self._create_query_engine, "query", streaming=True)

    @async_test
    async def test_query_engine_async_stream(self):
        """Asynchronous streaming query engine test."""

        await self._async_test(
            self._create_query_engine, "aquery", streaming=True
        )

    # Chat engine tests

    def test_chat_engine_sync(self):
        """Synchronous chat engine test."""

        self._sync_test(self._create_chat_engine, "chat")

    @async_test
    async def test_chat_engine_async(self):
        """Asynchronous chat engine test."""

        await self._async_test(self._create_chat_engine, "achat")

    def test_chat_engine_sync_stream(self):
        """Synchronous streaming chat engine test."""

        self._sync_test(self._create_chat_engine, "chat", streaming=True)

    @skip("Bug in llama_index.")
    @async_test
    async def test_chat_engine_async_stream(self):
        """Asynchronous streaming chat engine test."""

        await self._async_test(
            self._create_chat_engine, "achat", streaming=True
        )


if __name__ == "__main__":
    main()
