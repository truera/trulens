"""
Tests for TruLlama.
"""

import asyncio
import unittest
from unittest import main

from llama_index import SimpleWebPageReader
from llama_index import VectorStoreIndex
from llama_index import ServiceContext, set_global_service_context
from llama_index.llms import OpenAI

from tests.unit.test import JSONTestCase

from trulens_eval import Tru
from trulens_eval.keys import check_keys
from trulens_eval.tru_llama import TruLlama
import trulens_eval.utils.python  # makes sure asyncio gets instrumented

check_keys("OPENAI_API_KEY", "HUGGINGFACE_API_KEY")


class TestLlamaIndex(JSONTestCase):

    def setUp(self):

        # need temp = 0 for consistent tests
        llm = OpenAI(temperature=0.0)
        service_context = ServiceContext.from_defaults(llm=llm)
        set_global_service_context(service_context)

        self.documents = SimpleWebPageReader(html_to_text=True).load_data(
            ["http://paulgraham.com/worked.html"]
        )
        self.index = VectorStoreIndex.from_documents(self.documents)

    def test_query_engine_async(self):
        asyncio.run(self._test_query_engine_async())

    async def _test_query_engine_async(self):
        # Check that the instrumented async aquery method produces the same result as the query method.

        query_engine = self.index.as_query_engine()
        
        tru_query_engine = TruLlama(query_engine)
        llm_response_async, record_async = await tru_query_engine.aquery_with_record(
            "What did the author do growing up?"
        )

        query_engine = self.index.as_query_engine()
        tru_query_engine = TruLlama(query_engine)
        llm_response_sync, record_sync = tru_query_engine.query_with_record(
            "What did the author do growing up?"
        )

        self.assertJSONEqual(
            llm_response_sync,
            llm_response_async,
            numeric_places=2  # node scores and token counts are imprecise
        )

        self.assertJSONEqual(
            record_sync,
            record_async,
            skips=set(
                [
                    "calls",  # async/sync have different set of internal calls, so cannot easily compare
                    "name",
                    "app_id",
                    "ts",
                    "start_time",
                    "end_time",
                    "record_id"
                ]
            )
        )

    @unittest.skip("Streaming records not yet recorded properly.")
    def test_query_engine_stream(self):
        # Check that the instrumented query method produces the same result
        # regardless of streaming option.

        query_engine = self.index.as_query_engine()
        tru_query_engine = TruLlama(query_engine)
        llm_response, record = tru_query_engine.query_with_record(
            "What did the author do growing up?"
        )

        query_engine = self.index.as_query_engine(streaming=True)
        tru_query_engine = TruLlama(query_engine)
        llm_response_stream, record_stream = tru_query_engine.query_with_record(
            "What did the author do growing up?"
        )

        self.assertJSONEqual(
            llm_response_stream.get_response(),
            llm_response.response,
            numeric_places=2  # node scores and token counts are imprecise
        )

        self.assertJSONEqual(
            record_stream,
            record,
            skips=set(
                [
                    # "calls",
                    "name",
                    "app_id",
                    "ts",
                    "start_time",
                    "end_time",
                    "record_id"
                ]
            )
        )

    def test_chat_engine_async(self):
        asyncio.run(self._test_chat_engine_async())

    async def _test_chat_engine_async(self):
        # Check that the instrumented async achat method produces the same result as the chat method.

        chat_engine = self.index.as_chat_engine()
        tru_chat_engine = TruLlama(chat_engine)
        llm_response_async, record_async = await tru_chat_engine.achat_with_record(
            "What did the author do growing up?"
        )

        chat_engine = self.index.as_chat_engine()
        tru_chat_engine = TruLlama(chat_engine)
        llm_response_sync, record_sync = tru_chat_engine.chat_with_record(
            "What did the author do growing up?"
        )

        self.assertJSONEqual(
            llm_response_sync,
            llm_response_async,
            numeric_places=2  # node scores and token counts are imprecise
        )

        self.assertJSONEqual(
            record_sync,
            record_async,
            skips=set(
                [
                    "calls",  # async/sync have different set of internal calls, so cannot easily compare
                    "name",
                    "app_id",
                    "ts",
                    "start_time",
                    "end_time",
                    "record_id"
                ]
            )
        )


if __name__ == '__main__':
    main()
