"""
Tests for TruLlama.
"""

import unittest
from unittest import main

from llama_index import ServiceContext
from llama_index import set_global_service_context
from llama_index import VectorStoreIndex
from llama_index.llms import OpenAI
from llama_index.readers.web import SimpleWebPageReader
from tests.unit.test import JSONTestCase

from trulens_eval.keys import check_keys
from trulens_eval.tru_llama import TruLlama
from trulens_eval.utils.asynchro import sync

check_keys("OPENAI_API_KEY", "HUGGINGFACE_API_KEY")


class TestLlamaIndex(JSONTestCase):
    # TODO: Figure out why use of async test cases causes "more than one record
    # collected"
    # Need to use this:
    # from unittest import IsolatedAsyncioTestCase

    def setUp(self):

        # NOTE: Need temp = 0 for consistent tests. Some tests are still
        # non-deterministic despite this temperature, perhaps there is some
        # other temperature setting or this one is not taken up.
        llm = OpenAI(temperature=0.0)
        service_context = ServiceContext.from_defaults(llm=llm)
        set_global_service_context(service_context)

        # llama_index 0.8.15 bug: need to provide metadata_fn
        self.documents = SimpleWebPageReader(
            html_to_text=True, metadata_fn=lambda url: dict(url=url)
        ).load_data(["http://paulgraham.com/worked.html"])
        self.index = VectorStoreIndex.from_documents(self.documents)

    def test_query_engine_async(self):
        # Check that the instrumented async aquery method produces the same result as the query method.

        query_engine = self.index.as_query_engine()

        # This test does not run correctly if async is used, i.e. not using
        # `sync` to convert to sync.

        tru_query_engine_recorder = TruLlama(query_engine)
        with tru_query_engine_recorder as recording:
            llm_response_async = sync(
                query_engine.aquery, "What did the author do growing up?"
            )
            print("llm_response_async=", llm_response_async)

        record_async = recording.get()

        query_engine = self.index.as_query_engine()
        tru_query_engine_recorder = TruLlama(query_engine)
        with tru_query_engine_recorder as recording:
            llm_response_sync = query_engine.query(
                "What did the author do growing up?"
            )
            print("llm_response_sync=", llm_response_sync)
        record_sync = recording.get()

        self.assertJSONEqual(
            llm_response_sync,
            llm_response_async,
            numeric_places=2  # node scores and token counts are imprecise
        )

        self.assertJSONEqual(
            record_sync.model_dump(),
            record_async.model_dump(),
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
        tru_query_engine_recorder = TruLlama(query_engine)
        with tru_query_engine_recorder as recording:
            llm_response = query_engine.query(
                "What did the author do growing up?"
            )
        record = recording.get()

        query_engine = self.index.as_query_engine(streaming=True)
        tru_query_engine_recorder = TruLlama(query_engine)
        with tru_query_engine_recorder as stream_recording:
            llm_response_stream = query_engine.query(
                "What did the author do growing up?"
            )
        record_stream = stream_recording.get()

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

    async def test_chat_engine_async(self):
        # Check that the instrumented async achat method produces the same result as the chat method.

        chat_engine = self.index.as_chat_engine()
        tru_chat_engine_recorder = TruLlama(chat_engine)
        with tru_chat_engine_recorder as arecording:
            llm_response_async = await chat_engine.achat(
                "What did the author do growing up?"
            )
        record_async = arecording.records[0]

        chat_engine = self.index.as_chat_engine()
        tru_chat_engine_recorder = TruLlama(chat_engine)
        with tru_chat_engine_recorder as recording:
            llm_response_sync = chat_engine.chat(
                "What did the author do growing up?"
            )
        record_sync = recording.records[0]

        self.assertJSONEqual(
            llm_response_sync,
            llm_response_async,
            numeric_places=2  # node scores and token counts are imprecise
        )

        self.assertJSONEqual(
            record_sync.model_dump(),
            record_async.model_dump(),
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
