"""
Tests for TruLlama.
"""

import unittest
from unittest import main

from trulens.core.utils.asynchro import sync
from trulens.core.utils.keys import check_keys

from tests.unit.test import JSONTestCase
from tests.unit.test import optional_test


# All tests require optional packages.
@optional_test
class TestLlamaIndex(JSONTestCase):
    # TODO: Figure out why use of async test cases causes "more than one record
    # collected"
    # Need to use this:
    # from unittest import IsolatedAsyncioTestCase

    def setUp(self):
        check_keys("OPENAI_API_KEY", "HUGGINGFACE_API_KEY")
        import os

        from llama_index.core import SimpleDirectoryReader
        from llama_index.core import VectorStoreIndex

        os.system(
            "wget https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt -P data/"
        )

        documents = SimpleDirectoryReader("data").load_data()
        self.index = VectorStoreIndex.from_documents(documents)

    def test_query_engine_async(self):
        # Check that the instrumented async aquery method produces the same result as the query method.

        from trulens.instrument.llamaindex import TruLlama

        query_engine = self.index.as_query_engine()

        # This test does not run correctly if async is used, i.e. not using
        # `sync` to convert to sync.

        tru_query_engine_recorder = TruLlama(query_engine)
        llm_response_async, record_async = sync(
            tru_query_engine_recorder.awith_record,
            query_engine.aquery,
            "What did the author do growing up?",
        )

        query_engine = self.index.as_query_engine()
        tru_query_engine_recorder = TruLlama(query_engine)
        llm_response_sync, record_sync = tru_query_engine_recorder.with_record(
            query_engine.query, "What did the author do growing up?"
        )

        # llm response is probabilistic, so just test if async response is also a string. not that it is same as sync response.
        self.assertIsInstance(llm_response_async.response, str)

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
                    "record_id",
                    "cost",  # cost is not being correctly tracked in async
                    "main_output",  # response is not deterministic, so cannot easily compare across runs
                ]
            ),
        )

    @unittest.skip("Streaming records not yet recorded properly.")
    def test_query_engine_stream(self):
        # Check that the instrumented query method produces the same result
        # regardless of streaming option.

        from trulens.instrument.llamaindex import TruLlama

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
            numeric_places=2,  # node scores and token counts are imprecise
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
                    "record_id",
                ]
            ),
        )

    async def test_chat_engine_async(self):
        # Check that the instrumented async achat method produces the same result as the chat method.

        from trulens.instrument.llamaindex import TruLlama

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
            numeric_places=2,  # node scores and token counts are imprecise
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
                    "record_id",
                ]
            ),
        )


if __name__ == "__main__":
    main()
