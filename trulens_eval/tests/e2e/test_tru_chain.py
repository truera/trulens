"""
Tests for TruChain. Some of the tests are outdated.
"""

import unittest
from unittest import main

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.chains import LLMChain
from langchain.llms.openai import OpenAI
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from tests.unit.test import JSONTestCase
from tests.unit.test import optional_test

from trulens_eval import Tru
from trulens_eval.feedback.provider.endpoint import Endpoint
from trulens_eval.keys import check_keys
from trulens_eval.schema import FeedbackMode
from trulens_eval.schema import Record
from trulens_eval.utils.asynchro import sync


class TestTruChain(JSONTestCase):
    """Test TruChain class."""
    # TODO: See problem in TestTruLlama.
    # USE IsolatedAsyncioTestCase

    @classmethod
    def setUpClass(cls):
        # Cannot reset on each test as they might be done in parallel.
        Tru().reset_database()

    def setUp(self):

        check_keys(
            "OPENAI_API_KEY", "HUGGINGFACE_API_KEY", "PINECONE_API_KEY",
            "PINECONE_ENV"
        )

    @optional_test
    def test_multiple_instruments(self):
        # Multiple wrapped apps use the same components. Make sure paths are
        # correctly tracked.

        prompt = PromptTemplate.from_template(
            """Honestly answer this question: {question}."""
        )
        llm = OpenAI(temperature=0.0, streaming=False, cache=False)

        chain1 = LLMChain(llm=llm, prompt=prompt)

        memory = ConversationSummaryBufferMemory(
            memory_key="chat_history",
            input_key="question",
            llm=llm,  # same llm now appears in a different spot
        )
        chain2 = LLMChain(llm=llm, prompt=prompt, memory=memory)

    def _create_basic_chain(self, app_id: str = None):

        from langchain_openai import ChatOpenAI

        # Create simple QA chain.
        tru = Tru()
        prompt = PromptTemplate.from_template(
            """Honestly answer this question: {question}."""
        )

        # Get sync results.
        llm = ChatOpenAI(temperature=0.0)
        chain = LLMChain(llm=llm, prompt=prompt)

        # Note that without WITH_APP mode, there might be a delay between return
        # of a with_record and the record appearing in the db.
        tc = tru.Chain(
            chain, app_id=app_id, feedback_mode=FeedbackMode.WITH_APP
        )

        return tc

    @optional_test
    def test_record_metadata_plain(self):
        # Test inclusion of metadata in records.

        # Need unique app_id per test as they may be run in parallel and have
        # same ids.
        tc = self._create_basic_chain(app_id="metaplain")

        message = "What is 1+2?"
        meta = "this is plain metadata"

        _, rec = tc.with_record(tc.app, message, record_metadata=meta)

        # Check record has metadata.
        self.assertEqual(rec.meta, meta)

        # Check the record has the metadata when retrieved back from db.
        recs, _ = Tru().get_records_and_feedback([tc.app_id])
        self.assertGreater(len(recs), 0)
        rec = Record.model_validate_json(recs.iloc[0].record_json)
        self.assertEqual(rec.meta, meta)

        # Check updating the record metadata in the db.
        new_meta = "this is new meta"
        rec.meta = new_meta
        Tru().update_record(rec)
        recs, _ = Tru().get_records_and_feedback([tc.app_id])
        self.assertGreater(len(recs), 0)
        rec = Record.model_validate_json(recs.iloc[0].record_json)
        self.assertNotEqual(rec.meta, meta)
        self.assertEqual(rec.meta, new_meta)

        # Check adding meta to a record that initially didn't have it.
        # Record with no meta:
        _, rec = tc.with_record(tc.app, message)
        self.assertEqual(rec.meta, None)
        recs, _ = Tru().get_records_and_feedback([tc.app_id])
        self.assertGreater(len(recs), 1)
        rec = Record.model_validate_json(recs.iloc[1].record_json)
        self.assertEqual(rec.meta, None)

        # Update it to add meta:
        rec.meta = new_meta
        Tru().update_record(rec)
        recs, _ = Tru().get_records_and_feedback([tc.app_id])
        self.assertGreater(len(recs), 1)
        rec = Record.model_validate_json(recs.iloc[1].record_json)
        self.assertEqual(rec.meta, new_meta)

    @optional_test
    def test_record_metadata_json(self):
        # Test inclusion of metadata in records.

        # Need unique app_id per test as they may be run in parallel and have
        # same ids.
        tc = self._create_basic_chain(app_id="metajson")

        message = "What is 1+2?"
        meta = dict(field1="hello", field2="there")

        _, rec = tc.with_record(tc.app, message, record_metadata=meta)

        # Check record has metadata.
        self.assertEqual(rec.meta, meta)

        # Check the record has the metadata when retrieved back from db.
        recs, feedbacks = Tru().get_records_and_feedback([tc.app_id])
        self.assertGreater(len(recs), 0)
        rec = Record.model_validate_json(recs.iloc[0].record_json)
        self.assertEqual(rec.meta, meta)

        # Check updating the record metadata in the db.
        new_meta = dict(hello="this is new meta")
        rec.meta = new_meta
        Tru().update_record(rec)

        recs, _ = Tru().get_records_and_feedback([tc.app_id])
        self.assertGreater(len(recs), 0)
        rec = Record.model_validate_json(recs.iloc[0].record_json)
        self.assertNotEqual(rec.meta, meta)
        self.assertEqual(rec.meta, new_meta)

    @optional_test
    def test_async_with_task(self):
        # Check whether an async call that makes use of Task (via
        # asyncio.gather) can still track costs.

        # TODO: move to a different test file as TruChain is not involved.

        from langchain_openai import ChatOpenAI

        msg = HumanMessage(content="Hello there")

        prompt = PromptTemplate.from_template(
            """Honestly answer this question: {question}."""
        )
        llm = ChatOpenAI(temperature=0.0, streaming=False, cache=False)
        chain = LLMChain(llm=llm, prompt=prompt)

        async def test1():
            # Does not create a task:
            result = await chain.llm._agenerate(messages=[msg])
            return result

        res1, costs1 = Endpoint.track_all_costs(lambda: sync(test1))

        async def test2():
            # Creates a task internally via asyncio.gather:
            result = await chain.acall(inputs=dict(question="hello there"))
            return result

        res2, costs2 = Endpoint.track_all_costs(lambda: sync(test2))

        # Results are not the same as they involve different prompts but should
        # not be empty at least:
        self.assertGreater(len(res1.generations[0].text), 5)
        self.assertGreater(len(res2['text']), 5)

        # And cost tracking should have counted some number of tokens.
        # TODO: broken
        # self.assertGreater(costs1[0].cost.n_tokens, 3)
        # self.assertGreater(costs2[0].cost.n_tokens, 3)

        # If streaming were used, should count some number of chunks.
        # TODO: test with streaming
        # self.assertGreater(costs1[0].cost.n_stream_chunks, 0)
        # self.assertGreater(costs2[0].cost.n_stream_chunks, 0)

    @optional_test
    def test_async_with_record(self):
        """Check that the async awith_record produces the same stuff as the
        sync with_record."""

        from langchain_openai import ChatOpenAI

        # Create simple QA chain.
        tru = Tru()
        prompt = PromptTemplate.from_template(
            """Honestly answer this question: {question}."""
        )

        message = "What is 1+2?"

        # Get sync results.
        llm = ChatOpenAI(temperature=0.0)
        chain = LLMChain(llm=llm, prompt=prompt)
        tc = tru.Chain(chain)
        sync_res, sync_record = tc.with_record(
            tc.app, inputs=dict(question=message)
        )

        # Get async results.
        llm = ChatOpenAI(temperature=0.0)
        chain = LLMChain(llm=llm, prompt=prompt)
        tc = tru.Chain(chain)
        async_res, async_record = sync(
            tc.awith_record,
            tc.app.acall,
            inputs=dict(question=message),
        )

        self.assertJSONEqual(async_res, sync_res)

        self.assertJSONEqual(
            async_record.model_dump(),
            sync_record.model_dump(),
            skips=set(
                [
                    "id",
                    "name",
                    "ts",
                    "start_time",
                    "end_time",
                    "record_id",
                    "tid",
                    "pid",
                    "app_id",
                    "cost"  # TODO(piotrm): cost tracking not working with async
                ]
            )
        )

    @optional_test
    @unittest.skip("bug in langchain")
    def test_async_token_gen(self):
        # Test of chain acall methods as requested in https://github.com/truera/trulens/issues/309 .

        from langchain_openai import ChatOpenAI

        tru = Tru()
        # hugs = feedback.Huggingface()
        # f_lang_match = Feedback(hugs.language_match).on_input_output()

        async_callback = AsyncIteratorCallbackHandler()
        prompt = PromptTemplate.from_template(
            """Honestly answer this question: {question}."""
        )
        llm = ChatOpenAI(
            temperature=0.0, streaming=True, callbacks=[async_callback]
        )
        agent = LLMChain(llm=llm, prompt=prompt)
        agent_recorder = tru.Chain(agent)  #, feedbacks=[f_lang_match])

        message = "What is 1+2? Explain your answer."
        with agent_recorder as recording:
            async_res = sync(agent.acall, inputs=dict(question=message))

        async_record = recording.records[0]

        with agent_recorder as recording:
            sync_res = agent(inputs=dict(question=message))

        sync_record = recording.records[0]

        self.assertJSONEqual(async_res, sync_res)

        self.assertJSONEqual(
            async_record.model_dump(),
            sync_record.model_dump(),
            skips=set(
                [
                    "id",
                    "cost",  # usage info in streaming mode seems to not be available for openai by default https://community.openai.com/t/usage-info-in-api-responses/18862
                    "name",
                    "ts",
                    "start_time",
                    "end_time",
                    "record_id",
                    "tid",
                    "pid",
                    "run_id"
                ]
            )
        )

        # Check that we counted the number of chunks at least.
        self.assertGreater(async_record.cost.n_stream_chunks, 0)


if __name__ == '__main__':
    main()
