"""Tests for TruChain.

Some of the tests are outdated.
"""

from typing import Optional
from unittest import main

from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models.base import ChatOpenAI
from trulens.apps.langchain import TruChain
from trulens.core import TruSession
from trulens.core.feedback.endpoint import Endpoint
from trulens.core.schema import base as base_schema
from trulens.core.schema.feedback import FeedbackMode
from trulens.core.schema.record import Record
from trulens.core.utils.keys import check_keys

from tests.test import JSONTestCase
from tests.test import async_test
from tests.test import optional_test


class TestTruChain(JSONTestCase):
    """Test TruChain apps."""

    @classmethod
    def setUpClass(cls):
        # Cannot reset on each test as they might be done in parallel.
        TruSession().reset_database()

    def setUp(self):
        check_keys(
            "OPENAI_API_KEY",
            "HUGGINGFACE_API_KEY",
        )

    def _create_basic_chain(self, app_name: Optional[str] = None):
        # Create simple QA chain.
        prompt = PromptTemplate.from_template(
            """Honestly answer this question: {question}."""
        )

        # Get sync results.
        llm = ChatOpenAI(temperature=0.0)
        chain = prompt | llm | StrOutputParser()

        # Note that without WITH_APP mode, there might be a delay between return
        # of a with_record and the record appearing in the db.
        tc = TruChain(
            chain, app_name=app_name, feedback_mode=FeedbackMode.WITH_APP
        )

        return tc

    def _check_generation_costs(self, cost: base_schema.Cost):
        # Check that all cost fields that should be filled in for a successful generations are non-zero.

        self.assertGreater(cost.n_requests, 0)
        self.assertGreater(cost.n_successful_requests, 0)
        self.assertGreater(cost.n_tokens, 0)
        self.assertGreater(cost.n_prompt_tokens, 0)
        self.assertGreater(cost.n_completion_tokens, 0)
        self.assertGreater(cost.cost, 0.0)

    @optional_test
    def test_record_metadata_plain(self):
        # Test inclusion of metadata in records.

        # Need unique app_id per test as they may be run in parallel and have
        # same ids.
        session = TruSession()
        tc = self._create_basic_chain(app_name="metaplain")

        message = "What is 1+2?"
        meta = "this is plain metadata"

        _, rec = tc.with_record(tc.app.invoke, message, record_metadata=meta)

        # Check record has metadata.
        self.assertEqual(rec.meta, meta)

        # Check the record has the metadata when retrieved back from db.
        recs, _ = session.get_records_and_feedback([tc.app_id])
        self.assertGreater(len(recs), 0)
        rec = Record.model_validate_json(recs.iloc[0].record_json)
        self.assertEqual(rec.meta, meta)

        # Check updating the record metadata in the db.
        new_meta = "this is new meta"
        rec.meta = new_meta
        session.update_record(rec)
        recs, _ = session.get_records_and_feedback([tc.app_id])
        self.assertGreater(len(recs), 0)
        rec = Record.model_validate_json(recs.iloc[0].record_json)
        self.assertNotEqual(rec.meta, meta)
        self.assertEqual(rec.meta, new_meta)

        # Check adding meta to a record that initially didn't have it.
        # Record with no meta:
        _, rec = tc.with_record(tc.app.invoke, message)
        self.assertEqual(rec.meta, None)
        recs, _ = session.get_records_and_feedback([tc.app_id])
        self.assertGreater(len(recs), 1)
        rec = Record.model_validate_json(recs.iloc[1].record_json)
        self.assertEqual(rec.meta, None)

        # Update it to add meta:
        rec.meta = new_meta
        session.update_record(rec)
        recs, _ = session.get_records_and_feedback([tc.app_id])
        self.assertGreater(len(recs), 1)
        rec = Record.model_validate_json(recs.iloc[1].record_json)
        self.assertEqual(rec.meta, new_meta)

    @optional_test
    def test_record_metadata_json(self):
        # Test inclusion of metadata in records.

        # Need unique app_id per test as they may be run in parallel and have
        # same ids.
        tc = self._create_basic_chain(app_name="metajson")

        message = "What is 1+2?"
        meta = dict(field1="hello", field2="there")

        _, rec = tc.with_record(tc.app.invoke, message, record_metadata=meta)

        # Check record has metadata.
        self.assertEqual(rec.meta, meta)

        # Check the record has the metadata when retrieved back from db.
        recs, _ = TruSession().get_records_and_feedback([tc.app_id])
        self.assertGreater(len(recs), 0)
        rec = Record.model_validate_json(recs.iloc[0].record_json)
        self.assertEqual(rec.meta, meta)

        # Check updating the record metadata in the db.
        new_meta = dict(hello="this is new meta")
        rec.meta = new_meta
        TruSession().update_record(rec)

        recs, _ = TruSession().get_records_and_feedback([tc.app_id])
        self.assertGreater(len(recs), 0)
        rec = Record.model_validate_json(recs.iloc[0].record_json)
        self.assertNotEqual(rec.meta, meta)
        self.assertEqual(rec.meta, new_meta)

    @optional_test
    @async_test
    async def test_async_with_task(self):
        # Check whether an async call that makes use of Task (via
        # asyncio.gather) can still track costs.

        # TODO: move to a different test file as TruChain is not involved.

        msg = HumanMessage(content="Hello there")

        prompt = PromptTemplate.from_template(
            """Honestly answer this question: {question}."""
        )
        llm = ChatOpenAI(temperature=0.0, streaming=False, cache=False)
        chain = prompt | llm | StrOutputParser()

        async def test1():
            # Does not create a task:
            result = await chain.middle[0]._agenerate(messages=[msg])
            return result

        res1_, tally1 = Endpoint.track_all_costs_tally(test1)
        res1 = await res1_

        costs1 = tally1()
        with self.subTest(part="costs1"):
            self._check_generation_costs(costs1)

        async def test2():
            # Creates a task internally via asyncio.gather:
            result = await chain.ainvoke(input=dict(question="hello there"))
            return result

        res2_, tally2 = Endpoint.track_all_costs_tally(test2)
        res2 = await res2_

        costs2 = tally2()
        with self.subTest(part="costs2"):
            self._check_generation_costs(costs2)

        # Results are not the same as they involve different prompts but should
        # not be empty at least:
        self.assertGreater(len(res1.generations[0].text), 5)
        self.assertGreater(len(res2), 5)

        # If streaming were used, should count some number of chunks.
        # TODO: test with streaming
        # self.assertGreater(costs1[0].cost.n_stream_chunks, 0)
        # self.assertGreater(costs2[0].cost.n_stream_chunks, 0)

    @optional_test
    @async_test
    async def test_async_stream_token_gen(self):
        from langchain_openai import ChatOpenAI

        async_callback = AsyncIteratorCallbackHandler()
        prompt = PromptTemplate.from_template(
            """Honestly answer this question: {question}."""
        )
        llm = ChatOpenAI(
            temperature=0.0, streaming=True, callbacks=[async_callback]
        )
        agent = prompt | llm | StrOutputParser()
        agent_recorder = TruChain(agent)  # , feedbacks=[f_lang_match])

        message = "What is 1+2? Explain your answer."
        async with agent_recorder as recording:
            async_res = await agent.ainvoke(input=dict(question=message))

        # Need to iterate stream before the full record gets populated.
        async for chunk in async_callback.aiter():
            print("chunk: ", chunk)

        async_record = recording.records[0]

        with agent_recorder as recording:
            sync_res = agent.invoke(input=dict(question=message))

        sync_record = recording.records[0]

        self.assertJSONEqual(async_res, sync_res)

        self.assertJSONEqual(
            async_record.model_dump(),
            sync_record.model_dump(),
            skips=set([
                "id",
                "call_id",
                "cost",  # usage info in streaming mode seems to not be available for openai by default https://community.openai.com/t/usage-info-in-api-responses/18862
                "name",
                "ts",
                "start_time",
                "end_time",
                "record_id",
                "tid",
                "pid",
                "run_id",
            ]),
        )

        # Check that we counted the number of chunks at least.
        self.assertGreater(async_record.cost.n_stream_chunks, 0)

    @optional_test
    @async_test
    async def test_async_with_record(self):
        """Check that the async awith_record produces the same stuff as the
        sync with_record."""

        from langchain_openai import ChatOpenAI

        # Create simple QA chain.
        prompt = PromptTemplate.from_template(
            """Honestly answer this question: {question}."""
        )

        message = "What is 1+2?"

        # Get sync results.
        llm = ChatOpenAI(temperature=0.0)
        chain = prompt | llm | StrOutputParser()
        tc = TruChain(chain)
        sync_res, sync_record = tc.with_record(
            tc.app.invoke, input=dict(question=message)
        )
        with self.subTest(part="sync costs"):
            self._check_generation_costs(sync_record.cost)

        # Get async results.
        llm = ChatOpenAI(temperature=0.0)
        chain = prompt | llm | StrOutputParser()
        tc = TruChain(chain)
        async_res, async_record = await tc.awith_record(
            tc.app.ainvoke,
            input=dict(question=message),
        )
        with self.subTest(part="async costs"):
            self._check_generation_costs(async_record.cost)

        # These are sometimes different despite temperature=0.0. So check that
        # they both mention "3" in the response.
        # self.assertJSONEqual(async_res, sync_res)
        self.assertIn("3", async_res)
        self.assertIn("3", sync_res)

        self.assertJSONEqual(
            async_record.model_dump(),
            sync_record.model_dump(),
            skips=set([
                "id",
                "name",
                "ts",
                "start_time",
                "end_time",
                "record_id",
                "call_id",
                "tid",
                "pid",
                "app_id",
                "main_output",  # see prior note about non-deterministic answers
                "content",  # same issue
                "rets",  # same issue
                "total_tokens",  # same
                "output_tokens",  # same
                "completion_tokens",  # same
                "cost",  # same
            ]),
        )


if __name__ == "__main__":
    main()
