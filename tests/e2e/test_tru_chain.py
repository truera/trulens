"""Tests for TruChain."""

from typing import Optional
from unittest import main
import weakref

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai.chat_models.base import ChatOpenAI
from trulens.apps.langchain import TruChain
from trulens.core import session as mod_session
from trulens.core.schema import base as base_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.utils import keys as key_utils

from tests.test import TruTestCase
from tests.test import async_test
from tests.test import optional_test


@optional_test  # all tests are optional as langchain is optional
class TestTruChain(TruTestCase):
    """Test TruChain apps."""

    ANSWERS = {"What is 1+2?": set(["1+2 equals 3.", "The answer is 3."])}
    """Common answers to a simple question.

    Multiple answers are needed despite using temperature=0 for some reason.
    """

    @staticmethod
    def _get_question_and_answers(index: int):
        return list(TestTruChain.ANSWERS.items())[index]

    @classmethod
    def setUpClass(cls):
        # Cannot reset on each test as they might be done in parallel.
        mod_session.TruSession().reset_database()

    def setUp(self):
        key_utils.check_keys(
            "OPENAI_API_KEY",
            "HUGGINGFACE_API_KEY",
        )

    def _create_basic_chain(
        self, app_name: Optional[str] = None, streaming: bool = False
    ):
        # Create simple QA chain.
        prompt = PromptTemplate.from_template(
            """Honestly answer this question: {question}."""
        )

        # Get sync results.
        llm = ChatOpenAI(temperature=0, streaming=streaming)
        chain = prompt | llm | StrOutputParser()

        # Note that without WITH_APP mode, there might be a delay between return
        # of a with_record and the record appearing in the db.
        tc = TruChain(
            chain,
            app_name=app_name,
            feedback_mode=feedback_schema.FeedbackMode.WITH_APP,
        )

        return chain, tc

    def _check_generation_costs(self, cost: base_schema.Cost):
        # Check that all cost fields that should be filled in for a successful generations are non-zero.

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
        # Check that all cost fields that should be filled in for a successful generations are non-zero.

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

    def test_sync(self):
        """Synchronous (`invoke`) test."""

        chain, recorder = self._create_basic_chain(streaming=False)

        message, expected_answers = self._get_question_and_answers(0)

        with recorder as recording:
            result = chain.invoke(input=dict(question=message))

        record = recording.get()

        self.assertIn(result, expected_answers)

        self._check_generation_costs(record.cost)

        # Check that recorder is garbage collected.
        recorder_ref = weakref.ref(recorder)
        chain_ref = weakref.ref(chain)
        del recorder, recording, record, chain
        self.assertCollected(recorder_ref)
        self.assertCollected(chain_ref)

    @async_test
    async def test_async(self):
        """Asynchronous (`ainvoke`) test."""

        chain, recorder = self._create_basic_chain(streaming=False)

        message, expected_answers = self._get_question_and_answers(0)

        async with recorder as recording:
            result = await chain.ainvoke(input=dict(question=message))

        record = recording.get()

        self.assertIn(result, expected_answers)

        self._check_generation_costs(record.cost)

        # Check that recorder is garbage collected.
        recorder_ref = weakref.ref(recorder)
        # chain_ref = weakref.ref(chain)
        del recorder, recording, record, chain, result
        self.assertCollected(recorder_ref)

        # self.assertCollected(chain_ref) # fails due to memory leak bug in nltk or related

    def test_sync_stream(self):
        """Synchronous stream (`stream`) test."""

        chain, recorder = self._create_basic_chain(streaming=True)

        message, expected_answers = self._get_question_and_answers(0)

        result = ""
        with recorder as recording:
            for chunk in chain.stream(input=dict(question=message)):
                result += chunk

        record = recording.get()

        self.assertIn(result, expected_answers)

        self._check_stream_generation_costs(record.cost)

        # Check that recorder is garbage collected.
        recorder_ref = weakref.ref(recorder)
        chain_ref = weakref.ref(chain)
        del recorder, recording, record, chain, result
        self.assertCollected(recorder_ref)
        self.assertCollected(chain_ref)

    @async_test
    async def test_async_stream(self):
        """Asynchronous stream (`astream`) test."""

        chain, recorder = self._create_basic_chain(streaming=True)

        message, expected_answers = self._get_question_and_answers(0)

        result = ""
        async with recorder as recording:
            async for chunk in chain.astream(input=dict(question=message)):
                result += chunk

        record = recording.get()

        self.assertIn(result, expected_answers)

        self._check_stream_generation_costs(record.cost)

        # Check that recorder is garbage collected.
        recorder_ref = weakref.ref(recorder)
        # chain_ref = weakref.ref(chain)
        del recorder, recording, record, chain
        self.assertCollected(recorder_ref)
        # self.assertCollected(chain_ref) # fails due to memory leak bug in nltk or related

    def test_record_metadata_plain(self):
        """Test inclusion of metadata in records."""

        # Need unique app_id per test as they may be run in parallel and have
        # same ids.
        session = mod_session.TruSession()
        chain, recorder = self._create_basic_chain(app_name="metaplain")

        message, _ = self._get_question_and_answers(0)
        meta = "this is plain metadata"

        with recorder as recording:
            recording.record_metadata = meta
            chain.invoke(input=dict(question=message))

        record = recording.get()

        with self.subTest("Check the record has the metadata"):
            self.assertEqual(record.meta, meta)

        with self.subTest(
            "Check the record has the metadata when retrieved back from db"
        ):
            recs, _ = session.get_records_and_feedback([recorder.app_id])
            self.assertGreater(len(recs), 0)
            rec = record_schema.Record.model_validate_json(
                recs.iloc[-1].record_json
            )
            self.assertEqual(rec.meta, meta)

        with self.subTest("Check updating the record metadata in the db."):
            new_meta = "this is new meta"
            rec.meta = new_meta
            session.update_record(rec)
            recs, _ = session.get_records_and_feedback([recorder.app_id])
            self.assertGreater(len(recs), 0)
            rec = record_schema.Record.model_validate_json(
                recs[recs.record_id == rec.record_id].record_json[0]
            )
            self.assertNotEqual(rec.meta, meta)
            self.assertEqual(rec.meta, new_meta)

        with self.subTest(
            "Check adding meta to a record that initially didn't have it."
        ):
            with recorder as recording:
                chain.invoke(input=dict(question=message))

            with self.subTest("with no metadata"):
                rec = recording.get()
                self.assertEqual(rec.meta, None)
                recs, _ = session.get_records_and_feedback([recorder.app_id])
                self.assertGreater(len(recs), 1)
                rec = record_schema.Record.model_validate_json(
                    recs[recs.record_id == rec.record_id].record_json[0]
                )
                self.assertEqual(rec.meta, None)

            with self.subTest("Updated with metadata"):
                rec.meta = new_meta
                session.update_record(rec)
                recs, _ = session.get_records_and_feedback([recorder.app_id])
                self.assertGreater(len(recs), 1)
                rec = record_schema.Record.model_validate_json(
                    recs[recs.record_id == rec.record_id].record_json[0]
                )
                self.assertEqual(rec.meta, new_meta)

    def test_record_metadata_json(self):
        """Test inclusion of json metadata in records."""

        # Need unique app_id per test as they may be run in parallel and have
        # same ids.
        chain, recorder = self._create_basic_chain(app_name="metajson")

        message, _ = self._get_question_and_answers(0)
        meta = dict(field1="hello", field2="there")

        with recorder as recording:
            recording.record_metadata = meta
            chain.invoke(input=dict(question=message))
        record = recording.get()

        # Check record has metadata.
        self.assertEqual(record.meta, meta)

        # Check the record has the metadata when retrieved back from db.
        recs, _ = mod_session.TruSession().get_records_and_feedback([
            recorder.app_id
        ])
        self.assertGreater(len(recs), 0)
        rec = record_schema.Record.model_validate_json(recs.iloc[0].record_json)
        self.assertEqual(rec.meta, meta)

        # Check updating the record metadata in the db.
        new_meta = dict(hello="this is new meta")
        rec.meta = new_meta
        mod_session.TruSession().update_record(rec)

        recs, _ = mod_session.TruSession().get_records_and_feedback([
            recorder.app_id
        ])
        self.assertGreater(len(recs), 0)
        rec = record_schema.Record.model_validate_json(recs.iloc[0].record_json)
        self.assertNotEqual(rec.meta, meta)
        self.assertEqual(rec.meta, new_meta)


if __name__ == "__main__":
    main()
