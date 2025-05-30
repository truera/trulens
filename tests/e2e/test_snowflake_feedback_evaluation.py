"""
Tests server-side feedback evaluations in Snowflake.
"""

import time

import pytest
from trulens.apps.basic import TruBasicApp
import trulens.connectors.snowflake.utils.server_side_evaluation_artifacts as ssea
import trulens.connectors.snowflake.utils.server_side_evaluation_stored_procedure as ssesp
from trulens.core import session as core_session
from trulens.core.feedback import feedback as core_feedback
from trulens.core.schema import feedback as feedback_schema
import trulens.providers.cortex.provider as cortex_provider
from trulens.providers.openai import provider as openai_provider

from tests.util.snowflake_test_case import SnowflakeTestCase


def silly_feedback_function(q: str) -> float:
    return 0.1


class TestSnowflakeFeedbackEvaluation(SnowflakeTestCase):
    def _suspend_task(self) -> None:
        self._snowpark_session.sql(
            f"ALTER TASK {self._database}.{self._schema}.{ssea._TASK_NAME} SUSPEND"
        ).collect()

    def _wait_till_feedbacks_done(
        self, num_expected_feedbacks: int, timeout_in_seconds: int = 120
    ) -> None:
        start_time = time.time()
        while time.time() - start_time < timeout_in_seconds:
            res = self._snowpark_session.sql(
                f"SELECT STATUS FROM {self._database}.{self._schema}.TRULENS_FEEDBACKS"
            ).collect()
            if len(res) == num_expected_feedbacks and all([
                curr.STATUS == "done" for curr in res
            ]):
                return
            time.sleep(1)
        raise ValueError("Feedback evaluation didn't complete in time!")

    def _call_stored_procedure(self) -> None:
        self._snowpark_session.sql(
            f"CALL {self._database}.{self._schema}.{ssea._WRAPPER_STORED_PROCEDURE_NAME}()"
        ).collect()

    def _get_cortex_relevance_feedback_function(
        self,
    ) -> core_feedback.SnowflakeFeedback:
        return core_feedback.SnowflakeFeedback(
            cortex_provider.Cortex(
                self._snowpark_session, retry_timeout=60
            ).relevance
        ).on_input_output()

    def _start_evaluator_as_snowflake(self, session: core_session.TruSession):
        session.start_evaluator(
            run_location=feedback_schema.FeedbackRunLocation.SNOWFLAKE,
            return_when_done=True,
        )

    @pytest.mark.optional
    def test_local_deferred_mode(self) -> None:
        session = core_session.TruSession()
        session.reset_database()
        f = core_feedback.Feedback(silly_feedback_function).on_default()
        tru_app = TruBasicApp(
            text_to_text=lambda t: f"returning {t}",
            feedbacks=[f],
            feedback_mode=feedback_schema.FeedbackMode.DEFERRED,
        )
        with tru_app:
            tru_app.main_call("test_local_deferred_mode")
        time.sleep(2)
        records_and_feedback = session.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(len(records_and_feedback[1]), 0)
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        session.start_evaluator()
        time.sleep(2)
        session.stop_evaluator()
        records_and_feedback = session.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(records_and_feedback[1], ["silly_feedback_function"])
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function"].iloc[0],
            0.1,
        )

    @pytest.mark.optional
    def test_snowflake_deferred_mode(self) -> None:
        session = self.get_session("test_snowflake_deferred_mode")
        self._suspend_task()
        f_local = core_feedback.Feedback(silly_feedback_function).on_default()
        f_snowflake = self._get_cortex_relevance_feedback_function()
        tru_app = TruBasicApp(
            text_to_text=lambda _: "Tokyo is the capital of Japan.",
            feedbacks=[f_local, f_snowflake],
            feedback_mode=feedback_schema.FeedbackMode.DEFERRED,
        )
        with tru_app:
            tru_app.main_call("What is the capital of Japan?")
        time.sleep(2)
        session.start_evaluator()
        time.sleep(2)
        session.stop_evaluator()
        records_and_feedback = session.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(records_and_feedback[1], ["silly_feedback_function"])
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function"].iloc[0],
            0.1,
        )
        self._start_evaluator_as_snowflake(session)
        records_and_feedback = session.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertListEqual(
            sorted(records_and_feedback[1]),
            ["relevance", "silly_feedback_function"],
        )
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function"].iloc[0],
            0.1,
        )
        self.assertGreaterEqual(
            records_and_feedback[0]["relevance"].iloc[0],
            0.8,
        )

    @pytest.mark.optional
    def test_snowflake_feedback_is_always_deferred(self) -> None:
        session = self.get_session("test_snowflake_feedback_is_always_deferred")
        self._suspend_task()
        f_local = core_feedback.Feedback(silly_feedback_function).on_default()
        f_snowflake = self._get_cortex_relevance_feedback_function()
        tru_app = TruBasicApp(
            text_to_text=lambda _: "Tokyo is the capital of Japan.",
            feedbacks=[f_local, f_snowflake],
        )
        with tru_app:
            tru_app.main_call("What is the capital of Japan?")
        time.sleep(2)
        records_and_feedback = session.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(records_and_feedback[1], ["silly_feedback_function"])
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function"].iloc[0],
            0.1,
        )
        self._start_evaluator_as_snowflake(session)
        records_and_feedback = session.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertListEqual(
            sorted(records_and_feedback[1]),
            ["relevance", "silly_feedback_function"],
        )
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function"].iloc[0],
            0.1,
        )
        self.assertGreaterEqual(
            records_and_feedback[0]["relevance"].iloc[0],
            0.8,
        )

    @pytest.mark.optional
    def test_snowflake_feedback_setup(self) -> None:
        self.get_session("test_snowflake_feedback_setup")
        TruBasicApp(
            text_to_text=lambda t: f"returning {t}",
        )
        # Test stage exists.
        res = self._snowpark_session.sql(
            f"SHOW TERSE STAGES IN SCHEMA {self._database}.{self._schema}"
        ).collect()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].name, ssea._STAGE_NAME)
        self.assertEqual(res[0].database_name, self._database.upper())
        self.assertEqual(res[0].schema_name, self._schema.upper())
        # Test stream exists.
        res = self._snowpark_session.sql(
            f"SHOW TERSE STREAMS IN SCHEMA {self._database}.{self._schema}"
        ).collect()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].name, ssea._STREAM_NAME)
        self.assertEqual(res[0].database_name, self._database.upper())
        self.assertEqual(res[0].schema_name, self._schema.upper())
        self.assertEqual(res[0].tableOn, "TRULENS_FEEDBACKS")
        # Test stored procedure exists.
        res = self._snowpark_session.sql(
            f"SHOW TERSE PROCEDURES LIKE '{ssea._STORED_PROCEDURE_NAME}' IN SCHEMA {self._database}.{self._schema}"
        ).collect()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].name, ssea._STORED_PROCEDURE_NAME)
        self.assertEqual(res[0].schema_name, self._schema.upper())
        # Test task exists.
        res = self._snowpark_session.sql(
            f"SHOW TERSE TASKS IN SCHEMA {self._database}.{self._schema}"
        ).collect()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].name, ssea._TASK_NAME)
        self.assertEqual(res[0].database_name, self._database.upper())
        self.assertEqual(res[0].schema_name, self._schema.upper())

    @pytest.mark.optional
    def test_snowflake_feedback_ran(self) -> None:
        session = self.get_session("test_snowflake_feedback_ran")
        f_snowflake = self._get_cortex_relevance_feedback_function()
        tru_app = TruBasicApp(
            text_to_text=lambda _: "Tokyo is the capital of Japan.",
            feedbacks=[f_snowflake],
        )
        with tru_app:
            tru_app.main_call("What is the capital of Japan?")
        self._wait_till_feedbacks_done(num_expected_feedbacks=1)
        records_and_feedback = session.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertListEqual(
            records_and_feedback[1],
            ["relevance"],
        )
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertGreaterEqual(
            records_and_feedback[0]["relevance"].iloc[0],
            0.8,
        )
        self._call_stored_procedure()  # The stream will have data again due to the computed feedbacks that were deferred but now ran.
        res = self._snowpark_session.sql(
            f"SELECT SYSTEM$STREAM_HAS_DATA('{self._database}.{self._schema}.{ssea._STREAM_NAME}')"
        ).collect()
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 1)
        self.assertFalse(res[0][0])

    @pytest.mark.optional
    def test_japanese(self) -> None:
        session = self.get_session("test_japanese")
        tru_app = TruBasicApp(
            text_to_text=lambda _: "幸士君が世界で一番かわいい赤ちゃんです！",
        )
        with tru_app:
            tru_app.main_call("世界で一番かわいい赤ちゃんは誰ですか?")
        records_and_feedback = session.get_records_and_feedback()
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        res = self.run_query(
            f"SELECT INPUT, OUTPUT FROM {self._database}.{self._schema}.TRULENS_RECORDS"
        )
        self.assertEqual(1, len(res))
        self.assertEqual(
            res[0].INPUT, '"世界で一番かわいい赤ちゃんは誰ですか?"'
        )
        self.assertEqual(
            res[0].OUTPUT, '"幸士君が世界で一番かわいい赤ちゃんです！"'
        )

    @pytest.mark.optional
    def test_snowflake_feedback_only_runs_cortex(self) -> None:
        self._get_cortex_relevance_feedback_function()  # no error
        with self.assertRaisesRegex(
            ValueError,
            "`SnowflakeFeedback` can only support feedback functions defined in `trulens-providers-cortex` package's, `trulens.providers.cortex.provider.Cortex` class!",
        ):
            core_feedback.SnowflakeFeedback(openai_provider.OpenAI().relevance)

    @pytest.mark.optional
    def test_stored_procedure(self) -> None:
        session = self.get_session("test_stored_procedure")
        self._suspend_task()
        f_snowflake = self._get_cortex_relevance_feedback_function()
        tru_app = TruBasicApp(
            text_to_text=lambda _: "Tokyo is the capital of Japan.",
            feedbacks=[f_snowflake],
        )
        with tru_app:
            tru_app.main_call("What is the capital of Japan?")
        time.sleep(2)
        records_and_feedback = session.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(len(records_and_feedback[1]), 0)
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        with self.get_snowpark_session_with_schema(
            self._schema
        ) as snowpark_session:
            ssesp.run(snowpark_session)
        records_and_feedback = session.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertListEqual(
            records_and_feedback[1],
            ["relevance"],
        )
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertGreaterEqual(
            records_and_feedback[0]["relevance"].iloc[0],
            0.8,
        )
