"""
Tests server-side feedback evaluations in Snowflake.
"""

import time
from unittest import main

from trulens.core import Feedback
from trulens.core import SnowflakeFeedback
from trulens.core import Tru
from trulens.core import TruBasicApp
from trulens.core.schema.feedback import FeedbackMode
from trulens.core.schema.feedback import FeedbackRunLocation

from tests.test import optional_test
from tests.util.snowflake_test_case import SnowflakeTestCase


def silly_feedback_function_1(q: str) -> float:
    return 0.1


def silly_feedback_function_2(q: str) -> float:
    return 0.2


class TestSnowflakeFeedbackEvaluation(SnowflakeTestCase):
    @optional_test
    def test_local_deferred_mode(self) -> None:
        tru = Tru()
        tru.reset_database()
        f = Feedback(silly_feedback_function_1).on_default()
        tru_app = TruBasicApp(
            text_to_text=lambda t: f"returning {t}",
            feedbacks=[f],
            feedback_mode=FeedbackMode.DEFERRED,
        )
        with tru_app:
            tru_app.main_call("test_local_deferred_mode")
        time.sleep(2)
        records_and_feedback = tru.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(len(records_and_feedback[1]), 0)
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        tru.start_evaluator()
        time.sleep(2)
        tru.stop_evaluator()
        records_and_feedback = tru.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(records_and_feedback[1], ["silly_feedback_function_1"])
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function_1"].iloc[0],
            0.1,
        )

    @optional_test
    def test_snowflake_deferred_mode(self) -> None:
        tru = self.get_tru("test_snowflake_deferred_mode")
        f_local = Feedback(silly_feedback_function_1).on_default()
        f_snowflake = SnowflakeFeedback(silly_feedback_function_2).on_default()
        tru_app = TruBasicApp(
            text_to_text=lambda t: f"returning {t}",
            feedbacks=[f_local, f_snowflake],
            feedback_mode=FeedbackMode.DEFERRED,
        )
        with tru_app:
            tru_app.main_call("test_snowflake_deferred_mode")
        time.sleep(2)
        tru.start_evaluator()
        time.sleep(2)
        tru.stop_evaluator()
        records_and_feedback = tru.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(records_and_feedback[1], ["silly_feedback_function_1"])
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function_1"].iloc[0],
            0.1,
        )
        tru.start_evaluator(run_location=FeedbackRunLocation.SNOWFLAKE)
        time.sleep(2)
        tru.stop_evaluator()
        records_and_feedback = tru.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertListEqual(
            sorted(records_and_feedback[1]),
            ["silly_feedback_function_1", "silly_feedback_function_2"],
        )
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function_1"].iloc[0],
            0.1,
        )
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function_2"].iloc[0],
            0.2,
        )

    @optional_test
    def test_snowflake_feedback_is_always_deferred(self) -> None:
        tru = self.get_tru("test_snowflake_feedback_is_always_deferred")
        f_local = Feedback(silly_feedback_function_1).on_default()
        f_snowflake = SnowflakeFeedback(silly_feedback_function_2).on_default()
        tru_app = TruBasicApp(
            text_to_text=lambda t: f"returning {t}",
            feedbacks=[f_local, f_snowflake],
        )
        with tru_app:
            tru_app.main_call("test_snowflake_feedback_is_always_deferred")
        time.sleep(2)
        records_and_feedback = tru.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(records_and_feedback[1], ["silly_feedback_function_1"])
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function_1"].iloc[0],
            0.1,
        )
        tru.start_evaluator(run_location=FeedbackRunLocation.SNOWFLAKE)
        time.sleep(2)
        tru.stop_evaluator()
        records_and_feedback = tru.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertListEqual(
            sorted(records_and_feedback[1]),
            ["silly_feedback_function_1", "silly_feedback_function_2"],
        )
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function_1"].iloc[0],
            0.1,
        )
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function_2"].iloc[0],
            0.2,
        )

    @optional_test
    def test_snowflake_feedback_setup(self) -> None:
        self.get_tru("test_snowflake_feedback_setup")
        TruBasicApp(
            text_to_text=lambda t: f"returning {t}",
        )
        # Test stream exists.
        # TODO: parameterize sql better.
        res = self._snowflake_session.sql(
            f"SHOW TERSE STREAMS IN SCHEMA {self._database_name}.{self._schema_name}"
        ).collect()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].name, "TRULENS_FEEDBACK_EVALS_STREAM")
        self.assertEqual(res[0].database_name, self._database_name.upper())
        self.assertEqual(res[0].schema_name, self._schema_name.upper())
        self.assertEqual(res[0].tableOn, "TRULENS_FEEDBACKS")
        # Test task exists.
        res = self._snowflake_session.sql(
            f"SHOW TERSE TASKS IN SCHEMA {self._database_name}.{self._schema_name}"
        ).collect()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].name, "TRULENS_FEEDBACK_EVAL_TASK")
        self.assertEqual(res[0].database_name, self._database_name.upper())
        self.assertEqual(res[0].schema_name, self._schema_name.upper())
        # Test stored procedure exists.
        res = self._snowflake_session.sql(
            f"SHOW TERSE PROCEDURES LIKE 'TRULENS_RUN_DEFERRED_FEEDBACKS' IN SCHEMA {self._database_name}.{self._schema_name} "
        ).collect()
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0].name, "TRULENS_RUN_DEFERRED_FEEDBACKS")
        self.assertEqual(res[0].schema_name, self._schema_name.upper())
        # TODO(this_pr): Also check for stage, secret, EAI, network rule


if __name__ == "__main__":
    main()
