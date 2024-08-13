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

from tests.unit.utils import optional_test
from tests.util.snowflake_test_case import SnowflakeTestCase


def silly_feedback_function(q: str) -> float:
    return 0.1


class TestSnowflakeFeedbackEvaluation(SnowflakeTestCase):
    @optional_test
    def test_local_deferred_mode(self) -> None:
        tru = Tru()
        tru.reset_database()
        f = Feedback(silly_feedback_function).on_default()
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
        self.assertEqual(records_and_feedback[1], ["silly_feedback_function"])
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function"].iloc[0],
            0.1,
        )

    @optional_test
    def test_snowflake_deferred_mode(self) -> None:
        # TODO(this_pr): refactor so that I'm not duplicating so much code between this test and the local one!
        tru = self.get_tru("test_snowflake_deferred_mode")
        f = SnowflakeFeedback(silly_feedback_function).on_default()
        tru_app = TruBasicApp(
            text_to_text=lambda t: f"returning {t}",
            feedbacks=[f],
            feedback_mode=FeedbackMode.DEFERRED,
        )
        with tru_app:
            tru_app.main_call("test_snowflake_deferred_mode")
        tru.start_evaluator()
        time.sleep(2)
        tru.stop_evaluator()
        records_and_feedback = tru.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(len(records_and_feedback[1]), 0)
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        tru.start_evaluator(run_location=FeedbackRunLocation.SNOWFLAKE)
        time.sleep(2)
        tru.stop_evaluator()
        records_and_feedback = tru.get_records_and_feedback()
        self.assertEqual(len(records_and_feedback), 2)
        self.assertEqual(records_and_feedback[1], ["silly_feedback_function"])
        self.assertEqual(records_and_feedback[0].shape[0], 1)
        self.assertEqual(
            records_and_feedback[0]["silly_feedback_function"].iloc[0],
            0.1,
        )


if __name__ == "__main__":
    main()
