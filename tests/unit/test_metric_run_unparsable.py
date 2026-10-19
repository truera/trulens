"""Metric.run keeps unparsable judge sentinels out of its aggregate.

These live outside test_feedback.py because that module imports
test_otel_tru_chain, whose module-level skip under LangChain 1.x also skips
every test in the importing file.
"""

import os

# CRITICAL: Set OTEL environment variable BEFORE any TruLens imports
# This must be done before any TruSession is created to avoid freezing the experimental flag
os.environ["TRULENS_OTEL_TRACING"] = "0"

from unittest import TestCase

from trulens.core.feedback import feedback as core_feedback
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import select as select_schema
from trulens.core.utils import constants as constants_utils

# Get the "globally importable" feedback implementations.
from tests.unit import feedbacks as test_feedbacks


class TestMetricRunUnparsable(TestCase):
    """A judge failure is not averaged in as a low score."""

    def test_unparsable_score_excluded_from_aggregation(self) -> None:
        f = core_feedback.Feedback(imp=test_feedbacks.echo_score).on(
            val=select_schema.Select.RecordCalls.somemethod.args.num[:]
        )

        # The selected values stand in for judge replies: the last one is the
        # sentinel that providers report when a reply had no parseable score.
        source_data = {
            "__record__": {
                "app": {
                    "somemethod": {
                        "args": {
                            "num": [
                                0.5,
                                0.1,
                                constants_utils.UNPARSABLE_SCORE,
                            ]
                        }
                    }
                }
            }
        }

        res = f.run(source_data=source_data)

        sentinel_included = (0.5 + 0.1 + constants_utils.UNPARSABLE_SCORE) / 3
        self.assertNotAlmostEqual(sentinel_included, (0.5 + 0.1) / 2)
        # Make sure the sentinel-included average is not accidentally equal to
        # the average of the parsable scores.

        self.assertAlmostEqual(res.result, (0.5 + 0.1) / 2)
        # Only the parsable scores should have been aggregated.

        self.assertEqual(res.status, feedback_schema.FeedbackResultStatus.DONE)

    def test_genuine_negative_scores_still_aggregate(self) -> None:
        """Only the exact sentinel is dropped from the aggregate."""

        f = core_feedback.Feedback(imp=test_feedbacks.echo_score).on(
            val=select_schema.Select.RecordCalls.somemethod.args.num[:]
        )

        source_data = {
            "__record__": {
                "app": {"somemethod": {"args": {"num": [0.5, -0.5]}}}
            }
        }

        res = f.run(source_data=source_data)

        self.assertAlmostEqual(res.result, 0.0)
        # A genuine negative score such as -0.5 is still aggregated.

    def test_all_unparsable_scores_report_the_sentinel(self) -> None:
        """With nothing parsable left the failure stays visible."""

        f = core_feedback.Feedback(imp=test_feedbacks.echo_score).on(
            val=select_schema.Select.RecordCalls.somemethod.args.num[:]
        )

        source_data = {
            "__record__": {
                "app": {
                    "somemethod": {
                        "args": {"num": [constants_utils.UNPARSABLE_SCORE] * 2}
                    }
                }
            }
        }

        res = f.run(source_data=source_data)

        self.assertEqual(res.result, constants_utils.UNPARSABLE_SCORE)
        self.assertEqual(res.status, feedback_schema.FeedbackResultStatus.DONE)
