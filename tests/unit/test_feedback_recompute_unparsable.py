"""A stored unparsable judge score is not "already computed".

This lives outside test_otel_feedback_computation.py because that module
imports test_otel_tru_chain, whose module-level skip under LangChain 1.x also
skips every test in the importing file.
"""

import os

# CRITICAL: Set OTEL environment variable BEFORE any TruLens imports
# This must be done before any TruSession is created to avoid freezing the experimental flag
os.environ["TRULENS_OTEL_TRACING"] = "0"

from unittest import TestCase

import pandas as pd
from trulens.core.feedback.feedback_function_input import FeedbackFunctionInput
from trulens.core.utils.constants import UNPARSABLE_SCORE
from trulens.feedback.computer import _remove_already_computed_feedbacks
from trulens.otel.semconv.trace import SpanAttributes


class TestRecomputeUnparsableFeedback(TestCase):
    def test__remove_already_computed_feedbacks_recomputes_unparsable_score(
        self,
    ) -> None:
        """A stored UNPARSABLE_SCORE means the judge failed, not that the
        feedback was computed, so the input must be offered for computation
        again. A real score, including zero, still marks it as computed."""

        def eval_root(score: float | None) -> dict:
            attributes = {
                SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.EVAL_ROOT,
                SpanAttributes.EVAL_ROOT.METRIC_NAME: "feedback1",
                SpanAttributes.RECORD_ID: "record_id1",
                SpanAttributes.EVAL_ROOT.SPAN_GROUP: "span_group1",
                SpanAttributes.EVAL_ROOT.ARGS_SPAN_ID + ".a": "span_id1a",
                SpanAttributes.EVAL_ROOT.ARGS_SPAN_ATTRIBUTE
                + ".a": "span_attribute1a",
            }
            if score is not None:
                attributes[SpanAttributes.EVAL_ROOT.SCORE] = score
            return attributes

        flattened_inputs = [
            (
                "record_id1",
                "span_group1",
                {
                    "a": FeedbackFunctionInput(
                        span_id="span_id1a", span_attribute="span_attribute1a"
                    )
                },
            )
        ]
        for label, score, expected in (
            ("parsable score", 0.5, []),
            ("zero score", 0.0, []),
            ("unparsable sentinel", UNPARSABLE_SCORE, flattened_inputs),
            ("no score", None, flattened_inputs),
        ):
            with self.subTest(label=label):
                events = pd.DataFrame({"record_attributes": [eval_root(score)]})
                res = _remove_already_computed_feedbacks(
                    events, "feedback1", flattened_inputs
                )
                self.assertEqual(expected, res)
