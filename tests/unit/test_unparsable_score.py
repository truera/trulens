"""The unparsable-score sentinel is one shared value with one helper."""

import math

import pytest
from trulens.core.utils import constants as constants_utils
from trulens.feedback import UNPARSABLE_SCORE
from trulens.feedback import is_unparsable_score
from trulens.feedback import llm_provider


def test_sentinel_is_shared_between_core_and_feedback():
    assert UNPARSABLE_SCORE == -1.0
    assert llm_provider.UNPARSABLE_SCORE == constants_utils.UNPARSABLE_SCORE
    assert (
        llm_provider.is_unparsable_score is constants_utils.is_unparsable_score
    )


@pytest.mark.parametrize(
    "score, expected",
    [
        (-1.0, True),
        (-1, True),
        (0.0, False),
        (0.5, False),
        (1.0, False),
        (-0.5, False),
        (math.nan, False),
        (math.inf, False),
    ],
)
def test_is_unparsable_score(score, expected):
    assert is_unparsable_score(score) is expected


def test_mean_graded_score_skips_the_sentinel():
    assert llm_provider._mean_graded_score([1.0, UNPARSABLE_SCORE]) == 1.0
    assert (
        llm_provider._mean_graded_score([UNPARSABLE_SCORE]) == UNPARSABLE_SCORE
    )


def test_criteria_ab_test_skips_unparsable_replies():
    """A variant that fails to parse must not charge the judge -1.0 (#2997)."""
    from trulens.core.utils.constants import UNPARSABLE_SCORE

    criteria_ab_test = pytest.importorskip("trulens.benchmark.criteria_ab_test")
    CriteriaABTest = criteria_ab_test.CriteriaABTest

    golden = [
        {"query": f"q{i}", "expected_response": "r", "expected_score": 1.0}
        for i in range(4)
    ]

    def good(q, r, **kwargs):
        return (1.0, {})

    def sometimes_unparsable(q, r, **kwargs):
        return (UNPARSABLE_SCORE, {}) if q == "q3" else (1.0, {})

    report = CriteriaABTest(
        variant_a={"name": "a", "fn": good},
        variant_b={"name": "b", "fn": sometimes_unparsable},
        golden_set=golden,
    ).run()

    assert list(report.scores_a) == [1.0, 1.0, 1.0]
    assert list(report.scores_b) == [1.0, 1.0, 1.0]


def test_cross_model_alignment_skips_unparsable_replies():
    """A judge that fails to parse must not enter the alignment as -1.0 (#2997)."""
    from trulens.core.utils.constants import UNPARSABLE_SCORE

    cross_model_alignment = pytest.importorskip(
        "trulens.benchmark.cross_model_alignment"
    )
    CrossModelAlignment = cross_model_alignment.CrossModelAlignment

    golden = [
        {"query": f"q{i}", "expected_response": "r", "expected_score": 1.0}
        for i in range(4)
    ]

    class ReliableProvider:
        def judge_fn(self, q, r):
            return (1.0, {})

    class FlakyProvider:
        def judge_fn(self, q, r):
            return (UNPARSABLE_SCORE, {}) if q == "q3" else (1.0, {})

    report = CrossModelAlignment(
        judges=[
            {"name": "reliable", "provider": ReliableProvider()},
            {"name": "flaky", "provider": FlakyProvider()},
        ],
        feedback_method="judge_fn",
        golden_set=golden,
        args_fn=lambda r: (r["query"], r["expected_response"]),
    ).run()

    assert [len(scores) for scores in report.scores] == [3, 3]
