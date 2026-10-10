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
