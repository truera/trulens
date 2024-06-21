"""
Test suites meant for testing the reliability and robustness of the regex
pattern matching of feedback scores from LLM responses.
"""

import pytest

from trulens_eval.utils.generated import ParseError
from trulens_eval.utils.generated import re_0_10_rating

test_data = [
    ("The relevance score is 7.", 7),
    ("I rate this an 8 out of 10.", 8),
    ("In the range of 0-10, I give this a 9.", 0),
    # Currently does not have ideal handling as it returns the minimum integer found.
    ("This should be a 10!", 10),
    ("The score is 5", 5),
    ("A perfect score: 10.", 10),
    ("Irrelevant text 123 Main Street.", None),
    ("Score: 9.", 9),
    ("7", 7),
    ("This deserves a 6, I believe.", 6),
    ("Not relevant. Score: 0.", 0),
    ("Some text here. Score: 10.0", 10),
    ("Score: 4.5", 4),
    ("Score is 8.333", 8)
]


@pytest.mark.parametrize("test_input,expected", test_data)
def test_re_0_10_rating(test_input, expected):
    """Check that re_0_10_rating can extract the correct score from a string."""

    try:
        result = re_0_10_rating(test_input)
    except ParseError:
        result = None

    assert result == expected, f"Failed on {test_input}: expected {expected}, got {result}"
