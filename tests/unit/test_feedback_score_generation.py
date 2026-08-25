"""
Test suites meant for testing the reliability and robustness of the regex
pattern matching of feedback scores from LLM responses.
"""

import pytest
from trulens.feedback import generated as feedback_generated

test_data = [
    ("The relevance score is 7.", 7),
    ("I rate this an 8 out of 10.", 8),
    ("In the range of 0-10, I give this a 9.", 9),
    ("This should be a 10!", 10),
    ("The score is 5", 5),
    ("A perfect score: 10.", 10),
    ("Irrelevant text 123 Main Street.", None),
    ("Score: 9.", 9),
    ("7", 7),
    ("This deserves a 6, I believe.", 6),
    ("Not relevant. Score: 0.", 0),
    ("Some text here. Score: 10.0", 10.0),
    ("Score: 4.5", 4.5),
    ("Score is 8.333", 8.333),
]


@pytest.mark.parametrize("test_input,expected", test_data)
def test_re_0_10_rating(test_input, expected):
    """Check that re_0_10_rating can extract the correct score from a string."""

    try:
        result = feedback_generated.re_0_10_rating(test_input)
    except feedback_generated.ParseError:
        result = None

    assert result == expected, (
        f"Failed on {test_input}: expected {expected}, got {result}"
    )


# A judge that states its own scale before answering used to have those bounds
# read as candidate ratings, so the minimum was returned instead of the rating.
scale_statement_data = [
    ("On a scale of 0 to 10, I rate this 8", 8),
    ("Using a 0 to 10 scale, the answer deserves a 7.", 7),
    ("Rate from 0-10. Score: 9", 9),
    ("Between 0 and 10, this is a 6.", 6),
    ("Scoring 0 through 10, I give it a 4.", 4),
    ("Legend: 0 = irrelevant, 10 = perfect. I give this a 8.", 8),
]


@pytest.mark.parametrize("test_input,expected", scale_statement_data)
def test_re_0_10_rating_ignores_a_stated_scale(test_input, expected):
    """The stated bounds are not candidate ratings."""

    result = feedback_generated.re_0_10_rating(test_input)

    assert result == expected, (
        f"Failed on {test_input}: expected {expected}, got {result}"
    )


configured_scale_data = [
    ("On a scale of 0 to 3, I rate this 3", 3),
    ("Score 0-3: the context is fully relevant, so 3", 3),
    ("Rating scale 0 to 3. My rating: 2", 2),
    ("Given the criteria (0 = irrelevant, 3 = highly relevant), I give 3", 3),
]


@pytest.mark.parametrize("test_input,expected", configured_scale_data)
def test_re_configured_rating_ignores_a_stated_scale(test_input, expected):
    """Same, on the 0-3 scale re_configured_rating defaults to."""

    result = feedback_generated.re_configured_rating(test_input)

    assert result == expected, (
        f"Failed on {test_input}: expected {expected}, got {result}"
    )


out_of_data = [
    ("The rating is 1 out of 3.", 1),
]


@pytest.mark.parametrize("test_input,expected", out_of_data)
def test_re_configured_rating_keeps_out_of_phrasing(test_input, expected):
    """ "N out of M" is not a scale statement and keeps its existing handling."""

    result = feedback_generated.re_configured_rating(test_input)

    assert result == expected, (
        f"Failed on {test_input}: expected {expected}, got {result}"
    )
