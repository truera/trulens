"""
Test suites meant for testing the reliability and robustness of the regex
pattern matching of feedback scores from LLM responses.
"""

from typing import ClassVar

import pytest
from trulens.core.feedback import endpoint as core_endpoint
from trulens.feedback import generated as feedback_generated
from trulens.feedback import llm_provider

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


# A judge's reason often contains numbers of its own. All in-range numbers used to
# be candidates and the smallest won, so a count in the reason beat the score.
labelled_score_data = [
    ("Score: 3\nReason: answers 2 of the 3 parts", 0, 3, 3),
    ("The context mentions FY2024 and gives 1 exact figure. Score: 3", 0, 3, 3),
    ("Criteria 1: met. Criteria 2: met. Score: 9", 0, 10, 9),
    # Two labelled scores: the judge's last word is its final score.
    (
        "Score: 2. On reflection the context covers it all. Final score: 3",
        0,
        3,
        3,
    ),
    # "2/5" is not on the 0-3 scale, so it is not read as a labelled score.
    ("The context covers 2/5 cases. Score: 3", 0, 3, 3),
    # A "Score:" label is preferred over a fraction in the reason.
    ("Score: 3. It covers 2/3 of the question's parts.", 0, 3, 3),
]


@pytest.mark.parametrize(
    "test_input,min_score_val,max_score_val,expected", labelled_score_data
)
def test_re_configured_rating_prefers_a_labelled_score(
    test_input, min_score_val, max_score_val, expected
):
    """A score the judge labels wins over other numbers in its reason."""

    result = feedback_generated.re_configured_rating(
        test_input, min_score_val, max_score_val
    )

    assert result == expected, (
        f"Failed on {test_input}: expected {expected}, got {result}"
    )


# Replies that were already read correctly, and keep their result.
labelled_score_control_data = [
    ("3", 0, 3, 3),
    ("Score: 8", 0, 10, 8),
    ("8/10", 0, 10, 8),
    ("Rating: 8 out of 10", 0, 10, 8),
    ("**Score:** 8", 0, 10, 8),
    ("Final score: 2 (on a 0-3 scale)", 0, 3, 2),
    # The case the min() was written for.
    ("1 out of 3", 0, 3, 1),
]


@pytest.mark.parametrize(
    "test_input,min_score_val,max_score_val,expected",
    labelled_score_control_data,
)
def test_re_configured_rating_labelled_score_controls(
    test_input, min_score_val, max_score_val, expected
):
    """Replies that were already read correctly are unchanged."""

    result = feedback_generated.re_configured_rating(
        test_input, min_score_val, max_score_val
    )

    assert result == expected, (
        f"Failed on {test_input}: expected {expected}, got {result}"
    )


def test_re_configured_rating_out_of_range_labelled_score_falls_back():
    """An out-of-range labelled score is ignored, as before: the other
    in-range numbers are read the old way, and with none the parse fails."""

    assert (
        feedback_generated.re_configured_rating(
            "Score: 12. It meets 2 of the 3 criteria.", 0, 10
        )
        == 2
    )
    with pytest.raises(feedback_generated.ParseError):
        feedback_generated.re_configured_rating("Score: 12", 0, 10)


class _FixedReplyProvider(llm_provider.LLMProvider):
    """A judge that always gives the same text reply."""

    model_config: ClassVar[dict] = {"extra": "allow"}

    reply: ClassVar[str] = ""

    def _create_chat_completion(
        self, prompt=None, messages=None, response_format=None, **kwargs
    ) -> str:
        return self.reply


@pytest.mark.parametrize(
    "reply",
    [
        "Score: 3\nReason: the context answers 2 of the 3 parts of the question.",
        "The context mentions FY2024 and gives 1 exact figure, fully relevant. "
        "Score: 3",
    ],
)
def test_context_relevance_reads_the_labelled_score(monkeypatch, reply):
    """A judge's top score normalizes to 1.0 even when its reason has numbers."""

    provider = _FixedReplyProvider(
        endpoint=core_endpoint.Endpoint(name="fake"), model_engine="fake"
    )
    monkeypatch.setattr(_FixedReplyProvider, "reply", reply)

    score = provider.context_relevance("question", "context")
    cot_score, _ = provider.context_relevance_with_cot_reasons(
        "question", "context"
    )

    assert score == 1.0
    assert cot_score == 1.0


# Raised in review on #2726: when stripping the scale leaves MORE than one candidate,
# the old code fell back to the un-stripped string and re-injected the bounds, so the
# floor won anyway. The stripped text is now used consistently.
multi_candidate_data = [
    ("scale of 0 to 3, I give 2 or 3", 2),
    ("Rating 0 to 3: between 2 and 3, call it 3", 2),
]


@pytest.mark.parametrize("test_input,expected", multi_candidate_data)
def test_re_configured_rating_stripped_text_used_throughout(
    test_input, expected
):
    """The stated bounds do not come back when several candidates remain."""

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
