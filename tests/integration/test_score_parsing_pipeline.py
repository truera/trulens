"""
Integration tests verifying score parsing through the full feedback pipeline
with various realistic LLM response formats.

Each test uses a mock LLM provider that returns a specific response format,
exercising the complete ``generate_score()`` path: prompt creation, endpoint
invocation, JSON extraction attempt, string-based score parsing via
``re_0_10_rating``, and final normalization to [0.0, 1.0].
"""

from unittest.mock import patch

import pytest

from trulens.feedback.generated import ParseError, re_0_10_rating


# ---------------------------------------------------------------------------
# Mock response variants
# ---------------------------------------------------------------------------

# Each tuple: (description, mock_response, expected_score, expect_parse_error)
# expected_score is the normalized value on [0.0, 1.0].
# Set expect_parse_error=True when the score cannot be extracted and should
# raise ParseError.
MOCK_RESPONSE_VARIANTS = [
    pytest.param(
        "plain number",
        "2",
        0.2,  # (2 - 0) / 10
        False,
        id="plain_number",
    ),
    pytest.param(
        "full explanation",
        "The relevance is moderate. Score: 2",
        0.2,
        False,
        id="with_explanation",
    ),
    pytest.param(
        "json with score key",
        '{"criteria": "relevance check", "supporting_evidence": "...", "score": 2}',
        0.2,
        False,
        id="json_format",
    ),
    pytest.param(
        "json with markdown fence",
        '```json\n{"score": 2}\n```',
        0.2,
        False,
        id="markdown_json",
    ),
    pytest.param(
        "whitespace-surrounded score",
        "\n  Score: 2  \n",
        0.2,
        False,
        id="whitespace_score",
    ),
    pytest.param(
        "float score",
        "2.5",
        0.25,  # (2.5 - 0) / 10
        False,
        id="float_format",
    ),
    pytest.param(
        "score at upper bound",
        "Score: 10.0",
        1.0,  # (10 - 0) / 10
        False,
        id="upper_bound",
    ),
    pytest.param(
        "score at lower bound",
        "Score: 0",
        0.0,
        False,
        id="lower_bound",
    ),
    pytest.param(
        "unparseable string",
        "No score here at all, just text.",
        None,
        True,
        id="unparseable",
    ),
]


# ---------------------------------------------------------------------------
# Unit-level: verify re_0_10_rating handles each format
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "description,mock_response,expected_normalized,expect_parse_error",
    [
        p
        for p in MOCK_RESPONSE_VARIANTS
        if not p.values[3]  # exclude unparseable
    ],
)
def test_re_0_10_rating_extracts_score(
    description, mock_response, expected_normalized, expect_parse_error
):
    """``re_0_10_rating`` extracts the raw score (not normalized) from each
    response format without raising ParseError."""
    try:
        raw_score = re_0_10_rating(mock_response)
    except ParseError:
        pytest.fail(
            f"ParseError raised unexpectedly for '{description}': "
            f"{mock_response!r}"
        )

    # raw_score is integer or float in range [0, 10]
    assert 0 <= raw_score <= 10, (
        f"Raw score {raw_score} out of valid range [0, 10] "
        f"for '{description}': {mock_response!r}"
    )

    normalized = raw_score / 10.0
    assert normalized == pytest.approx(expected_normalized, abs=1e-9), (
        f"Normalized score mismatch for '{description}': "
        f"expected ~{expected_normalized}, got ~{normalized} "
        f"(raw={raw_score}) from {mock_response!r}"
    )


def test_re_0_10_rating_parse_error():
    """``re_0_10_rating`` raises ParseError on truly unparseable input."""
    with pytest.raises(ParseError):
        re_0_10_rating("No score here at all, just text.")


# ---------------------------------------------------------------------------
# Integration: full Feedback pipeline with mock provider
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_openai():
    """Creates a mock OpenAI provider whose ``_create_chat_completion`` is
    patched to return controlled responses."""
    from trulens.providers.openai import OpenAI
    from trulens.providers.openai.endpoint import OpenAIEndpoint

    # Create a minimal endpoint; the API key is not needed since we mock
    # _create_chat_completion before any real network call.
    endpoint = OpenAIEndpoint(api_key="sk-test-placeholder")
    provider = OpenAI(endpoint=endpoint, model_engine="gpt-4o-mini")
    return provider


@pytest.mark.parametrize(
    "description,mock_response,expected_score,expect_parse_error",
    [p for p in MOCK_RESPONSE_VARIANTS if not p.values[3]],
)
def test_generate_score_full_pipeline(
    mock_openai, description, mock_response, expected_score, expect_parse_error
):
    """``generate_score()`` correctly parses each response format through the
    full feedback pipeline (completion call → JSON extraction → string parsing
    → normalization)."""
    with patch.object(
        mock_openai, "_create_chat_completion", return_value=mock_response
    ):
        result = mock_openai.generate_score(
            system_prompt="Evaluate the quality.",
            user_prompt="Score this response from 0 to 10.",
        )

        if isinstance(result, tuple):
            score, reason = result
        else:
            score = result

        assert isinstance(score, float), (
            f"generate_score() should return float for '{description}', "
            f"got {type(score).__name__}: {score!r}"
        )
        assert 0.0 <= score <= 1.0, (
            f"Normalized score {score} out of [0.0, 1.0] range "
            f"for '{description}'"
        )
        assert score == pytest.approx(expected_score, abs=1e-9), (
            f"Score mismatch for '{description}': "
            f"expected ~{expected_score}, got ~{score} "
            f"from response {mock_response!r}"
        )


@pytest.mark.parametrize(
    "description,mock_response,expected_score,expect_parse_error",
    [p for p in MOCK_RESPONSE_VARIANTS if p.values[3]],
)
def test_generate_score_unparseable_response(
    mock_openai, description, mock_response, expected_score, expect_parse_error
):
    """``generate_score()`` raises ValueError or returns invalid score on
    truly unparseable responses (cannot fall back to JSON or regex)."""
    with patch.object(
        mock_openai, "_create_chat_completion", return_value=mock_response
    ):
        with pytest.raises(Exception):
            mock_openai.generate_score(
                system_prompt="Evaluate the quality.",
                user_prompt="Score this response from 0 to 10.",
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "description,response_string,raw_score",
    [
        ("score 5", "5", 5),
        ("score 10", "The answer is 10 out of 10.", 10),
        ("score 0", "0", 0),
        (
            "json with extra fields",
            '{"score": 7, "criteria": "accuracy", "supporting_evidence": "..."}',
            7,
        ),
        (
            "json array of scores",
            '[{"score": 4}, {"score": 6}]',
            None,  # arrays get averaged → (4+6)/2 = 5, normalized 0.5
        ),
    ],
)
def test_re_0_10_rating_edge_cases(description, response_string, raw_score):
    """Edge case handling for various response formats."""
    try:
        result = re_0_10_rating(response_string)
    except ParseError:
        if raw_score is not None:
            pytest.fail(
                f"Unexpected ParseError for '{description}' → "
                f"expected raw_score={raw_score}"
            )
        return  # expected failure

    if raw_score is not None:
        assert result == raw_score, (
            f"Expected raw_score={raw_score} for '{description}', "
            f"got {result}"
        )


def test_generate_score_roundtrip_cacheable():
    """The score extraction pipeline produces consistent results when the
    same input is processed twice (idempotency check)."""
    from trulens.providers.openai import OpenAI
    from trulens.providers.openai.endpoint import OpenAIEndpoint

    endpoint = OpenAIEndpoint(api_key="sk-test-placeholder")
    provider = OpenAI(endpoint=endpoint, model_engine="gpt-4o-mini")
    mock_response = "The quality is good. Score: 7"

    scores = []
    for _ in range(3):
        with patch.object(
            provider, "_create_chat_completion", return_value=mock_response
        ):
            result = provider.generate_score(
                system_prompt="Evaluate quality.",
                user_prompt="Score from 0 to 10.",
            )
            if isinstance(result, tuple):
                score, _ = result
            else:
                score = result
            scores.append(score)

    assert all(s == scores[0] for s in scores), (
        f"generate_score() should be idempotent, got {scores}"
    )


def test_generate_score_with_custom_range():
    """Score normalization respects custom min/max values."""
    from trulens.providers.openai import OpenAI
    from trulens.providers.openai.endpoint import OpenAIEndpoint

    endpoint = OpenAIEndpoint(api_key="sk-test-placeholder")
    provider = OpenAI(endpoint=endpoint, model_engine="gpt-4o-mini")

    # Response score is 50, range is 0-100 → normalized = 0.5
    mock_response = "Score: 50"
    with patch.object(
        provider, "_create_chat_completion", return_value=mock_response
    ):
        result = provider.generate_score(
            system_prompt="Evaluate",
            user_prompt="Score from 0 to 100.",
            min_score_val=0,
            max_score_val=100,
        )
        if isinstance(result, tuple):
            score, _ = result
        else:
            score = result
        assert score == pytest.approx(0.5, abs=1e-9)
