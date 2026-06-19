"""Integration test verifying trulens.feedback.llm_provider.LLMProvider.generate_score
parses scores correctly end-to-end through all three response branches:
JSON, structured BaseFeedbackResponse, and plain-string regex fallback
(trulens.feedback.generated.re_configured_rating).

Complements tests/unit/test_feedback_score_generation.py, which tests
re_0_10_rating in isolation. This test exercises the real generate_score
control flow, including JSON-parse-first behavior and fallthrough on
malformed/non-scored JSON.
"""

from typing import ClassVar, Optional

import pytest
from trulens.feedback import generated as feedback_generated
from trulens.feedback import llm_provider


class _StubEndpoint:
    """Minimal real endpoint stub. Only implements what run_in_pace needs:
    pace_me() and retries. No network, no rate limiting.
    """

    def __init__(self):
        self.retries = 0

    def pace_me(self):
        return None

    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class MockLLMProvider(llm_provider.LLMProvider):
    """LLMProvider with a real (stub) endpoint, so generate_score's actual
    parsing branches execute rather than being bypassed. Only
    _create_chat_completion is overridden, to return a canned response per
    test case.
    """

    model_config: ClassVar[dict] = {"extra": "allow"}

    canned_response: Optional[object] = None

    def __init__(self, **kwargs):
        super().__init__(
            endpoint=_StubEndpoint(),
            model_engine="mock-model",
            **kwargs,
        )

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        response_format=None,
        **kwargs,
    ):
        return self.canned_response


@pytest.fixture
def provider():
    return MockLLMProvider()


# response, expected normalized score (0-1 scale, min=0 max=10), or "raises"
JSON_CASES = [
    ('{"score": 7}', 0.7),
    ('{"score": 0}', 0.0),
    ('{"score": 10}', 1.0),
    ('[{"score": 4}, {"score": 6}]', 0.5),  # list branch: averaged
]

PLAIN_STRING_CASES = [
    ("The relevance score is 7.", 0.7),
    ("I rate this an 8 out of 10.", 0.8),
    ("Score: 9.", 0.9),
    ("Score: 4.5", 0.45),
]

MALFORMED_JSON_FALLTHROUGH_CASES = [
    # Not valid JSON at all -> should fall through to regex branch.
    ('{"score": 7', 0.7),
    # Valid JSON but no "score" key -> falls through json.loads success
    # but fails the isinstance/key check, so falls through to regex branch
    # which should then fail to find a number and raise ParseError.
    ('{"rating": 7}', "raises"),
]


@pytest.mark.parametrize("response,expected", JSON_CASES)
def test_generate_score_json_branch(provider, response, expected):
    """generate_score should parse a JSON dict/list response directly,
    without going through the regex fallback at all."""
    provider.canned_response = response
    result = provider.generate_score(
        system_prompt="irrelevant for this test",
        min_score_val=0,
        max_score_val=10,
    )
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("response,expected", PLAIN_STRING_CASES)
def test_generate_score_plain_string_regex_branch(provider, response, expected):
    """Plain-text (non-JSON) responses should fall through to
    re_configured_rating and parse correctly."""
    provider.canned_response = response
    result = provider.generate_score(
        system_prompt="irrelevant for this test",
        min_score_val=0,
        max_score_val=10,
    )
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("response,expected", MALFORMED_JSON_FALLTHROUGH_CASES)
def test_generate_score_malformed_json_falls_through(
    provider, response, expected
):
    """Responses that fail json.loads, or parse as JSON but lack a usable
    'score' key, should fall through to the string/regex branch rather
    than crash. If the fallback also can't find a number, ParseError
    should propagate (not be silently swallowed)."""
    provider.canned_response = response
    if expected == "raises":
        with pytest.raises(feedback_generated.ParseError):
            provider.generate_score(
                system_prompt="irrelevant for this test",
                min_score_val=0,
                max_score_val=10,
            )
    else:
        result = provider.generate_score(
            system_prompt="irrelevant for this test",
            min_score_val=0,
            max_score_val=10,
        )
        assert result == pytest.approx(expected)
