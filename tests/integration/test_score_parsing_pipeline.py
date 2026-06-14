"""Integration tests for score parsing through the full feedback pipeline.

Addresses truera/trulens#2496.

The existing unit tests in ``tests/unit/test_feedback_score_generation.py``
exercise the raw regex parser (``re_0_10_rating``) with hardcoded strings, but
nothing verifies that ``LLMProvider.generate_score()`` correctly parses *and*
normalizes the many response formats real LLMs return when driven through the
full call path:

    generate_score
        -> endpoint.run_in_pace
            -> _create_chat_completion        (raw model text)
        -> JSON / regex parsing
        -> normalization to [0, 1]

Unlike the mock in
``tests/unit/test_feedback_criteria_and_additional_instructions.py`` (which
overrides ``generate_score`` itself and therefore never touches the parser),
the ``MockLLMProvider`` here overrides only ``_create_chat_completion`` so the
real parsing/normalization pipeline runs end to end.
"""

import math
from typing import Optional

import pytest
from trulens.feedback import generated as feedback_generated
from trulens.feedback import llm_provider


class _MockEndpoint:
    """Minimal endpoint that invokes the completion function inline.

    ``generate_score`` calls ``self.endpoint.run_in_pace(func=..., **kwargs)``;
    for tests we just run the function directly without any rate limiting.
    """

    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class MockLLMProvider(llm_provider.LLMProvider):
    """LLMProvider that returns a canned raw completion string.

    Only ``_create_chat_completion`` is overridden, so ``generate_score`` runs
    its real JSON/regex parsing and 0-1 normalization on the canned response.
    """

    model_config = {"extra": "allow"}

    def __init__(self, response: str, **kwargs):
        super().__init__(endpoint=None, model_engine="mock-model", **kwargs)
        # Pydantic model: set non-field attributes via object.__setattr__.
        object.__setattr__(self, "endpoint", _MockEndpoint())
        object.__setattr__(self, "_response", response)

    def _is_reasoning_model(self) -> bool:
        return False

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        response_format=None,
        **kwargs,
    ) -> str:
        return self._response


# (id, raw LLM response) for response shapes that should each normalize to a
# plain float in [0, 1]. These mirror the realistic formats real models emit.
_FLOAT_RESPONSE_CASES = [
    ("plain_number", "2"),
    ("number_with_explanation", "The relevance is moderate. Score: 2"),
    ("markdown_fenced_json", '```json\n{"score": 2}\n```'),
    ("whitespace_padding", "\n  Score: 2  \n"),
    ("float_value", "2.5"),
]


@pytest.mark.parametrize(
    "response",
    [case[1] for case in _FLOAT_RESPONSE_CASES],
    ids=[case[0] for case in _FLOAT_RESPONSE_CASES],
)
def test_generate_score_parses_to_normalized_float(response):
    """Each realistic response shape parses to a normalized float in [0, 1]."""
    provider = MockLLMProvider(response)

    score = provider.generate_score(
        system_prompt="System prompt.",
        user_prompt="User prompt.",
        min_score_val=0,
        max_score_val=10,
    )

    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0


def test_generate_score_normalizes_to_custom_range():
    """A raw rating is normalized against the requested score range."""
    # Raw "4" on a 1..5 scale -> (4 - 1) / (5 - 1) = 0.75.
    provider = MockLLMProvider("The score is 4.")

    score = provider.generate_score(
        system_prompt="System prompt.",
        min_score_val=1,
        max_score_val=5,
    )

    assert score == pytest.approx(0.75)


def test_generate_score_raises_parse_error_on_unparseable_response():
    """A response with no in-range rating surfaces a ParseError."""
    provider = MockLLMProvider("No numeric rating is present here.")

    with pytest.raises(feedback_generated.ParseError):
        provider.generate_score(
            system_prompt="System prompt.",
            min_score_val=0,
            max_score_val=10,
        )


@pytest.mark.xfail(
    strict=True,
    reason=(
        "generate_score returns a (score, reason) tuple for structured-JSON "
        "responses instead of a normalized float, which is inconsistent with "
        "its `-> float` return type and with every other response format "
        "(truera/trulens#2496)."
    ),
)
def test_generate_score_structured_json_returns_float():
    """Structured JSON should also yield a plain normalized float.

    Currently ``generate_score`` returns a tuple in this branch; this test is
    an executable record of that inconsistency and will start passing once the
    branch is normalized to return only the float.
    """
    response = (
        '{"criteria": "relevance", "supporting_evidence": "...", "score": 2}'
    )
    provider = MockLLMProvider(response)

    score = provider.generate_score(
        system_prompt="System prompt.",
        min_score_val=0,
        max_score_val=10,
    )

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
