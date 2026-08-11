"""Integration tests for score range validation on structured-JSON paths.

The string-response path already enforces the configured rating scale:
``re_configured_rating`` warns on out-of-range matches and raises
``ParseError`` when no in-range rating remains (see
``test_generate_score_raises_parse_error[out_of_range]`` in
``test_score_parsing_normalization.py``).

The structured-JSON fast paths in ``generate_score`` and
``generate_score_and_reasons`` previously skipped that validation, so an
out-of-scale score (e.g. ``{"score": 42}`` on a 0-10 scale) was silently
normalized to a value outside [0, 1], corrupting downstream aggregation.
These tests pin the JSON paths to the same semantics as the string path.
"""

import math
from typing import ClassVar

import pytest
from trulens.feedback import generated as feedback_generated
from trulens.feedback import llm_provider
from trulens.feedback import output_schemas as feedback_output_schemas

_MIN, _MAX = 0, 10


class _MockEndpoint:
    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class MockLLMProvider(llm_provider.LLMProvider):
    """Returns a canned completion through the real parser."""

    model_config: ClassVar[dict[str, str]] = {"extra": "allow"}

    def __init__(
        self,
        response: str | feedback_output_schemas.BaseFeedbackResponse,
        **kwargs,
    ):
        super().__init__(endpoint=None, model_engine="mock-model", **kwargs)
        object.__setattr__(self, "endpoint", _MockEndpoint())
        object.__setattr__(self, "_response", response)

    def _is_reasoning_model(self) -> bool:
        return False

    def _create_chat_completion(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        response_format=None,
        **kwargs,
    ):
        return self._response


@pytest.mark.parametrize(
    "response",
    ['{"score": 42}', '{"score": -3}'],
    ids=["above_max", "below_min"],
)
def test_generate_score_json_out_of_range_raises_parse_error(response):
    """Regression: JSON scores outside the scale must raise ParseError like
    the string path does, instead of normalizing past [0, 1]."""
    with pytest.raises(feedback_generated.ParseError):
        MockLLMProvider(response).generate_score(
            system_prompt="System prompt.",
            min_score_val=_MIN,
            max_score_val=_MAX,
        )


@pytest.mark.parametrize(
    "response",
    [
        '{"criteria": "c", "supporting_evidence": "e", "score": 100}',
        '{"criteria": "c", "supporting_evidence": "e", "score": -1}',
    ],
    ids=["above_max", "below_min"],
)
def test_generate_score_and_reasons_json_out_of_range_raises_parse_error(
    response,
):
    with pytest.raises(feedback_generated.ParseError):
        MockLLMProvider(response).generate_score_and_reasons(
            system_prompt="System prompt.",
            min_score_val=_MIN,
            max_score_val=_MAX,
        )


def test_generate_score_json_in_range_still_normalizes():
    """In-range JSON scores keep working after the validation was added."""
    result = MockLLMProvider('{"score": 2}').generate_score(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    score = result[0] if isinstance(result, tuple) else result
    assert score == pytest.approx(0.2)


def test_generate_score_and_reasons_json_in_range_still_normalizes():
    response = (
        '{"criteria": "relevance", "supporting_evidence": "...", "score": 2}'
    )

    score, reasons = MockLLMProvider(response).generate_score_and_reasons(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert score == pytest.approx(0.2)
    assert isinstance(reasons, dict)


def test_generate_score_list_skips_out_of_range_items():
    """List averaging must ignore out-of-scale items (mirroring the string
    parser, which filters out-of-range matches) instead of folding them in."""
    response = '[{"score": 2}, {"score": 42}]'

    result = MockLLMProvider(response).generate_score(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    score = result[0] if isinstance(result, tuple) else result
    # Only the in-range 2 contributes; 42 is skipped with a warning.
    assert score == pytest.approx(0.2)


def test_generate_score_and_reasons_unparseable_sentinel_not_normalized():
    """Regression: the -1.0 failure sentinel was previously normalized to
    -0.1 on a 0-10 scale; it must be returned raw, matching generate_score."""
    response = '{"criteria": "c", "supporting_evidence": "e", "score": "abc"}'

    score, _ = MockLLMProvider(response).generate_score_and_reasons(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert not math.isnan(score)
    assert score == -1.0
