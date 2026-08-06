"""Integration tests for score parsing through the full feedback pipeline.

Addresses truera/trulens#2496.

``tests/unit/test_feedback_score_generation.py`` exercises the raw parser
(``re_0_10_rating``) against hardcoded strings. It does not cover the provider
entry points feedback functions actually call -- ``generate_score`` and
``generate_score_and_reasons`` -- where parsing is followed by normalization to
[0, 1]. ``MockLLMProvider`` overrides only ``_create_chat_completion``, so the
real parse-and-normalize path runs end to end for every case below.
"""

import math
from typing import ClassVar

import pytest
from trulens.feedback import generated as feedback_generated
from trulens.feedback import llm_provider
from trulens.feedback import output_schemas as feedback_output_schemas

_RAW = 2
_MIN, _MAX = 0, 10
_EXPECTED = (_RAW - _MIN) / (_MAX - _MIN)  # 0.2


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


_SCORE_2_SHAPES = [
    ("plain_number", "2"),
    ("number_with_explanation", "The relevance is moderate. Score: 2"),
    ("markdown_fenced_json", '```json\n{"score": 2}\n```'),
    ("whitespace_padding", "\n  Score: 2  \n"),
    ("prose_only", "I would give this a score of 2 overall."),
]
_SCORE_2_IDS = [case[0] for case in _SCORE_2_SHAPES]
_SCORE_2_VALUES = [case[1] for case in _SCORE_2_SHAPES]


def _assert_normalized_score(score: float, expected: float = _EXPECTED):
    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(expected)


@pytest.mark.parametrize("response", _SCORE_2_VALUES, ids=_SCORE_2_IDS)
def test_generate_score_normalizes_text_shapes(response):
    score = MockLLMProvider(response).generate_score(
        system_prompt="System prompt.",
        user_prompt="User prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    _assert_normalized_score(score)


def test_generate_score_accepts_decimal_rating():
    score = MockLLMProvider("2.5").generate_score(
        system_prompt="System prompt.",
        user_prompt="User prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0


def test_generate_score_normalizes_to_custom_range():
    score = MockLLMProvider("The score is 4.").generate_score(
        system_prompt="System prompt.",
        min_score_val=1,
        max_score_val=5,
    )

    assert score == pytest.approx(0.75)


@pytest.mark.parametrize(
    "response, expected",
    [("0", 0.0), ("10", 1.0)],
    ids=["floor", "ceiling"],
)
def test_generate_score_normalizes_scale_boundaries(response, expected):
    score = MockLLMProvider(response).generate_score(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    _assert_normalized_score(score, expected)


@pytest.mark.parametrize(
    "response",
    ["No numeric rating here.", "42"],
    ids=["no_number", "out_of_range"],
)
def test_generate_score_raises_parse_error(response):
    with pytest.raises(feedback_generated.ParseError):
        MockLLMProvider(response).generate_score(
            system_prompt="System prompt.",
            min_score_val=_MIN,
            max_score_val=_MAX,
        )


def test_generate_score_parses_structured_output_object():
    response = feedback_output_schemas.BaseFeedbackResponse(score=_RAW)

    score = MockLLMProvider(response).generate_score(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    _assert_normalized_score(score)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("response", _SCORE_2_VALUES, ids=_SCORE_2_IDS)
def test_generate_score_and_reasons_normalizes_text_shapes(response):
    score, reasons = MockLLMProvider(response).generate_score_and_reasons(
        system_prompt="System prompt.",
        user_prompt="User prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    _assert_normalized_score(score)
    assert isinstance(reasons, dict)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_generate_score_and_reasons_accepts_decimal_rating():
    score, reasons = MockLLMProvider("2.5").generate_score_and_reasons(
        system_prompt="System prompt.",
        user_prompt="User prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0
    assert isinstance(reasons, dict)


def test_generate_score_and_reasons_parses_structured_cot_object():
    response = feedback_output_schemas.ChainOfThoughtResponse(
        criteria="relevance",
        supporting_evidence="The context partially answers the question.",
        score=_RAW,
    )

    score, reasons = MockLLMProvider(response).generate_score_and_reasons(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    _assert_normalized_score(score)
    assert "relevance" in reasons["reason"]


def test_generate_score_and_reasons_normalizes_structured_json():
    response = (
        '{"criteria": "relevance", "supporting_evidence": "...", "score": 2}'
    )

    score, reasons = MockLLMProvider(response).generate_score_and_reasons(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    _assert_normalized_score(score)
    assert isinstance(reasons, dict)


@pytest.mark.parametrize("response", _SCORE_2_VALUES, ids=_SCORE_2_IDS)
def test_context_relevance_pipeline_normalizes_text_shapes(response):
    score = MockLLMProvider(response).context_relevance(
        question="What is the capital of France?",
        context="Paris is the capital of France.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    _assert_normalized_score(score)


def test_context_relevance_pipeline_accepts_decimal_rating():
    score = MockLLMProvider("2.5").context_relevance(
        question="What is the capital of France?",
        context="Paris is the capital of France.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert isinstance(score, float)
    assert not math.isnan(score)
    assert 0.0 <= score <= 1.0


@pytest.mark.xfail(
    strict=True,
    reason=(
        "generate_score returns a (score, reason) tuple for structured-JSON "
        "responses instead of a normalized float, inconsistent with its "
        "`-> float` annotation, with generate_score_and_reasons, and with "
        "every other response shape (truera/trulens#2496)."
    ),
)
def test_generate_score_structured_json_returns_float():
    response = (
        '{"criteria": "relevance", "supporting_evidence": "...", "score": 2}'
    )

    score = MockLLMProvider(response).generate_score(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    _assert_normalized_score(score)
