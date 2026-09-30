"""Unit tests for score range validation on structured-JSON paths.

No external service is contacted: the provider is mocked at the transport
boundary, so these run in the ``make test-unit`` sweep that CI executes.

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

Providers that support structured outputs do not return JSON text at all, they
return the parsed model, and the last-chance reformat fallback can produce one
too. Both of those paths skipped the validation as well, so the same 42 came
back as ``4.2`` and looked like a valid rating. These cover them against the
same expectation.
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


class ScriptedLLMProvider(MockLLMProvider):
    """Answers the judge call and the reformat call differently.

    ``generate_score_and_reasons`` consults ``_create_chat_completion`` once
    for the graded reply and, only if that reply came back as plain text, a
    second time to force it into ``ChainOfThoughtResponse``. Responses are
    handed out in order; the last one repeats for any further call.
    """

    def __init__(self, responses: list, **kwargs):
        super().__init__(responses[0], **kwargs)
        object.__setattr__(self, "_responses", list(responses))
        object.__setattr__(self, "_cursor", [0])

    def _create_chat_completion(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        response_format=None,
        **kwargs,
    ):
        index = self._cursor[0]
        self._cursor[0] = index + 1
        if index < len(self._responses):
            return self._responses[index]
        return self._responses[-1]


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


def test_generate_score_and_reasons_text_without_score_line_keeps_sentinel():
    """Regression: the same leak on the text fallback. A reply that carries
    supporting evidence but no "Score:" line leaves the -1 sentinel in place,
    and normalizing it turned a parse failure into a plausible -0.1 rating on a
    0-10 scale. The JSON path above was fixed for exactly this; the text branch
    was not."""
    response = "Criteria: c\nSupporting Evidence: cut off before the score"

    score, _ = MockLLMProvider(response).generate_score_and_reasons(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert not math.isnan(score)
    assert score == -1.0


def test_generate_score_and_reasons_text_with_score_line_still_normalizes():
    """The ordinary text path must keep normalizing, so the sentinel change
    above cannot be satisfied by skipping normalization altogether."""
    response = "Criteria: c\nSupporting Evidence: e\nScore: 7"

    score, _ = MockLLMProvider(response).generate_score_and_reasons(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert score == pytest.approx(0.7)


# Providers with structured-output support return the parsed model rather than
# JSON text -- OpenAI `response.output_parsed`, Anthropic
# `response_format.model_validate(...)`, Cortex
# `response_format.model_validate_json(...)`, Google `response.parsed`. That is
# the main path for feedback on those providers, so it has to enforce the scale
# the same way.
@pytest.mark.parametrize(
    "response",
    [
        feedback_output_schemas.BaseFeedbackResponse(score=42),
        feedback_output_schemas.BaseFeedbackResponse(score=-3),
    ],
    ids=["above_max", "below_min"],
)
def test_generate_score_structured_object_out_of_range_raises_parse_error(
    response,
):
    with pytest.raises(feedback_generated.ParseError):
        MockLLMProvider(response).generate_score(
            system_prompt="System prompt.",
            min_score_val=_MIN,
            max_score_val=_MAX,
        )


@pytest.mark.parametrize(
    "response",
    [
        feedback_output_schemas.ChainOfThoughtResponse(
            criteria="c", supporting_evidence="e", score=100
        ),
        feedback_output_schemas.ChainOfThoughtResponse(
            criteria="c", supporting_evidence="e", score=-1
        ),
    ],
    ids=["above_max", "below_min"],
)
def test_generate_score_and_reasons_structured_object_out_of_range_raises(
    response,
):
    with pytest.raises(feedback_generated.ParseError):
        MockLLMProvider(response).generate_score_and_reasons(
            system_prompt="System prompt.",
            min_score_val=_MIN,
            max_score_val=_MAX,
        )


def test_generate_score_structured_object_in_range_still_normalizes():
    response = feedback_output_schemas.BaseFeedbackResponse(score=2)

    score = MockLLMProvider(response).generate_score(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert score == pytest.approx(0.2)


def test_generate_score_and_reasons_structured_object_in_range_normalizes():
    response = feedback_output_schemas.ChainOfThoughtResponse(
        criteria="c", supporting_evidence="e", score=2
    )

    score, reasons = MockLLMProvider(response).generate_score_and_reasons(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert score == pytest.approx(0.2)
    assert isinstance(reasons, dict)


# The last-chance reformat fallback. Both of its branches sit inside a
# `try ... except Exception: pass`, so rejecting an out-of-scale coerced score
# does not surface as a ParseError from there: the reply's own text gets parsed
# instead. What must never happen is the coerced out-of-scale number being
# returned as a rating.
_COT_OUT_OF_RANGE = feedback_output_schemas.ChainOfThoughtResponse(
    criteria="c", supporting_evidence="e", score=42
)


def test_generate_score_and_reasons_reformat_object_out_of_range_falls_back():
    """An out-of-scale score from the coerced object must not be returned; the
    judge's own in-range text is what the caller gets."""
    provider = ScriptedLLMProvider([
        "Criteria: c\nSupporting Evidence: e\nScore: 3",
        _COT_OUT_OF_RANGE,
    ])

    score, reasons = provider.generate_score_and_reasons(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.3)
    assert isinstance(reasons, dict)


def test_generate_score_and_reasons_reformat_json_out_of_range_falls_back():
    """Same, for the reformat branch that reads back JSON text instead of an
    object -- it has its own swallowing `except Exception`."""
    provider = ScriptedLLMProvider([
        "Criteria: c\nSupporting Evidence: e\nScore: 3",
        '{"criteria": "c", "supporting_evidence": "e", "score": 100}',
    ])

    score, reasons = provider.generate_score_and_reasons(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(0.3)
    assert isinstance(reasons, dict)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_generate_score_and_reasons_reformat_object_out_of_range_raises():
    """When the reply's text carries nothing in range either, the caller ends up
    seeing the ParseError from the fallback parse -- not a normalized 4.2."""
    provider = ScriptedLLMProvider(["42", _COT_OUT_OF_RANGE])

    with pytest.raises(feedback_generated.ParseError):
        provider.generate_score_and_reasons(
            system_prompt="System prompt.",
            min_score_val=_MIN,
            max_score_val=_MAX,
        )


def test_generate_score_and_reasons_reformat_object_out_of_range_sentinel():
    """And when the text parses as evidence without a score, the rejected
    reformat score leaves the documented unparsable sentinel behind instead of a
    normalized out-of-scale rating."""
    provider = ScriptedLLMProvider([
        "Criteria: c\nSupporting Evidence: e",
        _COT_OUT_OF_RANGE,
    ])

    score, _ = provider.generate_score_and_reasons(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert score == -1.0


def test_generate_score_and_reasons_reformat_in_range_still_normalizes():
    """The in-range reformat path keeps working, so the fallback semantics above
    cannot be obtained by dropping the reformat result."""
    provider = ScriptedLLMProvider([
        "The relevance is moderate. Score: 2",
        feedback_output_schemas.ChainOfThoughtResponse(
            criteria="c", supporting_evidence="e", score=2
        ),
    ])

    score, reasons = provider.generate_score_and_reasons(
        system_prompt="System prompt.",
        min_score_val=_MIN,
        max_score_val=_MAX,
    )

    assert score == pytest.approx(0.2)
    assert isinstance(reasons, dict)
