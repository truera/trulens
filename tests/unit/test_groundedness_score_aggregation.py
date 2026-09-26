"""Unit tests for how groundedness aggregates per-statement judge scores.

No external service is contacted: the provider is mocked at the transport
boundary, so these run in the ``make test-unit`` sweep that CI executes.

``generate_score_and_reasons`` returns ``-1.0`` as its deliberate "could not
parse a score" sentinel (see ``test_score_range_validation.py``, which pins
that sentinel on both the JSON and the text path). The groundedness measures
used to average that sentinel in with ``np.mean`` over every statement's score,
so a judge failure on one statement dragged the reported groundedness down and
read as a model verdict. These tests pin the aggregation to the statements the
judge actually graded.
"""

import json
import math
from typing import ClassVar, Optional

import pytest
from trulens.core.feedback import feedback as core_feedback
from trulens.feedback import llm_provider

_MAX_SCORE = 3

_FIRST = "Alpha is grounded."
_SECOND = "BROKEN judge reply here."
_STATEMENT = f"{_FIRST} {_SECOND}"
_ALL = "*"


class _MockEndpoint:
    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class _SplittingStubProvider(llm_provider.LLMProvider):
    """Splits the statement in two, then fails the judge call for one of them.

    The sentence-splitter call is the one made without a ``response_format``;
    it answers with two lines. Each judge call gets a top-of-scale score, except
    for the statement named by ``failing`` whose reply carries no usable score
    and so comes back as the -1.0 sentinel.
    """

    model_config: ClassVar[dict[str, str]] = {"extra": "allow"}

    def __init__(self, failing: Optional[str] = _SECOND):
        super().__init__(endpoint=None, model_engine="mock-model")
        object.__setattr__(self, "endpoint", _MockEndpoint())
        object.__setattr__(self, "_failing", failing)

    def _is_reasoning_model(self) -> bool:
        return False

    def _create_chat_completion(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        response_format=None,
        **kwargs,
    ):
        if response_format is None:
            return f"{_FIRST}\n{_SECOND}"

        cot = {"criteria": "c", "supporting_evidence": "e"}
        text = " ".join(m["content"] for m in messages or [])
        if self._failing is not None and (
            self._failing == _ALL or self._failing in text
        ):
            return json.dumps({**cot, "score": "N/A"})
        return json.dumps({**cot, "score": _MAX_SCORE})


def _configs() -> "core_feedback.GroundednessConfigs":
    # use_sent_tokenize=False keeps NLTK out of the test; the stub answers the
    # sentence-splitter call instead.
    return core_feedback.GroundednessConfigs(
        use_sent_tokenize=False, filter_trivial_statements=False
    )


def test_groundedness_ignores_unparsable_statement_score():
    """Regression: one graded statement at the top of the scale and one whose
    judge reply had no usable score averaged to 0.0, which reads as "nothing is
    grounded" rather than "one judge call failed"."""
    provider = _SplittingStubProvider()

    score, meta = provider.groundedness_measure_with_cot_reasons(
        source="src",
        statement=_STATEMENT,
        groundedness_configs=_configs(),
        max_score_val=_MAX_SCORE,
    )

    assert not math.isnan(score)
    # Only the statement the judge actually graded contributes.
    assert score == pytest.approx(1.0)
    # The per-statement reasons still record what happened, sentinel included.
    assert [r["score"] for r in meta["reasons"]] == [pytest.approx(1.0), -1.0]


def test_groundedness_consider_answerability_ignores_unparsable_statement_score():
    """The answerability variant averages the same way, so it needs the same
    guard."""
    provider = _SplittingStubProvider()

    score, _ = (
        provider.groundedness_measure_with_cot_reasons_consider_answerability(
            source="src",
            statement=_STATEMENT,
            question="Is the statement grounded?",
            groundedness_configs=_configs(),
            max_score_val=_MAX_SCORE,
        )
    )

    assert not math.isnan(score)
    assert score == pytest.approx(1.0)


def test_groundedness_reports_sentinel_when_no_statement_is_graded():
    """When every judge call fails there is nothing to average. np.mean of an
    empty list is NaN, which is not a usable feedback value either; report the
    same sentinel the per-statement scores carry."""
    provider = _SplittingStubProvider(failing=_ALL)

    score, _ = provider.groundedness_measure_with_cot_reasons(
        source="src",
        statement=_STATEMENT,
        groundedness_configs=_configs(),
        max_score_val=_MAX_SCORE,
    )

    assert score == -1.0


def test_groundedness_still_averages_all_graded_statements():
    """Guard against a fix that drops the averaging entirely."""
    provider = _SplittingStubProvider(failing=None)

    score, meta = provider.groundedness_measure_with_cot_reasons(
        source="src",
        statement=_STATEMENT,
        groundedness_configs=_configs(),
        max_score_val=_MAX_SCORE,
    )

    assert score == pytest.approx(1.0)
    assert [r["score"] for r in meta["reasons"]] == [
        pytest.approx(1.0),
        pytest.approx(1.0),
    ]
