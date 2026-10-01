"""Unit tests for how comprehensiveness aggregates its per-key-point verdicts.

``comprehensiveness_with_cot_reasons`` asks the judge about each key point of
the source separately and averages the verdicts. A judge reply that carries no
text leaves that key point out of the average, so a run where *every* reply was
empty used to report 0.0 -- which the method's own docstring defines as "not
comprehensive". ``_mean_graded_score`` reports ``UNPARSABLE_SCORE`` for the
same situation, and these tests pin the two to each other.

No external service is contacted: the provider is stubbed at the
``_create_chat_completion`` boundary, the same boundary the method under test
uses, so the real aggregation runs.
"""

from typing import ClassVar, List, Optional

import pytest
from trulens.feedback import llm_provider

_MAX_SCORE = 3

_KEY_POINTS = "Point one.\nPoint two.\nPoint three."


class _MockEndpoint:
    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class _StubProvider(llm_provider.LLMProvider):
    """Answers each key-point judge call with a canned reply.

    ``reply`` is either one string used for every key point, or a list so a
    single key point can fail while the others succeed.
    """

    model_config: ClassVar[dict[str, str]] = {"extra": "allow"}

    def __init__(self, reply: str | List[str]):
        super().__init__(endpoint=None, model_engine="mock-model")
        object.__setattr__(self, "endpoint", _MockEndpoint())
        object.__setattr__(self, "_reply", reply)
        object.__setattr__(self, "_calls", 0)

    def _is_reasoning_model(self) -> bool:
        return False

    def _create_chat_completion(self, *args: object, **kwargs: object) -> str:
        if isinstance(self._reply, list):
            reply = self._reply[min(self._calls, len(self._reply) - 1)]
        else:
            reply = self._reply
        object.__setattr__(self, "_calls", self._calls + 1)
        return reply

    def _generate_key_points(self, source: str, **kwargs: object) -> str:
        return _KEY_POINTS


def _comprehensiveness(
    reply: str | List[str], max_score_val: int = _MAX_SCORE
) -> tuple:
    provider = _StubProvider(reply)
    return provider.comprehensiveness_with_cot_reasons(
        source="a source document",
        summary="a summary",
        max_score_val=max_score_val,
    )


def test_comprehensiveness_reports_sentinel_when_no_key_point_is_graded():
    """Regression: an empty judge reply for every key point averaged to 0.0,
    which reads as "the summary is not comprehensive at all" rather than "the
    judge never answered"."""
    score, _ = _comprehensiveness("")

    assert score == llm_provider.UNPARSABLE_SCORE


def test_comprehensiveness_ignores_ungraded_key_points():
    """One key point answered at the top of the scale, one with no text. The
    graded one is the only one that contributes, and the run is still graded
    overall so it is not the sentinel."""
    score, _ = _comprehensiveness([f"{_MAX_SCORE}: fully included", ""])

    assert score == pytest.approx(1.0)


def test_comprehensiveness_still_averages_all_graded_key_points():
    """Guard against a fix that drops the averaging: every key point answered,
    two at the bottom of the scale and one at the top, so the mean is 1/3."""
    score, _ = _comprehensiveness([
        f"{_MAX_SCORE}: fully included",
        "0: absent",
        "0: absent",
    ])

    assert score == pytest.approx(1 / 3)


def test_comprehensiveness_normalizes_by_the_requested_scale():
    """The score is normalized by the scale the caller asked for, so a 0-1
    judge and a 0-3 judge describing the same verdict agree."""
    one_point, _ = _comprehensiveness("1: partially included", max_score_val=1)
    three_points, _ = _comprehensiveness("3: fully included", max_score_val=3)
    assert one_point == pytest.approx(1.0)
    assert three_points == pytest.approx(1.0)

    zero_of_one, _ = _comprehensiveness("0: absent", max_score_val=1)
    zero_of_three, _ = _comprehensiveness("0: absent", max_score_val=3)
    assert zero_of_one == pytest.approx(0.0)
    assert zero_of_three == pytest.approx(0.0)


_ = Optional
