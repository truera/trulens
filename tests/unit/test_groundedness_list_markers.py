"""Unit tests for how groundedness strips numbered-list markers.

No external service is contacted: the provider is mocked at the transport
boundary, so these run in the ``make test-unit`` sweep that CI executes.

After splitting a statement, groundedness drops items that are only a list
marker such as ``"1."`` or ``"2)"``. The filter used to drop every bare number,
so an answer that is a single number (``"1969."``, ``"42"``) was removed before
the judge saw it, and groundedness reported 0.0 or the unparsable sentinel for
a supported answer.
"""

import json
import re
from typing import ClassVar

import pytest
from trulens.core.feedback import feedback as core_feedback
from trulens.feedback import llm_provider
from trulens.feedback.templates import rag as templates_rag

_MAX_SCORE = 3
_STATEMENT_LINE = re.compile(r"^STATEMENT: (.*)$", re.MULTILINE)


class _MockEndpoint:
    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class _SplitStubProvider(llm_provider.LLMProvider):
    """Answers the sentence-splitter call with fixed text, keeps every
    statement in the trivial-statement filter, classifies nothing as an
    abstention, and grades every statement at the top of the scale while
    recording the exact statement each groundedness judge call received."""

    model_config: ClassVar[dict[str, str]] = {"extra": "allow"}

    def __init__(self, split_reply: str):
        super().__init__(endpoint=None, model_engine="mock-model")
        object.__setattr__(self, "endpoint", _MockEndpoint())
        object.__setattr__(self, "_split_reply", split_reply)
        object.__setattr__(self, "judged", [])
        object.__setattr__(self, "trivial_inputs", [])

    def _is_reasoning_model(self) -> bool:
        return False

    def _create_chat_completion(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        response_format=None,
        **kwargs,
    ):
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else (prompt or "")
        if system == templates_rag.Trivial.system_prompt:
            # Keep every statement: return the list the filter was given.
            given = re.search(r"\[.*\]", user, re.DOTALL).group(0)
            self.trivial_inputs.append(given)
            return given
        if system.startswith("You are a ABSTENTION classifier"):
            return json.dumps({
                "criteria": "c",
                "supporting_evidence": "e",
                "score": 0,
            })
        if response_format is None:
            return self._split_reply
        if user.startswith("SOURCE:"):
            self.judged.append(_STATEMENT_LINE.search(user).group(1))
        return json.dumps({
            "criteria": "c",
            "supporting_evidence": "e",
            "score": _MAX_SCORE,
        })


def _groundedness(provider, method, source, statement, filter_trivial):
    configs = core_feedback.GroundednessConfigs(
        # use_sent_tokenize=False keeps NLTK out of the test; the stub answers
        # the sentence-splitter call instead.
        use_sent_tokenize=False,
        filter_trivial_statements=filter_trivial,
    )
    if method == "answerability":
        return provider.groundedness_measure_with_cot_reasons_consider_answerability(
            source=source,
            statement=statement,
            question="What is the answer?",
            groundedness_configs=configs,
            max_score_val=_MAX_SCORE,
        )
    return provider.groundedness_measure_with_cot_reasons(
        source=source,
        statement=statement,
        groundedness_configs=configs,
        max_score_val=_MAX_SCORE,
    )


_METHODS = pytest.mark.parametrize("method", ["standard", "answerability"])
_FILTER = pytest.mark.parametrize("filter_trivial", [True, False])


def _main_filter(hypotheses):
    """The filter as it was before: drop every item that is only a number."""
    return [h for h in hypotheses if not re.match(r"^\s*\d+[\.)]?\s*$", h)]


@pytest.mark.parametrize(
    "hypotheses",
    [
        ["1.", "The sky is blue.", "2.", "Grass is green."],
        ["1.", "The sky is blue.", "2."],
        ["1.", "2.", "Grass is green."],
        ["2)", "Rome is in Italy."],
        ["It was 1969.", "42"],
        ["1969.", "42"],
        ["1.", "1969.", "2.", "1972."],
        ["1.", "2."],
        ["Plain text."],
        [""],
        [],
    ],
)
def test_drop_list_markers_matches_old_filter_otherwise(hypotheses):
    assert llm_provider._drop_list_markers(hypotheses) == _main_filter(
        hypotheses
    )


@pytest.mark.parametrize(
    ("hypotheses", "expected"),
    [
        (["1969."], ["1969."]),
        (["42"], ["42"]),
        (["2)"], ["2)"]),
        (["1969.", ""], ["1969."]),
        (["42", " "], ["42"]),
        (["", "1969.", ""], ["1969."]),
    ],
)
def test_drop_list_markers_keeps_an_answer_that_is_a_single_number(
    hypotheses, expected
):
    assert llm_provider._drop_list_markers(hypotheses) == expected


@_METHODS
@_FILTER
@pytest.mark.parametrize(
    ("source", "split_reply", "answer"),
    [
        ("The first crewed Moon landing was in 1969.", "1969.", "1969."),
        ("The first crewed Moon landing was in 1969.", "1969.\n", "1969."),
        ("The answer is 42.", "42", "42"),
    ],
)
def test_groundedness_grades_a_number_only_answer(
    method, filter_trivial, source, split_reply, answer
):
    """Regression: the answer was dropped as a list marker, so nothing was
    graded and groundedness returned 0.0 or the -1.0 sentinel."""
    provider = _SplitStubProvider(split_reply)

    score, _ = _groundedness(provider, method, source, answer, filter_trivial)

    assert provider.judged == [answer]
    assert provider.trivial_inputs == (
        [str([answer])] if filter_trivial else []
    )
    assert score == pytest.approx(1.0)


@_METHODS
@_FILTER
def test_groundedness_still_drops_list_markers(method, filter_trivial):
    provider = _SplitStubProvider(
        "1.\nAlpha is grounded.\n2.\nBeta is grounded."
    )

    score, _ = _groundedness(
        provider,
        method,
        "Alpha is grounded. Beta is grounded.",
        "1. Alpha is grounded. 2. Beta is grounded.",
        filter_trivial,
    )

    # Statements are graded concurrently, so compare without order.
    assert sorted(provider.judged) == [
        "Alpha is grounded.",
        "Beta is grounded.",
    ]
    assert score == pytest.approx(1.0)


@_METHODS
def test_groundedness_does_not_grade_markers_of_a_numbers_only_list(method):
    """Several bare numbers may be list labels ("1. 1969. 2. 1972."), so they
    are still all dropped, as before; none is sent to the judge."""
    provider = _SplitStubProvider("1.\n1969.\n2.\n1972.")

    score, _ = _groundedness(
        provider,
        method,
        "Apollo 11 landed in 1969 and Apollo 17 in 1972.",
        "1. 1969. 2. 1972.",
        False,
    )

    assert provider.judged == []
    assert score == pytest.approx(-1.0)
