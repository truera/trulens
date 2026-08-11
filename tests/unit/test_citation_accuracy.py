"""Unit tests for the citation-accuracy feedback function.

These exercise the prompt construction and delegation of
``LLMProvider.citation_accuracy`` and ``citation_accuracy_with_cot_reasons``
without standing up a real provider/endpoint: the static context-joining helper
and the output-space resolver are real, and ``generate_score`` /
``generate_score_and_reasons`` are mocked.
"""

from types import MethodType
from unittest.mock import MagicMock

import pytest
from trulens.feedback.llm_provider import LLMProvider
from trulens.feedback.templates.rag import CitationAccuracy


def _mock_self():
    mock = MagicMock(spec=LLMProvider)
    # Use the real static joining helper and output-space resolver, not mocks.
    mock._join_context_passages = LLMProvider._join_context_passages
    mock._determine_output_space = MethodType(
        LLMProvider._determine_output_space, mock
    )
    return mock


def test_join_context_passages_joins_a_list():
    out = LLMProvider._join_context_passages(["alpha", "beta"])
    assert out == "alpha\n\nbeta"


def test_join_context_passages_passes_string_through():
    assert LLMProvider._join_context_passages("one block") == "one block"


def test_join_context_passages_adds_no_citation_numbering():
    # Unlike _number_citation_sources, this must not invent [N] markers.
    out = LLMProvider._join_context_passages(["alpha", "beta"])
    assert "[1]" not in out
    assert "[2]" not in out


def test_citation_accuracy_builds_prompt_and_delegates():
    mock = _mock_self()
    mock.generate_score = MagicMock(return_value=1.0)

    score = LLMProvider.citation_accuracy(
        mock,
        response="The treaty was signed in 1991, according to the summary.",
        context=["It was signed in 1991.", "Unrelated passage."],
    )

    assert score == 1.0
    mock.generate_score.assert_called_once()
    _, kwargs = mock.generate_score.call_args
    system_prompt = kwargs["system_prompt"]
    user_prompt = kwargs["user_prompt"]
    assert "CITATION ACCURACY grader" in system_prompt
    # The list of passages reaches the judge as one joined block.
    assert "It was signed in 1991.\n\nUnrelated passage." in user_prompt
    assert "according to the summary" in user_prompt
    # Likert 0-3 output space by default.
    assert kwargs["min_score_val"] == 0
    assert kwargs["max_score_val"] == 3


def test_citation_accuracy_accepts_a_plain_string_context():
    mock = _mock_self()
    mock.generate_score = MagicMock(return_value=1.0)

    LLMProvider.citation_accuracy(
        mock, response="a", context="a single context block"
    )

    _, kwargs = mock.generate_score.call_args
    assert "a single context block" in kwargs["user_prompt"]


def test_citation_accuracy_with_cot_reasons_delegates_and_returns_reasons():
    mock = _mock_self()
    mock.generate_score_and_reasons = MagicMock(
        return_value=(0.0, {"reason": "cited claim is not in the context"})
    )

    score, reasons = LLMProvider.citation_accuracy_with_cot_reasons(
        mock,
        response="The treaty was signed in 1987, per the summary.",
        context=["It was signed in 1991."],
    )

    assert score == 0.0
    assert "not in the context" in reasons["reason"]
    _, kwargs = mock.generate_score_and_reasons.call_args
    # The CoT variant swaps the trailing label for the reasons template.
    assert "Supporting Evidence:" in kwargs["user_prompt"]
    assert "CITATION ACCURACY:" not in kwargs["user_prompt"]


def test_citation_accuracy_with_cot_reasons_joins_list_context():
    mock = _mock_self()
    mock.generate_score_and_reasons = MagicMock(return_value=(1.0, {}))

    LLMProvider.citation_accuracy_with_cot_reasons(
        mock, response="a", context=["alpha", "beta"]
    )

    _, kwargs = mock.generate_score_and_reasons.call_args
    assert "alpha\n\nbeta" in kwargs["user_prompt"]


def test_citation_accuracy_respects_score_range_kwargs():
    mock = _mock_self()
    mock.generate_score = MagicMock(return_value=1.0)

    LLMProvider.citation_accuracy(
        mock,
        response="a",
        context="p",
        min_score_val=0,
        max_score_val=1,
    )

    _, kwargs = mock.generate_score.call_args
    assert kwargs["min_score_val"] == 0
    assert kwargs["max_score_val"] == 1
    # The binary scale is described in the system prompt, not the 0-3 one.
    assert "0 or 1" in kwargs["system_prompt"]


def test_citation_accuracy_rejects_unsupported_score_range():
    mock = _mock_self()
    mock.generate_score = MagicMock(return_value=1.0)

    with pytest.raises(ValueError):
        LLMProvider.citation_accuracy(
            mock,
            response="a",
            context="p",
            min_score_val=0,
            max_score_val=7,
        )


def test_citation_accuracy_accepts_criteria_and_additional_instructions():
    mock = _mock_self()
    mock.generate_score = MagicMock(return_value=1.0)

    LLMProvider.citation_accuracy(
        mock,
        response="a",
        context="p",
        criteria="Only judge citations attached to numeric claims.",
        additional_instructions="Do not penalize uncited claims.",
    )

    _, kwargs = mock.generate_score.call_args
    system_prompt = kwargs["system_prompt"]
    assert "Only judge citations attached to numeric claims." in system_prompt
    assert "Additional Instructions:" in system_prompt
    assert "Do not penalize uncited claims." in system_prompt


def test_citation_accuracy_accepts_examples():
    mock = _mock_self()
    mock.generate_score = MagicMock(return_value=1.0)

    LLMProvider.citation_accuracy(
        mock,
        response="a",
        context="p",
        examples=[
            (
                {
                    "response": "Signed in 1991, per the summary.",
                    "context": "It was signed in 1991.",
                },
                3,
            ),
        ],
    )

    _, kwargs = mock.generate_score.call_args
    system_prompt = kwargs["system_prompt"]
    assert "Use the following examples to guide" in system_prompt
    assert "Score: 3" in system_prompt


def test_citation_accuracy_handles_deprecated_custom_instructions_kwarg():
    mock = _mock_self()
    mock.generate_score = MagicMock(return_value=1.0)

    with pytest.warns(DeprecationWarning):
        LLMProvider.citation_accuracy(
            mock,
            response="a",
            context="p",
            custom_instructions="Be strict about paraphrase.",
        )

    _, kwargs = mock.generate_score.call_args
    assert "Be strict about paraphrase." in kwargs["system_prompt"]


def test_template_formats():
    # The default system prompt is pre-composed for the 0-3 output space.
    assert "CITATION ACCURACY grader" in CitationAccuracy.system_prompt
    assert "0 to 3" in CitationAccuracy.system_prompt
    CitationAccuracy.user_prompt.format(response="a", context="p")


def test_generate_system_prompt_defaults_match_class_prompt():
    generated = CitationAccuracy.generate_system_prompt(
        min_score=0, max_score=3
    )
    assert generated == CitationAccuracy.system_prompt


def test_missing_citation_penalty_is_documented_in_criteria():
    # The behavioral difference from CitationAttribution is load-bearing:
    # this metric penalizes support that is present but left uncited.
    assert "missing" in CitationAccuracy.criteria.lower()
