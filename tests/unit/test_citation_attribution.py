"""Unit tests for the citation-attribution feedback function.

These exercise the prompt construction and delegation of
``LLMProvider.citation_attribution`` and ``citation_attribution_with_cot_reasons``
without standing up a real provider/endpoint: the static source-numbering helper
and the output-space resolver are real, and ``generate_score`` /
``generate_score_and_reasons`` are mocked.
"""

from types import MethodType
from unittest.mock import MagicMock

import pytest
from trulens.feedback.llm_provider import LLMProvider
from trulens.feedback.templates.rag import CitationAttribution


def _mock_self():
    mock = MagicMock(spec=LLMProvider)
    # Use the real static numbering helper and output-space resolver, not mocks.
    mock._number_citation_sources = LLMProvider._number_citation_sources
    mock._determine_output_space = MethodType(
        LLMProvider._determine_output_space, mock
    )
    return mock


def test_number_citation_sources_numbers_a_list():
    out = LLMProvider._number_citation_sources(["alpha", "beta"])
    assert "[1] alpha" in out
    assert "[2] beta" in out


def test_number_citation_sources_passes_string_through():
    assert (
        LLMProvider._number_citation_sources("[1] pre-numbered")
        == "[1] pre-numbered"
    )


def test_citation_attribution_builds_prompt_and_delegates():
    mock = _mock_self()
    mock.generate_score = MagicMock(return_value=1.0)

    score = LLMProvider.citation_attribution(
        mock,
        question="When did it happen?",
        source=["It happened in 1991.", "Something else in 2003."],
        statement="It happened in 1991 [1].",
    )

    assert score == 1.0
    mock.generate_score.assert_called_once()
    _, kwargs = mock.generate_score.call_args
    system_prompt = kwargs["system_prompt"]
    user_prompt = kwargs["user_prompt"]
    assert "CITATION-ATTRIBUTION classifier" in system_prompt
    # User prompt carries the numbered sources and the cited statement.
    assert "[1] It happened in 1991." in user_prompt
    assert "[2] Something else in 2003." in user_prompt
    assert "It happened in 1991 [1]." in user_prompt
    # Binary output space by default.
    assert kwargs["min_score_val"] == 0
    assert kwargs["max_score_val"] == 1


def test_citation_attribution_with_cot_reasons_delegates_and_returns_reasons():
    mock = _mock_self()
    mock.generate_score_and_reasons = MagicMock(
        return_value=(
            0.0,
            {"reason": "claim cited to a non-supporting passage"},
        )
    )

    score, reasons = LLMProvider.citation_attribution_with_cot_reasons(
        mock,
        question="When did it happen?",
        source=["It happened in 1991.", "Something else in 2003."],
        statement="It happened in 1991 [2].",
    )

    assert score == 0.0
    assert "non-supporting passage" in reasons["reason"]
    _, kwargs = mock.generate_score_and_reasons.call_args
    # The CoT variant swaps the trailing label for the reasons template.
    assert "Supporting Evidence:" in kwargs["user_prompt"]
    assert "CITATION ATTRIBUTION:" not in kwargs["user_prompt"]


def test_citation_attribution_respects_score_range_kwargs():
    mock = _mock_self()
    mock.generate_score = MagicMock(return_value=0.67)

    LLMProvider.citation_attribution(
        mock,
        question="q",
        source=["p"],
        statement="a [1]",
        min_score_val=0,
        max_score_val=3,
    )

    _, kwargs = mock.generate_score.call_args
    assert kwargs["min_score_val"] == 0
    assert kwargs["max_score_val"] == 3
    # The 0-3 scale is described in the system prompt, not the binary one.
    assert "0 to 3" in kwargs["system_prompt"]


def test_citation_attribution_rejects_unsupported_score_range():
    mock = _mock_self()
    mock.generate_score = MagicMock(return_value=1.0)

    with pytest.raises(ValueError):
        LLMProvider.citation_attribution(
            mock,
            question="q",
            source=["p"],
            statement="a [1]",
            min_score_val=0,
            max_score_val=7,
        )


def test_citation_attribution_accepts_criteria_and_additional_instructions():
    mock = _mock_self()
    mock.generate_score = MagicMock(return_value=1.0)

    LLMProvider.citation_attribution(
        mock,
        question="q",
        source=["p"],
        statement="a [1]",
        criteria="Only penalize citations attached to numeric claims.",
        additional_instructions="Treat a missing marker as acceptable.",
    )

    _, kwargs = mock.generate_score.call_args
    system_prompt = kwargs["system_prompt"]
    assert (
        "Only penalize citations attached to numeric claims." in system_prompt
    )
    assert "Additional Instructions:" in system_prompt
    assert "Treat a missing marker as acceptable." in system_prompt


def test_citation_attribution_accepts_examples():
    mock = _mock_self()
    mock.generate_score = MagicMock(return_value=1.0)

    LLMProvider.citation_attribution(
        mock,
        question="q",
        source=["p"],
        statement="a [1]",
        examples=[
            (
                {
                    "question": "When?",
                    "source": "[1] In 1991.",
                    "statement": "In 1991 [1].",
                },
                1,
            ),
        ],
    )

    _, kwargs = mock.generate_score.call_args
    system_prompt = kwargs["system_prompt"]
    assert "Use the following examples to guide" in system_prompt
    assert "Score: 1" in system_prompt


def test_template_formats():
    # The default system prompt is pre-composed for the binary output space.
    assert (
        "CITATION-ATTRIBUTION classifier" in CitationAttribution.system_prompt
    )
    assert "0 or 1" in CitationAttribution.system_prompt
    CitationAttribution.user_prompt.format(
        question="q", source="[1] p", statement="a [1]"
    )


def test_generate_system_prompt_defaults_match_class_prompt():
    generated = CitationAttribution.generate_system_prompt(
        min_score=0, max_score=1
    )
    assert generated == CitationAttribution.system_prompt
