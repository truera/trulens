"""Unit tests for the citation-attribution feedback function.

These exercise the prompt construction and delegation of
``LLMProvider.citation_attribution`` and ``citation_attribution_with_cot_reasons``
without standing up a real provider/endpoint: the static source-numbering helper
is real, and ``generate_score`` / ``generate_score_and_reasons`` are mocked.
"""

from unittest.mock import MagicMock

from trulens.feedback.llm_provider import LLMProvider
from trulens.feedback.templates.rag import CitationAttribution


def _mock_self():
    mock = MagicMock(spec=LLMProvider)
    # Use the real static numbering helper, not a mock.
    mock._number_citation_sources = LLMProvider._number_citation_sources
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
    args, kwargs = mock.generate_score.call_args
    system_prompt, user_prompt = args[0], args[1]
    # System prompt is the attribution classifier with the 0/1 range filled in.
    assert "CITATION-ATTRIBUTION classifier" in system_prompt
    # User prompt carries the numbered sources and the cited statement.
    assert "[1] It happened in 1991." in user_prompt
    assert "[2] Something else in 2003." in user_prompt
    assert "It happened in 1991 [1]." in user_prompt
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
    args, _ = mock.generate_score_and_reasons.call_args
    # The CoT variant uses the step-by-step system prompt.
    assert "Think step by step" in args[0]


def test_template_formats():
    CitationAttribution.system_prompt.format(min_score=0, max_score=1)
    CitationAttribution.cot_system_prompt.format(min_score=0, max_score=1)
    CitationAttribution.user_prompt.format(
        question="q", source="[1] p", statement="a [1]"
    )
