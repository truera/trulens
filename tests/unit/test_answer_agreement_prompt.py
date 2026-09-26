"""Unit tests for the prompt `_get_answer_agreement` sends to the judge.

No external service is contacted: the provider is mocked at the transport
boundary, so these run in the ``make test-unit`` sweep that CI executes.

The response being graded used to be concatenated straight onto the end of
`AGREEMENT_SYSTEM`, which already ends with the instruction "give the integer
score and nothing more". The judge therefore could not tell the response apart
from the instruction, and an empty response left the prompt with nothing to
grade at all -- the number the judge returned anyway was reported as the
agreement of an empty answer.
"""

from typing import ClassVar

import pytest
from trulens.feedback import groundtruth as feedback_groundtruth
from trulens.feedback import llm_provider
from trulens.feedback.templates import quality as templates_quality

_QUESTION = "What is the capital of France?"
_EXPECTED = "Paris"
_ANSWER = "Paris is the capital of France."


class _MockEndpoint:
    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class _RecordingProvider(llm_provider.LLMProvider):
    """Records the judge prompt and always answers with the same rating."""

    model_config: ClassVar[dict[str, str]] = {"extra": "allow"}

    def __init__(self, reply: str = "10"):
        super().__init__(endpoint=None, model_engine="mock-model")
        object.__setattr__(self, "endpoint", _MockEndpoint())
        object.__setattr__(self, "prompts", [])
        object.__setattr__(self, "_reply", reply)

    def _is_reasoning_model(self) -> bool:
        return False

    def _create_chat_completion(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        response_format=None,
        **kwargs,
    ):
        self.prompts.append(prompt)
        return self._reply


def test_empty_response_scores_zero_without_asking_the_judge():
    """An empty response agrees with nothing, so it must not be reported as
    whatever integer the judge happened to return for a prompt that never
    showed a response."""
    provider = _RecordingProvider()

    rating = provider._get_answer_agreement(_QUESTION, "", _EXPECTED)

    assert rating == "0"
    assert provider.prompts == []


def test_whitespace_only_response_scores_zero_without_asking_the_judge():
    provider = _RecordingProvider()

    rating = provider._get_answer_agreement(_QUESTION, "   \n\t ", _EXPECTED)

    assert rating == "0"
    assert provider.prompts == []


def test_response_is_labelled_in_the_judge_prompt():
    """The response must be delimited so it cannot be read as part of the
    trailing instruction."""
    provider = _RecordingProvider()

    provider._get_answer_agreement(_QUESTION, _ANSWER, _EXPECTED)

    assert len(provider.prompts) == 1
    prompt = provider.prompts[0]
    instruction = templates_quality.AGREEMENT_SYSTEM % (
        _QUESTION,
        _EXPECTED,
    )
    # The template is used whole, and the response follows it rather than being
    # appended straight onto the instruction with nothing in between.
    assert instruction.rstrip().endswith(
        "On a NEW LINE, give the integer score and nothing more."
    )
    tail = prompt[len(instruction) :]
    assert _ANSWER in tail
    assert tail.strip() != _ANSWER  # something labels it
    assert tail.index(_ANSWER) > 0  # and the label comes first


def test_agreement_measure_reports_zero_for_an_empty_response():
    """End to end through GroundTruthAgreement: the returned value must be in
    [0, 1] and must not be the judge's number for a blank answer."""
    provider = _RecordingProvider()
    agreement = feedback_groundtruth.GroundTruthAgreement(
        [{"query": _QUESTION, "expected_response": _EXPECTED}],
        provider=provider,
    )

    score, meta = agreement.agreement_measure(_QUESTION, "")

    assert score == 0.0
    assert meta["ground_truth_response"] == _EXPECTED


def test_agreement_measure_still_uses_the_judge_for_a_real_response():
    provider = _RecordingProvider(reply="8")

    agreement = feedback_groundtruth.GroundTruthAgreement(
        [{"query": _QUESTION, "expected_response": _EXPECTED}],
        provider=provider,
    )

    score, _ = agreement.agreement_measure(_QUESTION, _ANSWER)

    assert score == pytest.approx(0.8)
    assert len(provider.prompts) == 1


def test_missing_ground_truth_still_returns_nan():
    """Guard: the existing no-ground-truth behaviour is unchanged."""
    provider = _RecordingProvider()
    agreement = feedback_groundtruth.GroundTruthAgreement(
        [{"query": _QUESTION, "expected_response": _EXPECTED}],
        provider=provider,
    )

    result = agreement.agreement_measure("an unrelated question", _ANSWER)

    assert result != result  # NaN
    assert provider.prompts == []
