"""Unit tests for the reference answer `model_agreement` asks the judge for.

No external service is contacted: the provider is mocked at the transport
boundary, so these run in the ``make test-unit`` sweep that CI executes.

`model_agreement` first asks the model to answer the prompt itself, then grades
the agent's response against that answer. The first call used to send only
`CORRECT_SYSTEM`, so the model never saw the prompt it was meant to answer.
"""

from typing import ClassVar

import pytest
from trulens.feedback import llm_provider
from trulens.feedback.templates import quality as templates_quality

_QUESTION = "What is the capital of France?"
_ANSWER = "Paris is the capital of France."


class _MockEndpoint:
    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class _RecordingProvider(llm_provider.LLMProvider):
    """Records every chat completion request and answers with fixed text."""

    model_config: ClassVar[dict[str, str]] = {"extra": "allow"}

    def __init__(self):
        super().__init__(endpoint=None, model_engine="mock-model")
        object.__setattr__(self, "endpoint", _MockEndpoint())
        object.__setattr__(self, "calls", [])

    def _is_reasoning_model(self) -> bool:
        return False

    def _create_chat_completion(
        self,
        prompt: str | None = None,
        messages: list | None = None,
        response_format=None,
        **kwargs,
    ):
        self.calls.append({"prompt": prompt, "messages": messages})
        return "Paris" if len(self.calls) == 1 else "2"


def test_reference_answer_request_includes_the_prompt():
    """The model is asked to answer the same prompt the agent answered."""
    provider = _RecordingProvider()

    with pytest.deprecated_call():
        provider.model_agreement(_QUESTION, _ANSWER)

    reference_call = provider.calls[0]
    assert reference_call["prompt"] is None
    assert reference_call["messages"] == [
        {"role": "system", "content": templates_quality.CORRECT_SYSTEM},
        {"role": "user", "content": _QUESTION},
    ]


def test_reference_answer_is_passed_to_the_judge():
    """The agent's response is graded against the model's own answer."""
    provider = _RecordingProvider()

    with pytest.deprecated_call():
        provider.model_agreement(_QUESTION, _ANSWER)

    assert len(provider.calls) == 2
    judge_prompt = provider.calls[1]["prompt"]
    instruction = templates_quality.AGREEMENT_SYSTEM % (_QUESTION, "Paris")
    assert judge_prompt.startswith(instruction)
    assert _ANSWER in judge_prompt[len(instruction) :]
