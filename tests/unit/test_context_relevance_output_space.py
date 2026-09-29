"""Unit tests for how context relevance honours the configured rating scale.

No external service is contacted: the judge is stubbed at the transport
boundary, so these run in the ``make test-unit`` sweep that CI executes.

``context_relevance`` and ``context_relevance_with_cot_reasons`` take the same
``min_score_val`` / ``max_score_val`` pair and both promise "a value between 0
and 1" in their docstrings, but only the non-COT one builds its prompt from the
scale the caller asked for. The COT one takes a shortcut to a hardcoded 0-3
prompt whenever no criteria or additional instructions are given, so on any
other scale the judge is asked for a 0-3 rating and its answer is then
normalized against a scale it was never shown.
"""

import json
import re
from typing import ClassVar, Optional

import pytest
from trulens.feedback import llm_provider
from trulens.feedback import templates as feedback_templates
from trulens.feedback.templates import base as templates_base
from trulens.feedback.templates import rag as templates_rag

_QUESTION = "What is the capital of France?"
_CONTEXT = "Paris is the capital and most populous city of France."

# The scale a judge states it was given: "0 to 3", "0 or 1", "0 to 10".
_STATED_SCALE = re.compile(r"(\d+)\s*(?:to|or)\s*(\d+)")

_COT = {
    "criteria": "the context answers the question",
    "supporting_evidence": "Paris",
}


class _MockEndpoint:
    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class _ObedientJudgeProvider(llm_provider.LLMProvider):
    """A judge that answers with the top of whichever scale it was shown.

    Reading the scale out of the system prompt is what a well-behaved judge
    does, so the score this stub returns is the score a real judge returns for
    the prompt the function under test actually built.
    """

    model_config: ClassVar[dict[str, str]] = {"extra": "allow"}

    def __init__(self):
        super().__init__(endpoint=None, model_engine="mock-model")
        object.__setattr__(self, "endpoint", _MockEndpoint())
        object.__setattr__(self, "system_prompts", [])

    def _create_chat_completion(
        self,
        prompt: Optional[str] = None,
        messages: Optional[list] = None,
        response_format=None,
        **kwargs,
    ) -> str:
        system = ""
        for message in messages or []:
            if message.get("role") == "system":
                system = message.get("content", "")
        self.system_prompts.append(system)

        scale = _STATED_SCALE.search(system)
        assert scale is not None, f"no rating scale in prompt:\n{system}"
        return json.dumps({**_COT, "score": int(scale.group(2))})


def _provider() -> _ObedientJudgeProvider:
    return _ObedientJudgeProvider()


def test_cot_context_relevance_normalizes_against_the_scale_it_asked_for():
    """Regression: the COT variant graded 0-3 and normalized by 10-0, so a
    perfectly relevant context came back as 0.3 -- a score that looks like an
    ordinary mediocre result and is well inside [0, 1]."""
    provider = _provider()

    score, _ = provider.context_relevance_with_cot_reasons(
        question=_QUESTION,
        context=_CONTEXT,
        max_score_val=10,
    )

    assert score == pytest.approx(1.0)


@pytest.mark.parametrize("max_score_val", [1, 3, 10])
def test_both_context_relevance_variants_grade_on_the_same_scale(
    max_score_val: int,
):
    """The COT and non-COT variants must not disagree on a fully relevant
    context, whichever scale the caller configured. On 0-1 the COT variant did
    not merely disagree, it raised: the judge was told to answer 0-3 and its
    "3" was then read against the 0-1 scale the caller asked for."""
    cot_provider = _provider()
    plain_provider = _provider()

    plain_score = plain_provider.context_relevance(
        question=_QUESTION,
        context=_CONTEXT,
        max_score_val=max_score_val,
    )
    cot_score, _ = cot_provider.context_relevance_with_cot_reasons(
        question=_QUESTION,
        context=_CONTEXT,
        max_score_val=max_score_val,
    )

    assert plain_score == pytest.approx(1.0)
    assert cot_score == pytest.approx(plain_score)


def test_cot_context_relevance_keeps_the_default_prompt_on_the_default_scale():
    """Guard against a fix that drops the default 0-3 prompt, which is a
    deliberate, more detailed one."""
    provider = _provider()

    score, _ = provider.context_relevance_with_cot_reasons(
        question=_QUESTION,
        context=_CONTEXT,
    )

    assert provider.system_prompts[0] == (
        templates_rag.ContextRelevance.default_cot_prompt
    )
    assert score == pytest.approx(1.0)


def test_cot_context_relevance_rejects_an_unsupported_scale():
    """Every other feedback function validates the scale through
    ``_determine_output_space``; the shortcut skipped that check too."""
    provider = _provider()

    with pytest.raises(ValueError, match="Invalid score range"):
        provider.context_relevance_with_cot_reasons(
            question=_QUESTION,
            context=_CONTEXT,
            max_score_val=5,
        )


def test_both_prompts_the_cot_variant_can_pick_describe_zero_to_three():
    """What makes the shortcut safe is that on the default scale the two prompts
    agree. This is the invariant the fix leans on."""
    generated = templates_rag.ContextRelevance.generate_system_prompt(
        min_score=0,
        max_score=3,
        output_space=templates_base.OutputSpace.LIKERT_0_3.name,
    )
    default = templates_rag.ContextRelevance.default_cot_prompt

    assert _STATED_SCALE.search(default).groups() == ("0", "3")
    assert _STATED_SCALE.search(generated).groups() == ("0", "3")
    # The prompt is re-exported under this name; keep the reference honest.
    assert feedback_templates.rag.ContextRelevance.default_cot_prompt == default
