"""Judge replies that wrap their JSON in a markdown code fence.

Read as plain text, a fenced reply falls through to `re_configured_rating`,
which returns the smallest in-range number anywhere in the reply. A count in
the criteria ("Checked 1 claim") then beats the score.
"""

from typing import ClassVar

import pytest
from trulens.feedback import llm_provider


class _MockEndpoint:
    def run_in_pace(self, func, *args, **kwargs):
        return func(*args, **kwargs)


class _CannedProvider(llm_provider.LLMProvider):
    """Returns one canned reply through the real parser."""

    model_config: ClassVar[dict[str, str]] = {"extra": "allow"}

    def __init__(self, response: str, **kwargs):
        super().__init__(endpoint=None, model_engine="mock-model", **kwargs)
        object.__setattr__(self, "endpoint", _MockEndpoint())
        object.__setattr__(self, "_response", response)

    def _is_reasoning_model(self) -> bool:
        return False

    def _create_chat_completion(self, *args, **kwargs):
        return self._response


_COT = (
    '{"criteria": "Checked 1 claim", "supporting_evidence": "ok", "score": 3}'
)

_FENCED = [
    f"```json\n{_COT}\n```",
    f"```JSON\r\n{_COT}\r\n```\r\n",
    f"```\n{_COT}\n```",
    f"  ```json {_COT} ```  ",
    f"Here is my evaluation:\n```json\n{_COT}\n```",
    f"```json\n{_COT}\n```\nLet me know if you need more detail.",
    f"Draft:\n```\nnot json\n```\nFinal:\n```json\n{_COT}\n```",
    f"Final:\n```json\n{_COT}\n```\nScratch:\n```\nnot json\n```",
]
_IDS = [
    "json_tag",
    "upper_tag_crlf",
    "no_tag",
    "single_line",
    "text_before",
    "text_after",
    "last_block_parses",
    "earlier_block_parses",
]


@pytest.mark.parametrize("response", _FENCED, ids=_IDS)
def test_generate_score_reads_fenced_json(response):
    score = _CannedProvider(response).generate_score(
        system_prompt="System prompt.", min_score_val=0, max_score_val=3
    )

    assert score == pytest.approx(1.0)


@pytest.mark.parametrize("response", _FENCED, ids=_IDS)
def test_generate_score_and_reasons_reads_fenced_json(response):
    score, reasons = _CannedProvider(response).generate_score_and_reasons(
        system_prompt="System prompt.", min_score_val=0, max_score_val=3
    )

    assert score == pytest.approx(1.0)
    assert "Checked 1 claim" in reasons["reason"]


def test_unfenced_json_with_backticks_in_a_value_is_read_whole():
    reply = (
        '{"criteria": "Checked 1 claim, see ```note```",'
        ' "supporting_evidence": "ok", "score": 3}'
    )

    score = _CannedProvider(reply).generate_score(
        system_prompt="System prompt.", min_score_val=0, max_score_val=3
    )

    assert score == pytest.approx(1.0)


def test_fenced_text_without_json_still_uses_rating_regex():
    score = _CannedProvider("```\nI rate this 2.\n```").generate_score(
        system_prompt="System prompt.", min_score_val=0, max_score_val=3
    )

    assert score == pytest.approx(2 / 3)
