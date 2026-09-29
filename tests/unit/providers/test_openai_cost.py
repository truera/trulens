"""Regression test: OpenAICallback must report per-call, not cumulative, cost.

OpenAICallback.handle_generation reads its `langchain_handler`'s counters
after calling `on_llm_end` and adds them wholesale to `self.cost`. That
handler accumulates every field (`total_tokens`, `successful_requests`, …)
over its whole lifetime rather than resetting per call, so call *k* re-adds
the running total instead of that call's own usage: a cost scope that makes
N calls ends up reporting N(N+1)/2 times a single call's tokens/cost instead
of N times. See https://github.com/truera/trulens/issues/2839.
"""

from __future__ import annotations

import unittest

try:
    from langchain_core.outputs import Generation
    from langchain_core.outputs import LLMResult
    from trulens.core.feedback import endpoint as core_endpoint
    from trulens.providers.openai.endpoint import OpenAICallback
except Exception:  # pragma: no cover
    OpenAICallback = None


def _llm_result(
    model: str, prompt_tokens: int, completion_tokens: int
) -> LLMResult:
    return LLMResult(
        generations=[[Generation(text="hi")]],
        llm_output={
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "model_name": model,
        },
    )


class TestOpenAICallbackCostAccumulation(unittest.TestCase):
    def setUp(self):
        if OpenAICallback is None:
            self.skipTest("trulens-providers-openai not available.")

    def test_handle_generation_adds_per_call_delta_not_running_total(self):
        callback = OpenAICallback(
            endpoint=core_endpoint.Endpoint(name="test-endpoint")
        )

        # Three identical calls, each worth 100 prompt + 20 completion
        # tokens, matching what a single call by itself would report.
        calls = [
            _llm_result("gpt-3.5-turbo", 100, 20),
            _llm_result("gpt-3.5-turbo", 100, 20),
            _llm_result("gpt-3.5-turbo", 100, 20),
        ]

        single_callback = OpenAICallback(
            endpoint=core_endpoint.Endpoint(name="test-endpoint")
        )
        single_callback.handle_generation(_llm_result("gpt-3.5-turbo", 100, 20))
        per_call_tokens = single_callback.cost.n_tokens
        per_call_prompt_tokens = single_callback.cost.n_prompt_tokens
        per_call_completion_tokens = single_callback.cost.n_completion_tokens
        per_call_cost = single_callback.cost.cost

        for llm_result in calls:
            callback.handle_generation(llm_result)

        n = len(calls)
        self.assertEqual(callback.cost.n_tokens, n * per_call_tokens)
        self.assertEqual(
            callback.cost.n_prompt_tokens, n * per_call_prompt_tokens
        )
        self.assertEqual(
            callback.cost.n_completion_tokens,
            n * per_call_completion_tokens,
        )
        self.assertEqual(callback.cost.n_successful_requests, n)
        self.assertAlmostEqual(callback.cost.cost, n * per_call_cost)


if __name__ == "__main__":
    unittest.main()
