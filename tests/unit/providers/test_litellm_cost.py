"""Regression tests for LiteLLM cost tracking.

LiteLLMCallback accumulated tokens across calls but replaced the cost with
each call's own cost, so a feedback that made several LiteLLM calls reported
only the last call's cost. A call whose cost could not be computed also set
the cost to None, dropping what earlier calls had recorded.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

try:
    import litellm
    from trulens.core.schema import base as base_schema
    from trulens.providers.litellm import endpoint as litellm_endpoint
except Exception:  # pragma: no cover
    litellm_endpoint = None

_MODEL = "gemini/gemini-2.5-flash"


def _response(prompt_tokens: int, completion_tokens: int):
    return litellm.ModelResponse(
        model=_MODEL,
        choices=[{"message": {"role": "assistant", "content": "ok"}}],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


class TestLiteLLMCostCallback(unittest.TestCase):
    def setUp(self):
        if litellm_endpoint is None:
            self.skipTest("trulens-providers-litellm not available.")
        self.callback = litellm_endpoint.LiteLLMCallback.model_construct(
            endpoint=SimpleNamespace(litellm_provider="gemini"),
            cost=base_schema.Cost(),
        )

    def test_cost_accumulates_across_calls(self):
        responses = [_response(1000, 100), _response(2000, 50)]
        expected = sum(litellm.completion_cost(r) for r in responses)
        self.assertGreater(expected, 0)

        for response in responses:
            self.callback.handle_generation(response)

        self.assertAlmostEqual(self.callback.cost.cost, expected)
        self.assertEqual(self.callback.cost.n_tokens, 3150)

    def test_failed_cost_computation_keeps_earlier_cost(self):
        first = _response(1000, 100)
        expected = litellm.completion_cost(first)
        self.callback.handle_generation(first)

        with mock.patch.object(
            litellm_endpoint,
            "completion_cost",
            side_effect=ValueError("model not mapped"),
        ):
            self.callback.handle_generation(_response(500, 20))

        self.assertAlmostEqual(self.callback.cost.cost, expected)
        self.assertEqual(self.callback.cost.n_tokens, 1620)


if __name__ == "__main__":
    unittest.main()
