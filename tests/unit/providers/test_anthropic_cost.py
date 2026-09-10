"""Regression test: Anthropic cost callback must not crash.

AnthropicCallback.handle_generation and AnthropicEndpoint.handle_wrapped_call
built cost with core_endpoint.Cost, but trulens.core.feedback.endpoint has no
Cost symbol, so every Anthropic call through the cost callback raised
AttributeError. They also passed currency= where the field is cost_currency.
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest

try:
    from trulens.core.schema import base as base_schema
    from trulens.providers.anthropic.endpoint import AnthropicCallback
except Exception:  # pragma: no cover
    AnthropicCallback = None


class TestAnthropicCostCallback(unittest.TestCase):
    def setUp(self):
        if AnthropicCallback is None:
            self.skipTest("trulens-providers-anthropic not available.")

    def test_handle_generation_accumulates_cost_without_crashing(self):
        cb = AnthropicCallback.model_construct(cost=base_schema.Cost())
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=10, output_tokens=5),
            model="claude-3-5-sonnet-20241022",
        )

        cb.handle_generation(response)  # raised AttributeError on main

        self.assertEqual(cb.cost.n_tokens, 15)
        self.assertEqual(cb.cost.n_prompt_tokens, 10)
        self.assertEqual(cb.cost.cost_currency, "USD")


if __name__ == "__main__":
    unittest.main()
