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
    from trulens.providers.anthropic.endpoint import (
        AnthropicCallback,
        AnthropicCostComputer,
    )
    from trulens.otel.semconv.trace import SpanAttributes
except Exception:  # pragma: no cover
    AnthropicCallback = None
    AnthropicCostComputer = None
    SpanAttributes = None


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


class TestAnthropicCostComputerUnknownModel(unittest.TestCase):
    def setUp(self):
        if AnthropicCostComputer is None:
            self.skipTest("trulens-providers-anthropic not available.")

    def _make_response(self, model: str, input_tokens: int = 100, output_tokens: int = 50):
        return SimpleNamespace(
            usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
            model=model,
        )

    def test_unknown_model_omits_cost_from_result(self):
        """handle_response must not stamp COST.COST=0 for an unrecognised model."""
        cost_info = AnthropicCostComputer.handle_response(
            self._make_response("claude-does-not-exist-99999")
        )
        self.assertNotIn(
            SpanAttributes.COST.COST,
            cost_info,
            "COST.COST must be absent for an unknown model so the caller's "
            "running total is not polluted by a fake zero.",
        )

    def test_unknown_model_still_records_tokens(self):
        """Token counts must be present even when the model price is unknown."""
        cost_info = AnthropicCostComputer.handle_response(
            self._make_response("claude-does-not-exist-99999", 100, 50)
        )
        self.assertEqual(cost_info[SpanAttributes.COST.NUM_TOKENS], 150)
        self.assertEqual(cost_info[SpanAttributes.COST.NUM_PROMPT_TOKENS], 100)
        self.assertEqual(cost_info[SpanAttributes.COST.NUM_COMPLETION_TOKENS], 50)

    def test_known_model_includes_positive_cost(self):
        """handle_response must include a positive COST.COST for a known model."""
        cost_info = AnthropicCostComputer.handle_response(
            self._make_response("claude-3-5-sonnet-20241022", 1000, 200)
        )
        self.assertIn(SpanAttributes.COST.COST, cost_info)
        self.assertGreater(
            cost_info[SpanAttributes.COST.COST],
            0,
            "Expected a positive cost for claude-3-5-sonnet-20241022 with "
            "1000 input and 200 output tokens.",
        )

    def test_callback_cost_stays_zero_for_unknown_model(self):
        """AnthropicCallback must not add 0.0 to its cost accumulator for unknown models.

        Before the fix, cost=0.0 was always written; after, it is left at the
        Cost field default.  Both result in 0.0 cost, but the intent (leave
        alone vs. stamp a fake zero) is now correct.  The key assertion is
        that COST.COST is absent from handle_response so there is no
        misleading span attribute.
        """
        cb = AnthropicCallback.model_construct(cost=base_schema.Cost())
        cb.handle_generation(self._make_response("claude-does-not-exist-99999"))
        # Token counts accumulate; cost stays at the default (0.0 from Cost())
        self.assertEqual(cb.cost.n_tokens, 150)
        # cost.cost == 0.0 because it was never written, not because we priced at zero
        self.assertEqual(cb.cost.cost, 0.0)


if __name__ == "__main__":
    unittest.main()
