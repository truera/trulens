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
    from trulens.otel.semconv.trace import SpanAttributes
    from trulens.providers.anthropic.endpoint import AnthropicCallback
    from trulens.providers.anthropic.endpoint import AnthropicCostComputer
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

    def _make_response(
        self, model: str, input_tokens: int = 100, output_tokens: int = 50
    ):
        return SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=input_tokens, output_tokens=output_tokens
            ),
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
        self.assertEqual(
            cost_info[SpanAttributes.COST.NUM_COMPLETION_TOKENS], 50
        )

    def test_known_model_includes_positive_cost(self):
        """handle_response must include a positive COST.COST for a known model."""
        from unittest.mock import patch

        from trulens.providers.anthropic import endpoint

        fake_table = {
            "claude-test-priced": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 2e-6,
                "litellm_provider": "anthropic",
            }
        }
        with patch.object(endpoint, "LITELLM_MODEL_COSTS_TABLE", fake_table):
            cost_info = AnthropicCostComputer.handle_response(
                self._make_response("claude-test-priced", 1000, 200)
            )
        self.assertIn(SpanAttributes.COST.COST, cost_info)
        self.assertGreater(
            cost_info[SpanAttributes.COST.COST],
            0,
            "Expected a positive cost for a patched known model with "
            "1000 input and 200 output tokens.",
        )

    def test_callback_cost_stays_zero_for_unknown_model(self):
        """AnthropicCallback must not stamp COST.COST for unknown models.

        Before the fix, cost=0.0 was always written as a span attribute;
        after, COST.COST is absent so no misleading zero is recorded.
        Both paths yield cb.cost.cost == 0.0, so that assertion does not
        distinguish them -- asserting the span attribute is absent does.
        """
        cb = AnthropicCallback.model_construct(cost=base_schema.Cost())
        cb.handle_generation(self._make_response("claude-does-not-exist-99999"))
        # Token counts accumulate regardless of whether the model is priced
        self.assertEqual(cb.cost.n_tokens, 150)
        # assertNotIn fails on old code (stamped COST.COST=0.0); assertEqual
        # cb.cost.cost==0.0 passes on both old and new, so it does not guard.
        cost_info = AnthropicCostComputer.handle_response(
            self._make_response("claude-does-not-exist-99999")
        )
        self.assertNotIn(
            SpanAttributes.COST.COST,
            cost_info,
            "COST.COST span attribute must be absent for unknown models; "
            "the default 0.0 on Cost() is not the same as a stamped zero.",
        )


if __name__ == "__main__":
    unittest.main()


class TestAnthropicCostComputerPromptCache(unittest.TestCase):
    """Anthropic's input_tokens excludes tokens read from or written to the
    prompt cache, so they are reported in separate usage fields."""

    def setUp(self):
        if AnthropicCostComputer is None:
            self.skipTest("trulens-providers-anthropic not available.")

    def test_cache_tokens_count_as_prompt_tokens_and_are_priced(self):
        from trulens.providers.anthropic.endpoint import (
            LITELLM_MODEL_COSTS_TABLE,
        )

        model = "claude-sonnet-4-5"
        response = SimpleNamespace(
            usage=SimpleNamespace(
                input_tokens=100,
                cache_read_input_tokens=1000,
                cache_creation_input_tokens=200,
                output_tokens=50,
            ),
            model=model,
        )

        cost_info = AnthropicCostComputer.handle_response(response)

        pricing = LITELLM_MODEL_COSTS_TABLE[model]
        expected_cost = (
            100 * pricing["input_cost_per_token"]
            + 1000 * pricing["cache_read_input_token_cost"]
            + 200 * pricing["cache_creation_input_token_cost"]
            + 50 * pricing["output_cost_per_token"]
        )
        self.assertEqual(cost_info[SpanAttributes.COST.NUM_PROMPT_TOKENS], 1300)
        self.assertEqual(cost_info[SpanAttributes.COST.NUM_TOKENS], 1350)
        self.assertAlmostEqual(
            cost_info[SpanAttributes.COST.COST], round(expected_cost, 8)
        )

    def test_usage_without_cache_fields_is_unchanged(self):
        response = SimpleNamespace(
            usage=SimpleNamespace(input_tokens=100, output_tokens=50),
            model="claude-sonnet-4-5",
        )

        cost_info = AnthropicCostComputer.handle_response(response)

        self.assertEqual(cost_info[SpanAttributes.COST.NUM_PROMPT_TOKENS], 100)
        self.assertEqual(cost_info[SpanAttributes.COST.NUM_TOKENS], 150)
