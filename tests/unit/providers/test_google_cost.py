"""Regression tests for Google (Gemini) cost tracking.

Gemini bills thinking tokens (`thoughts_token_count`) at the output rate, but
the cost was computed from prompt and candidate tokens only. The OTEL cost
computer also built a `GoogleEndpoint` for every response just to reach the
pricing helper, which raised without a `GOOGLE_API_KEY`/`GEMINI_API_KEY` (for
example on Vertex AI with application default credentials).
"""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

try:
    from trulens.core.schema import base as base_schema
    from trulens.otel.semconv.trace import SpanAttributes
    from trulens.providers.google import endpoint as google_endpoint
except Exception:  # pragma: no cover
    google_endpoint = None

_MODEL = "gemini-2.5-flash"


class TestGoogleCost(unittest.TestCase):
    def setUp(self):
        if google_endpoint is None:
            self.skipTest("trulens-providers-google not available.")
        self.price = google_endpoint.LITELLM_MODEL_COSTS_TABLE[_MODEL]

    def _expected(self, prompt: int, output: int) -> float:
        return (
            prompt * self.price["input_cost_per_token"]
            + output * self.price["output_cost_per_token"]
        )

    def test_cost_computer_bills_thinking_tokens_without_an_api_key(self):
        response = SimpleNamespace(
            model_version=_MODEL,
            usage_metadata=SimpleNamespace(
                total_token_count=2150,
                prompt_token_count=100,
                candidates_token_count=50,
                thoughts_token_count=2000,
            ),
        )

        with mock.patch.dict("os.environ", clear=True):
            attributes = google_endpoint.GoogleCostComputer.handle_response(
                response
            )

        self.assertAlmostEqual(
            attributes[SpanAttributes.COST.COST],
            self._expected(prompt=100, output=50 + 2000),
        )
        self.assertEqual(
            attributes[SpanAttributes.COST.NUM_REASONING_TOKENS], 2000
        )

    def test_callback_bills_thinking_tokens(self):
        callback = google_endpoint.GoogleCallback.model_construct(
            cost=base_schema.Cost()
        )
        response = {
            "model_version": _MODEL,
            "usage_metadata": {
                "total_token_count": 2150,
                "prompt_token_count": 100,
                "candidates_token_count": 50,
                "thoughts_token_count": 2000,
            },
        }

        callback.handle_generation(response)

        self.assertAlmostEqual(
            callback.cost.cost, self._expected(prompt=100, output=50 + 2000)
        )

    def test_cost_without_thinking_tokens_is_unchanged(self):
        response = SimpleNamespace(
            model_version=_MODEL,
            usage_metadata=SimpleNamespace(
                total_token_count=150,
                prompt_token_count=100,
                candidates_token_count=50,
                thoughts_token_count=None,
            ),
        )

        with mock.patch.dict("os.environ", clear=True):
            attributes = google_endpoint.GoogleCostComputer.handle_response(
                response
            )

        self.assertAlmostEqual(
            attributes[SpanAttributes.COST.COST],
            self._expected(prompt=100, output=50),
        )


if __name__ == "__main__":
    unittest.main()
