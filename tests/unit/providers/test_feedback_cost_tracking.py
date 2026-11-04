"""Regression tests: feedback calls on Anthropic and Google record their cost.

Feedback evaluation tallies cost with `Endpoint.track_all_costs_tally`. That
only creates callbacks for the endpoints listed in `Endpoint.ENDPOINT_SETUPS`,
and a provider's `handle_wrapped_call` is only reached through the wrapper its
endpoint installs on the SDK method. `AnthropicEndpoint` and `GoogleEndpoint`
had neither, so an evaluation backed by either reported zero tokens and zero
cost (#2870).

Each test builds the real endpoint around a real SDK client and runs the SDK
method the provider calls; only the outgoing request is replaced.
"""

from __future__ import annotations

import unittest
from unittest import mock

from trulens.core.feedback import endpoint as core_endpoint

try:
    import anthropic
    from trulens.providers.anthropic import endpoint as anthropic_endpoint
except Exception:  # pragma: no cover
    anthropic_endpoint = None

try:
    from google import genai
    from google.genai import types as genai_types
    from trulens.providers.google import endpoint as google_endpoint
except Exception:  # pragma: no cover
    google_endpoint = None


def _expected_cost(price: dict, prompt: int, completion: int) -> float:
    return (
        prompt * price["input_cost_per_token"]
        + completion * price["output_cost_per_token"]
    )


class TestAnthropicFeedbackCost(unittest.TestCase):
    _MODEL = "claude-sonnet-4-5"

    def setUp(self):
        if anthropic_endpoint is None:
            self.skipTest("trulens-providers-anthropic not available.")

    def test_feedback_call_records_tokens_and_cost(self):
        client = anthropic.Anthropic(api_key="test-key")
        endpoint = anthropic_endpoint.AnthropicEndpoint(client=client)
        message = anthropic.types.Message.model_construct(
            id="msg_1",
            type="message",
            role="assistant",
            model=self._MODEL,
            content=[],
            stop_reason="end_turn",
            usage=anthropic.types.Usage.model_construct(
                input_tokens=1200, output_tokens=300
            ),
        )

        # The same call `Anthropic._create_chat_completion` makes.
        with mock.patch.object(client.messages, "_post", return_value=message):
            _, tally = core_endpoint.Endpoint.track_all_costs_tally(
                endpoint.run_in_pace,
                func=endpoint.client.client.messages.create,
                model=self._MODEL,
                max_tokens=16,
                messages=[{"role": "user", "content": "hi"}],
            )

        cost = tally()
        self.assertEqual(cost.n_tokens, 1500)
        self.assertEqual(cost.n_prompt_tokens, 1200)
        self.assertEqual(cost.n_completion_tokens, 300)
        self.assertAlmostEqual(
            cost.cost,
            _expected_cost(
                anthropic_endpoint.LITELLM_MODEL_COSTS_TABLE[self._MODEL],
                prompt=1200,
                completion=300,
            ),
        )


class TestGoogleFeedbackCost(unittest.TestCase):
    _MODEL = "gemini-2.5-flash"

    def setUp(self):
        if google_endpoint is None:
            self.skipTest("trulens-providers-google not available.")

    def test_feedback_call_records_tokens_and_cost(self):
        endpoint = google_endpoint.GoogleEndpoint(
            client=genai.Client(api_key="test-key")
        )
        response = genai_types.GenerateContentResponse(
            candidates=[
                genai_types.Candidate(
                    content=genai_types.Content(
                        role="model", parts=[genai_types.Part(text="3")]
                    ),
                    finish_reason=genai_types.FinishReason.STOP,
                )
            ],
            usage_metadata=genai_types.GenerateContentResponseUsageMetadata(
                prompt_token_count=1200,
                candidates_token_count=300,
                total_token_count=1500,
            ),
            model_version=self._MODEL,
        )

        # The same call `Google._create_chat_completion` makes.
        with mock.patch.object(
            genai.models.Models, "_generate_content", return_value=response
        ):
            _, tally = core_endpoint.Endpoint.track_all_costs_tally(
                endpoint.client.models.generate_content,
                model=self._MODEL,
                contents="hi",
            )

        cost = tally()
        self.assertEqual(cost.n_tokens, 1500)
        self.assertEqual(cost.n_prompt_tokens, 1200)
        self.assertEqual(cost.n_completion_tokens, 300)
        self.assertAlmostEqual(
            cost.cost,
            _expected_cost(
                google_endpoint.LITELLM_MODEL_COSTS_TABLE[self._MODEL],
                prompt=1200,
                completion=300,
            ),
        )


if __name__ == "__main__":
    unittest.main()
