"""Regression tests: Gemini usage is tracked for every finish reason.

`GoogleEndpoint.handle_wrapped_call` only passed a response to the callbacks
when the candidate's `finish_reason` was `STOP`. Responses cut off by
`MAX_TOKENS` or blocked by `SAFETY` still carry `usage_metadata` and are
billed, but their tokens and cost were dropped.
"""

from __future__ import annotations

import unittest
from unittest import mock

try:
    from google.genai.types import GenerateContentResponse
    from trulens.core.schema import base as base_schema
    from trulens.providers.google import endpoint as google_endpoint
except Exception:  # pragma: no cover
    google_endpoint = None

_MODEL = "gemini-2.5-flash"


def _response(finish_reason: str) -> "GenerateContentResponse":
    return GenerateContentResponse.model_validate({
        "candidates": [{"finish_reason": finish_reason}],
        "usage_metadata": {
            "prompt_token_count": 10,
            "candidates_token_count": 5,
            "total_token_count": 15,
        },
        "model_version": _MODEL,
    })


class TestGoogleFinishReason(unittest.TestCase):
    def setUp(self):
        if google_endpoint is None:
            self.skipTest("trulens-providers-google not available.")

    def _tokens_counted(self, finish_reason: str) -> int:
        callback = google_endpoint.GoogleCallback.model_construct(
            cost=base_schema.Cost()
        )
        endpoint = mock.Mock(
            global_callback=google_endpoint.GoogleCallback.model_construct(
                cost=base_schema.Cost()
            )
        )
        google_endpoint.GoogleEndpoint.handle_wrapped_call(
            endpoint,
            mock.Mock(),
            mock.Mock(),
            _response(finish_reason),
            callback,
        )
        self.assertEqual(
            callback.cost.n_tokens, endpoint.global_callback.cost.n_tokens
        )
        return callback.cost.n_tokens

    def test_stop_is_counted(self):
        self.assertEqual(self._tokens_counted("STOP"), 15)

    def test_max_tokens_is_counted(self):
        self.assertEqual(self._tokens_counted("MAX_TOKENS"), 15)

    def test_safety_is_counted(self):
        self.assertEqual(self._tokens_counted("SAFETY"), 15)
