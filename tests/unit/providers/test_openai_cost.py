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


class TestOpenAICacheReadTokens(unittest.TestCase):
    def setUp(self):
        if OpenAICallback is None:
            self.skipTest("trulens-providers-openai not available.")

    def test_cached_tokens_helper(self):
        from trulens.providers.openai.endpoint import _cached_tokens

        usage = {"prompt_tokens_details": {"cached_tokens": 128}}
        self.assertEqual(_cached_tokens(usage, "prompt_tokens_details"), 128)
        self.assertEqual(_cached_tokens(usage, "input_tokens_details"), 0)
        self.assertEqual(_cached_tokens({}, "prompt_tokens_details"), 0)
        self.assertEqual(_cached_tokens(None, "prompt_tokens_details"), 0)
        self.assertEqual(
            _cached_tokens(
                {"prompt_tokens_details": {"cached_tokens": None}},
                "prompt_tokens_details",
            ),
            0,
        )

    def test_streamed_usage_records_cached_tokens(self):
        from trulens.otel.semconv.trace import SpanAttributes
        from trulens.providers.openai.endpoint import cost_attributes

        callback = OpenAICallback(
            endpoint=core_endpoint.Endpoint(name="test-endpoint")
        )
        callback.handle_generation_usage(
            "gpt-4o",
            {
                "prompt_tokens": 200,
                "completion_tokens": 10,
                "total_tokens": 210,
                "prompt_tokens_details": {"cached_tokens": 150},
            },
        )

        self.assertEqual(callback.cost.n_cache_read_tokens, 150)
        attrs = cost_attributes(callback, "gpt-4o")
        self.assertEqual(attrs[SpanAttributes.COST.NUM_CACHE_READ_TOKENS], 150)

    def test_cost_attributes_omit_cache_read_when_zero(self):
        from trulens.otel.semconv.trace import SpanAttributes
        from trulens.providers.openai.endpoint import cost_attributes

        callback = OpenAICallback(
            endpoint=core_endpoint.Endpoint(name="test-endpoint")
        )
        callback.handle_generation_usage(
            "gpt-4o",
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )

        attrs = cost_attributes(callback, "gpt-4o")
        self.assertNotIn(SpanAttributes.COST.NUM_CACHE_READ_TOKENS, attrs)

    def _handle(self, response, model):
        from trulens.providers.openai.endpoint import OpenAIEndpoint

        callback = OpenAICallback(
            endpoint=core_endpoint.Endpoint(name="test-endpoint")
        )
        OpenAIEndpoint._handle_response(model, response, [callback])
        return callback

    def test_chat_completion_non_streaming_records_cache_read(self):
        from openai.types.chat import ChatCompletion

        response = ChatCompletion.model_validate({
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hi"},
                }
            ],
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 10,
                "total_tokens": 210,
                "prompt_tokens_details": {"cached_tokens": 150},
            },
        })

        callback = self._handle(response, "gpt-4o")

        self.assertEqual(callback.cost.n_cache_read_tokens, 150)
        self.assertEqual(callback.cost.n_prompt_tokens, 200)

    def test_responses_api_non_streaming_records_cache_read(self):
        from openai.types.responses.response import Response

        response = Response.model_validate({
            "id": "resp-1",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 200,
                "output_tokens": 10,
                "total_tokens": 210,
                "input_tokens_details": {
                    "cached_tokens": 150,
                    "cache_write_tokens": 0,
                },
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        })

        callback = self._handle(response, "gpt-5")

        self.assertEqual(callback.cost.n_cache_read_tokens, 150)
        self.assertEqual(callback.cost.n_prompt_tokens, 200)

    def test_repeated_non_streaming_calls_accumulate_cache_read(self):
        from openai.types.chat import ChatCompletion
        from trulens.providers.openai.endpoint import OpenAIEndpoint

        def completion(cached):
            return ChatCompletion.model_validate({
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "hi"},
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 5,
                    "total_tokens": 105,
                    "prompt_tokens_details": {"cached_tokens": cached},
                },
            })

        callback = OpenAICallback(
            endpoint=core_endpoint.Endpoint(name="test-endpoint")
        )
        for cached in (40, 60):
            OpenAIEndpoint._handle_response(
                "gpt-4o", completion(cached), [callback]
            )

        self.assertEqual(callback.cost.n_cache_read_tokens, 100)


if __name__ == "__main__":
    unittest.main()
