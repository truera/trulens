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

import json
import unittest

try:
    import httpx
    from langchain_core.outputs import Generation
    from langchain_core.outputs import LLMResult
    import openai
    from trulens.core.feedback import endpoint as core_endpoint
    from trulens.providers.openai import endpoint as openai_endpoint
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

    def test_track_cost_accumulates_streamed_usage(self):
        """Stream usage adds to earlier calls in the public cost scope."""
        model = "o3-mini"
        for first_streamed in (True, False):
            with self.subTest(first_streamed=first_streamed):
                usages = iter([(100, 20, 5), (50, 10, 3)])

                def respond(request: httpx.Request) -> httpx.Response:
                    prompt, completion, reasoning = next(usages)
                    usage = {
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": prompt + completion,
                        "completion_tokens_details": {
                            "reasoning_tokens": reasoning
                        },
                    }
                    common = {
                        "id": f"chatcmpl-{prompt}",
                        "created": 0,
                        "model": model,
                    }
                    if not json.loads(request.content).get("stream"):
                        return httpx.Response(
                            200,
                            json={
                                **common,
                                "object": "chat.completion",
                                "choices": [
                                    {
                                        "index": 0,
                                        "finish_reason": "stop",
                                        "message": {
                                            "role": "assistant",
                                            "content": "hi",
                                        },
                                    }
                                ],
                                "usage": usage,
                            },
                        )
                    chunks = [
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": "hi"},
                                    "finish_reason": None,
                                }
                            ]
                        },
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {},
                                    "finish_reason": "stop",
                                }
                            ]
                        },
                        {"choices": [], "usage": usage},
                    ]
                    data = "".join(
                        "data: "
                        + json.dumps({
                            **common,
                            "object": "chat.completion.chunk",
                            **chunk,
                        })
                        + "\n\n"
                        for chunk in chunks
                    )
                    return httpx.Response(
                        200,
                        content=data + "data: [DONE]\n\n",
                        headers={"Content-Type": "text/event-stream"},
                    )

                with openai.OpenAI(
                    api_key="sk-test",
                    base_url="https://example.invalid/v1",
                    http_client=httpx.Client(
                        transport=httpx.MockTransport(respond)
                    ),
                ) as client:
                    endpoint = openai_endpoint.OpenAIEndpoint(client=client)

                    def generate(streamed):
                        kwargs = (
                            {"stream_options": {"include_usage": True}}
                            if streamed
                            else {}
                        )
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": "hi"}],
                            stream=streamed,
                            **kwargs,
                        )
                        if streamed:
                            return "".join(
                                chunk.choices[0].delta.content or ""
                                for chunk in response
                                if chunk.choices
                            )
                        return response.choices[0].message.content

                    outputs, callback = endpoint.track_cost(
                        lambda: [generate(first_streamed), generate(True)]
                    )

                self.assertEqual(outputs, ["hi", "hi"])
                self.assertEqual(callback.cost.n_tokens, 180)
                self.assertEqual(callback.cost.n_prompt_tokens, 150)
                self.assertEqual(callback.cost.n_completion_tokens, 30)
                self.assertEqual(callback.cost.n_reasoning_tokens, 8)
                self.assertAlmostEqual(
                    callback.cost.cost,
                    openai_endpoint.token_cost_for_model(model, 150, 30),
                )


if __name__ == "__main__":
    unittest.main()
