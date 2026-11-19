"""Regression tests: Cortex usage is tracked for every finish reason.

`CortexEndpoint.handle_wrapped_call` only passed a streamed response to the
callbacks when the choice's `finish_reason` was `stop`. Completions cut off by
`length` or ended by `tool_calls` still carry `usage` and are billed, but
their tokens and cost were dropped.
"""

import json
from unittest import mock

import pytest

pytest.importorskip(
    "trulens.providers.cortex",
    reason="trulens-providers-cortex is not installed",
)

from snowflake.cortex._sse_client import Event  # noqa: E402
from trulens.core.schema import base as base_schema  # noqa: E402
from trulens.providers.cortex import endpoint as cortex_endpoint  # noqa: E402

_MODEL = "mistral-large"


def _event(finish_reason):
    return Event(
        data=json.dumps({
            "model": _MODEL,
            "choices": [{"finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        })
    )


def _tokens_counted(finish_reason):
    callback = cortex_endpoint.CortexCallback.model_construct(
        cost=base_schema.Cost()
    )
    endpoint = mock.Mock(
        global_callback=cortex_endpoint.CortexCallback.model_construct(
            cost=base_schema.Cost()
        )
    )
    cortex_endpoint.CortexEndpoint.handle_wrapped_call(
        endpoint, mock.Mock(), mock.Mock(), _event(finish_reason), callback
    )
    assert callback.cost.n_tokens == endpoint.global_callback.cost.n_tokens
    return callback.cost.n_tokens


@pytest.mark.parametrize("finish_reason", ["stop", "length", "tool_calls"])
def test_usage_counted_for_every_finish_reason(finish_reason):
    assert _tokens_counted(finish_reason) == 15
