# pyright: reportMissingImports=false, reportMissingModuleSource=false
from unittest.mock import Mock

import pytest


def _make_chunk(
    content=None, finish_reason=None, usage=None, model="gpt-4o-mini"
):
    return Mock(
        model=model,
        choices=[Mock(delta=Mock(content=content), finish_reason=finish_reason)]
        if content is not None or finish_reason is not None
        else [],
        usage=usage,
    )


@pytest.mark.optional
def test_finalize_stream_uses_real_usage_from_trailing_chunk():
    """The bug this fixes: token_usage was hard-coded to `{}` on the
    finish_reason=="stop" chunk, so streamed calls never got real cost
    tracked even when the caller requested
    stream_options={"include_usage": True} -- the real usage arrives on a
    *separate* trailing chunk with empty choices, which finish_reason=="stop"
    handling never even looked at."""
    from trulens.providers.openai.endpoint import OpenAICallback
    from trulens.providers.openai.endpoint import OpenAIEndpoint

    callback = OpenAICallback(endpoint=OpenAIEndpoint(api_key="test"))

    callback.handle_generation_chunk(_make_chunk(content="Hello"))
    callback.handle_generation_chunk(_make_chunk(content=" world"))
    callback.handle_generation_chunk(_make_chunk(finish_reason="stop"))
    # Nothing should be finalized yet -- real usage hasn't arrived, and we
    # don't yet know whether it's coming.
    assert callback.cost.n_successful_requests == 0

    usage = Mock(
        spec=["prompt_tokens", "completion_tokens", "total_tokens"],
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    callback.handle_generation_chunk(_make_chunk(usage=usage))
    # Still not finalized -- that only happens when the stream is actually
    # exhausted (finalize_stream), not on any particular chunk pattern.
    assert callback.cost.n_successful_requests == 0

    callback.finalize_stream(model_name="gpt-4o-mini")

    assert callback.cost.n_successful_requests == 1
    assert callback.cost.n_prompt_tokens == 10
    assert callback.cost.n_completion_tokens == 5
    assert callback.cost.n_tokens == 15
    assert callback.cost.cost > 0  # gpt-4o-mini has known per-token pricing


@pytest.mark.optional
def test_finalize_stream_without_usage_still_records_request():
    """When the caller doesn't request stream_options={"include_usage":
    True}, no usage chunk ever arrives -- finalize_stream should still flush
    the accumulated text (same as the old behavior), just with unknown
    (zero) cost, not silently drop the request."""
    from trulens.providers.openai.endpoint import OpenAICallback
    from trulens.providers.openai.endpoint import OpenAIEndpoint

    callback = OpenAICallback(endpoint=OpenAIEndpoint(api_key="test"))

    callback.handle_generation_chunk(_make_chunk(content="Hello"))
    callback.handle_generation_chunk(_make_chunk(finish_reason="stop"))
    callback.finalize_stream(model_name="gpt-4o-mini")

    assert callback.cost.n_successful_requests == 1
    assert callback.cost.cost == 0


@pytest.mark.optional
def test_finalize_stream_is_a_noop_with_nothing_accumulated():
    from trulens.providers.openai.endpoint import OpenAICallback
    from trulens.providers.openai.endpoint import OpenAIEndpoint

    callback = OpenAICallback(endpoint=OpenAIEndpoint(api_key="test"))
    callback.finalize_stream(model_name="gpt-4o-mini")
    assert callback.cost.n_successful_requests == 0


@pytest.mark.optional
def test_handle_response_stream_accumulates_real_cost_end_to_end():
    """Full path: OpenAICostComputer.handle_response wraps the stream, the
    caller consumes it exactly like real application code would, and the
    *same* callback object's cost (reachable via the endpoint actually used
    by the app, e.g. provider.endpoint.global_callback in a real provider)
    ends up with real, non-zero cost -- not just $0 like before this fix."""
    import openai
    from trulens.providers.openai.endpoint import OpenAIEndpoint

    endpoint = OpenAIEndpoint(api_key="test")

    usage = Mock(
        spec=["prompt_tokens", "completion_tokens", "total_tokens"],
        prompt_tokens=8,
        completion_tokens=4,
        total_tokens=12,
    )
    stream = openai.Stream(
        cast_to=object, client=Mock(spec=openai.OpenAI), response=Mock()
    )

    def _iterator():
        yield _make_chunk(content="Hi")
        yield _make_chunk(finish_reason="stop")
        yield _make_chunk(usage=usage)

    stream._iterator = _iterator()

    from trulens.providers.openai.endpoint import OpenAICallback

    callback = OpenAICallback(endpoint=endpoint)
    wrapped = OpenAIEndpoint._handle_response(
        model_name="gpt-4o-mini", response=stream, callbacks=[callback]
    )
    list(wrapped)  # simulate the app consuming the stream

    assert callback.cost.n_successful_requests == 1
    assert callback.cost.n_prompt_tokens == 8
    assert callback.cost.n_completion_tokens == 4
    assert callback.cost.cost > 0
