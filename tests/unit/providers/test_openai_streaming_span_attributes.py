# pyright: reportMissingImports=false, reportMissingModuleSource=false
from typing import Any, Dict
from unittest.mock import Mock

import openai
from opentelemetry import trace
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
import pytest
from trulens.otel.semconv.trace import SpanAttributes


def _make_chunk(content: str, usage=None):
    return Mock(
        model="gpt-4o-mini",
        choices=[Mock(delta=Mock(content=content), finish_reason=None)],
        usage=usage,
    )


def _make_sync_stream(chunks):
    stream = openai.Stream(
        cast_to=object, client=Mock(spec=openai.OpenAI), response=Mock()
    )

    def _iterator():
        yield from chunks

    stream._iterator = _iterator()
    return stream


@pytest.fixture
def otel_setup():
    from trulens.experimental.otel_tracing.core.session import (
        _set_up_tracer_provider,
    )

    exporter = InMemorySpanExporter()
    _set_up_tracer_provider()
    processor = SimpleSpanProcessor(exporter)
    trace.get_tracer_provider().add_span_processor(processor)
    try:
        yield exporter
    finally:
        processor.shutdown()


@pytest.mark.optional
def test_streaming_attrs_returned_immediately(otel_setup):
    from trulens.providers.openai.endpoint import OpenAICostComputer

    stream = _make_sync_stream([
        _make_chunk("Hello"),
        _make_chunk(" world"),
    ])

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        ret = OpenAICostComputer.handle_response(stream)
        # Fully consume the stream (handle_response rewires
        # `stream._iterator` in place, so iterating `stream` itself drives
        # the wrapped iterator).
        list(stream)

    assert ret[SpanAttributes.GENERATION.IS_STREAMING] is True
    assert ret[SpanAttributes.GENERATION.TIME_TO_FIRST_TOKEN_MS] >= 0


@pytest.mark.optional
def test_chunks_received_and_tokens_per_second_recorded_on_open_span(
    otel_setup,
):
    from trulens.providers.openai.endpoint import OpenAICostComputer

    usage = Mock(
        spec=["prompt_tokens", "completion_tokens", "total_tokens"],
        prompt_tokens=12,
        completion_tokens=7,
        total_tokens=19,
    )
    stream = _make_sync_stream([
        _make_chunk("Hello"),
        _make_chunk(" world"),
        _make_chunk("", usage=usage),
    ])

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        OpenAICostComputer.handle_response(stream)
        list(stream)  # fully consume while span is still open

    spans = otel_setup.get_finished_spans()
    assert len(spans) == 1
    attrs: Dict[str, Any] = spans[0].attributes
    assert attrs[SpanAttributes.GENERATION.CHUNKS_RECEIVED] == 3
    assert attrs[SpanAttributes.GENERATION.TOKENS_PER_SECOND] > 0
    assert attrs[SpanAttributes.COST.NUM_PROMPT_TOKENS] == 12
    assert attrs[SpanAttributes.COST.NUM_COMPLETION_TOKENS] == 7
    assert attrs[SpanAttributes.COST.NUM_TOKENS] == 19


@pytest.mark.optional
def test_chunks_received_omitted_without_usage(otel_setup):
    from trulens.providers.openai.endpoint import OpenAICostComputer

    stream = _make_sync_stream([_make_chunk("Hello"), _make_chunk(" world")])

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        OpenAICostComputer.handle_response(stream)
        list(stream)

    attrs = otel_setup.get_finished_spans()[0].attributes
    assert attrs[SpanAttributes.GENERATION.CHUNKS_RECEIVED] == 2
    assert SpanAttributes.GENERATION.TOKENS_PER_SECOND not in attrs


@pytest.mark.optional
def test_stream_completion_after_span_ended_is_a_safe_noop(otel_setup):
    from trulens.providers.openai.endpoint import OpenAICostComputer

    stream = _make_sync_stream([_make_chunk("Hello"), _make_chunk(" world")])

    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("test_span"):
        OpenAICostComputer.handle_response(stream)
    # Span has ended -- consuming the rest of the stream now must not raise.
    list(stream)

    attrs = otel_setup.get_finished_spans()[0].attributes
    assert SpanAttributes.GENERATION.CHUNKS_RECEIVED not in attrs


class _FakeChatCompletion:
    """Minimal stand-in for a non-streaming `ChatCompletion`: has `.model`
    like the real thing, and supports `in` (pydantic models iterate as
    key/value pairs) the same way `_handle_response`'s generic response
    handling expects."""

    model = "gpt-4o-mini"
    usage = None

    def __contains__(self, _key):
        return False


@pytest.mark.optional
def test_non_streaming_response_unaffected():
    from trulens.providers.openai.endpoint import OpenAICostComputer

    ret = OpenAICostComputer.handle_response(_FakeChatCompletion())
    assert SpanAttributes.GENERATION.IS_STREAMING not in ret
