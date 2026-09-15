import asyncio
import os
import sys
import unittest
from unittest.mock import patch

from opentelemetry.baggage import set_baggage
import opentelemetry.context as context_api
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from trulens.core.otel.instrument import instrument
from trulens.core.otel.mcp import _wrap_call_tool
from trulens.core.otel.mcp import instrument_mcp
from trulens.core.otel.recording import Recording
from trulens.experimental.otel_tracing.core.session import (
    _set_up_tracer_provider,
)
from trulens.otel.semconv.trace import SpanAttributes


class _Result:
    def __init__(self, is_error: bool = False) -> None:
        self.isError = is_error
        self.content = [{"text": "ok"}]


class _FakeSession:
    server_info = {"name": "weather"}

    async def call_tool(self, name, arguments=None, **kwargs):
        return _Result(is_error=name == "missing")


class TestMCPInstrumentation(unittest.TestCase):
    def setUp(self) -> None:
        instrument.enable_all_instrumentation()
        self.exporter = InMemorySpanExporter()
        _set_up_tracer_provider()
        self.processor = SimpleSpanProcessor(self.exporter)
        from opentelemetry import trace

        trace.get_tracer_provider().add_span_processor(self.processor)
        self.token = context_api.attach(
            set_baggage("__trulens_recording__", Recording(None))
        )
        self.record_id_token = context_api.attach(
            set_baggage(SpanAttributes.RECORD_ID, "test-record")
        )
        self.capture = os.environ.get("TRULENS_CAPTURE_TOOL_PAYLOADS")
        os.environ["TRULENS_CAPTURE_TOOL_PAYLOADS"] = "true"

    def tearDown(self) -> None:
        if self.capture is None:
            os.environ.pop("TRULENS_CAPTURE_TOOL_PAYLOADS", None)
        else:
            os.environ["TRULENS_CAPTURE_TOOL_PAYLOADS"] = self.capture
        context_api.detach(self.token)
        context_api.detach(self.record_id_token)
        self.processor.shutdown()

    def test_records_metadata_payloads_and_sdk_error_results(self) -> None:
        _wrap_call_tool(_FakeSession)

        async def run() -> None:
            session = _FakeSession()
            await session.call_tool(
                "missing", {"city": "Chicago", "api_key": "secret"}
            )

        asyncio.run(run())
        span = self.exporter.get_finished_spans()[-1]
        self.assertEqual(
            span.attributes[SpanAttributes.MCP.TOOL_NAME], "missing"
        )
        self.assertEqual(
            span.attributes[SpanAttributes.MCP.SERVER_NAME], "weather"
        )
        self.assertIn(
            "[REDACTED]", span.attributes[SpanAttributes.MCP.INPUT_ARGUMENTS]
        )
        self.assertTrue(span.attributes[SpanAttributes.MCP.OUTPUT_IS_ERROR])
        self.assertGreaterEqual(
            span.attributes[SpanAttributes.MCP.EXECUTION_TIME_MS], 0
        )

    def test_cancellation_propagates_and_does_not_double_wrap(self) -> None:
        class BlockingSession:
            async def call_tool(self, name, arguments=None):
                await asyncio.Event().wait()

        _wrap_call_tool(BlockingSession)
        first = BlockingSession.__dict__["call_tool"]
        _wrap_call_tool(BlockingSession)
        self.assertIs(BlockingSession.__dict__["call_tool"], first)

        async def run() -> None:
            task = asyncio.create_task(BlockingSession().call_tool("slow"))
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task

        asyncio.run(run())
        self.assertEqual(len(self.exporter.get_finished_spans()), 1)

    def test_sdk_is_optional_until_activation(self) -> None:
        with patch.dict(sys.modules, {"mcp": None}):
            with self.assertRaisesRegex(ImportError, "optional 'mcp' package"):
                instrument_mcp()
