"""Tests for W3C trace context propagation on outbound HTTP requests.

TruLens instruments the caller, but the provider runs on the far side of an
HTTP call, so without propagation the provider's server span starts a separate
trace. These cover the helper that writes the active span's context and the
httpx transport that applies it to every outbound request.
"""

import unittest

import httpx
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import format_trace_id
from opentelemetry.trace import get_current_span
from trulens.core.otel import propagation


class _TracingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = TracerProvider()
        self.provider.add_span_processor(
            SimpleSpanProcessor(InMemorySpanExporter())
        )
        self.tracer = self.provider.get_tracer(__name__)


class TestInjectTraceContext(_TracingTestCase):
    def test_injects_active_span_context(self) -> None:
        with self.tracer.start_as_current_span("outbound"):
            headers = {}
            trace_id = format_trace_id(
                get_current_span().get_span_context().trace_id
            )
            self.assertTrue(propagation.inject_trace_context(headers))

        self.assertIn("traceparent", headers)
        self.assertIn(trace_id, headers["traceparent"])

    def test_injects_nothing_without_an_active_span(self) -> None:
        headers = {}
        self.assertFalse(propagation.inject_trace_context(headers))
        self.assertEqual({}, headers)

    def test_preserves_existing_headers(self) -> None:
        with self.tracer.start_as_current_span("outbound"):
            headers = {"authorization": "Bearer test"}
            propagation.inject_trace_context(headers)

        self.assertEqual("Bearer test", headers["authorization"])


class TestTracingTransport(_TracingTestCase):
    def _client_recording_headers(self) -> tuple:
        recorded = {}

        def handler(request: httpx.Request) -> httpx.Response:
            recorded.update(request.headers)
            return httpx.Response(200, json={"ok": True})

        transport = propagation.tracing_transport(httpx.MockTransport(handler))
        return httpx.Client(transport=transport), recorded

    def test_transport_injects_on_outbound_request(self) -> None:
        client, recorded = self._client_recording_headers()
        with self.tracer.start_as_current_span("outbound"):
            trace_id = format_trace_id(
                get_current_span().get_span_context().trace_id
            )
            client.get("https://example.invalid/v1/models")

        self.assertIn("traceparent", recorded)
        self.assertIn(trace_id, recorded["traceparent"])

    def test_transport_sends_request_without_an_active_span(self) -> None:
        # A client with the transport must still work outside a trace.
        client, recorded = self._client_recording_headers()
        response = client.get("https://example.invalid/v1/models")

        self.assertEqual(200, response.status_code)
        self.assertNotIn("traceparent", recorded)


if __name__ == "__main__":
    unittest.main()
