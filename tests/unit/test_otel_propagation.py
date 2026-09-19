"""Tests for W3C trace context propagation on outbound HTTP requests.

TruLens instruments the caller, but the provider runs on the far side of an
HTTP call, so without propagation the provider's server span starts a separate
trace. These cover the helper that writes the active span's context and the
sync and async httpx transports that apply it to every outbound request.
"""

import asyncio
import unittest

import httpx
from opentelemetry.context import Context
from opentelemetry.propagate import set_global_textmap
from opentelemetry.propagators.textmap import TextMapPropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import format_trace_id
from opentelemetry.trace import get_current_span
from trulens.core.otel import propagation


class _NoOpPropagator(TextMapPropagator):
    """A propagator that is valid but writes nothing, like propagation off."""

    @property
    def fields(self) -> set:
        return set()

    def extract(
        self, carrier: object, context: object = None, getter: object = None
    ) -> Context:
        return context or Context()

    def inject(
        self, carrier: object, context: object = None, setter: object = None
    ) -> None:
        return None


class _TracingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = TracerProvider()
        self.provider.add_span_processor(
            SimpleSpanProcessor(InMemorySpanExporter())
        )
        self.tracer = self.provider.get_tracer(__name__)

    def current_trace_id(self) -> str:
        """Format the active span's trace id for asserting on headers."""
        return format_trace_id(get_current_span().get_span_context().trace_id)


class TestInjectTraceContext(_TracingTestCase):
    def test_injects_active_span_context(self) -> None:
        with self.tracer.start_as_current_span("outbound"):
            headers = {}
            trace_id = self.current_trace_id()
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

    def test_reports_no_injection_when_propagator_writes_nothing(self) -> None:
        # A valid span context is not enough on its own: with a no-op
        # propagator installed the header mapping stays untouched, and the
        # return value has to say so.
        from opentelemetry.propagate import get_global_textmap

        original = get_global_textmap()
        set_global_textmap(_NoOpPropagator())
        self.addCleanup(set_global_textmap, original)

        with self.tracer.start_as_current_span("outbound"):
            headers = {}
            self.assertFalse(propagation.inject_trace_context(headers))

        self.assertEqual({}, headers)


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
            trace_id = self.current_trace_id()
            client.get("https://example.invalid/v1/models")

        self.assertIn("traceparent", recorded)
        self.assertIn(trace_id, recorded["traceparent"])

    def test_transport_sends_request_without_an_active_span(self) -> None:
        # A client with the transport must still work outside a trace.
        client, recorded = self._client_recording_headers()
        response = client.get("https://example.invalid/v1/models")

        self.assertEqual(200, response.status_code)
        self.assertNotIn("traceparent", recorded)

    def test_close_is_delegated_to_the_inner_transport(self) -> None:
        # The inner transport owns the connection pool, so closing the wrapper
        # has to close the inner one too.
        inner = _CloseSpy()
        transport = propagation.tracing_transport(inner)
        transport.close()

        self.assertTrue(inner.closed)

    def test_close_reaches_the_inner_transport_through_client_close(
        self,
    ) -> None:
        inner = _CloseSpy()
        client = httpx.Client(transport=propagation.tracing_transport(inner))
        client.close()

        self.assertTrue(inner.closed)


class TestAsyncTracingTransport(_TracingTestCase):
    def _client_recording_headers(self) -> tuple:
        recorded = {}

        async def handler(request: httpx.Request) -> httpx.Response:
            recorded.update(request.headers)
            return httpx.Response(200, json={"ok": True})

        transport = propagation.async_tracing_transport(
            httpx.MockTransport(handler)
        )
        return httpx.AsyncClient(transport=transport), recorded

    def test_transport_injects_on_outbound_request(self) -> None:
        client, recorded = self._client_recording_headers()

        async def request() -> None:
            with self.tracer.start_as_current_span("outbound"):
                trace_id = self.current_trace_id()
                await client.get("https://example.invalid/v1/models")
            self.recorded_trace_id = trace_id

        asyncio.run(request())
        asyncio.run(client.aclose())

        self.assertIn("traceparent", recorded)
        self.assertIn(self.recorded_trace_id, recorded["traceparent"])

    def test_transport_sends_request_without_an_active_span(self) -> None:
        client, recorded = self._client_recording_headers()

        async def request() -> httpx.Response:
            return await client.get("https://example.invalid/v1/models")

        response = asyncio.run(request())
        asyncio.run(client.aclose())

        self.assertEqual(200, response.status_code)
        self.assertNotIn("traceparent", recorded)

    def test_aclose_is_delegated_to_the_inner_transport(self) -> None:
        inner = _CloseSpy()
        transport = propagation.async_tracing_transport(inner)
        asyncio.run(transport.aclose())

        self.assertTrue(inner.closed)


class _CloseSpy(httpx.MockTransport):
    """MockTransport that records whether it was closed."""

    def __init__(self) -> None:
        super().__init__(lambda request: httpx.Response(200))
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.closed = True


if __name__ == "__main__":
    unittest.main()
