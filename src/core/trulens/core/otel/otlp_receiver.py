"""Lightweight OTLP/HTTP receiver for TruSession.

Accepts OTLP protobuf or JSON span payloads on POST /v1/traces and writes
them to the TruSession's configured database via the existing OTEL exporter
pipeline.

This allows any OTLP-capable client (TypeScript, Go, Java, …) to send spans
to a running TruSession and have them stored alongside Python-emitted spans,
viewable in the TruLens dashboard.

The receiver requires no additional dependencies beyond what is already used
by the OTEL tracing path. It uses the stdlib ``http.server`` and the
``opentelemetry-proto`` package (pulled in transitively by the SDK) for
protobuf decoding.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from trulens.core import session as core_session

logger = logging.getLogger(__name__)

_CONTENT_TYPE_PROTOBUF = "application/x-protobuf"
_CONTENT_TYPE_JSON = "application/json"


class _OTLPRequestHandler(BaseHTTPRequestHandler):
    """HTTP request handler for OTLP /v1/traces."""

    # Injected by OTLPReceiver
    tru_session: core_session.TruSession

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        logger.debug("OTLP receiver: " + format, *args)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/traces":
            self.send_response(404)
            self.end_headers()
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        content_type = self.headers.get("Content-Type", "")

        try:
            spans = self._decode_spans(body, content_type)
            self._ingest_spans(spans)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")
        except Exception:
            logger.exception("OTLP receiver: error processing spans")
            self.send_response(500)
            self.end_headers()

    def _decode_spans(self, body: bytes, content_type: str) -> list[dict]:
        """Decode an OTLP ExportTraceServiceRequest into a list of span dicts."""
        if _CONTENT_TYPE_PROTOBUF in content_type:
            return self._decode_proto(body)
        return self._decode_json(body)

    @staticmethod
    def _decode_proto(body: bytes) -> list[dict]:
        try:
            from opentelemetry.proto.collector.trace.v1 import trace_service_pb2
        except ImportError as exc:
            raise ImportError(
                "opentelemetry-proto is required for protobuf decoding. "
                "Install it with: pip install opentelemetry-proto"
            ) from exc

        req = trace_service_pb2.ExportTraceServiceRequest()
        req.ParseFromString(body)
        spans = []
        for resource_spans in req.resource_spans:
            resource_attrs = {
                kv.key: _proto_any_value(kv.value)
                for kv in resource_spans.resource.attributes
            }
            for scope_spans in resource_spans.scope_spans:
                for span in scope_spans.spans:
                    span_dict = {
                        "name": span.name,
                        "trace_id": span.trace_id.hex(),
                        "span_id": span.span_id.hex(),
                        "parent_span_id": span.parent_span_id.hex()
                        if span.parent_span_id
                        else None,
                        "start_time_unix_nano": span.start_time_unix_nano,
                        "end_time_unix_nano": span.end_time_unix_nano,
                        "attributes": {
                            kv.key: _proto_any_value(kv.value)
                            for kv in span.attributes
                        },
                        "resource_attributes": resource_attrs,
                        "status": {
                            "code": span.status.code,
                            "message": span.status.message,
                        },
                    }
                    spans.append(span_dict)
        return spans

    @staticmethod
    def _decode_json(body: bytes) -> list[dict]:
        payload = json.loads(body.decode("utf-8"))
        spans = []
        for resource_spans in payload.get("resourceSpans", []):
            resource_attrs = {
                kv["key"]: kv["value"].get(
                    "stringValue",
                    kv["value"].get("intValue", kv["value"].get("boolValue")),
                )
                for kv in resource_spans.get("resource", {}).get(
                    "attributes", []
                )
            }
            for scope_spans in resource_spans.get("scopeSpans", []):
                for span in scope_spans.get("spans", []):
                    span_dict = {
                        "name": span.get("name"),
                        "trace_id": span.get("traceId"),
                        "span_id": span.get("spanId"),
                        "parent_span_id": span.get("parentSpanId"),
                        "start_time_unix_nano": span.get("startTimeUnixNano"),
                        "end_time_unix_nano": span.get("endTimeUnixNano"),
                        "attributes": {
                            kv["key"]: kv["value"].get(
                                "stringValue",
                                kv["value"].get(
                                    "intValue",
                                    kv["value"].get("boolValue"),
                                ),
                            )
                            for kv in span.get("attributes", [])
                        },
                        "resource_attributes": resource_attrs,
                        "status": span.get("status", {}),
                    }
                    spans.append(span_dict)
        return spans

    def _ingest_spans(self, spans: list[dict]) -> None:
        """Write decoded spans to the TruSession database."""
        if not spans:
            return
        try:
            self.tru_session.connector.add_otel_spans(spans)
        except AttributeError:
            # Fallback: log a warning — the connector may not support this yet.
            logger.warning(
                "OTLP receiver: connector does not implement add_otel_spans; "
                "%d span(s) dropped. Please upgrade trulens-core.",
                len(spans),
            )


def _proto_any_value(av) -> object:  # type: ignore[no-untyped-def]
    """Extract a Python value from an OTLP AnyValue proto."""
    kind = av.WhichOneof("value")
    if kind == "string_value":
        return av.string_value
    if kind == "int_value":
        return av.int_value
    if kind == "double_value":
        return av.double_value
    if kind == "bool_value":
        return av.bool_value
    if kind == "array_value":
        return [_proto_any_value(v) for v in av.array_value.values]
    if kind == "kvlist_value":
        return {
            kv.key: _proto_any_value(kv.value) for kv in av.kvlist_value.values
        }
    return None


class OTLPReceiver:
    """Wraps an HTTPServer to serve as an OTLP/HTTP receiver.

    Args:
        session: The :class:`~trulens.core.session.TruSession` to ingest spans
            into.
        host: The hostname to bind to.
        port: The port to listen on.
    """

    def __init__(
        self,
        session: core_session.TruSession,
        host: str = "0.0.0.0",
        port: int = 4318,
    ) -> None:
        self._session = session
        self._host = host
        self._port = port

        # Bind the session to the handler class dynamically so each request
        # handler has access to it without a global variable.
        handler = type(
            "_BoundOTLPRequestHandler",
            (_OTLPRequestHandler,),
            {"tru_session": session},
        )
        self._server = HTTPServer((host, port), handler)

    def serve_forever(self) -> None:
        """Block and serve requests until :meth:`shutdown` is called."""
        self._server.serve_forever()

    def shutdown(self) -> None:
        """Stop the server."""
        self._server.shutdown()
