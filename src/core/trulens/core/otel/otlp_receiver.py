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

from datetime import datetime
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from trulens.core import session as core_session

from trulens.core.schema import app as app_schema
from trulens.core.schema import event as event_schema
from trulens.otel.semconv import trace as semconv_trace

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
        if self.path == "/v1/traces":
            self._handle_traces()
        elif self.path == "/v1/register":
            self._handle_register()
        else:
            self.send_response(404)
            self.end_headers()

    # ------------------------------------------------------------------
    # Body reading
    # ------------------------------------------------------------------

    def _read_body(self) -> bytes:
        """Read the full request body, handling chunked encoding."""
        content_length_str = self.headers.get("Content-Length")
        transfer_encoding = self.headers.get("Transfer-Encoding", "")

        if content_length_str is not None:
            body = self.rfile.read(int(content_length_str))
        elif "chunked" in transfer_encoding.lower():
            body = self._read_chunked()
        else:
            body = b""

        content_encoding = self.headers.get("Content-Encoding", "")
        if content_encoding == "gzip":
            import gzip

            body = gzip.decompress(body)

        return body

    # ------------------------------------------------------------------
    # POST /v1/register — app registration
    # ------------------------------------------------------------------

    def _handle_register(self) -> None:
        try:
            body = self._read_body()
            payload = json.loads(body.decode("utf-8"))
            app_name = payload["app_name"]
            app_version = payload.get("app_version", "base")

            app_def = app_schema.AppDefinition(
                app_name=app_name,
                app_version=app_version,
                root_class=None,
                app={},
            )
            app_id = self.tru_session.connector.add_app(app_def)

            logger.info(
                "OTLP receiver: registered app %r version=%r id=%s",
                app_name,
                app_version,
                app_id,
            )
            self._send_json(200, {"app_id": app_id})
        except Exception:
            logger.exception("OTLP receiver: error registering app")
            self._send_json(500, {"error": "registration failed"})

    # ------------------------------------------------------------------
    # POST /v1/traces — OTLP span ingestion
    # ------------------------------------------------------------------

    def _handle_traces(self) -> None:
        body = self._read_body()
        content_type = self.headers.get("Content-Type", "")

        logger.debug(
            "OTLP POST /v1/traces Content-Type=%r body_len=%d",
            content_type,
            len(body),
        )

        try:
            spans = self._decode_spans(body, content_type)
            self._ingest_spans(spans)
            self._send_json(200, {})
        except Exception:
            logger.exception("OTLP receiver: error processing spans")
            self.send_response(500)
            self.end_headers()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _send_json(self, code: int, obj: dict) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

    def _read_chunked(self) -> bytes:
        """Decode HTTP chunked transfer encoding from the request body."""
        body = b""
        while True:
            size_line = self.rfile.readline().strip()
            if not size_line:
                break
            # Chunk size is hex, optionally followed by extensions after ";"
            chunk_size = int(size_line.split(b";")[0], 16)
            if chunk_size == 0:
                break
            chunk = self.rfile.read(chunk_size)
            body += chunk
            self.rfile.read(2)  # consume trailing \r\n after chunk data
        return body

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
                kv["key"]: _json_any_value(kv["value"])
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
                            kv["key"]: _json_any_value(kv["value"])
                            for kv in span.get("attributes", [])
                        },
                        "resource_attributes": resource_attrs,
                        "status": span.get("status", {}),
                    }
                    spans.append(span_dict)
        return spans

    @staticmethod
    def _enrich_cost(attrs: dict[str, Any]) -> None:
        """Compute cost from token counts + model if not already set."""
        cost_key = semconv_trace.SpanAttributes.COST.COST
        model_key = semconv_trace.SpanAttributes.COST.MODEL
        prompt_key = semconv_trace.SpanAttributes.COST.NUM_PROMPT_TOKENS
        comp_key = semconv_trace.SpanAttributes.COST.NUM_COMPLETION_TOKENS
        currency_key = semconv_trace.SpanAttributes.COST.CURRENCY

        if attrs.get(cost_key):
            return

        model = attrs.get(model_key)
        n_prompt = attrs.get(prompt_key, 0)
        n_completion = attrs.get(comp_key, 0)
        if not model or (not n_prompt and not n_completion):
            return

        try:
            from langchain_community.callbacks.openai_info import (
                get_openai_token_cost_for_model,
            )

            cost = 0.0
            if n_prompt:
                cost += get_openai_token_cost_for_model(
                    model, n_prompt, is_completion=False
                )
            if n_completion:
                cost += get_openai_token_cost_for_model(
                    model, n_completion, is_completion=True
                )
            attrs[cost_key] = cost
            attrs.setdefault(currency_key, "USD")
        except Exception:
            logger.debug(
                "Could not compute cost for model %r; skipping.",
                model,
                exc_info=True,
            )

    def _ingest_spans(self, spans: list[dict]) -> None:
        """Convert decoded span dicts to Event objects and write them."""
        if not spans:
            return

        events: list[event_schema.Event] = []
        for s in spans:
            attrs: dict[str, Any] = dict(s.get("attributes") or {})
            self._enrich_cost(attrs)
            res_attrs: dict[str, Any] = dict(s.get("resource_attributes") or {})

            # The Python pipeline copies app identity attributes from
            # record_attributes into resource_attributes as a workaround.
            for k in [
                semconv_trace.ResourceAttributes.APP_ID,
                semconv_trace.ResourceAttributes.APP_NAME,
                semconv_trace.ResourceAttributes.APP_VERSION,
            ]:
                if k in attrs:
                    res_attrs[k] = attrs[k]

            # Skip spans that have no app_name — they aren't TruLens spans.
            if not res_attrs.get(semconv_trace.ResourceAttributes.APP_NAME):
                continue

            span_id = s.get("span_id", "")
            parent_id = s.get("parent_span_id") or ""
            trace_id = s.get("trace_id", "")

            status_raw = s.get("status") or {}
            status_code = status_raw.get("code", 0)
            # OTLP StatusCode 2 == ERROR
            status_str = (
                "STATUS_CODE_ERROR" if status_code == 2 else "STATUS_CODE_UNSET"
            )

            event = event_schema.Event(
                event_id=span_id,
                record={
                    "name": s.get("name", ""),
                    "kind": 1,  # SPAN_KIND_INTERNAL
                    "parent_span_id": parent_id,
                    "status": status_str,
                },
                record_attributes=attrs,
                record_type=event_schema.EventRecordType.SPAN,
                resource_attributes=res_attrs,
                start_timestamp=_nano_to_datetime(
                    s.get("start_time_unix_nano")
                ),
                timestamp=_nano_to_datetime(s.get("end_time_unix_nano")),
                trace={
                    "span_id": span_id,
                    "trace_id": trace_id,
                    "parent_id": parent_id,
                },
            )
            events.append(event)

        if not events:
            logger.debug(
                "OTLP receiver: all %d span(s) filtered out "
                "(no app_name attribute).",
                len(spans),
            )
            return

        logger.info(
            "OTLP receiver: ingesting %d event(s) into database.",
            len(events),
        )
        self.tru_session.connector.add_events(events)


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


def _json_any_value(val: dict) -> Any:
    """Extract a Python value from an OTLP JSON AnyValue dict."""
    if "stringValue" in val:
        return val["stringValue"]
    if "intValue" in val:
        raw = val["intValue"]
        return int(raw) if isinstance(raw, str) else raw
    if "doubleValue" in val:
        return val["doubleValue"]
    if "boolValue" in val:
        return val["boolValue"]
    if "arrayValue" in val:
        return [_json_any_value(v) for v in val["arrayValue"].get("values", [])]
    if "kvlistValue" in val:
        return {
            kv["key"]: _json_any_value(kv["value"])
            for kv in val["kvlistValue"].get("values", [])
        }
    return None


def _nano_to_datetime(ts: Any) -> datetime:
    """Convert a nanosecond UNIX timestamp to a datetime."""
    if ts is None:
        return datetime.now()
    if isinstance(ts, str):
        ts = int(ts)
    return datetime.fromtimestamp(ts * 1e-9)


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
