# ruff: noqa: E402

"""OTEL Compatible OTLP Collector.

!!! Warning:
    WORK ONGOING; please do not use.

See [specification](https://opentelemetry.io/docs/specs/otlp/). See also the
other side of this connection in
[OTLPSpanExporter][opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter].

Not all of the specification is currently supported. Please update this
docstring as more of the spec is handled.

- Ignores most of http spec including path.

- Only proto payloads are supported.

- No compression is supported.

- Only spans are supported.


"""

from __future__ import annotations

import json
import logging
from pprint import pprint
import threading

import pydantic
from trulens.experimental.otel_tracing import _feature

_feature._FeatureSetup.assert_optionals_installed()  # checks to make sure otel is installed

import uvicorn

logger = logging.getLogger(__name__)


class CollectorRequest(pydantic.BaseModel):
    payload: str = "notset"


class CollectorResponse(pydantic.BaseModel):
    status: int = 404


class Collector:
    """OTLP Traces Collector."""

    @staticmethod
    async def _uvicorn_handle(scope, receive, send):
        """Main uvicorn handler."""

        print("scope:")
        pprint(scope)
        if scope.get("type") != "http":
            return

        request = await receive()
        print("request:")
        pprint(request)

        if request.get("type") != "http.request":
            return

        headers = dict(scope.get("headers", {}))
        method = scope.get("method", None)
        if method != "POST":
            return

        body = request.get("body", None)
        if body is None:
            return
        content_type = headers.get(b"content-type", None)

        if content_type == b"application/json":
            body = json.loads(body.decode("utf-8"))
        elif content_type == b"application/x-protobuf":
            from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
                ExportTraceServiceRequest as PB2ExportTraceServiceRequest,
            )

            body = PB2ExportTraceServiceRequest().FromString(body)

            for resource_and_span in body.resource_spans:
                resource = resource_and_span.resource
                print("resource:")
                pprint(resource)
                spans = resource_and_span.scope_spans
                for span in spans:
                    print("span:")
                    pprint(span)

        else:
            return

        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                [b"content-type", b"application/json"],
            ],
        })

        await send({
            "type": "http.response.body",
            "body": CollectorResponse(status=200)
            .model_dump_json()
            .encode("utf-8"),
        })

    def __init__(self):
        self.app = self._uvicorn_handle
        self.server_thread = threading.Thread(target=self._run)

    def _run(self):
        self.config = uvicorn.Config(app=self.app, port=5000)
        self.server = uvicorn.Server(self.config)
        import asyncio

        self.loop = asyncio.new_event_loop()
        self.loop.run_until_complete(self.server.serve())

    def start(self):
        self.server_thread.start()

    def stop(self):
        self.loop.close()
