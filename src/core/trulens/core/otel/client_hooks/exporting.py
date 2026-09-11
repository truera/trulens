"""Destination configuration and span export for coding-agent hooks."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from trulens.core import session as core_session
from trulens.core.experimental import Feature

logger = logging.getLogger(__name__)


def _local_session() -> core_session.TruSession:
    database_url = os.environ.get("TRULENS_DATABASE_URL")
    if database_url:
        return core_session.TruSession(database_url=database_url)
    database_path = Path(
        os.environ.get(
            "TRULENS_DATABASE_PATH",
            str(Path.home() / ".trulens" / "client-hooks.sqlite"),
        )
    ).expanduser()
    database_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    return core_session.TruSession(database_url=f"sqlite:///{database_path}")


def _snowflake_session() -> core_session.TruSession:
    try:
        from snowflake.snowpark import Session
        from trulens.connectors.snowflake import SnowflakeConnector
    except ImportError as exc:
        raise ImportError(
            "Snowflake hook export requires trulens-connectors-snowflake."
        ) from exc
    connection_name = os.environ.get("TRULENS_SNOWFLAKE_CONNECTION")
    if not connection_name:
        raise ValueError(
            "Set TRULENS_SNOWFLAKE_CONNECTION for Snowflake export."
        )
    builder = Session.builder.config("connection_name", connection_name)
    database = os.environ.get("TRULENS_SNOWFLAKE_DATABASE")
    schema = os.environ.get("TRULENS_SNOWFLAKE_SCHEMA")
    if database:
        builder = builder.config("database", database)
    if schema:
        builder = builder.config("schema", schema)
    snowpark_session = builder.create()
    connector = SnowflakeConnector(snowpark_session=snowpark_session)
    return core_session.TruSession(connector=connector)


def _ai_gateway_session() -> core_session.TruSession:
    gateway_url = os.environ.get("TRULENS_AI_GATEWAY_URL")
    if not gateway_url:
        raise ValueError("Set TRULENS_AI_GATEWAY_URL for AI Gateway export.")
    token_file = os.environ.get("TRULENS_AI_GATEWAY_PAT_FILE")
    if token_file:
        with open(Path(token_file).expanduser()) as handle:
            token = handle.read().strip()
    else:
        token = os.environ.get("TRULENS_AI_GATEWAY_TOKEN")
    if not token:
        raise ValueError(
            "Set TRULENS_AI_GATEWAY_PAT_FILE or TRULENS_AI_GATEWAY_TOKEN for "
            "AI Gateway export."
        )
    # AI Gateways serve OTLP over HTTP/protobuf (not gRPC), so build the
    # exporter directly rather than via otel_exporter="otlp".
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
        OTLPSpanExporter,
    )

    exporter = OTLPSpanExporter(
        endpoint=f"{gateway_url}/telemetry/v1/traces",
        headers={"Authorization": f"Bearer {token}"},
    )
    return core_session.TruSession(
        _experimental_otel_exporter=exporter,
        experimental_feature_flags={Feature.OTEL_TRACING: True},
    )


def create_session() -> core_session.TruSession:
    """Create the configured destination TruLens session.

    Supported ``TRULENS_DESTINATION`` values: local (default), database,
    snowflake, otlp, and ai_gateway.
    """

    destination = os.environ.get("TRULENS_DESTINATION", "local").lower()
    if destination in {"local", "database"}:
        return _local_session()
    if destination == "snowflake":
        return _snowflake_session()
    if destination == "otlp":
        endpoint = os.environ.get("TRULENS_OTLP_ENDPOINT")
        protocol = os.environ.get("TRULENS_OTLP_PROTOCOL") or os.environ.get(
            "OTEL_EXPORTER_OTLP_PROTOCOL"
        )
        return core_session.TruSession(
            otel_exporter="otlp",
            otlp_endpoint=endpoint,
            otlp_protocol=protocol,
        )
    if destination == "ai_gateway":
        return _ai_gateway_session()
    raise ValueError(
        "TRULENS_DESTINATION must be local, database, snowflake, otlp, or "
        "ai_gateway."
    )


def export_spans(
    spans: Sequence[ReadableSpan],
    *,
    session: Optional[core_session.TruSession] = None,
) -> bool:
    """Synchronously export one complete trace batch."""

    if not spans:
        return True
    active_session = session or create_session()
    exporter = active_session.experimental_otel_exporter
    if exporter is None:
        logger.warning(
            "Hook export destination produced no OTel exporter; spans were "
            "not exported."
        )
        return False
    result = exporter.export(spans)
    flushed = active_session.force_flush()
    return result == SpanExportResult.SUCCESS and flushed is not False
