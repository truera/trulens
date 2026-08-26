"""Destination configuration and span export for coding-agent hooks."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence

from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExportResult
from trulens.core import session as core_session


def _local_session() -> core_session.TruSession:
    database_url = os.environ.get("TRULENS_HOOKS_DATABASE_URL")
    if database_url:
        return core_session.TruSession(database_url=database_url)
    database_path = Path(
        os.environ.get(
            "TRULENS_HOOKS_DATABASE_PATH",
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
    connection_name = os.environ.get("TRULENS_HOOKS_SNOWFLAKE_CONNECTION")
    if not connection_name:
        raise ValueError(
            "Set TRULENS_HOOKS_SNOWFLAKE_CONNECTION for Snowflake export."
        )
    snowpark_session = Session.builder.config(
        "connection_name", connection_name
    ).create()
    connector = SnowflakeConnector(snowpark_session=snowpark_session)
    return core_session.TruSession(connector=connector)


def create_session() -> core_session.TruSession:
    """Create the configured database, Snowflake, or OTLP TruLens session."""

    destination = os.environ.get("TRULENS_HOOKS_DESTINATION", "local").lower()
    if destination in {"local", "database"}:
        return _local_session()
    if destination == "snowflake":
        return _snowflake_session()
    if destination == "otlp":
        endpoint = os.environ.get("TRULENS_HOOKS_OTLP_ENDPOINT")
        return core_session.TruSession(
            otel_exporter="otlp", otlp_endpoint=endpoint
        )
    raise ValueError(
        "TRULENS_HOOKS_DESTINATION must be local, database, snowflake, or otlp."
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
        return False
    result = exporter.export(spans)
    flushed = active_session.force_flush()
    return result == SpanExportResult.SUCCESS and flushed is not False
