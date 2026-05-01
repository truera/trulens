"""Seed script: populates a SQLite DB with 10k OTEL records for dashboard perf testing.

Usage:
    TRULENS_OTEL_TRACING=1 python scripts/seed_dashboard_test.py [--db test_perf.sqlite] [--records 10000]
"""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import os
import random
import time
import uuid

os.environ["TRULENS_OTEL_TRACING"] = "1"

from trulens.core.database import sqlalchemy as db_sqlalchemy
from trulens.core.schema import app as app_schema
from trulens.core.schema import event as event_schema
from trulens.otel.semconv.trace import ResourceAttributes
from trulens.otel.semconv.trace import SpanAttributes

APP_CONFIGS = [
    ("rag_chatbot", ["v1.0", "v1.1", "v2.0"]),
    ("support_agent", ["v0.9", "v1.0", "v1.1"]),
    ("doc_search", ["v1.0", "v2.0", "v2.1"]),
    ("code_assistant", ["v0.1", "v0.2", "v1.0"]),
    ("summarizer", ["v1.0", "v1.1", "v2.0"]),
]

FEEDBACK_NAMES = [
    "Answer Relevance",
    "Context Relevance",
    "Groundedness",
    "Harmfulness",
]

PINNED_VERSIONS = {
    ("rag_chatbot", "v2.0"),
    ("support_agent", "v1.1"),
}


def make_app_id(app_name: str, app_version: str) -> str:
    return app_schema.AppDefinition._compute_app_id(app_name, app_version)


def make_record_root_event(
    record_id: str,
    trace_id: str,
    app_name: str,
    app_version: str,
    app_id: str,
    base_time: datetime,
    latency_s: float,
) -> event_schema.Event:
    span_id = uuid.uuid4().hex[:16]
    start = base_time
    end = start + timedelta(seconds=latency_s)
    cost = round(random.uniform(0.001, 0.05), 6)
    tokens = random.randint(100, 2000)

    return event_schema.Event(
        event_id=span_id,
        record={
            "name": f"{app_name}.main",
            "kind": "SPAN_KIND_TRULENS",
            "status": "STATUS_CODE_UNSET",
        },
        record_attributes={
            SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.RECORD_ROOT.value,
            SpanAttributes.RECORD_ID: record_id,
            SpanAttributes.RECORD_ROOT.INPUT: f"User query #{random.randint(1, 10000)}",
            SpanAttributes.RECORD_ROOT.OUTPUT: "Generated response for query.",
            SpanAttributes.COST.COST: cost,
            SpanAttributes.COST.CURRENCY: "USD",
            SpanAttributes.COST.NUM_TOKENS: tokens,
        },
        record_type=event_schema.EventRecordType.SPAN,
        resource_attributes={
            ResourceAttributes.APP_NAME: app_name,
            ResourceAttributes.APP_VERSION: app_version,
            ResourceAttributes.APP_ID: app_id,
        },
        start_timestamp=start,
        timestamp=end,
        trace={
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_id": "",
        },
    )


def make_eval_root_event(
    record_id: str,
    trace_id: str,
    app_name: str,
    app_version: str,
    app_id: str,
    metric_name: str,
    score: float,
    base_time: datetime,
) -> event_schema.Event:
    span_id = uuid.uuid4().hex[:16]
    eval_start = base_time + timedelta(seconds=random.uniform(0.1, 0.5))
    eval_end = eval_start + timedelta(seconds=random.uniform(0.5, 2.0))

    return event_schema.Event(
        event_id=span_id,
        record={
            "name": f"eval.{metric_name}",
            "kind": "SPAN_KIND_TRULENS",
            "status": "STATUS_CODE_UNSET",
        },
        record_attributes={
            SpanAttributes.SPAN_TYPE: SpanAttributes.SpanType.EVAL_ROOT.value,
            SpanAttributes.RECORD_ID: record_id,
            SpanAttributes.EVAL_ROOT.METRIC_NAME: metric_name,
            SpanAttributes.EVAL_ROOT.SCORE: score,
            SpanAttributes.COST.COST: round(random.uniform(0.001, 0.01), 6),
            SpanAttributes.COST.CURRENCY: "USD",
        },
        record_type=event_schema.EventRecordType.SPAN,
        resource_attributes={
            ResourceAttributes.APP_NAME: app_name,
            ResourceAttributes.APP_VERSION: app_version,
            ResourceAttributes.APP_ID: app_id,
        },
        start_timestamp=eval_start,
        timestamp=eval_end,
        trace={
            "trace_id": trace_id,
            "span_id": span_id,
            "parent_id": "",
        },
    )


def seed(db_path: str, num_records: int, batch_size: int = 500):
    db_url = f"sqlite:///{db_path}"
    print(f"Creating DB at {db_url} with {num_records} records...")

    db = db_sqlalchemy.SQLAlchemyDB.from_db_url(db_url)
    db.migrate_database()

    app_versions_flat = [
        (name, ver) for name, versions in APP_CONFIGS for ver in versions
    ]
    records_per_version = num_records // len(app_versions_flat)
    remainder = num_records % len(app_versions_flat)

    total_events = 0
    base_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    for idx, (app_name, app_version) in enumerate(app_versions_flat):
        count = records_per_version + (1 if idx < remainder else 0)
        app_id = make_app_id(app_name, app_version)

        print(
            f"  Seeding {count} records for {app_name}/{app_version} (app_id={app_id[:8]}...)"
        )

        batch = []
        for i in range(count):
            record_id = uuid.uuid4().hex
            trace_id = uuid.uuid4().hex
            record_time = base_time + timedelta(
                seconds=random.uniform(0, 86400 * 90)
            )
            latency = round(random.uniform(0.5, 5.0), 3)

            batch.append(
                make_record_root_event(
                    record_id=record_id,
                    trace_id=trace_id,
                    app_name=app_name,
                    app_version=app_version,
                    app_id=app_id,
                    base_time=record_time,
                    latency_s=latency,
                )
            )

            for fb_name in FEEDBACK_NAMES:
                score = round(random.uniform(0.0, 1.0), 4)
                batch.append(
                    make_eval_root_event(
                        record_id=record_id,
                        trace_id=trace_id,
                        app_name=app_name,
                        app_version=app_version,
                        app_id=app_id,
                        metric_name=fb_name,
                        score=score,
                        base_time=record_time,
                    )
                )

            if len(batch) >= batch_size:
                db.insert_events(batch)
                total_events += len(batch)
                batch = []

        if batch:
            db.insert_events(batch)
            total_events += len(batch)

    # Insert app definitions with pinned metadata for select versions
    _seed_app_definitions(db, app_versions_flat)

    print(
        f"\nDone! Inserted {total_events} events ({num_records} records x ~5 events each)."
    )
    print(f"DB file: {db_path}")
    print(f"Pinned versions: {PINNED_VERSIONS}")


def _seed_app_definitions(db, app_versions_flat):
    for app_name, app_version in app_versions_flat:
        app_id = make_app_id(app_name, app_version)
        metadata = {}
        if (app_name, app_version) in PINNED_VERSIONS:
            metadata["trulens"] = {"dashboard": {"pinned": True}}

        app_def = app_schema.AppDefinition(
            app_id=app_id,
            app_name=app_name,
            app_version=app_version,
            root_class={
                "name": "TruCustomApp",
                "module": {"module_name": "trulens.apps.custom"},
            },
            app={},
            metadata=metadata,
        )
        db.insert_app(app_def)


def main():
    parser = argparse.ArgumentParser(description="Seed TruLens test DB")
    parser.add_argument(
        "--db", default="test_perf.sqlite", help="SQLite DB path"
    )
    parser.add_argument(
        "--records", type=int, default=10000, help="Number of records"
    )
    args = parser.parse_args()

    if os.path.exists(args.db):
        os.remove(args.db)
        print(f"Removed existing {args.db}")

    start = time.perf_counter()
    seed(args.db, args.records)
    elapsed = time.perf_counter() - start
    print(f"Seeding took {elapsed:.1f}s")


if __name__ == "__main__":
    main()
