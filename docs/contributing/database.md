# 🗄️ Database Schema

TruLens stores trace and evaluation data in a database. This document covers the
current OTEL-based schema, the legacy schema, and migration procedures.

## Current Schema (OTEL)

As of TruLens 1.x with OpenTelemetry instrumentation, all trace data is stored
in a single **events table**. This table follows the
[Snowflake Event Table](https://docs.snowflake.com/en/developer-guide/logging-tracing/event-table-columns)
specification for OTEL spans.

### Events Table

| Column | Type | Description |
| ------ | ---- | ----------- |
| `event_id` | VARCHAR(256) | Primary key |
| `record` | JSON | Span metadata: name, kind, parent_span_id, status |
| `record_attributes` | JSON | User and TruLens semantic convention attributes |
| `record_type` | ENUM | Always "SPAN" for TruLens |
| `resource_attributes` | JSON | Reserved for resource-level attributes |
| `start_timestamp` | TIMESTAMP | When the span started |
| `timestamp` | TIMESTAMP | When the span concluded |
| `trace` | JSON | Span context: trace_id, parent_id, span_id |

The events table is append-only. All trace hierarchy (apps, records, spans) and
evaluation results are encoded in the span attributes using TruLens semantic
conventions (see `trulens.otel.semconv`).

### Ground Truth Tables

In addition to the events table, TruLens maintains two tables for evaluation
ground truth data. These are written to via
`TruSession.add_ground_truth_to_dataset()`:

| Table | Description |
| ----- | ----------- |
| `trulens_dataset` | Dataset definitions (dataset_id, dataset_json) |
| `trulens_ground_truth` | Ground truth entries linked to datasets |
| `trulens_dataset_version` | Immutable, content-addressed snapshots of a dataset |
| `trulens_dataset_version_item` | The examples belonging to one snapshot |

```text
dataset (1) ──────< ground_truth (many)
         (1) ──────< dataset_version (many) ──────< dataset_version_item (many)
```

These tables are separate from trace storage and are used to store expected
outputs for evaluation comparisons.

`trulens_dataset` and `trulens_ground_truth` are mutable: writing to a dataset
changes what it contains. `trulens_dataset_version` rows are immutable — the
version id is a hash of the version's ordered contents, so any change to those
contents is a new row rather than an update to an existing one, and
`(dataset_id, version_index)` is unique so version history cannot fork. Datasets
written before versioning existed are exposed as version zero by a
compatibility loader that reads, but never rewrites, their ground truth
payloads. See
[Dataset Versions](../component_guides/evaluation/dataset_versions.md).

### Benefits of OTEL Schema

- **Single table for traces**: Simpler schema, no foreign key relationships
- **Standardized**: Compatible with OTEL tooling and exporters
- **Flexible**: New attributes can be added without schema migrations

---

## Legacy Schema

Prior to the OTEL migration, TruLens stored traces and evaluations in a
relational schema. These tables are **no longer written to** but may still exist
in older databases.

### Legacy Tables

| Table | Description |
| ----- | ----------- |
| `trulens_apps` | App definitions (app_id, app_name, app_version, app_json) |
| `trulens_records` | Invocation records with input/output and timing |
| `trulens_feedbacks` | Feedback evaluation results |
| `trulens_feedback_defs` | Feedback function definitions |

The ORM definitions for all tables are in `src/core/trulens/core/database/orm.py`.

### Legacy Table Relationships

```text
apps (1) ──────< records (many)
                    │
                    └──< feedbacks (many) >── feedback_defs (1)
```

---

## Legacy Migrations

These notes apply to schema changes for the **legacy tables only**. The OTEL
events table does not use Alembic migrations.

### Creating a New Schema Revision

If you need to modify the legacy ORM:

1. Make changes to SQLAlchemy ORM models in `src/core/trulens/core/database/orm.py`

2. Generate an Alembic migration script:

    ```bash
    cd src/core/trulens/core/database/migrations
    SQLALCHEMY_URL="sqlite:///../../../../../../default.sqlite" \
        alembic revision --autogenerate -m "<description>" --rev-id "<version>"
    ```

3. Review the generated script in `migrations/versions/`

4. Create a fresh database with the new schema:

    ```bash
    rm default.sqlite
    python -c "from trulens.core.session import TruSession; TruSession()"
    ```

5. Register the version in `migrations/data.py`:
   - Add to `sql_alchemy_migration_versions`
   - Update `sqlalchemy_upgrade_paths` if backfill is needed

### Migration Versions

Existing migrations in `src/core/trulens/core/database/migrations/versions/`:

| Version | Description |
| ------- | ----------- |
| 1 | Initial schema |
| 2 | Add run_location to feedback_defs |
| 3 | Add ground_truth and dataset tables |
| 4 | Set feedback_definition_id not null |
| 5 | Add app_name and app_version fields |
| 6 | Populate app_name and version data |
| 7 | Make app_name and version not null |
| 8 | Update records app_id |
| 9 | Update app_json |
| 10 | Create events table (OTEL) |
| 11 | Add events table indexes |
| 12 | Create runs table |
| 13 | Add dataset_version and dataset_version_item tables |

---

## Connector Implementation

Database operations are abstracted through connectors in
`src/core/trulens/core/database/connector/`. When implementing a new connector
(e.g., `trulens-connectors-*`), you need to handle:

- **OTEL events**: Writing spans to the events table
- **Legacy support** (optional): Reading from legacy tables for backwards compatibility

See the default SQLAlchemy connector in `connector/default.py` for reference.
