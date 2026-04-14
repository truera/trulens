from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

import pandas as pd
from trulens.core.dao.run import RunDaoBase
from trulens.core.enums import Mode
from trulens.core.run import SUPPORTED_ENTRY_TYPES
from trulens.core.run import SupportedEntryType

logger = logging.getLogger(__name__)

RUN_METADATA_NON_MAP_FIELD_MASKS = [
    "labels",
    "llm_judge_name",
]
RUN_ENTITY_EDITABLE_ATTRIBUTES = ["description", "run_status"]


class DefaultRunDao(RunDaoBase):
    """SQLAlchemy-backed RunDao for OSS users.

    Persists run metadata to the ``runs`` table in the local database
    (SQLite, PostgreSQL, etc.) and handles ingestion/metrics locally
    instead of via Snowflake system functions.
    """

    def __init__(self, db):
        self._db = db

    def _run_to_json(self, orm_run) -> str:
        metadata = json.loads(orm_run.run_metadata_json)
        source_info = json.loads(orm_run.source_info_json)
        return json.dumps({
            "object_name": orm_run.object_name,
            "object_type": orm_run.object_type,
            "object_version": orm_run.object_version,
            "run_name": orm_run.run_name,
            "run_status": orm_run.run_status,
            "description": orm_run.description,
            "run_metadata": metadata,
            "source_info": source_info,
        })

    def create_new_run(
        self,
        object_name: str,
        object_type: str,
        run_name: str,
        dataset_name: str,
        source_type: str,
        dataset_spec: dict,
        object_version: Optional[str] = None,
        description: Optional[str] = "",
        label: Optional[str] = "",
        llm_judge_name: Optional[str] = "",
        mode: Optional[Mode] = Mode.APP_INVOCATION,
    ) -> pd.DataFrame:
        orm = self._db.orm
        now = time.time()

        run_metadata = {
            "labels": [label] if label else [],
            "llm_judge_name": llm_judge_name or "",
            "mode": mode.value if mode else Mode.APP_INVOCATION.value,
        }
        source_info = {
            "name": dataset_name,
            "column_spec": dataset_spec,
            "source_type": source_type,
        }

        with self._db.session.begin() as session:
            existing = (
                session.query(orm.Run)
                .filter(orm.Run.run_name == run_name)
                .first()
            )
            if existing is not None:
                raise ValueError(
                    f"Run '{run_name}' already exists."
                )

            new_run = orm.Run(
                run_name=run_name,
                object_name=object_name,
                object_type=object_type,
                object_version=object_version,
                run_status="ACTIVE",
                description=description,
                run_metadata_json=json.dumps(run_metadata),
                source_info_json=json.dumps(source_info),
                created_at=now,
                updated_at=now,
            )
            session.add(new_run)

        return self.get_run(
            run_name=run_name,
            object_name=object_name,
            object_type=object_type,
            object_version=object_version,
        )

    def get_run(
        self,
        run_name: str,
        object_name: str,
        object_type: str,
        object_version: Optional[str] = None,
    ) -> pd.DataFrame:
        orm = self._db.orm
        with self._db.session.begin() as session:
            row = (
                session.query(orm.Run)
                .filter(
                    orm.Run.run_name == run_name,
                    orm.Run.object_name == object_name,
                )
                .first()
            )
            if row is None:
                return pd.DataFrame()
            json_str = self._run_to_json(row)
        return pd.DataFrame({"col": [json_str]})

    def list_all_runs(
        self,
        object_name: str,
        object_type: str,
    ) -> pd.DataFrame:
        orm = self._db.orm
        with self._db.session.begin() as session:
            rows = (
                session.query(orm.Run)
                .filter(orm.Run.object_name == object_name)
                .all()
            )
            if not rows:
                return pd.DataFrame()
            runs_list = []
            for row in rows:
                metadata = json.loads(row.run_metadata_json)
                source_info = json.loads(row.source_info_json)
                runs_list.append({
                    "object_name": row.object_name,
                    "object_type": row.object_type,
                    "object_version": row.object_version,
                    "run_name": row.run_name,
                    "run_status": row.run_status,
                    "description": row.description,
                    "run_metadata": metadata,
                    "source_info": source_info,
                })
        return pd.DataFrame({"col": [json.dumps(runs_list)]})

    def delete_run(
        self,
        run_name: str,
        object_name: str,
        object_type: str,
        object_version: Optional[str] = None,
    ) -> None:
        orm = self._db.orm
        with self._db.session.begin() as session:
            session.query(orm.Run).filter(
                orm.Run.run_name == run_name,
                orm.Run.object_name == object_name,
            ).delete()

    def upsert_run_metadata_fields(
        self,
        run_name: str,
        object_name: str,
        object_type: str,
        object_version: Optional[str] = None,
        entry_id: Optional[str] = None,
        entry_type: Optional[str] = None,
        **field_updates: Any,
    ) -> None:
        if entry_type and entry_type not in SUPPORTED_ENTRY_TYPES:
            raise ValueError(f"Unsupported entry type: {entry_type}")

        keys = list(field_updates.keys())
        for key in keys:
            if entry_type:
                field_updates[f"{entry_type}.{entry_id}.{key}"] = (
                    field_updates.pop(key)
                )

        orm = self._db.orm
        with self._db.session.begin() as session:
            row = (
                session.query(orm.Run)
                .filter(
                    orm.Run.run_name == run_name,
                    orm.Run.object_name == object_name,
                )
                .first()
            )
            if row is None:
                raise ValueError(f"Run '{run_name}' does not exist.")

            existing_run_metadata = json.loads(row.run_metadata_json)

            update_description = "description" in field_updates
            description_val = field_updates.pop("description", None)

            update_run_status = "run_status" in field_updates
            run_status_val = field_updates.pop("run_status", None)

            updated_metadata = self._apply_field_masks(
                existing_run_metadata, field_updates
            )

            row.run_metadata_json = json.dumps(updated_metadata)
            row.updated_at = time.time()

            if update_description:
                row.description = description_val
            if update_run_status:
                row.run_status = run_status_val

    def _apply_field_masks(
        self, existing: dict, updates: dict
    ) -> dict:
        for key, value in updates.items():
            parts = key.split(".")

            if parts[0] in SUPPORTED_ENTRY_TYPES:
                group = parts[0]
                if len(parts) < 3:
                    raise ValueError(
                        f"Expected format '{group}.<id>.<field>', got: {key}"
                    )
                entry_id = parts[1]
                nested_keys = parts[2:]

                if group not in existing or existing[group] is None:
                    existing[group] = {}
                if entry_id not in existing[group]:
                    existing[group][entry_id] = {"id": entry_id}

                d = existing[group][entry_id]
                for k in nested_keys[:-1]:
                    if k not in d or not isinstance(d[k], dict):
                        d[k] = {}
                    d = d[k]
                d[nested_keys[-1]] = value

            elif key in RUN_METADATA_NON_MAP_FIELD_MASKS:
                existing[key] = value
            elif key not in RUN_ENTITY_EDITABLE_ATTRIBUTES:
                raise ValueError(f"Unsupported key: {key}")

        return existing

    def start_ingestion_query(
        self,
        object_name: str,
        object_version: Optional[str],
        object_type: str,
        run_name: str,
        input_records_count: int,
    ) -> None:
        from trulens.core.utils.json import obj_id_of_obj

        invocation_id = obj_id_of_obj(
            obj={
                "run_name": run_name,
                "input_records_count": input_records_count,
            },
            prefix="invocation_metadata",
        )
        now_ms = int(round(time.time() * 1000))

        self.upsert_run_metadata_fields(
            run_name=run_name,
            object_name=object_name,
            object_type=object_type,
            object_version=object_version,
            entry_id=invocation_id,
            entry_type=SupportedEntryType.INVOCATIONS.value,
            id=invocation_id,
            input_records_count=input_records_count,
            start_time_ms=now_ms,
            end_time_ms=now_ms,
            **{
                "completion_status.status": "COMPLETED",
                "completion_status.record_count": input_records_count,
            },
        )
        logger.info(
            f"Marked ingestion complete for run '{run_name}' with {input_records_count} records."
        )

    def call_compute_metrics_query(
        self,
        metrics: List[str],
        object_name: str,
        object_version: Optional[str],
        object_type: str,
        run_name: str,
    ) -> None:
        if metrics:
            logger.warning(
                "Server-side (string) metrics are not supported in OSS mode. "
                "Skipping: %s. Use Metric objects for client-side evaluation.",
                metrics,
            )
