from datetime import datetime
import json
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from trulens.connectors.snowflake.dao.external_agent import ExternalAgentDao
from trulens.connectors.snowflake.dao.sql_utils import (
    clean_up_snowflake_identifier,
)
from trulens.core.database import base as core_db
from trulens.core.schema import app as app_schema
from trulens.core.schema import types as types_schema
from trulens.core.schema.event import Event
from trulens.core.utils import serial as serial_utils
from trulens.otel.semconv.trace import SpanAttributes

from snowflake.snowpark import Session


def _parse_variant_value(value):
    """Best-effort parse a Snowflake VARIANT/OBJECT value into a Python
    object.

    Snowpark's ``to_pandas`` returns VARIANT/OBJECT columns as JSON strings.
    Downstream callers in TruLens expect dict/list values, so this helper
    parses strings via ``json.loads`` and returns the original value
    unchanged when parsing is not applicable.
    """
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        if not value:
            return value
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _attach_parent_id_to_trace(row):
    """Ensure ``trace["parent_id"]`` is set on each event row.

    ``SNOWFLAKE.LOCAL.GET_AI_OBSERVABILITY_EVENTS`` returns ``TRACE`` objects
    that contain only ``span_id`` and ``trace_id``; the parent linkage lives
    on ``RECORD.parent_span_id`` for non-root spans. Callers downstream
    (e.g. ``Evaluator._events_under_record_root``) reconstruct the span
    tree from ``trace["parent_id"]``, so populate it here as the canonical
    location.
    """
    trace = row.get("trace")
    if not isinstance(trace, dict):
        return trace
    if "parent_id" in trace and trace["parent_id"] is not None:
        return trace
    record = row.get("record")
    parent_span_id = None
    if isinstance(record, dict):
        parent_span_id = record.get("parent_span_id")
    trace["parent_id"] = parent_span_id
    return trace


class SnowflakeEventTableDB(core_db.DB):
    """Connector to the account level event table in Snowflake."""

    _snowpark_session: Session
    _external_agent_dao: ExternalAgentDao

    table_prefix: str = core_db.DEFAULT_DATABASE_PREFIX

    def __init__(self, snowpark_session: Session):
        super().__init__(table_prefix="")
        self._snowpark_session = snowpark_session
        self._external_agent_dao = ExternalAgentDao(snowpark_session)

    def _is_cortex_agent(self, app_name: types_schema.AppName) -> bool:
        q = "SHOW AGENTS IN ACCOUNT"
        df = self._snowpark_session.sql(q).to_pandas()
        if app_name in df['"name"'].values:
            # Technically, because we can have multiple cortex agents with the
            # same name but in different schemas, this is a bit problematic, but
            # there's not much we can do about it unfortunately.
            return True
        return False

    def get_records_and_feedback(
        self,
        app_ids: Optional[List[types_schema.AppID]] = None,
        app_name: Optional[types_schema.AppName] = None,
        app_version: Optional[types_schema.AppVersion] = None,
        app_versions: Optional[List[types_schema.AppVersion]] = None,
        run_name: Optional[types_schema.RunName] = None,
        record_ids: Optional[List[types_schema.RecordID]] = None,
        matched_record_ids: Optional[List[types_schema.RecordID]] = None,
        include_conversation_context: bool = False,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        """See [DB.get_records_and_feedback][trulens.core.database.base.DB.get_records_and_feedback]."""
        if include_conversation_context:
            raise NotImplementedError(
                "Conversation context expansion is not supported for Snowflake event tables."
            )
        if matched_record_ids is not None:
            record_ids = matched_record_ids
        page = slice(
            offset or 0,
            None if limit is None else (offset or 0) + limit,
        )
        if record_ids is not None:
            record_ids = record_ids[page]
        df = self._get_events(
            app_ids=app_ids,
            app_name=app_name,
            app_version=app_version,
            app_versions=app_versions,
            run_name=run_name,
            record_ids=record_ids,
            start_time=start_time,
            end_time=end_time,
        )
        events = []
        for _, row in df.iterrows():
            # `_get_events` already parses the VARIANT columns, so these may be
            # dicts rather than JSON strings.
            trace = _parse_variant_value(row["trace"])
            if "parent_id" not in trace:
                trace["parent_id"] = ""
            if "trace_id" not in trace:
                trace["trace_id"] = ""
            events.append(
                Event(
                    event_id=trace["span_id"],
                    record=_parse_variant_value(row["record"]),
                    record_attributes=_parse_variant_value(
                        row["record_attributes"]
                    ),
                    record_type=row["record_type"],
                    resource_attributes=_parse_variant_value(
                        row["resource_attributes"]
                    ),
                    start_timestamp=row["start_timestamp"],
                    timestamp=row["timestamp"],
                    trace=trace,
                )
            )
        if record_ids is None and (offset is not None or limit is not None):
            events = self._get_events_of_record_page(events, page)
        return self._get_records_and_feedback_otel_from_events(
            events=events, app_ids=app_ids, app_name=app_name
        )

    def _get_events_of_record_page(
        self, events: List[Event], page: slice
    ) -> List[Event]:
        """Keep the events of one page of records.

        Records are the ones with a `RECORD_ROOT` span, newest first with ties
        broken by record id, as in the SQLAlchemy event table.
        """
        record_ids = [
            self._get_event_record_attributes_otel(event).get(
                SpanAttributes.RECORD_ID
            )
            for event in events
        ]
        root_starts = {}
        for record_id, event in zip(record_ids, events):
            span_type = self._get_event_record_attributes_otel(event).get(
                SpanAttributes.SPAN_TYPE
            )
            if (
                record_id
                and span_type == SpanAttributes.SpanType.RECORD_ROOT.value
            ):
                root_starts[record_id] = min(
                    event.start_timestamp,
                    root_starts.get(record_id, event.start_timestamp),
                )
        newest_first = sorted(
            sorted(root_starts), key=root_starts.get, reverse=True
        )
        selected = set(newest_first[page])
        return [
            event
            for record_id, event in zip(record_ids, events)
            if record_id in selected
        ]

    def get_events(
        self,
        app_name: Optional[types_schema.AppName] = None,
        app_version: Optional[types_schema.AppVersion] = None,
        run_name: Optional[types_schema.RunName] = None,
        record_ids: Optional[List[types_schema.RecordID]] = None,
        start_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get events from the database.

        Args:
            app_name: The app name to filter events by.
            app_version: The app version to filter events by.
            run_name: The run name to filter events by.
            record_ids: The record ids to filter events by.
            start_time: The minimum time to consider events from.

        Returns:
            A pandas DataFrame of all relevant events.
        """
        return self._get_events(
            app_name=app_name,
            app_version=app_version,
            run_name=run_name,
            record_ids=record_ids,
            start_time=start_time,
        )

    def _get_events(
        self,
        app_ids: Optional[List[types_schema.AppID]] = None,
        app_name: Optional[types_schema.AppName] = None,
        app_version: Optional[types_schema.AppVersion] = None,
        app_versions: Optional[List[types_schema.AppVersion]] = None,
        run_name: Optional[types_schema.RunName] = None,
        record_ids: Optional[List[types_schema.RecordID]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        if record_ids == []:
            return pd.DataFrame()
        if app_name is None:
            raise ValueError("app_name is required!")
        if app_ids is not None:
            raise ValueError("app_ids is not supported!")
        database = clean_up_snowflake_identifier(
            self._snowpark_session.get_current_database()
        )
        schema = clean_up_snowflake_identifier(
            self._snowpark_session.get_current_schema()
        )
        q = """
        SELECT
            *
        FROM TABLE(SNOWFLAKE.LOCAL.GET_AI_OBSERVABILITY_EVENTS(
            ?, ?, ?, ?
        ))
        """
        where_clauses = []
        extra_params = []
        if app_version:
            app_version_str = f"'{app_version}'"
            where_clauses.append(
                f"IFNULL(RECORD_ATTRIBUTES:\"snow.ai.observability.object.version.name\", 'base') = {app_version_str}"
            )
        if app_versions:
            app_versions_str = ", ".join([f"'{curr}'" for curr in app_versions])
            where_clauses.append(
                f"IFNULL(RECORD_ATTRIBUTES:\"snow.ai.observability.object.version.name\", 'base') IN ({app_versions_str})"
            )
        if run_name:
            run_name_str = f"'{run_name}'"
            where_clauses.append(
                f'RECORD_ATTRIBUTES:"ai.observability.run.name" = {run_name_str} OR RECORD_ATTRIBUTES:"snow.ai.observability.run.name" = {run_name_str}'
            )
        if record_ids:
            record_ids_str = ", ".join([f"'{curr}'" for curr in record_ids])
            where_clauses.append(
                f'RECORD_ATTRIBUTES:"{SpanAttributes.RECORD_ID}" IN ({record_ids_str})'
            )
        if start_time:
            where_clauses.append("START_TIMESTAMP >= ?")
            extra_params.append(start_time)
        if end_time:
            where_clauses.append("START_TIMESTAMP < ?")
            extra_params.append(end_time)
        if where_clauses:
            q += " WHERE " + " AND ".join(where_clauses)
        df = None
        for app_type in ["EXTERNAL AGENT", "CORTEX AGENT"]:
            params = [database, schema, app_name, app_type] + extra_params
            try:
                df = self._snowpark_session.sql(q, params=params).to_pandas()
                if not df.empty:
                    break
            except Exception:
                # TODO: There doesn't seem to be a way to tell if an app is an
                # external agent or a cortex agent yet so we're trying both
                # in a hacky way right now for the time being.
                pass
        df.columns = df.columns.str.lower()
        # Snowpark's ``to_pandas`` serializes VARIANT/OBJECT columns as JSON
        # strings. Downstream consumers (e.g. the evaluator thread, client-side
        # metric computation) expect dicts on these columns, so normalize at
        # the data-access boundary.
        _VARIANT_COLUMNS = (
            "record",
            "record_attributes",
            "resource_attributes",
            "trace",
            "value",
            "scope",
        )
        for _col in _VARIANT_COLUMNS:
            if _col in df.columns:
                df[_col] = df[_col].apply(_parse_variant_value)
        # ``GET_AI_OBSERVABILITY_EVENTS`` does not currently expose
        # ``parent_id`` on the ``TRACE`` object. The parent linkage is on
        # ``RECORD.parent_span_id`` for non-root spans. Normalize so every
        # consumer can rely on ``trace["parent_id"]``.
        if "trace" in df.columns:
            df["trace"] = df.apply(_attach_parent_id_to_trace, axis=1)
        return df

    def check_db_revision(*args, **kwargs):
        # The account level event table does not have versioning.
        pass

    def batch_insert_feedback(*args, **kwargs):
        raise NotImplementedError()

    def batch_insert_ground_truth(*args, **kwargs):
        raise NotImplementedError()

    def batch_insert_record(*args, **kwargs):
        raise NotImplementedError()

    def delete_app(*args, **kwargs):
        raise NotImplementedError()

    def get_app(*args, **kwargs):
        raise NotImplementedError()

    def get_apps(
        self, app_name: Optional[types_schema.AppName] = None
    ) -> Iterable[serial_utils.JSONized[app_schema.AppDefinition]]:
        """See [DB.get_apps][trulens.core.database.base.DB.get_apps]."""
        if app_name is None:
            agents = self._external_agent_dao._list_agents()
            if agents.empty or "name" not in agents.columns:
                return

            app_names = agents["name"].values
        else:
            app_names = [app_name]
        for app_name in app_names:
            if self._is_cortex_agent(app_name):
                agent_versions = ["base"]
            else:
                versions = self._external_agent_dao.list_agent_versions(
                    app_name
                )
                if versions.empty or "name" not in versions.columns:
                    continue

                agent_versions = versions["name"].values
            for app_version in agent_versions:
                app_id = app_schema.AppDefinition._compute_app_id(
                    app_name, app_version
                )
                app_defn = app_schema.AppDefinition(
                    app_id=app_id,
                    app_name=app_name,
                    app_version=app_version,
                    root_class={
                        "name": "Placeholder",
                        "module": {"module_name": "placeholder"},
                    },
                    app={},
                )
                yield serial_utils.JSONized(app_defn)

    def get_datasets(*args, **kwargs):
        raise NotImplementedError()

    def get_db_revision(*args, **kwargs):
        raise NotImplementedError()

    def get_feedback(*args, **kwargs):
        raise NotImplementedError()

    def get_feedback_count_by_status(*args, **kwargs):
        raise NotImplementedError()

    def get_feedback_defs(
        self,
        feedback_definition_id: Optional[
            types_schema.FeedbackDefinitionID
        ] = None,
    ) -> pd.DataFrame:
        """See [DB.get_feedback_defs][trulens.core.database.base.DB.get_feedback_defs]."""
        # TODO(otel): Find all feedback definitions!
        return pd.DataFrame(
            columns=["feedback_definition_id", "feedback_json"],
        )

    def get_ground_truth(*args, **kwargs):
        raise NotImplementedError()

    def get_ground_truths_by_dataset(*args, **kwargs):
        raise NotImplementedError()

    def get_virtual_ground_truth(*args, **kwargs):
        raise NotImplementedError()

    def insert_app(*args, **kwargs):
        raise NotImplementedError()

    def insert_dataset(*args, **kwargs):
        raise NotImplementedError()

    def insert_event(*args, **kwargs):
        raise NotImplementedError()

    def insert_feedback(*args, **kwargs):
        raise NotImplementedError()

    def insert_feedback_definition(*args, **kwargs):
        raise NotImplementedError()

    def insert_ground_truth(*args, **kwargs):
        raise NotImplementedError()

    def insert_record(*args, **kwargs):
        raise NotImplementedError()

    def migrate_database(*args, **kwargs):
        raise NotImplementedError()

    def reset_database(*args, **kwargs):
        raise NotImplementedError()
