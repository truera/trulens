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


class SnowflakeEventTableDB(core_db.DB):
    """Connector to the account level event table in Snowflake."""

    _snowpark_session: Session
    _external_agent_dao: ExternalAgentDao

    table_prefix: str = core_db.DEFAULT_DATABASE_PREFIX

    def __init__(self, snowpark_session: Session):
        super().__init__(table_prefix="")
        self._snowpark_session = snowpark_session
        self._external_agent_dao = ExternalAgentDao(snowpark_session)

    def get_records_and_feedback(
        self,
        app_ids: Optional[List[types_schema.AppID]] = None,
        app_name: Optional[types_schema.AppName] = None,
        app_version: Optional[types_schema.AppVersion] = None,
        app_versions: Optional[List[types_schema.AppVersion]] = None,
        record_ids: Optional[List[types_schema.RecordID]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        """Get records from the database.

        Args:
            app_ids: If given, retrieve only the records for the given apps.
                Otherwise all apps are retrieved.
            app_name: If given, retrieve only the records for the given app name.
            app_version: If given, retrieve only the records for the given app version.
            app_versions: If given, retrieve only the records for the given app versions.
            record_ids: Optional list of record IDs to filter by. Defaults to None.
            offset: Database row offset.
            limit: Limit on rows (records) returned.

        Returns:
            A DataFrame with the records.

            A list of column names that contain feedback results.
        """
        df = self._get_events(
            app_ids=app_ids,
            app_name=app_name,
            app_version=app_version,
            app_versions=app_versions,
            record_ids=record_ids,
        )
        events = []
        for _, row in df.iterrows():
            trace = json.loads(row["TRACE"])
            if "parent_id" not in trace:
                trace["parent_id"] = ""
            events.append(
                Event(
                    event_id=trace["span_id"],
                    record=json.loads(row["RECORD"]),
                    record_attributes=json.loads(row["RECORD_ATTRIBUTES"]),
                    record_type=row["RECORD_TYPE"],
                    resource_attributes=json.loads(row["RESOURCE_ATTRIBUTES"]),
                    start_timestamp=row["START_TIMESTAMP"],
                    timestamp=row["TIMESTAMP"],
                    trace=trace,
                )
            )
        return self._get_records_and_feedback_otel_from_events(
            events=events, app_ids=app_ids, app_name=app_name
        )

    def get_events(
        self,
        app_name: Optional[types_schema.AppName] = None,
        app_version: Optional[types_schema.AppVersion] = None,
        record_ids: Optional[List[types_schema.RecordID]] = None,
        start_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Get events from the database.

        Args:
            app_name: The app name to filter events by.
            app_version: The app version to filter events by.
            record_ids: The record ids to filter events by.
            start_time: The minimum time to consider events from.

        Returns:
            A pandas DataFrame of all relevant events.
        """
        return self._get_events(
            app_name=app_name,
            app_version=app_version,
            record_ids=record_ids,
            start_time=start_time,
        )

    def _get_events(
        self,
        app_ids: Optional[List[types_schema.AppID]] = None,
        app_name: Optional[types_schema.AppName] = None,
        app_version: Optional[types_schema.AppVersion] = None,
        app_versions: Optional[List[types_schema.AppVersion]] = None,
        record_ids: Optional[List[types_schema.RecordID]] = None,
        start_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
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
        if app_version:
            app_version_str = f"'{app_version}'"
            where_clauses.append(
                f'(RECORD_ATTRIBUTES:"snow.ai.observability.agent.version" = {app_version_str} OR RECORD_ATTRIBUTES:"snow.ai.observability.object.version.name" = {app_version_str})'
            )
        if app_versions:
            app_versions_str = ", ".join([f"'{curr}'" for curr in app_versions])
            where_clauses.append(
                f'(RECORD_ATTRIBUTES:"snow.ai.observability.agent.version" IN ({app_versions_str}) OR RECORD_ATTRIBUTES:"snow.ai.observability.object.version.name" IN ({app_versions_str}))'
            )
        if record_ids:
            record_ids_str = ", ".join([f"'{curr}'" for curr in record_ids])
            where_clauses.append(
                f'RECORD_ATTRIBUTES:"{SpanAttributes.RECORD_ID}" IN ({record_ids_str})'
            )
        if start_time:
            start_time_str = f"'{start_time}'"
            where_clauses.append(f"TIMESTAMP >= {start_time_str}")
        if where_clauses:
            q += " WHERE " + " AND ".join(where_clauses)
        df = None
        for app_type in ["EXTERNAL AGENT", "CORTEX AGENT"]:
            try:
                df = self._snowpark_session.sql(
                    q, params=[database, schema, app_name, app_type]
                ).to_pandas()
                if not df.empty:
                    break
            except Exception:
                # TODO: There doesn't seem to be a way to tell if an app is an
                # external agent or a cortex agent yet so we're trying both
                # in a hacky way right now for the time being.
                pass
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
            app_names = self._external_agent_dao._list_agents()["name"].values
        else:
            app_names = [app_name]
        for app_name in app_names:
            for _, row in self._external_agent_dao.list_agent_versions(
                app_name
            ).iterrows():
                app_version = row["name"]
                app_id = (
                    app_schema.AppDefinition._compute_app_id(
                        app_name, app_version
                    ),
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

    def get_feedback_defs(*args, **kwargs):
        raise NotImplementedError()

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
