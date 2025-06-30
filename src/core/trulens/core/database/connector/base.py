from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from concurrent import futures
import datetime
import logging
import queue
from threading import Thread
import time
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import pandas as pd
from trulens.core._utils.pycompat import Future  # code style exception
from trulens.core.database import base as core_db
from trulens.core.otel.utils import is_otel_tracing_enabled
from trulens.core.schema import app as app_schema
from trulens.core.schema import event as event_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import types as types_schema
from trulens.core.utils import serial as serial_utils
from trulens.core.utils import text as text_utils

logger = logging.getLogger(__name__)


class DBConnector(ABC, text_utils.WithIdentString):
    """Base class for DB connector implementations."""

    RECORDS_BATCH_TIMEOUT_IN_SEC: int = 10
    """Time to wait before inserting a batch of records into the database."""

    batch_record_queue = queue.Queue()

    batch_thread = None

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.db})"

    # for WithIdentString:
    def _ident_str(self) -> str:
        return self.db._ident_str()

    @property
    @abstractmethod
    def db(self) -> core_db.DB:
        """Get the database instance."""
        ...

    def reset_database(self):
        """Reset the database. Clears all tables.

        See [DB.reset_database][trulens.core.database.base.DB.reset_database].
        """
        self.db.reset_database()

    def migrate_database(self, **kwargs: Any):
        """Migrates the database.

        This should be run whenever there are breaking changes in a database
        created with an older version of _trulens_.

        Args:
            **kwargs: Keyword arguments to pass to
                [migrate_database][trulens.core.database.base.DB.migrate_database]
                of the current database.

        See [DB.migrate_database][trulens.core.database.base.DB.migrate_database].
        """
        self.db.migrate_database(**kwargs)

    def add_record(
        self, record: Optional[record_schema.Record] = None, **kwargs
    ) -> types_schema.RecordID:
        """Add a record to the database.

        Args:
            record: The record to add.

            **kwargs: [Record][trulens.core.schema.record.Record] fields to add to the
                given record or a new record if no `record` provided.

        Returns:
            Unique record identifier [str][] .

        """
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")

        if record is None:
            record = record_schema.Record(**kwargs)
        else:
            record.update(**kwargs)
        return self.db.insert_record(record=record)

    def add_record_nowait(
        self,
        record: record_schema.Record,
    ) -> None:
        """Add a record to the queue to be inserted in the next batch."""
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")
        if self.batch_thread is None:
            self.batch_thread = Thread(target=self._batch_loop, daemon=True)
            self.batch_thread.start()
        self.batch_record_queue.put(record)

    def _batch_loop(self):
        apps = {}
        while True:
            time.sleep(self.RECORDS_BATCH_TIMEOUT_IN_SEC)
            records = []
            while True:
                try:
                    record = self.batch_record_queue.get_nowait()
                    records.append(record)
                except queue.Empty:
                    break
            if records:
                try:
                    self.db.batch_insert_record(records)
                except Exception as e:
                    # Re-queue the records that failed to be inserted
                    for record in records:
                        self.batch_record_queue.put(record)
                    logger.error(
                        "Re-queued records due to insertion error {}", e
                    )
                    continue
                feedback_results = []
                for record in records:
                    app_id = record.app_id
                    if app_id not in apps:
                        apps[app_id] = self.get_app(app_id=app_id)
                    app = apps[app_id]

                    feedback_definitions = app.get("feedback_definitions", [])
                    # TODO(Dave): Modify this to add only client side feedback results
                    for feedback_definition_id in feedback_definitions:
                        feedback_results.append(
                            feedback_schema.FeedbackResult(
                                feedback_definition_id=feedback_definition_id,
                                record_id=record.record_id,
                                name="feedback_name",  # this will be updated later by deferred evaluator
                            )
                        )
                try:
                    self.db.batch_insert_feedback(feedback_results)
                except Exception as e:
                    logger.error("Failed to insert feedback results {}", e)

    def add_app(self, app: app_schema.AppDefinition) -> types_schema.AppID:
        """
        Add an app to the database and return its unique id.

        Args:
            app: The app to add to the database.

        Returns:
            A unique app identifier [str][].

        """

        return self.db.insert_app(app=app)

    def delete_app(self, app_id: types_schema.AppID) -> None:
        """
        Deletes an app from the database based on its app_id.

        Args:
            app_id (schema.AppID): The unique identifier of the app to be deleted.
        """
        self.db.delete_app(app_id=app_id)
        logger.info(f"App with ID {app_id} has been successfully deleted.")

    def add_feedback_definition(
        self, feedback_definition: feedback_schema.FeedbackDefinition
    ) -> types_schema.FeedbackDefinitionID:
        """
        Add a feedback definition to the database and return its unique id.

        Args:
            feedback_definition: The feedback definition to add to the database.

        Returns:
            A unique feedback definition identifier [str][].
        """

        return self.db.insert_feedback_definition(
            feedback_definition=feedback_definition
        )

    def add_feedback(
        self,
        feedback_result_or_future: Optional[
            Union[
                feedback_schema.FeedbackResult,
                Future[feedback_schema.FeedbackResult],
            ]
        ] = None,
        **kwargs: Any,
    ) -> types_schema.FeedbackResultID:
        """Add a single feedback result or future to the database and return its unique id.

        Args:
            feedback_result_or_future: If a [Future][concurrent.futures.Future]
                is given, call will wait for the result before adding it to the
                database. If `kwargs` are given and a
                [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] is also
                given, the `kwargs` will be used to update the
                [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] otherwise a
                new one will be created with `kwargs` as arguments to its
                constructor.

            **kwargs: Fields to add to the given feedback result or to create a
                new [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] with.

        Returns:
            A unique result identifier [str][].

        """
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")
        if feedback_result_or_future is None:
            if "result" in kwargs and "status" not in kwargs:
                # If result already present, set status to done.
                kwargs["status"] = feedback_schema.FeedbackResultStatus.DONE

            feedback_result = feedback_schema.FeedbackResult(**kwargs)

        elif isinstance(feedback_result_or_future, Future):
            futures.wait([feedback_result_or_future])
            feedback_result: feedback_schema.FeedbackResult = (
                feedback_result_or_future.result()
            )
            feedback_result.update(**kwargs)

        elif isinstance(
            feedback_result_or_future, feedback_schema.FeedbackResult
        ):
            feedback_result = feedback_result_or_future
            feedback_result.update(**kwargs)
        else:
            raise ValueError(
                f"Unknown type {type(feedback_result_or_future)} in feedback_results."
            )

        if feedback_result.feedback_definition_id is None:
            feedback_result.feedback_definition_id = "anonymous"  # or "human" feedback that does not come with a feedback definition

        return self.db.insert_feedback(feedback_result=feedback_result)

    def add_feedbacks(
        self,
        feedback_results: Iterable[
            Union[
                feedback_schema.FeedbackResult,
                Future[feedback_schema.FeedbackResult],
            ]
        ],
    ) -> List[types_schema.FeedbackResultID]:
        """Add multiple feedback results to the database and return their unique ids.
        # TODO: This is slow and should be batched or otherwise optimized in the future.

        Args:
            feedback_results: An iterable with each iteration being a [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] or
                [Future][concurrent.futures.Future] of the same. Each given future will be waited.

        Returns:
            List of unique result identifiers [str][] in the same order as input
                `feedback_results`.
        """
        if is_otel_tracing_enabled():
            raise RuntimeError("Not supported with OTel tracing enabled!")
        return [
            self.add_feedback(
                feedback_result_or_future=feedback_result_or_future
            )
            for feedback_result_or_future in feedback_results
        ]

    def get_app(
        self, app_id: types_schema.AppID
    ) -> Optional[serial_utils.JSONized[app_schema.AppDefinition]]:
        """Look up an app from the database.

        This method produces the JSON-ized version of the app. It can be deserialized back into an [AppDefinition][trulens.core.schema.app.AppDefinition] with [model_validate][pydantic.BaseModel.model_validate]:

        Example:
            ```python
            from trulens.core.schema import app
            app_json = session.get_app(app_id="Custom Application v1")
            app = app.AppDefinition.model_validate(app_json)
            ```

        Warning:
            Do not rely on deserializing into [App][trulens.core.app.App] as
            its implementations feature attributes not meant to be deserialized.

        Args:
            app_id: The unique identifier [str][] of the app to look up.

        Returns:
            JSON-ized version of the app.
        """

        return self.db.get_app(app_id)

    def get_apps(self) -> List[serial_utils.JSONized[app_schema.AppDefinition]]:
        """Look up all apps from the database.

        Returns:
            A list of JSON-ized version of all apps in the database.

        Warning:
            Same Deserialization caveats as [get_app][trulens.core.session.TruSession.get_app].
        """

        return list(self.db.get_apps())

    def get_records_and_feedback(
        self,
        app_ids: Optional[List[types_schema.AppID]] = None,
        app_name: Optional[types_schema.AppName] = None,
        app_version: Optional[types_schema.AppVersion] = None,
        record_ids: Optional[List[types_schema.RecordID]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Get records, their feedback results, and feedback names.

        Args:
            app_ids: A list of app ids to filter records by. If empty or not given, all
                apps' records will be returned.

            app_name: A name of the app to filter records by. If given, only records for
                this app will be returned.

            record_ids: An optional list of record ids to filter records by.

            offset: Record row offset.

            limit: Limit on the number of records to return.

        Returns:
            DataFrame of records with their feedback results.

            List of feedback names that are columns in the DataFrame.
        """

        df, feedback_columns = self.db.get_records_and_feedback(
            app_ids=app_ids,
            app_name=app_name,
            app_version=app_version,
            record_ids=record_ids,
            offset=offset,
            limit=limit,
        )

        df["app_name"] = df["app_json"].apply(lambda x: x.get("app_name"))
        df["app_version"] = df["app_json"].apply(lambda x: x.get("app_version"))

        return df, list(feedback_columns)

    def get_leaderboard(
        self,
        app_ids: Optional[List[types_schema.AppID]] = None,
        group_by_metadata_key: Optional[str] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
    ) -> pd.DataFrame:
        """Get a leaderboard for the given apps.

        Args:
            app_ids: A list of app ids to filter records by. If empty or not given, all
                apps will be included in leaderboard.

            group_by_metadata_key: A key included in record metadata that you want to group results by.

            limit: Limit on the number of records to aggregate to produce the leaderboard.

            offset: Record row offset to select which records to use to aggregate the leaderboard.

        Returns:
            DataFrame of apps with their feedback results aggregated.
            If group_by_metadata_key is provided, the DataFrame will be grouped by the specified key.
        """

        if app_ids is None:
            app_ids = []

        df, feedback_cols = self.get_records_and_feedback(
            app_ids, limit=limit, offset=offset
        )
        feedback_cols = sorted(feedback_cols)

        df["app_name"] = df["app_json"].apply(lambda x: x.get("app_name"))
        df["app_version"] = df["app_json"].apply(lambda x: x.get("app_version"))

        # TODO: refactor implementation for total_cost map in OTEL implementation of _get_records_and_feedback (see comment: https://github.com/truera/trulens/pull/1939#discussion_r2054802093)
        col_agg_list = feedback_cols + ["latency", "total_cost"]

        if group_by_metadata_key is not None:
            df["meta"] = [df["record_json"][i]["meta"] for i in range(len(df))]

            df[str(group_by_metadata_key)] = [
                item.get(group_by_metadata_key, None)
                if isinstance(item, dict)
                else None
                for item in df["meta"]
            ]
            return (
                df.groupby([
                    "app_name",
                    "app_version",
                    str(group_by_metadata_key),
                ])[col_agg_list]
                .mean()
                .sort_values(by=feedback_cols, ascending=False)
            )
        else:
            return (
                df.groupby(["app_name", "app_version"])[col_agg_list]
                .mean()
                .sort_values(by=feedback_cols, ascending=False)
            )

    def add_event(self, event: event_schema.Event):
        """
        Add an event to the database.

        Args:
            event: The event to add to the database.
        """
        return self.db.insert_event(event=event)

    def add_events(self, events: List[event_schema.Event]):
        """
        Add multiple events to the database.
        # TODO: This is slow and should be batched or otherwise optimized in the future.

        Args:
            events: A list of events to add to the database.
        """
        return [self.add_event(event=event) for event in events]

    def get_events(
        self,
        app_name: Optional[str] = None,
        app_version: Optional[str] = None,
        record_ids: Optional[List[str]] = None,
        start_time: Optional[datetime.datetime] = None,
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
        return self.db.get_events(
            app_name=app_name,
            app_version=app_version,
            record_ids=record_ids,
            start_time=start_time,
        )
