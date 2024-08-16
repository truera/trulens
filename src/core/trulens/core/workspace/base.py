from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from concurrent import futures
import json
import logging
import queue
from threading import Thread
import time
from typing import (
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import pandas
from trulens.core import feedback
from trulens.core.database.base import DB
from trulens.core.schema import app as mod_app_schema
from trulens.core.schema import feedback as mod_feedback_schema
from trulens.core.schema import record as mod_record_schema
from trulens.core.schema import types as mod_types_schema
from trulens.core.utils import serial
from trulens.core.utils.python import Future  # code style exception

logger = logging.getLogger(__name__)


class BaseWorkspace(ABC):
    """Base class for workspace implementations."""

    RECORDS_BATCH_TIMEOUT: int = 10
    """Time to wait before inserting a batch of records into the database."""

    batch_record_queue = queue.Queue()

    batch_thread = None

    @property
    @abstractmethod
    def db(self) -> DB: ...

    def reset_database(self):
        """Reset the database. Clears all tables.

        See [DB.reset_database][trulens.core.database.base.DB.reset_database].
        """
        self.db.reset_database()

    def migrate_database(self, **kwargs):
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
        self, record: Optional[mod_record_schema.Record] = None, **kwargs
    ) -> mod_types_schema.RecordID:
        """Add a record to the database.

        Args:
            record: The record to add.

            **kwargs: [Record][trulens.core.schema.record.Record] fields to add to the
                given record or a new record if no `record` provided.

        Returns:
            Unique record identifier [str][] .

        """

        if record is None:
            record = mod_record_schema.Record(**kwargs)
        else:
            record.update(**kwargs)
        return self.db.insert_record(record=record)

    def add_record_nowait(
        self,
        record: mod_record_schema.Record,
    ) -> None:
        """Add a record to the queue to be inserted in the next batch."""
        if self.batch_thread is None:
            self.batch_thread = Thread(target=self.batch_loop, daemon=True)
            self.batch_thread.start()
        self.batch_record_queue.put(record)

    def batch_loop(self):
        while True:
            time.sleep(self.RECORDS_BATCH_TIMEOUT)
            records: List[mod_record_schema.Record] = []
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
                apps = {}
                for record in records:
                    app_id = record.app_id
                    app = apps.setdefault(app_id, self.get_app(app_id=app_id))
                    feedback_definitions = app.get("feedback_definitions", [])
                    # TODO(Dave): Modify this to add only client side feedback results
                    for feedback_definition_id in feedback_definitions:
                        feedback_results.append(
                            mod_feedback_schema.FeedbackResult(
                                feedback_definition_id=feedback_definition_id,
                                record_id=record.record_id,
                                name="feedback_name",  # this will be updated later by deferred evaluator
                            )
                        )
                try:
                    self.db.batch_insert_feedback(feedback_results)
                except Exception as e:
                    logger.error("Failed to insert feedback results {}", e)

    def run_feedback_functions(
        self,
        record: mod_record_schema.Record,
        feedback_functions: Sequence[feedback.Feedback],
        app: Optional[mod_app_schema.AppDefinition] = None,
        wait: bool = True,
    ) -> Union[
        Iterable[mod_feedback_schema.FeedbackResult],
        Iterable[Future[mod_feedback_schema.FeedbackResult]],
    ]:
        """Run a collection of feedback functions and report their result.

        Args:
            record: The record on which to evaluate the feedback
                functions.

            app: The app that produced the given record.
                If not provided, it is looked up from the given database `db`.

            feedback_functions: A collection of feedback
                functions to evaluate.

            wait: If set (default), will wait for results
                before returning.

        Yields:
            One result for each element of `feedback_functions` of
                [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] if `wait`
                is enabled (default) or [Future][concurrent.futures.Future] of
                [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] if `wait`
                is disabled.
        """

        if not isinstance(record, mod_record_schema.Record):
            raise ValueError(
                "`record` must be a `trulens.core.schema.record.Record` instance."
            )

        if not isinstance(feedback_functions, Sequence):
            raise ValueError("`feedback_functions` must be a sequence.")

        if not all(
            isinstance(ffunc, feedback.Feedback) for ffunc in feedback_functions
        ):
            raise ValueError(
                "`feedback_functions` must be a sequence of `trulens.core.Feedback` instances."
            )

        if not (app is None or isinstance(app, mod_app_schema.AppDefinition)):
            raise ValueError(
                "`app` must be a `trulens.core.schema.app.AppDefinition` instance."
            )

        if not isinstance(wait, bool):
            raise ValueError("`wait` must be a bool.")

        future_feedback_map: Dict[
            Future[mod_feedback_schema.FeedbackResult], feedback.Feedback
        ] = {
            p[1]: p[0]
            for p in mod_app_schema.AppDefinition._submit_feedback_functions(
                record=record,
                feedback_functions=feedback_functions,
                workspace=self,
                app=app,
            )
        }

        if wait:
            # In blocking mode, wait for futures to complete.
            for fut_result in futures.as_completed(future_feedback_map.keys()):
                # TODO: Do we want a version that gives the feedback for which
                # the result is being produced too? This is more useful in the
                # Future case as we cannot check associate a Future result to
                # its feedback before result is ready.

                # yield (future_feedback_map[fut_result], fut_result.result())
                yield fut_result.result()

        else:
            # In non-blocking, return the futures instead.
            for fut_result, _ in future_feedback_map.items():
                # TODO: see prior.

                # yield (feedback, fut_result)
                yield fut_result

    def add_app(
        self, app: mod_app_schema.AppDefinition
    ) -> mod_types_schema.AppID:
        """
        Add an app to the database and return its unique id.

        Args:
            app: The app to add to the database.

        Returns:
            A unique app identifier [str][].

        """

        return self.db.insert_app(app=app)

    def delete_app(self, app_id: mod_types_schema.AppID) -> None:
        """
        Deletes an app from the database based on its app_id.

        Args:
            app_id (schema.AppID): The unique identifier of the app to be deleted.
        """
        self.db.delete_app(app_id=app_id)
        logger.info(f"App with ID {app_id} has been successfully deleted.")

    def add_feedback_definition(
        self, feedback_definition: mod_feedback_schema.FeedbackDefinition
    ) -> mod_types_schema.FeedbackDefinitionID:
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
                mod_feedback_schema.FeedbackResult,
                Future[mod_feedback_schema.FeedbackResult],
            ]
        ] = None,
        **kwargs,
    ) -> mod_types_schema.FeedbackResultID:
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

        if feedback_result_or_future is None:
            if "result" in kwargs and "status" not in kwargs:
                # If result already present, set status to done.
                kwargs["status"] = mod_feedback_schema.FeedbackResultStatus.DONE

            feedback_result = mod_feedback_schema.FeedbackResult(**kwargs)

        elif isinstance(feedback_result_or_future, Future):
            futures.wait([feedback_result_or_future])
            feedback_result: mod_feedback_schema.FeedbackResult = (
                feedback_result_or_future.result()
            )
            feedback_result.update(**kwargs)

        elif isinstance(
            feedback_result_or_future, mod_feedback_schema.FeedbackResult
        ):
            feedback_result = feedback_result_or_future
            feedback_result.update(**kwargs)
        else:
            raise ValueError(
                f"Unknown type {type(feedback_result_or_future)} in feedback_results."
            )

        return self.db.insert_feedback(feedback_result=feedback_result)

    def add_feedbacks(
        self,
        feedback_results: Iterable[
            Union[
                mod_feedback_schema.FeedbackResult,
                Future[mod_feedback_schema.FeedbackResult],
            ]
        ],
    ) -> List[mod_types_schema.FeedbackResultID]:
        """Add multiple feedback results to the database and return their unique ids.

        Args:
            feedback_results: An iterable with each iteration being a [FeedbackResult][trulens.core.schema.feedback.FeedbackResult] or
                [Future][concurrent.futures.Future] of the same. Each given future will be waited.

        Returns:
            List of unique result identifiers [str][] in the same order as input
                `feedback_results`.
        """

        return [
            self.add_feedback(
                feedback_result_or_future=feedback_result_or_future
            )
            for feedback_result_or_future in feedback_results
        ]

    def get_app(
        self, app_id: mod_types_schema.AppID
    ) -> Optional[serial.JSONized[mod_app_schema.AppDefinition]]:
        """Look up an app from the database.

        This method produces the JSON-ized version of the app. It can be deserialized back into an [AppDefinition][trulens.core.schema.app.AppDefinition] with [model_validate][pydantic.BaseModel.model_validate]:

        Example:
            ```python
            from trulens.core.schema import app
            app_json = tru.get_app(app_id="Custom Application v1")
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

    def get_apps(self) -> List[serial.JSONized[mod_app_schema.AppDefinition]]:
        """Look up all apps from the database.

        Returns:
            A list of JSON-ized version of all apps in the database.

        Warning:
            Same Deserialization caveats as [get_app][trulens.core.tru.Tru.get_app].
        """

        return list(self.db.get_apps())

    def get_records_and_feedback(
        self,
        app_ids: Optional[List[mod_types_schema.AppID]] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Tuple[pandas.DataFrame, List[str]]:
        """Get records, their feedback results, and feedback names.

        Args:
            app_ids: A list of app ids to filter records by. If empty or not given, all
                apps' records will be returned.

            offset: Record row offset.

            limit: Limit on the number of records to return.

        Returns:
            DataFrame of records with their feedback results.

            List of feedback names that are columns in the DataFrame.
        """

        df, feedback_columns = self.db.get_records_and_feedback(
            app_ids, offset=offset, limit=limit
        )

        return df, list(feedback_columns)

    def get_leaderboard(
        self,
        app_ids: Optional[List[mod_types_schema.AppID]] = None,
        group_by_metadata_key: Optional[str] = None,
    ) -> pandas.DataFrame:
        """Get a leaderboard for the given apps.

        Args:
            app_ids: A list of app ids to filter records by. If empty or not given, all
                apps will be included in leaderboard.
            group_by_metadata_key: A key included in record metadata that you want to group results by.

        Returns:
            DataFrame of apps with their feedback results aggregated.
            If group_by_metadata_key is provided, the DataFrame will be grouped by the specified key.
        """

        if app_ids is None:
            app_ids = []

        df, feedback_cols = self.get_records_and_feedback(app_ids)

        col_agg_list = feedback_cols + ["latency", "total_cost"]

        if group_by_metadata_key is not None:
            df["meta"] = [
                json.loads(df["record_json"][i])["meta"] for i in range(len(df))
            ]

            df[str(group_by_metadata_key)] = [
                item.get(group_by_metadata_key, None)
                if isinstance(item, dict)
                else None
                for item in df["meta"]
            ]
            return (
                df.groupby(["app_id", str(group_by_metadata_key)])[col_agg_list]
                .mean()
                .sort_values(by=feedback_cols, ascending=False)
            )
        else:
            return (
                df.groupby("app_id")[col_agg_list]
                .mean()
                .sort_values(by=feedback_cols, ascending=False)
            )
