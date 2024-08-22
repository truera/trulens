from __future__ import annotations

from collections import defaultdict
from concurrent import futures
from datetime import datetime
import json
import logging
from multiprocessing import Process
from pprint import PrettyPrinter
import queue
import re
import threading
from threading import Thread
import time
from time import sleep
from typing import (
    Any,
    Callable,
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
from trulens.core.database.exceptions import DatabaseVersionException
from trulens.core.database.sqlalchemy import SQLAlchemyDB
from trulens.core.schema import app as mod_app_schema
from trulens.core.schema import dataset as mod_dataset_schema
from trulens.core.schema import feedback as mod_feedback_schema
from trulens.core.schema import groundtruth as mod_groundtruth_schema
from trulens.core.schema import record as mod_record_schema
from trulens.core.schema import types as mod_types_schema
from trulens.core.utils import python
from trulens.core.utils import serial
from trulens.core.utils import threading as tru_threading
from trulens.core.utils.imports import REQUIREMENT_SNOWFLAKE
from trulens.core.utils.imports import OptionalImports
from trulens.core.utils.python import Future  # code style exception
from trulens.core.utils.python import OpaqueWrapper
from trulens.core.utils.text import format_seconds

tqdm = None
with OptionalImports(messages=REQUIREMENT_SNOWFLAKE):
    from snowflake.core import CreateMode
    from snowflake.core import Root
    from snowflake.core.schema import Schema
    from snowflake.snowpark import Session
    from snowflake.sqlalchemy import URL
    from tqdm import tqdm

pp = PrettyPrinter()

logger = logging.getLogger(__name__)


class Tru(python.SingletonPerName):
    """Tru is the main class that provides an entry points to trulens.

    Tru lets you:

    - Log app prompts and outputs
    - Log app Metadata
    - Run and log feedback functions
    - Run streamlit dashboard to view experiment results

    By default, all data is logged to the current working directory to
    `"default.sqlite"`. Data can be logged to a SQLAlchemy-compatible url
    referred to by `database_url`.

    Supported App Types:
        [TruChain][trulens.instrument.langchain.TruChain]: Langchain
            apps.

        [TruLlama][trulens.instrument.llamaindex.TruLlama]: Llama Index
            apps.

        [TruRails][trulens.instrument.nemo.TruRails]: NeMo Guardrails apps.

        [TruBasicApp][trulens.core.TruBasicApp]:
            Basic apps defined solely using a function from `str` to `str`.

        [TruCustomApp][trulens.core.TruCustomApp]:
            Custom apps containing custom structures and methods. Requires annotation
            of methods to instrument.

        [TruVirtual][trulens.core.TruVirtual]: Virtual
            apps that do not have a real app to instrument but have a virtual
            structure and can log existing captured data as if they were trulens
            records.

    Args:
        database: Database to use. If not provided, an
            [SQLAlchemyDB][trulens.core.database.sqlalchemy.SQLAlchemyDB] database
            will be initialized based on the other arguments.

        database_url: Database URL. Defaults to a local SQLite
            database file at `"default.sqlite"` See [this
            article](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls)
            on SQLAlchemy database URLs. (defaults to
            `sqlite://DEFAULT_DATABASE_FILE`).

        database_file: Path to a local SQLite database file.

            **Deprecated**: Use `database_url` instead.

        database_prefix: Prefix for table names for trulens to use.
            May be useful in some databases hosting other apps.

        database_redact_keys: Whether to redact secret keys in data to be
            written to database (defaults to `False`)

        database_args: Additional arguments to pass to the database constructor.

        snowflake_connection_parameters: Connection arguments to Snowflake database to use.

        app_name: Name of the app.
    """

    RETRY_RUNNING_SECONDS: float = 60.0
    """How long to wait (in seconds) before restarting a feedback function that has already started

    A feedback function execution that has started may have stalled or failed in a bad way that did not record the
    failure.

    See also:
        [start_evaluator][trulens.core.tru.Tru.start_evaluator]

        [DEFERRED][trulens.core.schema.feedback.FeedbackMode.DEFERRED]
    """

    RETRY_FAILED_SECONDS: float = 5 * 60.0
    """How long to wait (in seconds) to retry a failed feedback function run."""

    DEFERRED_NUM_RUNS: int = 32
    """Number of futures to wait for when evaluating deferred feedback functions."""

    RECORDS_BATCH_TIMEOUT_IN_SEC: int = 10
    """Time to wait before inserting a batch of records into the database."""

    GROUND_TRUTHS_BATCH_SIZE: int = 100
    """Time to wait before inserting a batch of ground truths into the database."""

    db: Union[DB, OpaqueWrapper[DB]]
    """Database supporting this workspace.

    Will be an opqaue wrapper if it is not ready to use due to migration requirements.
    """

    _dashboard_urls: Optional[str] = None

    _evaluator_proc: Optional[Union[Process, Thread]] = None
    """[Process][multiprocessing.Process] or [Thread][threading.Thread] of the deferred feedback evaluator if started.

        Is set to `None` if evaluator is not running.
    """

    _dashboard_proc: Optional[Process] = None
    """[Process][multiprocessing.Process] executing the dashboard streamlit app.

    Is set to `None` if not executing.
    """

    _evaluator_stop: Optional[threading.Event] = None
    """Event for stopping the deferred evaluator which runs in another thread."""

    batch_record_queue = queue.Queue()

    batch_ground_truth_queue = queue.Queue()

    batch_thread = None

    def __new__(cls, *args, **kwargs) -> Tru:
        return super().__new__(cls, *args, **kwargs)

    def __init__(
        self,
        database: Optional[DB] = None,
        database_url: Optional[str] = None,
        database_file: Optional[str] = None,
        database_redact_keys: Optional[bool] = None,
        database_prefix: Optional[str] = None,
        database_args: Optional[Dict[str, Any]] = None,
        database_check_revision: bool = True,
        snowflake_connection_parameters: Optional[Dict[str, str]] = None,
        app_name: Optional[str] = None,
    ):
        """
        Args:
            database_check_revision: Whether to check the database revision on
                init. This prompt determine whether database migration is required.
        """

        if database_args is None:
            database_args = {}

        if snowflake_connection_parameters is not None:
            if database is not None:
                raise ValueError(
                    "`database` must be `None` if `snowflake_connection_parameters` is set!"
                )
            if database_url is not None:
                raise ValueError(
                    "`database_url` must be `None` if `snowflake_connection_parameters` is set!"
                )
            if not app_name:
                raise ValueError(
                    "`app_name` must be set if `snowflake_connection_parameters` is set!"
                )
            schema_name = self._validate_and_compute_schema_name(app_name)
            database_url = self._create_snowflake_database_url(
                snowflake_connection_parameters, schema_name
            )

        database_args.update({
            k: v
            for k, v in {
                "database_url": database_url,
                "database_file": database_file,
                "database_redact_keys": database_redact_keys,
                "database_prefix": database_prefix,
            }.items()
            if v is not None
        })

        if python.safe_hasattr(self, "db"):
            # Already initialized by SingletonByName mechanism. Give warning if
            # any option was specified (not None) as it will be ignored.
            for v in database_args.values():
                if v is not None:
                    logger.warning(
                        "Tru was already initialized. Cannot change database configuration after initialization."
                    )
                    self.warning()
                    break

            return

        if database is not None:
            if not isinstance(database, DB):
                raise ValueError(
                    "`database` must be a `trulens.core.database.base.DB` instance."
                )

            self.db = database
        else:
            self.db = SQLAlchemyDB.from_tru_args(**database_args)

        if database_check_revision:
            try:
                self.db.check_db_revision()
            except DatabaseVersionException as e:
                print(e)
                self.db = OpaqueWrapper(obj=self.db, e=e)

        # TODO: if snowflake_connection_parameters is not None:
        #    # initialize stream for feedback eval table.
        #    # initialize task for stream that will import trulens and try to run the Tru.

    @staticmethod
    def _validate_and_compute_schema_name(name):
        if not re.match(r"^[A-Za-z0-9_]+$", name):
            raise ValueError(
                "`name` must contain only alphanumeric and underscore characters!"
            )
        return f"TRULENS_APP__{name.upper()}"

    @staticmethod
    def _create_snowflake_database_url(
        snowflake_connection_parameters: Dict[str, str], schema_name: str
    ) -> str:
        Tru._create_snowflake_schema_if_not_exists(
            snowflake_connection_parameters, schema_name
        )
        return URL(
            account=snowflake_connection_parameters["account"],
            user=snowflake_connection_parameters["user"],
            password=snowflake_connection_parameters["password"],
            database=snowflake_connection_parameters["database"],
            schema=schema_name,
            warehouse=snowflake_connection_parameters.get("warehouse", None),
            role=snowflake_connection_parameters.get("role", None),
        )

    @staticmethod
    def _create_snowflake_schema_if_not_exists(
        snowflake_connection_parameters: Dict[str, str], schema_name: str
    ):
        session = Session.builder.configs(
            snowflake_connection_parameters
        ).create()
        root = Root(session)
        schema = Schema(name=schema_name)
        root.databases[
            snowflake_connection_parameters["database"]
        ].schemas.create(schema, mode=CreateMode.if_not_exists)

    def reset_database(self):
        """Reset the database. Clears all tables.

        See [DB.reset_database][trulens.core.database.base.DB.reset_database].
        """

        if isinstance(self.db, OpaqueWrapper):
            db = self.db.unwrap()
        elif isinstance(self.db, DB):
            db = self.db
        else:
            raise RuntimeError("Unhandled database type.")

        db.reset_database()
        self.db = db

    def migrate_database(self, **kwargs: Dict[str, Any]):
        """Migrates the database.

        This should be run whenever there are breaking changes in a database
        created with an older version of _trulens_.

        Args:
            **kwargs: Keyword arguments to pass to
                [migrate_database][trulens.core.database.base.DB.migrate_database]
                of the current database.

        See [DB.migrate_database][trulens.core.database.base.DB.migrate_database].
        """

        if isinstance(self.db, OpaqueWrapper):
            db = self.db.unwrap()
        elif isinstance(self.db, DB):
            db = self.db
        else:
            raise RuntimeError("Unhandled database type.")

        db.migrate_database(**kwargs)
        self.db = db

    def add_record(
        self, record: Optional[mod_record_schema.Record] = None, **kwargs: dict
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

    update_record = add_record

    def add_record_nowait(
        self,
        record: mod_record_schema.Record,
    ) -> None:
        """Add a record to the queue to be inserted in the next batch."""
        if self.batch_thread is None:
            self.batch_thread = threading.Thread(
                target=self.batch_loop, daemon=True
            )
            self.batch_thread.start()
        self.batch_record_queue.put(record)

    def batch_loop(self):
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

    # TODO: this method is used by app.py, which represents poor code
    # organization.
    def _submit_feedback_functions(
        self,
        record: mod_record_schema.Record,
        feedback_functions: Sequence[feedback.Feedback],
        app: Optional[mod_app_schema.AppDefinition] = None,
        on_done: Optional[
            Callable[
                [
                    Union[
                        mod_feedback_schema.FeedbackResult,
                        Future[mod_feedback_schema.FeedbackResult],
                    ],
                    None,
                ]
            ]
        ] = None,
    ) -> List[
        Tuple[feedback.Feedback, Future[mod_feedback_schema.FeedbackResult]]
    ]:
        """Schedules to run the given feedback functions.

        Args:
            record: The record on which to evaluate the feedback functions.

            feedback_functions: A collection of feedback functions to evaluate.

            app: The app that produced the given record. If not provided, it is
                looked up from the database of this `Tru` instance

            on_done: A callback to call when each feedback function is done.

        Returns:

            List[Tuple[feedback.Feedback, Future[schema.FeedbackResult]]]

            Produces a list of tuples where the first item in each tuple is the
            feedback function and the second is the future of the feedback result.
        """

        app_id = record.app_id

        self.db: DB

        if app is None:
            app = mod_app_schema.AppDefinition.model_validate(
                self.db.get_app(app_id=app_id)
            )
            if app is None:
                raise RuntimeError(
                    f"App {app_id} not present in db. "
                    "Either add it with `tru.add_app` or provide `app_json` to `tru.run_feedback_functions`."
                )

        else:
            assert (
                app_id == app.app_id
            ), "Record was produced by a different app."

            if self.db.get_app(app_id=app.app_id) is None:
                logger.warning(
                    f"App {app_id} was not present in database. Adding it."
                )
                self.add_app(app=app)

        feedbacks_and_futures = []

        tp: tru_threading.TP = tru_threading.TP()

        for ffunc in feedback_functions:
            # Run feedback function and the on_done callback. This makes sure
            # that Future.result() returns only after on_done has finished.
            def run_and_call_callback(ffunc, app, record):
                temp = ffunc.run(app=app, record=record)
                if on_done is not None:
                    try:
                        on_done(temp)
                    finally:
                        return temp
                return temp

            fut: Future[mod_feedback_schema.FeedbackResult] = tp.submit(
                run_and_call_callback, ffunc=ffunc, app=app, record=record
            )

            # Have to roll the on_done callback into the submitted function
            # because the result() is returned before callback runs otherwise.
            # We want to do db work before result is returned.

            feedbacks_and_futures.append((ffunc, fut))

        return feedbacks_and_futures

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
            for p in self._submit_feedback_functions(
                record=record, feedback_functions=feedback_functions, app=app
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

    def add_feedback(
        self,
        feedback_result_or_future: Optional[
            Union[
                mod_feedback_schema.FeedbackResult,
                Future[mod_feedback_schema.FeedbackResult],
            ]
        ] = None,
        **kwargs: dict,
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

            feedback_result_or_future = mod_feedback_schema.FeedbackResult(
                **kwargs
            )

        else:
            if isinstance(feedback_result_or_future, Future):
                futures.wait([feedback_result_or_future])
                feedback_result_or_future: mod_feedback_schema.FeedbackResult = feedback_result_or_future.result()

            elif isinstance(
                feedback_result_or_future, mod_feedback_schema.FeedbackResult
            ):
                pass
            else:
                raise ValueError(
                    f"Unknown type {type(feedback_result_or_future)} in feedback_results."
                )

            feedback_result_or_future.update(**kwargs)

        return self.db.insert_feedback(
            feedback_result=feedback_result_or_future
        )

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

        ids = []

        for feedback_result_or_future in feedback_results:
            ids.append(
                self.add_feedback(
                    feedback_result_or_future=feedback_result_or_future
                )
            )

        return ids

    def get_app(
        self, app_id: mod_types_schema.AppID
    ) -> serial.JSONized[mod_app_schema.AppDefinition]:
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

        return self.db.get_apps()

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
            Dataframe of records with their feedback results.

            List of feedback names that are columns in the dataframe.
        """

        if app_ids is None:
            app_ids = []

        df, feedback_columns = self.db.get_records_and_feedback(
            app_ids, offset=offset, limit=limit
        )

        return df, feedback_columns

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
            Dataframe of apps with their feedback results aggregated.
            If group_by_metadata_key is provided, the dataframe will be grouped by the specified key.
        """

        if app_ids is None:
            app_ids = []

        df, feedback_cols = self.db.get_records_and_feedback(app_ids)

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

    def add_ground_truth_to_dataset(
        self,
        dataset_name: str,
        ground_truth_df: pandas.DataFrame,
        dataset_metadata: Optional[Dict[str, Any]] = None,
    ):
        """Create a new dataset, if not existing, and add ground truth data to it. If
        the dataset with the same name already exists, the ground truth data will be added to it.

        Args:
            dataset_name: Name of the dataset.
            ground_truth_df: DataFrame containing the ground truth data.
            dataset_metadata: Additional metadata to add to the dataset.
        """

        # Create and insert the dataset record
        dataset = mod_dataset_schema.Dataset(
            name=dataset_name,
            meta=dataset_metadata,
        )
        dataset_id = self.db.insert_dataset(dataset=dataset)

        buffer = []

        for _, row in ground_truth_df.iterrows():
            ground_truth = mod_groundtruth_schema.GroundTruth(
                dataset_id=dataset_id,
                query=row["query"],
                query_id=row.get("query_id", None),
                expected_response=row.get("expected_response", None),
                expected_chunks=row.get("expected_chunks", None),
                meta=row.get("meta", None),
            )
            buffer.append(ground_truth)

            if len(buffer) >= self.GROUND_TRUTHS_BATCH_SIZE:
                self.db.batch_insert_ground_truth(buffer)
                buffer.clear()

        # remaining ground truths in the buffer
        if buffer:
            self.db.batch_insert_ground_truth(buffer)

    def get_ground_truth(self, dataset_name: str) -> pandas.DataFrame:
        """Get ground truth data from the dataset.
        dataset_name: Name of the dataset.
        """

        return self.db.get_ground_truths_by_dataset(dataset_name)

    def start_evaluator(
        self,
        restart: bool = False,
        fork: bool = False,
        disable_tqdm: bool = False,
        run_location: Optional[mod_feedback_schema.FeedbackRunLocation] = None,
        return_when_done: bool = False,
    ) -> Optional[Union[Process, Thread]]:
        """
        Start a deferred feedback function evaluation thread or process.

        Args:
            restart: If set, will stop the existing evaluator before starting a
                new one.

            fork: If set, will start the evaluator in a new process instead of a
                thread. NOT CURRENTLY SUPPORTED.

            disable_tqdm: If set, will disable progress bar logging from the evaluator.

            run_location: Run only the evaluations corresponding to run_location.

            return_when_done: Instead of running asynchronously, will block until no feedbacks remain.

        Returns:
            If return_when_done is True, then returns None. Otherwise, the started process or thread
                that is executing the deferred feedback evaluator.

        Relevant constants:
            [RETRY_RUNNING_SECONDS][trulens.core.tru.Tru.RETRY_RUNNING_SECONDS]

            [RETRY_FAILED_SECONDS][trulens.core.tru.Tru.RETRY_FAILED_SECONDS]

            [DEFERRED_NUM_RUNS][trulens.core.tru.Tru.DEFERRED_NUM_RUNS]

            [MAX_THREADS][trulens.core.utils.threading.TP.MAX_THREADS]
        """

        assert not fork, "Fork mode not yet implemented."
        assert (
            (not fork) or (not return_when_done)
        ), "fork=True implies running asynchronously but return_when_done=True does not!"

        if self._evaluator_proc is not None:
            if restart:
                self.stop_evaluator()
            else:
                raise RuntimeError(
                    "Evaluator is already running in this process."
                )

        if not fork:
            self._evaluator_stop = threading.Event()

        def runloop(stop_when_none_left: bool = False):
            assert self._evaluator_stop is not None

            print(
                f"Will keep max of "
                f"{self.DEFERRED_NUM_RUNS} feedback(s) running."
            )
            print(
                f"Tasks are spread among max of "
                f"{tru_threading.TP.MAX_THREADS} thread(s)."
            )
            print(
                f"Will rerun running feedbacks after "
                f"{format_seconds(self.RETRY_RUNNING_SECONDS)}."
            )
            print(
                f"Will rerun failed feedbacks after "
                f"{format_seconds(self.RETRY_FAILED_SECONDS)}."
            )

            total = 0

            if tqdm:
                # Getting total counts from the database to start off the tqdm
                # progress bar initial values so that they offer accurate
                # predictions initially after restarting the process.
                queue_stats = self.db.get_feedback_count_by_status(
                    run_location=run_location
                )
                queue_done = (
                    queue_stats.get(
                        mod_feedback_schema.FeedbackResultStatus.DONE
                    )
                    or 0
                )
                queue_total = sum(queue_stats.values())

                # Show the overall counts from the database, not just what has been
                # looked at so far.
                tqdm_status = tqdm(
                    desc="Feedback Status",
                    initial=queue_done,
                    unit="feedbacks",
                    total=queue_total,
                    postfix={
                        status.name: count
                        for status, count in queue_stats.items()
                    },
                    disable=disable_tqdm,
                )

                # Show the status of the results so far.
                tqdm_total = tqdm(
                    desc="Done Runs",
                    initial=0,
                    unit="runs",
                    disable=disable_tqdm,
                )

                # Show what is being waited for right now.
                tqdm_waiting = tqdm(
                    desc="Waiting for Runs",
                    initial=0,
                    unit="runs",
                    disable=disable_tqdm,
                )

            runs_stats = defaultdict(int)

            futures_map: Dict[
                Future[mod_feedback_schema.FeedbackResult], pandas.Series
            ] = dict()

            while fork or not self._evaluator_stop.is_set():
                if len(futures_map) < self.DEFERRED_NUM_RUNS:
                    # Get some new evals to run if some already completed by now.
                    new_futures: List[
                        Tuple[
                            pandas.Series,
                            Future[mod_feedback_schema.FeedbackResult],
                        ]
                    ] = feedback.Feedback.evaluate_deferred(
                        tru=self,
                        limit=self.DEFERRED_NUM_RUNS - len(futures_map),
                        shuffle=True,
                        run_location=run_location,
                    )

                    # Will likely get some of the same ones that already have running.
                    for row, fut in new_futures:
                        if fut in futures_map:
                            # If the future is already in our set, check whether
                            # its status has changed and if so, note it in the
                            # runs_stats.
                            if futures_map[fut].status != row.status:
                                runs_stats[row.status.name] += 1

                        futures_map[fut] = row
                        total += 1

                    if tqdm:
                        tqdm_total.total = total
                        tqdm_total.refresh()

                if tqdm:
                    tqdm_waiting.total = self.DEFERRED_NUM_RUNS
                    tqdm_waiting.n = len(futures_map)
                    tqdm_waiting.refresh()

                # Note whether we have waited for some futures in this
                # iteration. Will control some extra wait time if there is no
                # work.
                did_wait = False

                if len(futures_map) > 0:
                    did_wait = True

                    futures_copy = list(futures_map.keys())

                    try:
                        for fut in futures.as_completed(
                            futures_copy, timeout=10
                        ):
                            del futures_map[fut]

                            if tqdm:
                                tqdm_waiting.update(-1)
                                tqdm_total.update(1)

                            feedback_result = fut.result()
                            runs_stats[feedback_result.status.name] += 1

                    except futures.TimeoutError:
                        pass

                if tqdm:
                    tqdm_total.set_postfix({
                        name: count for name, count in runs_stats.items()
                    })

                    queue_stats = self.db.get_feedback_count_by_status(
                        run_location=run_location
                    )
                    queue_done = (
                        queue_stats.get(
                            mod_feedback_schema.FeedbackResultStatus.DONE
                        )
                        or 0
                    )
                    queue_total = sum(queue_stats.values())

                    tqdm_status.n = queue_done
                    tqdm_status.total = queue_total
                    tqdm_status.set_postfix({
                        status.name: count
                        for status, count in queue_stats.items()
                    })

                # Check if any of the running futures should be stopped.
                futures_copy = list(futures_map.keys())
                for fut in futures_copy:
                    row = futures_map[fut]

                    if fut.running():
                        # Not checking status here as this will be not yet be set
                        # correctly. The computation in the future updates the
                        # database but this object is outdated.

                        elapsed = datetime.now().timestamp() - row.last_ts
                        if elapsed > self.RETRY_RUNNING_SECONDS:
                            fut.cancel()

                            # Not an actual status, but would be nice to
                            # indicate cancellations in run stats:
                            runs_stats["CANCELLED"] += 1

                            del futures_map[fut]

                if not did_wait:
                    if stop_when_none_left:
                        break
                    # Nothing to run/is running, wait a bit.
                    if fork:
                        sleep(10)
                    else:
                        self._evaluator_stop.wait(10)

            print("Evaluator stopped.")

        if return_when_done:
            runloop(stop_when_none_left=True)
            return None
        else:
            if fork:
                proc = Process(target=runloop)
            else:
                proc = Thread(target=runloop)
                proc.daemon = True
            # Start a persistent thread or process that evaluates feedback functions.
            self._evaluator_proc = proc
            proc.start()
            return proc

    run_evaluator = start_evaluator

    def stop_evaluator(self):
        """
        Stop the deferred feedback evaluation thread.
        """

        if self._evaluator_proc is None:
            raise RuntimeError("Evaluator not running this process.")

        if isinstance(self._evaluator_proc, Process):
            self._evaluator_proc.terminate()

        elif isinstance(self._evaluator_proc, Thread):
            self._evaluator_stop.set()
            self._evaluator_proc.join()
            self._evaluator_stop = None

        self._evaluator_proc = None
