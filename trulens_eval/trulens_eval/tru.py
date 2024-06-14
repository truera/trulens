from __future__ import annotations

from collections import defaultdict
from concurrent import futures
from datetime import datetime
from datetime import timedelta
import logging
from multiprocessing import Process
import os
from pathlib import Path
from pprint import PrettyPrinter
import subprocess
import sys
import threading
from threading import Thread
from time import sleep
from typing import (
    Any, Callable, Dict, Generic, Iterable, List, Optional, Sequence, Tuple,
    TypeVar, Union
)

import humanize
import pandas
from tqdm.auto import tqdm
from typing_extensions import Annotated
from typing_extensions import Doc

from trulens_eval.database import sqlalchemy
from trulens_eval.database.base import DB
from trulens_eval.database.exceptions import DatabaseVersionException
from trulens_eval.feedback import feedback
from trulens_eval.schema import app as mod_app_schema
from trulens_eval.schema import feedback as mod_feedback_schema
from trulens_eval.schema import record as mod_record_schema
from trulens_eval.schema import types as mod_types_schema
from trulens_eval.utils import notebook_utils
from trulens_eval.utils import python
from trulens_eval.utils import serial
from trulens_eval.utils import threading as tru_threading
from trulens_eval.utils.imports import static_resource
from trulens_eval.utils.python import Future  # code style exception
from trulens_eval.utils.python import OpaqueWrapper

pp = PrettyPrinter()

logger = logging.getLogger(__name__)

DASHBOARD_START_TIMEOUT: Annotated[int, Doc("Seconds to wait for dashboard to start")] \
    = 30


def humanize_seconds(seconds: float):
    return humanize.naturaldelta(timedelta(seconds=seconds))


class Tru(python.SingletonPerName):
    """Tru is the main class that provides an entry points to trulens-eval.
    
    Tru lets you:

    - Log app prompts and outputs
    - Log app Metadata
    - Run and log feedback functions
    - Run streamlit dashboard to view experiment results

    By default, all data is logged to the current working directory to
    `"default.sqlite"`. Data can be logged to a SQLAlchemy-compatible url
    referred to by `database_url`.

    Supported App Types:
        [TruChain][trulens_eval.tru_chain.TruChain]: Langchain
            apps.

        [TruLlama][trulens_eval.tru_llama.TruLlama]: Llama Index
            apps.

        [TruRails][trulens_eval.tru_rails.TruRails]: NeMo Guardrails apps.

        [TruBasicApp][trulens_eval.tru_basic_app.TruBasicApp]:
            Basic apps defined solely using a function from `str` to `str`.

        [TruCustomApp][trulens_eval.tru_custom_app.TruCustomApp]:
            Custom apps containing custom structures and methods. Requres annotation
            of methods to instrument.

        [TruVirtual][trulens_eval.tru_virtual.TruVirtual]: Virtual
            apps that do not have a real app to instrument but have a virtual            structure and can log existing captured data as if they were trulens
            records.

    Args:
        database: Database to use. If not provided, an
            [SQLAlchemyDB][trulens_eval.database.sqlalchemy.SQLAlchemyDB] database
            will be initialized based on the other arguments.

        database_url: Database URL. Defaults to a local SQLite
            database file at `"default.sqlite"` See [this
            article](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls)
            on SQLAlchemy database URLs. (defaults to
            `sqlite://DEFAULT_DATABASE_FILE`).

        database_file: Path to a local SQLite database file.

            **Deprecated**: Use `database_url` instead.

        database_prefix: Prefix for table names for trulens_eval to use. 
            May be useful in some databases hosting other apps.

        database_redact_keys: Whether to redact secret keys in data to be
            written to database (defaults to `False`)

        database_args: Additional arguments to pass to the database constructor.
    """

    RETRY_RUNNING_SECONDS: float = 60.0
    """How long to wait (in seconds) before restarting a feedback function that has already started
    
    A feedback function execution that has started may have stalled or failed in a bad way that did not record the
    failure.

    See also:
        [start_evaluator][trulens_eval.tru.Tru.start_evaluator]

        [DEFERRED][trulens_eval.schema.feedback.FeedbackMode.DEFERRED]
    """

    RETRY_FAILED_SECONDS: float = 5 * 60.0
    """How long to wait (in seconds) to retry a failed feedback function run."""

    DEFERRED_NUM_RUNS: int = 32
    """Number of futures to wait for when evaluating deferred feedback functions."""

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

    def __init__(
        self,
        database: Optional[DB] = None,
        database_url: Optional[str] = None,
        database_file: Optional[str] = None,
        database_redact_keys: Optional[bool] = None,
        database_prefix: Optional[str] = None,
        database_args: Optional[Dict[str, Any]] = None,
        database_check_revision: bool = True,
    ):
        """
        Args:
            database_check_revision: Whether to check the database revision on
                init. This prompt determine whether database migration is required.
        """

        if database_args is None:
            database_args = {}

        database_args.update(
            {
                k: v for k, v in {
                    'database_url': database_url,
                    'database_file': database_file,
                    'database_redact_keys': database_redact_keys,
                    'database_prefix': database_prefix
                }.items() if v is not None
            }
        )

        if python.safe_hasattr(self, "db"):
            # Already initialized by SingletonByName mechanism. Give warning if
            # any option was specified (not None) as it will be ignored.
            if sum((1 if v is not None else 0 for v in database_args.values())
                  ) > 0:
                logger.warning(
                    "Tru was already initialized. "
                    "Cannot change database configuration after initialization."
                )
                self.warning()

            return

        if database is not None:
            if not isinstance(database, DB):
                raise ValueError(
                    "`database` must be a `trulens_eval.database.base.DB` instance."
                )

            self.db = database
        else:
            self.db = sqlalchemy.SQLAlchemyDB.from_tru_args(**database_args)

        if database_check_revision:
            try:
                self.db.check_db_revision()
            except DatabaseVersionException as e:
                print(e)
                self.db = OpaqueWrapper(obj=self.db, e=e)

    def Chain(
        self, chain: langchain.chains.base.Chain, **kwargs: dict
    ) -> trulens_eval.tru_chain.TruChain:
        """Create a langchain app recorder with database managed by self.

        Args:
            chain: The langchain chain defining the app to be instrumented.

            **kwargs: Additional keyword arguments to pass to the
                [TruChain][trulens_eval.tru_chain.TruChain].
        """

        from trulens_eval.tru_chain import TruChain

        return TruChain(tru=self, app=chain, **kwargs)

    def Llama(
        self, engine: Union[llama_index.indices.query.base.BaseQueryEngine,
                            llama_index.chat_engine.types.BaseChatEngine],
        **kwargs: dict
    ) -> trulens_eval.tru_llama.TruLlama:
        """Create a llama-index app recorder with database managed by self.

        Args:
            engine: The llama-index engine defining
                the app to be instrumented.

            **kwargs: Additional keyword arguments to pass to
                [TruLlama][trulens_eval.tru_llama.TruLlama].
        """

        from trulens_eval.tru_llama import TruLlama

        return TruLlama(tru=self, app=engine, **kwargs)

    def Basic(
        self, text_to_text: Callable[[str], str], **kwargs: dict
    ) -> trulens_eval.tru_basic_app.TruBasicApp:
        """Create a basic app recorder with database managed by self.

        Args:
            text_to_text: A function that takes a string and returns a string.
                The wrapped app's functionality is expected to be entirely in
                this function.

            **kwargs: Additional keyword arguments to pass to
                [TruBasicApp][trulens_eval.tru_basic_app.TruBasicApp].
        """

        from trulens_eval.tru_basic_app import TruBasicApp

        return TruBasicApp(tru=self, text_to_text=text_to_text, **kwargs)

    def Custom(
        self, app: Any, **kwargs: dict
    ) -> trulens_eval.tru_custom_app.TruCustomApp:
        """Create a custom app recorder with database managed by self.

        Args:
            app: The app to be instrumented. This can be any python object.

            **kwargs: Additional keyword arguments to pass to
                [TruCustomApp][trulens_eval.tru_custom_app.TruCustomApp].
        """

        from trulens_eval.tru_custom_app import TruCustomApp

        return TruCustomApp(tru=self, app=app, **kwargs)

    def Virtual(
        self, app: Union[trulens_eval.tru_virtual.VirtualApp, Dict],
        **kwargs: dict
    ) -> trulens_eval.tru_virtual.TruVirtual:
        """Create a virtual app recorder with database managed by self.

        Args:
            app: The app to be instrumented. If not a
                [VirtualApp][trulens_eval.tru_virtual.VirtualApp], it is passed
                to [VirtualApp][trulens_eval.tru_virtual.VirtualApp] constructor
                to create it.

            **kwargs: Additional keyword arguments to pass to
                [TruVirtual][trulens_eval.tru_virtual.TruVirtual].
        """

        from trulens_eval.tru_virtual import TruVirtual

        return TruVirtual(tru=self, app=app, **kwargs)

    def reset_database(self):
        """Reset the database. Clears all tables.
        
        See [DB.reset_database][trulens_eval.database.base.DB.reset_database].
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
        created with an older version of _trulens_eval_.

        Args:
            **kwargs: Keyword arguments to pass to
                [migrate_database][trulens_eval.database.base.DB.migrate_database]
                of the current database.

        See [DB.migrate_database][trulens_eval.database.base.DB.migrate_database].
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
        self,
        record: Optional[mod_record_schema.Record] = None,
        **kwargs: dict
    ) -> mod_types_schema.RecordID:
        """Add a record to the database.

        Args:
            record: The record to add.

            **kwargs: [Record][trulens_eval.schema.record.Record] fields to add to the
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

    # TODO: this method is used by app.py, which represents poor code
    # organization.
    def _submit_feedback_functions(
        self,
        record: mod_record_schema.Record,
        feedback_functions: Sequence[feedback.Feedback],
        app: Optional[mod_app_schema.AppDefinition] = None,
        on_done: Optional[Callable[[
            Union[mod_feedback_schema.FeedbackResult,
                  Future[mod_feedback_schema.FeedbackResult]], None
        ]]] = None
    ) -> List[Tuple[feedback.Feedback,
                    Future[mod_feedback_schema.FeedbackResult]]]:
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
            assert app_id == app.app_id, "Record was produced by a different app."

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


            fut: Future[mod_feedback_schema.FeedbackResult] = \
                tp.submit(run_and_call_callback, ffunc=ffunc, app=app, record=record)

            # Have to roll the on_done callback into the submitted function
            # because the result() is returned before callback runs otherwise.
            # We want to do db work before result is returned.

            # if on_done is not None:
            #    fut.add_done_callback(on_done)

            feedbacks_and_futures.append((ffunc, fut))

        return feedbacks_and_futures

    def run_feedback_functions(
        self,
        record: mod_record_schema.Record,
        feedback_functions: Sequence[feedback.Feedback],
        app: Optional[mod_app_schema.AppDefinition] = None,
        wait: bool = True
    ) -> Union[Iterable[mod_feedback_schema.FeedbackResult],
               Iterable[Future[mod_feedback_schema.FeedbackResult]]]:
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
                [FeedbackResult][trulens_eval.schema.feedback.FeedbackResult] if `wait`
                is enabled (default) or [Future][concurrent.futures.Future] of
                [FeedbackResult][trulens_eval.schema.feedback.FeedbackResult] if `wait`
                is disabled.
        """

        if not isinstance(record, mod_record_schema.Record):
            raise ValueError(
                "`record` must be a `trulens_eval.schema.record.Record` instance."
            )

        if not isinstance(feedback_functions, Sequence):
            raise ValueError("`feedback_functions` must be a sequence.")

        if not all(isinstance(ffunc, feedback.Feedback)
                   for ffunc in feedback_functions):
            raise ValueError(
                "`feedback_functions` must be a sequence of `trulens_eval.feedback.feedback.Feedback` instances."
            )

        if not (app is None or isinstance(app, mod_app_schema.AppDefinition)):
            raise ValueError(
                "`app` must be a `trulens_eval.schema.app.AppDefinition` instance."
            )

        if not isinstance(wait, bool):
            raise ValueError("`wait` must be a bool.")

        future_feedback_map: Dict[Future[mod_feedback_schema.FeedbackResult],
                                  feedback.Feedback] = {
                                      p[1]: p[0]
                                      for p in self._submit_feedback_functions(
                                          record=record,
                                          feedback_functions=feedback_functions,
                                          app=app
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
            Union[mod_feedback_schema.FeedbackResult,
                  Future[mod_feedback_schema.FeedbackResult]]] = None,
        **kwargs: dict
    ) -> mod_types_schema.FeedbackResultID:
        """Add a single feedback result or future to the database and return its unique id.
        
        Args:
            feedback_result_or_future: If a [Future][concurrent.futures.Future]
                is given, call will wait for the result before adding it to the
                database. If `kwargs` are given and a
                [FeedbackResult][trulens_eval.schema.feedback.FeedbackResult] is also
                given, the `kwargs` will be used to update the
                [FeedbackResult][trulens_eval.schema.feedback.FeedbackResult] otherwise a
                new one will be created with `kwargs` as arguments to its
                constructor.

            **kwargs: Fields to add to the given feedback result or to create a
                new [FeedbackResult][trulens_eval.schema.feedback.FeedbackResult] with.

        Returns:
            A unique result identifier [str][].

        """

        if feedback_result_or_future is None:
            if 'result' in kwargs and 'status' not in kwargs:
                # If result already present, set status to done.
                kwargs['status'] = mod_feedback_schema.FeedbackResultStatus.DONE

            feedback_result_or_future = mod_feedback_schema.FeedbackResult(
                **kwargs
            )

        else:
            if isinstance(feedback_result_or_future, Future):
                futures.wait([feedback_result_or_future])
                feedback_result_or_future: mod_feedback_schema.FeedbackResult = feedback_result_or_future.result(
                )

            elif isinstance(feedback_result_or_future,
                            mod_feedback_schema.FeedbackResult):
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
        self, feedback_results: Iterable[
            Union[mod_feedback_schema.FeedbackResult,
                  Future[mod_feedback_schema.FeedbackResult]]]
    ) -> List[schema.FeedbackResultID]:
        """Add multiple feedback results to the database and return their unique ids.
        
        Args:
            feedback_results: An iterable with each iteration being a [FeedbackResult][trulens_eval.schema.feedback.FeedbackResult] or
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

        This method produces the JSON-ized version of the app. It can be deserialized back into an [AppDefinition][trulens_eval.schema.app.AppDefinition] with [model_validate][pydantic.BaseModel.model_validate]:
        
        Example:
            ```python
            from trulens_eval.schema import app
            app_json = tru.get_app(app_id="Custom Application v1")
            app = app.AppDefinition.model_validate(app_json)
            ```

        Warning:
            Do not rely on deserializing into [App][trulens_eval.app.App] as
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
            Same Deserialization caveats as [get_app][trulens_eval.tru.Tru.get_app].
        """

        return self.db.get_apps()

    def get_records_and_feedback(
        self,
        app_ids: Optional[List[mod_types_schema.AppID]] = None
    ) -> Tuple[pandas.DataFrame, List[str]]:
        """Get records, their feeback results, and feedback names.
        
        Args:
            app_ids: A list of app ids to filter records by. If empty or not given, all
                apps' records will be returned.

        Returns:
            Dataframe of records with their feedback results.
            
            List of feedback names that are columns in the dataframe.
        """

        if app_ids is None:
            app_ids = []

        df, feedback_columns = self.db.get_records_and_feedback(app_ids)

        return df, feedback_columns

    def get_leaderboard(
        self,
        app_ids: Optional[List[mod_types_schema.AppID]] = None
    ) -> pandas.DataFrame:
        """Get a leaderboard for the given apps.

        Args:
            app_ids: A list of app ids to filter records by. If empty or not given, all
                apps will be included in leaderboard.

        Returns:
            Dataframe of apps with their feedback results aggregated.
        """

        if app_ids is None:
            app_ids = []

        df, feedback_cols = self.db.get_records_and_feedback(app_ids)

        col_agg_list = feedback_cols + ['latency', 'total_cost']

        leaderboard = df.groupby('app_id')[col_agg_list].mean().sort_values(
            by=feedback_cols, ascending=False
        )

        return leaderboard

    def start_evaluator(
        self,
        restart: bool = False,
        fork: bool = False,
        disable_tqdm: bool = False
    ) -> Union[Process, Thread]:
        """
        Start a deferred feedback function evaluation thread or process.

        Args:
            restart: If set, will stop the existing evaluator before starting a
                new one.
            
            fork: If set, will start the evaluator in a new process instead of a
                thread. NOT CURRENTLY SUPPORTED.

            disable_tqdm: If set, will disable progress bar logging from the evaluator.

        Returns:
            The started process or thread that is executing the deferred feedback
                evaluator.

        Relevant constants:
            [RETRY_RUNNING_SECONDS][trulens_eval.tru.Tru.RETRY_RUNNING_SECONDS]

            [RETRY_FAILED_SECONDS][trulens_eval.tru.Tru.RETRY_FAILED_SECONDS]

            [DEFERRED_NUM_RUNS][trulens_eval.tru.Tru.DEFERRED_NUM_RUNS]

            [MAX_THREADS][trulens_eval.utils.threading.TP.MAX_THREADS]
        """

        assert not fork, "Fork mode not yet implemented."

        if self._evaluator_proc is not None:
            if restart:
                self.stop_evaluator()
            else:
                raise RuntimeError(
                    "Evaluator is already running in this process."
                )

        if not fork:
            self._evaluator_stop = threading.Event()

        def runloop():
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
                f"{humanize_seconds(self.RETRY_RUNNING_SECONDS)}."
            )
            print(
                f"Will rerun failed feedbacks after "
                f"{humanize_seconds(self.RETRY_FAILED_SECONDS)}."
            )

            total = 0

            # Getting total counts from the database to start off the tqdm
            # progress bar initial values so that they offer accurate
            # predictions initially after restarting the process.
            queue_stats = self.db.get_feedback_count_by_status()
            queue_done = queue_stats.get(
                mod_feedback_schema.FeedbackResultStatus.DONE
            ) or 0
            queue_total = sum(queue_stats.values())

            # Show the overall counts from the database, not just what has been
            # looked at so far.
            tqdm_status = tqdm(
                desc="Feedback Status",
                initial=queue_done,
                unit="feedbacks",
                total=queue_total,
                postfix={
                    status.name: count for status, count in queue_stats.items()
                },
                disable=disable_tqdm
            )

            # Show the status of the results so far.
            tqdm_total = tqdm(
                desc="Done Runs", initial=0, unit="runs", disable=disable_tqdm
            )

            # Show what is being waited for right now.
            tqdm_waiting = tqdm(
                desc="Waiting for Runs",
                initial=0,
                unit="runs",
                disable=disable_tqdm
            )

            runs_stats = defaultdict(int)

            futures_map: Dict[Future[mod_feedback_schema.FeedbackResult],
                              pandas.Series] = dict()

            while fork or not self._evaluator_stop.is_set():

                if len(futures_map) < self.DEFERRED_NUM_RUNS:
                    # Get some new evals to run if some already completed by now.
                    new_futures: List[Tuple[pandas.Series, Future[mod_feedback_schema.FeedbackResult]]] = \
                        feedback.Feedback.evaluate_deferred(
                            tru=self,
                            limit=self.DEFERRED_NUM_RUNS-len(futures_map),
                            shuffle=True
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

                    tqdm_total.total = total
                    tqdm_total.refresh()

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
                        for fut in futures.as_completed(futures_copy,
                                                        timeout=10):
                            del futures_map[fut]

                            tqdm_waiting.update(-1)
                            tqdm_total.update(1)

                            feedback_result = fut.result()
                            runs_stats[feedback_result.status.name] += 1

                    except futures.TimeoutError:
                        pass

                tqdm_total.set_postfix(
                    {name: count for name, count in runs_stats.items()}
                )

                queue_stats = self.db.get_feedback_count_by_status()
                queue_done = queue_stats.get(
                    mod_feedback_schema.FeedbackResultStatus.DONE
                ) or 0
                queue_total = sum(queue_stats.values())

                tqdm_status.n = queue_done
                tqdm_status.total = queue_total
                tqdm_status.set_postfix(
                    {
                        status.name: count
                        for status, count in queue_stats.items()
                    }
                )

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
                    # Nothing to run/is running, wait a bit.
                    if fork:
                        sleep(10)
                    else:
                        self._evaluator_stop.wait(10)

            print("Evaluator stopped.")

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

    def run_dashboard(
        self,
        port: Optional[int] = 8501,
        address: Optional[str] = None,
        force: bool = False,
        _dev: Optional[Path] = None
    ) -> Process:
        """Run a streamlit dashboard to view logged results and apps.

        Args:
           port: Port number to pass to streamlit through `server.port`.

           address: Address to pass to streamlit through `server.address`.
           
               **Address cannot be set if running from a colab
               notebook.**
        
           force: Stop existing dashboard(s) first. Defaults to `False`.

           _dev: If given, run dashboard with the given
              `PYTHONPATH`. This can be used to run the dashboard from outside
              of its pip package installation folder.

        Returns:
            The [Process][multiprocessing.Process] executing the streamlit
                dashboard.

        Raises:
            RuntimeError: Dashboard is already running. Can be avoided if `force`
                is set.

        """

        IN_COLAB = 'google.colab' in sys.modules
        if IN_COLAB and address is not None:
            raise ValueError("`address` argument cannot be used in colab.")

        if force:
            self.stop_dashboard(force=force)

        print("Starting dashboard ...")

        # Create .streamlit directory if it doesn't exist
        streamlit_dir = os.path.join(os.getcwd(), '.streamlit')
        os.makedirs(streamlit_dir, exist_ok=True)

        # Create config.toml file path
        config_path = os.path.join(streamlit_dir, 'config.toml')

        # Check if the file already exists
        if not os.path.exists(config_path):
            with open(config_path, 'w') as f:
                f.write('[theme]\n')
                f.write('primaryColor="#0A2C37"\n')
                f.write('backgroundColor="#FFFFFF"\n')
                f.write('secondaryBackgroundColor="F5F5F5"\n')
                f.write('textColor="#0A2C37"\n')
                f.write('font="sans serif"\n')
        else:
            print("Config file already exists. Skipping writing process.")

        # Create credentials.toml file path
        cred_path = os.path.join(streamlit_dir, 'credentials.toml')

        # Check if the file already exists
        if not os.path.exists(cred_path):
            with open(cred_path, 'w') as f:
                f.write('[general]\n')
                f.write('email=""\n')
        else:
            print("Credentials file already exists. Skipping writing process.")

        #run leaderboard with subprocess
        leaderboard_path = static_resource('Leaderboard.py')

        if Tru._dashboard_proc is not None:
            print("Dashboard already running at path:", Tru._dashboard_urls)
            return Tru._dashboard_proc

        env_opts = {}
        if _dev is not None:
            env_opts['env'] = os.environ
            env_opts['env']['PYTHONPATH'] = str(_dev)

        args = ["streamlit", "run", "--server.headless=True"]
        if port is not None:
            args.append(f"--server.port={port}")
        if address is not None:
            args.append(f"--server.address={address}")

        args += [
            leaderboard_path, "--", "--database-url",
            self.db.engine.url.render_as_string(hide_password=False),
            "--database-prefix", self.db.table_prefix
        ]

        proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **env_opts
        )

        started = threading.Event()
        tunnel_started = threading.Event()
        if notebook_utils.is_notebook():
            out_stdout, out_stderr = notebook_utils.setup_widget_stdout_stderr()
        else:
            out_stdout = None
            out_stderr = None

        if IN_COLAB:
            tunnel_proc = subprocess.Popen(
                ["npx", "localtunnel", "--port",
                 str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                **env_opts
            )

            def listen_to_tunnel(proc: subprocess.Popen, pipe, out, started):
                while proc.poll() is None:

                    line = pipe.readline()
                    if "url" in line:
                        started.set()
                        line = "Go to this url and submit the ip given here. " + line

                    if out is not None:
                        out.append_stdout(line)

                    else:
                        print(line)

            Tru.tunnel_listener_stdout = Thread(
                target=listen_to_tunnel,
                args=(
                    tunnel_proc, tunnel_proc.stdout, out_stdout, tunnel_started
                )
            )
            Tru.tunnel_listener_stderr = Thread(
                target=listen_to_tunnel,
                args=(
                    tunnel_proc, tunnel_proc.stderr, out_stderr, tunnel_started
                )
            )
            Tru.tunnel_listener_stdout.daemon = True
            Tru.tunnel_listener_stderr.daemon = True
            Tru.tunnel_listener_stdout.start()
            Tru.tunnel_listener_stderr.start()
            if not tunnel_started.wait(timeout=DASHBOARD_START_TIMEOUT
                                      ):  # This might not work on windows.
                raise RuntimeError("Tunnel failed to start in time. ")

        def listen_to_dashboard(proc: subprocess.Popen, pipe, out, started):
            while proc.poll() is None:
                line = pipe.readline()
                if IN_COLAB:
                    if "External URL: " in line:
                        started.set()
                        line = line.replace(
                            "External URL: http://", "Submit this IP Address: "
                        )
                        line = line.replace(f":{port}", "")
                        if out is not None:
                            out.append_stdout(line)
                        else:
                            print(line)
                        Tru._dashboard_urls = line  # store the url when dashboard is started
                else:
                    if "Network URL: " in line:
                        url = line.split(": ")[1]
                        url = url.rstrip()
                        print(f"Dashboard started at {url} .")
                        started.set()
                        Tru._dashboard_urls = line  # store the url when dashboard is started
                    if out is not None:
                        out.append_stdout(line)
                    else:
                        print(line)
            if out is not None:
                out.append_stdout("Dashboard closed.")
            else:
                print("Dashboard closed.")

        Tru.dashboard_listener_stdout = Thread(
            target=listen_to_dashboard,
            args=(proc, proc.stdout, out_stdout, started)
        )
        Tru.dashboard_listener_stderr = Thread(
            target=listen_to_dashboard,
            args=(proc, proc.stderr, out_stderr, started)
        )

        # Purposely block main process from ending and wait for dashboard.
        Tru.dashboard_listener_stdout.daemon = False
        Tru.dashboard_listener_stderr.daemon = False

        Tru.dashboard_listener_stdout.start()
        Tru.dashboard_listener_stderr.start()

        Tru._dashboard_proc = proc

        wait_period = DASHBOARD_START_TIMEOUT
        if IN_COLAB:
            # Need more time to setup 2 processes tunnel and dashboard
            wait_period = wait_period * 3

        # This might not work on windows.
        if not started.wait(timeout=wait_period):
            Tru._dashboard_proc = None
            raise RuntimeError(
                "Dashboard failed to start in time. "
                "Please inspect dashboard logs for additional information."
            )

        return proc

    start_dashboard = run_dashboard

    def stop_dashboard(self, force: bool = False) -> None:
        """
        Stop existing dashboard(s) if running.

        Args:
            force: Also try to find any other dashboard processes not
                started in this notebook and shut them down too.
              
                **This option is not supported under windows.**

        Raises:
             RuntimeError: Dashboard is not running in the current process. Can be avoided with `force`.
        """
        if Tru._dashboard_proc is None:
            if not force:
                raise RuntimeError(
                    "Dashboard not running in this workspace. "
                    "You may be able to shut other instances by setting the `force` flag."
                )

            else:
                if sys.platform.startswith("win"):
                    raise RuntimeError(
                        "Force stop option is not supported on windows."
                    )

                print("Force stopping dashboard ...")
                import os
                import pwd  # PROBLEM: does not exist on windows

                import psutil
                username = pwd.getpwuid(os.getuid())[0]
                for p in psutil.process_iter():
                    try:
                        cmd = " ".join(p.cmdline())
                        if "streamlit" in cmd and "Leaderboard.py" in cmd and p.username(
                        ) == username:
                            print(f"killing {p}")
                            p.kill()
                    except Exception as e:
                        continue

        else:
            Tru._dashboard_proc.kill()
            Tru._dashboard_proc = None
