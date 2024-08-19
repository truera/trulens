from __future__ import annotations

from collections import defaultdict
from concurrent import futures
from datetime import datetime
import logging
from multiprocessing import Process
import queue
import threading
from threading import Thread
from time import sleep
from typing import (
    Any,
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
from trulens.core.schema import app as mod_app_schema
from trulens.core.schema import feedback as mod_feedback_schema
from trulens.core.schema import record as mod_record_schema
from trulens.core.schema import types as mod_types_schema
from trulens.core.utils import python
from trulens.core.utils import serial
from trulens.core.utils import threading as tru_threading
from trulens.core.utils.imports import REQUIREMENT_SNOWFLAKE
from trulens.core.utils.imports import OptionalImports
from trulens.core.utils.python import Future  # code style exception
from trulens.core.utils.text import format_seconds
from trulens.core.workspace import BaseWorkspace

tqdm = None
with OptionalImports(messages=REQUIREMENT_SNOWFLAKE):
    from tqdm import tqdm

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
        workspace: Workspace to use. If not provided, a default
            [DefaultWorkspace][trulens.core.workspace.default.DefaultWorkspace]
            is created.
    """

    RETRY_RUNNING_SECONDS: float = 60.0
    """How long to wait (in seconds) before restarting a feedback function that
    has already started.

    A feedback function execution that has started may have stalled or failed in
    a bad way that did not record the failure.

    See also:
        [start_evaluator][trulens.core.tru.Tru.start_evaluator]

        [DEFERRED][trulens.core.schema.feedback.FeedbackMode.DEFERRED]
    """

    RETRY_FAILED_SECONDS: float = 5 * 60.0
    """How long to wait (in seconds) to retry a failed feedback function run."""

    DEFERRED_NUM_RUNS: int = 32
    """Number of futures to wait for when evaluating deferred feedback functions."""

    RECORDS_BATCH_TIMEOUT: int = 10
    """Time to wait before inserting a batch of records into the database."""

    _dashboard_urls: Optional[str] = None

    _evaluator_proc: Optional[Union[Process, Thread]] = None
    """[Process][multiprocessing.Process] or [Thread][threading.Thread] of the
    deferred feedback evaluator if started.

        Is set to `None` if evaluator is not running.
    """

    _dashboard_proc: Optional[Process] = None
    """[Process][multiprocessing.Process] executing the dashboard streamlit app.

    Is set to `None` if not executing.
    """

    _evaluator_stop: Optional[threading.Event] = None
    """Event for stopping the deferred evaluator which runs in another thread."""

    batch_record_queue = queue.Queue()

    batch_thread = None

    workspace: Optional[BaseWorkspace] = None

    def __new__(cls, *args, **kwargs) -> Tru:
        inst = super().__new__(cls, *args, **kwargs)
        assert isinstance(inst, Tru)
        return inst

    def __init__(self, workspace: Optional[BaseWorkspace] = None, **kwargs):
        """Create a new Tru instance with the given workspace.

        If workspace is not given, a workspace is selected based on the other
        arguments.
        """

        self.workspace = workspace or BaseWorkspace.from_tru_args(**kwargs)

    def reset_database(self):
        """Reset the database. Clears all tables.

        See [DB.reset_database][trulens.core.database.base.DB.reset_database].
        """
        self.workspace.reset_database()

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
        self.workspace.migrate_database(**kwargs)

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
        return self.workspace.add_record(record=record, **kwargs)

    def add_record_nowait(
        self,
        record: mod_record_schema.Record,
    ) -> None:
        """Add a record to the queue to be inserted in the next batch."""
        return self.workspace.add_record_nowait(record)

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
        return self.workspace.run_feedback_functions(
            record=record,
            feedback_functions=feedback_functions,
            app=app,
            wait=wait,
        )

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
        return self.workspace.add_app(app=app)

    def delete_app(self, app_id: mod_types_schema.AppID) -> None:
        """
        Deletes an app from the database based on its app_id.

        Args:
            app_id (schema.AppID): The unique identifier of the app to be deleted.
        """
        return self.workspace.delete_app(app_id=app_id)

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
        return self.workspace.add_feedback(
            feedback_result_or_future=feedback_result_or_future, **kwargs
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
        return self.workspace.add_feedbacks(feedback_results=feedback_results)

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

        return self.workspace.get_app(app_id)

    def get_apps(self) -> List[serial.JSONized[mod_app_schema.AppDefinition]]:
        """Look up all apps from the database.

        Returns:
            A list of JSON-ized version of all apps in the database.

        Warning:
            Same Deserialization caveats as [get_app][trulens.core.tru.Tru.get_app].
        """

        return self.workspace.get_apps()

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
        return self.workspace.get_records_and_feedback(
            app_ids=app_ids, offset=offset, limit=limit
        )

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
        return self.workspace.get_leaderboard(
            app_ids=app_ids, group_by_metadata_key=group_by_metadata_key
        )

    def start_evaluator(
        self,
        restart: bool = False,
        fork: bool = False,
        disable_tqdm: bool = False,
        run_location: Optional[mod_feedback_schema.FeedbackRunLocation] = None,
    ) -> Union[Process, Thread]:
        """
        Start a deferred feedback function evaluation thread or process.

        Args:
            restart: If set, will stop the existing evaluator before starting a
                new one.

            fork: If set, will start the evaluator in a new process instead of a
                thread. NOT CURRENTLY SUPPORTED.

            disable_tqdm: If set, will disable progress bar logging from the evaluator.

            run_location: Run only the evaluations corresponding to run_location.

        Returns:
            The started process or thread that is executing the deferred feedback
                evaluator.

        Relevant constants:
            [RETRY_RUNNING_SECONDS][trulens.core.tru.Tru.RETRY_RUNNING_SECONDS]

            [RETRY_FAILED_SECONDS][trulens.core.tru.Tru.RETRY_FAILED_SECONDS]

            [DEFERRED_NUM_RUNS][trulens.core.tru.Tru.DEFERRED_NUM_RUNS]

            [MAX_THREADS][trulens.core.utils.threading.TP.MAX_THREADS]
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
                queue_stats = self.workspace.db.get_feedback_count_by_status()
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

                    queue_stats = (
                        self.workspace.db.get_feedback_count_by_status()
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
