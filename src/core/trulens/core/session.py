from __future__ import annotations

from collections import defaultdict
from concurrent import futures
from datetime import datetime
import inspect
import logging
from multiprocessing import Process
import threading
from threading import Thread
from time import sleep
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import warnings

import pandas
import pydantic
from trulens.core import feedback
from trulens.core._utils import optional as optional_utils
from trulens.core.database.connector import DBConnector
from trulens.core.database.connector import DefaultDBConnector
from trulens.core.schema import app as mod_app_schema
from trulens.core.schema import dataset as mod_dataset_schema
from trulens.core.schema import feedback as mod_feedback_schema
from trulens.core.schema import groundtruth as mod_groundtruth_schema
from trulens.core.schema import record as mod_record_schema
from trulens.core.schema import types as mod_types_schema
from trulens.core.utils import deprecation as deprecation_utils
from trulens.core.utils import imports as import_utils
from trulens.core.utils import python
from trulens.core.utils import serial
from trulens.core.utils import text as text_utils
from trulens.core.utils import threading as tru_threading
from trulens.core.utils.imports import OptionalImports
from trulens.core.utils.python import Future  # code style exception
from trulens.core.utils.text import format_seconds

if TYPE_CHECKING:
    from trulens.core import app as base_app

tqdm = None
with OptionalImports(messages=optional_utils.REQUIREMENT_SNOWFLAKE):
    from tqdm import tqdm

logger = logging.getLogger(__name__)


class TruSession(pydantic.BaseModel, python.SingletonPerName):
    """TruSession is the main class that provides an entry points to trulens.

    TruSession lets you:

    - Log app prompts and outputs
    - Log app Metadata
    - Run and log feedback functions
    - Run streamlit dashboard to view experiment results

    By default, all data is logged to the current working directory to
    `"default.sqlite"`. Data can be logged to a SQLAlchemy-compatible url
    referred to by `database_url`.

    Supported App Types:
        [TruChain][trulens.apps.langchain.TruChain]: Langchain
            apps.

        [TruLlama][trulens.apps.llamaindex.TruLlama]: Llama Index
            apps.

        [TruRails][trulens.apps.nemo.TruRails]: NeMo Guardrails apps.

        [TruBasicApp][trulens.apps.basic.TruBasicApp]:
            Basic apps defined solely using a function from `str` to `str`.

        [TruCustomApp][trulens.apps.custom.TruCustomApp]:
            Custom apps containing custom structures and methods. Requires
            annotation of methods to instrument.

        [TruVirtual][trulens.apps.virtual.TruVirtual]: Virtual
            apps that do not have a real app to instrument but have a virtual
            structure and can log existing captured data as if they were trulens
            records.

    Args:
        connector: Database Connector to use. If not provided, a default
            [DefaultDBConnector][trulens.core.database.connector.default.DefaultDBConnector]
            is created.

        **kwargs: All other arguments are used to initialize
            [DefaultDBConnector][trulens.core.database.connector.default.DefaultDBConnector].
            Mutually exclusive with `connector`.
    """

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True, extra="forbid"
    )

    RETRY_RUNNING_SECONDS: float = 60.0
    """How long to wait (in seconds) before restarting a feedback function that has already started

    A feedback function execution that has started may have stalled or failed in a bad way that did not record the
    failure.

    See also:
        [start_evaluator][trulens.core.session.TruSession.start_evaluator]

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

    _evaluator_stop: Optional[threading.Event] = pydantic.PrivateAttr(None)
    """Event for stopping the deferred evaluator which runs in another thread."""

    _evaluator_proc: Optional[Union[Process, Thread]] = pydantic.PrivateAttr(
        None
    )

    _dashboard_urls: Optional[str] = pydantic.PrivateAttr(None)

    _dashboard_proc: Optional[Process] = pydantic.PrivateAttr(None)

    _tunnel_listener_stdout: Optional[Thread] = pydantic.PrivateAttr(None)

    _tunnel_listener_stderr: Optional[Thread] = pydantic.PrivateAttr(None)

    _dashboard_listener_stdout: Optional[Thread] = pydantic.PrivateAttr(None)

    _dashboard_listener_stderr: Optional[Thread] = pydantic.PrivateAttr(None)

    connector: Optional[DBConnector] = pydantic.Field(None, exclude=True)

    def __new__(cls, *args, **kwargs: Any) -> TruSession:
        inst = super().__new__(cls, *args, **kwargs)
        assert isinstance(inst, TruSession)
        return inst

    def __init__(self, connector: Optional[DBConnector] = None, **kwargs):
        if python.safe_hasattr(self, "connector"):
            # Already initialized by SingletonByName mechanism. Give warning if
            # any option was specified (not None) as it will be ignored.
            if connector is not None:
                logger.warning(
                    "Tru was already initialized. Cannot change database configuration after initialization."
                )
                self.warning()
            return
        connector_args = {
            k: v
            for k, v in kwargs.items()
            if k in inspect.signature(DefaultDBConnector.__init__).parameters
        }
        self_args = {k: v for k, v in kwargs.items() if k not in connector_args}

        if connector_args and connector is not None:
            extra_keys = ", ".join(
                map(lambda s: "`" + s + "`", connector_args.keys())
            )
            raise ValueError(
                f"Cannot provide both `connector` and connector argument(s) {extra_keys}."
            )

        super().__init__(
            connector=connector or DefaultDBConnector(**connector_args),
            **self_args,
        )

    def App(self, *args, app: Optional[Any] = None, **kwargs) -> base_app.App:
        """Create an App from the given App constructor arguments by guessing
        which app type they refer to.

        This method intentionally prints out the type of app being created to
        let user know in case the guess is wrong.
        """
        if app is None:
            # If app is not given as a keyword argument, check the positional args.

            if len(args) == 0:
                # Basic app can be specified using the text_to_text key argument.
                if "text_to_text" in kwargs:
                    from trulens.apps import basic

                    return basic.TruBasicApp(
                        *args, connector=self.connector, **kwargs
                    )

                raise ValueError("No app provided.")

            # Otherwise the app must be the first positional arg.
            app, args = args[0], args[1:]

        # Check for optional app types.
        if app.__module__.startswith("langchain"):
            with import_utils.OptionalImports(
                messages=optional_utils.REQUIREMENT_INSTRUMENT_LANGCHAIN
            ):
                from trulens.apps.langchain import tru_chain

            print(f"{text_utils.UNICODE_SQUID} Instrumenting LangChain app.")
            return tru_chain.TruChain(
                *args, app=app, connector=self.connector, **kwargs
            )

        elif app.__module__.startswith("llamaindex"):
            with import_utils.OptionalImports(
                messages=optional_utils.REQUIREMENT_INSTRUMENT_LLAMA
            ):
                from trulens.apps.llamaindex import tru_llama

            print(f"{text_utils.UNICODE_SQUID} Instrumenting LlamaIndex app.")
            return tru_llama.TruLlama(
                *args, app=app, connector=self.connector, **kwargs
            )

        elif app.__module__.startswith("nemoguardrails"):
            with import_utils.OptionalImports(
                messages=optional_utils.REQUIREMENT_INSTRUMENT_NEMO
            ):
                from trulens.apps.nemo import tru_rails

            print(
                f"{text_utils.UNICODE_SQUID} Instrumenting NeMo GuardRails app."
            )

            return tru_rails.TruRails(
                *args, app=app, connector=self.connector, **kwargs
            )

        # Check for virtual. Either VirtualApp or JSON app arg.
        from trulens.apps import virtual
        from trulens.core.utils import serial as serial_utils

        if isinstance(app, virtual.VirtualApp) or serial_utils.is_json(app):
            print(f"{text_utils.UNICODE_SQUID} Instrumenting virtual app.")
            return virtual.TruVirtual(
                *args, app=app, connector=self.connector, **kwargs
            )

        # Check for basic. Either TruWrapperApp or the text_to_text arg. Unsure
        # what we want to do if they provide both. Let's TruBasicApp handle it.
        from trulens.apps import basic

        if isinstance(app, basic.TruWrapperApp) or "text_to_text" in kwargs:
            print(f"{text_utils.UNICODE_SQUID} Instrumenting basic app.")

            return basic.TruBasicApp(
                *args, app=app, connector=self.connector, **kwargs
            )

        # If all else fails, assume it is a custom app.
        print(f"{text_utils.UNICODE_SQUID} Instrumenting custom app.")
        from trulens.apps import custom

        return custom.TruCustomApp(
            *args, app=app, connector=self.connector, **kwargs
        )

    @deprecation_utils.method_renamed("TruSession.App")
    def Basic(self, *args, **kwargs) -> base_app.App:
        from trulens.apps.basic import TruBasicApp

        return TruBasicApp(*args, connector=self.connector, **kwargs)

    @deprecation_utils.method_renamed("TruSession.App")
    def Custom(self, *args, **kwargs) -> base_app.App:
        from trulens.apps.custom import TruCustomApp

        return TruCustomApp(*args, connector=self.connector, **kwargs)

    @deprecation_utils.method_renamed("TruSession.App")
    def Virtual(self, *args, **kwargs) -> base_app.App:
        from trulens.apps.virtual import TruVirtual

        return TruVirtual(*args, connector=self.connector, **kwargs)

    @deprecation_utils.method_renamed("TruSession.App")
    def Chain(self, *args, **kwargs) -> base_app.App:
        with import_utils.OptionalImports(
            messages=optional_utils.REQUIREMENT_INSTRUMENT_LANGCHAIN
        ):
            from trulens.apps.langchain import tru_chain

        return tru_chain.TruChain(*args, connector=self.connector, **kwargs)

    @deprecation_utils.method_renamed("TruSession.App")
    def Llama(self, *args, **kwargs) -> base_app.App:
        with import_utils.OptionalImports(
            messages=optional_utils.REQUIREMENT_INSTRUMENT_LLAMA
        ):
            from trulens.apps.llamaindex import tru_llama

        return tru_llama.TruLlama(*args, connector=self.connector, **kwargs)

    @deprecation_utils.method_renamed("TruSession.App")
    def Rails(self, *args, **kwargs) -> base_app.App:
        with import_utils.OptionalImports(
            messages=optional_utils.REQUIREMENT_INSTRUMENT_NEMO
        ):
            from trulens.apps.nemo import tru_rails

        return tru_rails.TruRails(*args, connector=self.connector, **kwargs)

    @deprecation_utils.method_renamed("trulens.dashboard.run.find_unused_port")
    def find_unused_port(self, *args, **kwargs):
        from trulens.dashboard.run import find_unused_port

        return find_unused_port(*args, **kwargs)

    @deprecation_utils.method_renamed("trulens.dashboard.run.run_dashboard")
    def run_dashboard(self, *args, **kwargs):
        from trulens.dashboard.run import run_dashboard

        return run_dashboard(*args, session=self, **kwargs)

    @deprecation_utils.method_renamed("trulens.dashboard.run.run_dashboard")
    def start_dashboard(self, *args, **kwargs):
        from trulens.dashboard.run import run_dashboard

        return run_dashboard(*args, session=self, **kwargs)

    @deprecation_utils.method_renamed("trulens.dashboard.run.stop_dashboard")
    def stop_dashboard(self, *args, **kwargs):
        from trulens.dashboard.run import stop_dashboard

        return stop_dashboard(*args, session=self, **kwargs)

    @deprecation_utils.method_renamed("TruSession.connector.db.insert_record")
    def update_record(self, *args, **kwargs):
        assert self.connector is not None
        return self.connector.db.insert_record(*args, **kwargs)

    def reset_database(self):
        """Reset the database. Clears all tables.

        See [DB.reset_database][trulens.core.database.base.DB.reset_database].
        """
        self.connector.reset_database()

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
        self.connector.migrate_database(**kwargs)

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
        return self.connector.add_record(record=record, **kwargs)

    def add_record_nowait(
        self,
        record: mod_record_schema.Record,
    ) -> None:
        """Add a record to the queue to be inserted in the next batch."""
        return self.connector.add_record_nowait(record)

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
                connector=self.connector,
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
        return self.connector.add_app(app=app)

    def delete_app(self, app_id: mod_types_schema.AppID) -> None:
        """
        Deletes an app from the database based on its app_id.

        Args:
            app_id (schema.AppID): The unique identifier of the app to be deleted.
        """
        return self.connector.delete_app(app_id=app_id)

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
        return self.connector.add_feedback(
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
        return self.connector.add_feedbacks(feedback_results=feedback_results)

    def get_app(
        self, app_id: mod_types_schema.AppID
    ) -> Optional[serial.JSONized[mod_app_schema.AppDefinition]]:
        """Look up an app from the database.

        This method produces the JSON-ized version of the app. It can be deserialized back into an [AppDefinition][trulens.core.schema.app.AppDefinition] with [model_validate][pydantic.BaseModel.model_validate]:

        Example:
            ```python
            from trulens.core.schema import app
            app_json = session.get_app(app_id="app_hash_85ebbf172d02e733c8183ac035d0cbb2")
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

        return self.connector.get_app(app_id)

    def get_apps(self) -> List[serial.JSONized[mod_app_schema.AppDefinition]]:
        """Look up all apps from the database.

        Returns:
            A list of JSON-ized version of all apps in the database.

        Warning:
            Same Deserialization caveats as [get_app][trulens.core.session.TruSession.get_app].
        """

        return self.connector.get_apps()

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
        return self.connector.get_records_and_feedback(
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
        return self.connector.get_leaderboard(
            app_ids=app_ids, group_by_metadata_key=group_by_metadata_key
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
                self.connector.db.batch_insert_ground_truth(buffer)
                buffer.clear()

        # remaining ground truths in the buffer
        if buffer:
            self.connector.db.batch_insert_ground_truth(buffer)

    def get_ground_truth(self, dataset_name: str) -> pandas.DataFrame:
        """Get ground truth data from the dataset.
        dataset_name: Name of the dataset.
        """

        return self.connector.db.get_ground_truths_by_dataset(dataset_name)

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
            [RETRY_RUNNING_SECONDS][trulens.core.session.TruSession.RETRY_RUNNING_SECONDS]

            [RETRY_FAILED_SECONDS][trulens.core.session.TruSession.RETRY_FAILED_SECONDS]

            [DEFERRED_NUM_RUNS][trulens.core.session.TruSession.DEFERRED_NUM_RUNS]

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

            # TODO: a lot of the time we say `if tqdm`, but shouldn't we say `if not disable_tqdm`?
            if tqdm:
                # Getting total counts from the database to start off the tqdm
                # progress bar initial values so that they offer accurate
                # predictions initially after restarting the process.
                queue_stats = self.connector.db.get_feedback_count_by_status()
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
                        limit=self.DEFERRED_NUM_RUNS - len(futures_map),
                        shuffle=True,
                        session=self,
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
                        self.connector.db.get_feedback_count_by_status()
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


def Tru(*args, **kwargs) -> TruSession:
    warnings.warn(
        "Tru is deprecated, use TruSession instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return TruSession(*args, **kwargs)
