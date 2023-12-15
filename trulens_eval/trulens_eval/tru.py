from concurrent.futures import as_completed
from concurrent.futures import wait
import logging
from multiprocessing import Process
import os
from pathlib import Path
import subprocess
import sys
import threading
from threading import Thread
from time import sleep
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union
import warnings

import pkg_resources

from trulens_eval.database.sqlalchemy_db import SqlAlchemyDB
from trulens_eval.db import DB
from trulens_eval.db import JSON
from trulens_eval.feedback import Feedback
from trulens_eval.schema import AppDefinition
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import FeedbackResultStatus
from trulens_eval.schema import Record
from trulens_eval.utils.notebook_utils import is_notebook
from trulens_eval.utils.notebook_utils import setup_widget_stdout_stderr
from trulens_eval.utils.python import safe_hasattr
from trulens_eval.utils.python import SingletonPerName
from trulens_eval.utils.text import UNICODE_CHECK
from trulens_eval.utils.text import UNICODE_LOCK
from trulens_eval.utils.text import UNICODE_SQUID
from trulens_eval.utils.text import UNICODE_STOP
from trulens_eval.utils.text import UNICODE_YIELD
from trulens_eval.utils.threading import TP

logger = logging.getLogger(__name__)

# How long to wait (seconds) for streamlit to print out url when starting the
# dashboard.
DASHBOARD_START_TIMEOUT = 30


class Tru(SingletonPerName):
    """
    Tru is the main class that provides an entry points to trulens-eval. Tru lets you:

    * Log app prompts and outputs
    * Log app Metadata
    * Run and log feedback functions
    * Run streamlit dashboard to view experiment results

    By default, all data is logged to the current working directory to `default.sqlite`. 
    Data can be logged to a SQLAlchemy-compatible referred to by `database_url`.
    """
    DEFAULT_DATABASE_FILE = "default.sqlite"

    # Process or Thread of the deferred feedback function evaluator.
    evaluator_proc = None

    # Process of the dashboard app.
    dashboard_proc = None

    def Chain(__tru_self, chain, **kwargs):
        """
        Create a TruChain with database managed by self.
        """

        from trulens_eval.tru_chain import TruChain

        return TruChain(tru=__tru_self, app=chain, **kwargs)

    def Llama(self, engine, **kwargs):
        """
        Create a llama_index engine with database managed by self.
        """

        from trulens_eval.tru_llama import TruLlama

        return TruLlama(tru=self, app=engine, **kwargs)

    def Basic(self, text_to_text, **kwargs):
        from trulens_eval.tru_basic_app import TruBasicApp

        return TruBasicApp(tru=self, text_to_text=text_to_text, **kwargs)

    def Custom(self, app, **kwargs):
        from trulens_eval.tru_custom_app import TruCustomApp

        return TruCustomApp(tru=self, app=app, **kwargs)

    def __init__(
        self,
        database_url: Optional[str] = None,
        database_file: Optional[str] = None,
        database_redact_keys: bool = False
    ):
        """
        TruLens instrumentation, logging, and feedback functions for apps.

        Args:
           database_url: SQLAlchemy database URL. Defaults to a local
                                SQLite database file at 'default.sqlite'
                                See [this article](https://docs.sqlalchemy.org/en/20/core/engines.html#database-urls)
                                on SQLAlchemy database URLs.
           database_file: (Deprecated) Path to a local SQLite database file
           database_redact_keys: whether to redact secret keys in data to be written to database.
        """
        if safe_hasattr(self, "db"):
            if database_url is not None or database_file is not None:
                logger.warning(
                    f"Tru was already initialized. Cannot change database_url={database_url} or database_file={database_file} ."
                )

            # Already initialized by SingletonByName mechanism.
            return

        assert None in (database_url, database_file), \
            "Please specify at most one of `database_url` and `database_file`"

        if database_file:
            warnings.warn(
                "`database_file` is deprecated, use `database_url` instead as in `database_url='sqlite:///filename'.",
                DeprecationWarning,
                stacklevel=2
            )

        if database_url is None:
            database_url = f"sqlite:///{database_file or self.DEFAULT_DATABASE_FILE}"

        self.db: SqlAlchemyDB = SqlAlchemyDB.from_db_url(
            database_url, redact_keys=database_redact_keys
        )

        print(
            f"{UNICODE_SQUID} Tru initialized with db url {self.db.engine.url} ."
        )
        if database_redact_keys:
            print(
                f"{UNICODE_LOCK} Secret keys will not be included in the database."
            )
        else:
            print(
                f"{UNICODE_STOP} Secret keys may be written to the database. "
                "See the `database_redact_keys` option of `Tru` to prevent this."
            )

    def reset_database(self):
        """
        Reset the database. Clears all tables.
        """

        self.db.reset_database()

    def migrate_database(self):
        """
        Migrates the database. This should be run whenever there are breaking
        changes in a database created with an older version of trulens_eval.
        """

        self.db.migrate_database()

    def add_record(self, record: Optional[Record] = None, **kwargs):
        """
        Add a record to the database.

        Args:
        
            record: Record

            **kwargs: Record fields.
            
        Returns:
            RecordID: Unique record identifier.

        """

        if record is None:
            record = Record(**kwargs)
        else:
            record.update(**kwargs)

        return self.db.insert_record(record=record)

    update_record = add_record

    def _submit_feedback_functions(
        self,
        record: Record,
        feedback_functions: Sequence[Feedback],
        app: Optional[AppDefinition] = None,
        on_done: Optional[Callable[['Future[Tuple[Feedback,FeedbackResult]]'],
                                   None]] = None
    ) -> List['Future[Tuple[Feedback,FeedbackResult]]']:
        app_id = record.app_id

        self.db: DB

        if app is None:
            app = AppDefinition.model_validate(self.db.get_app(app_id=app_id))
            if app is None:
                raise RuntimeError(
                    "App {app_id} not present in db. "
                    "Either add it with `tru.add_app` or provide `app_json` to `tru.run_feedback_functions`."
                )

        else:
            assert app_id == app.app_id, "Record was produced by a different app."

            if self.db.get_app(app_id=app.app_id) is None:
                logger.warning(
                    "App {app_id} was not present in database. Adding it."
                )
                self.add_app(app=app)

        futures = []

        tp: TP = TP()

        for ffunc in feedback_functions:
            fut: 'Future[Tuple[Feedback,FeedbackResult]]' = \
                tp.submit(lambda f: (f, f.run(app=app, record=record)), ffunc)

            if on_done is not None:
                fut.add_done_callback(on_done)

            futures.append(fut)

        return futures

    def run_feedback_functions(
        self,
        record: Record,
        feedback_functions: Sequence[Feedback],
        app: Optional[AppDefinition] = None,
    ) -> Iterable[FeedbackResult]:
        """
        Run a collection of feedback functions and report their result.

        Parameters:

            record (Record): The record on which to evaluate the feedback
            functions.

            app (App, optional): The app that produced the given record.
            If not provided, it is looked up from the given database `db`.

            feedback_functions (Sequence[Feedback]): A collection of feedback
            functions to evaluate.

        Yields `FeedbackResult`, one for each element of `feedback_functions`
        potentially in random order.
        """

        for res in as_completed(self._submit_feedback_functions(
                record=record, feedback_functions=feedback_functions, app=app)):

            yield res.result()[1]

    def add_app(self, app: AppDefinition) -> None:
        """
        Add a app to the database.        
        """

        self.db.insert_app(app=app)

    def add_feedback(
        self,
        feedback_result: Optional[FeedbackResult] = None,
        **kwargs
    ) -> None:
        """
        Add a single feedback result to the database.
        """

        if feedback_result is None:
            if 'result' in kwargs and 'status' not in kwargs:
                # If result already present, set status to done.
                kwargs['status'] = FeedbackResultStatus.DONE

            feedback_result = FeedbackResult(**kwargs)
        else:
            feedback_result.update(**kwargs)

        self.db.insert_feedback(feedback_result=feedback_result)

    def add_feedbacks(self, feedback_results: Iterable[FeedbackResult]) -> None:
        """
        Add multiple feedback results to the database.
        """

        for feedback_result in feedback_results:
            self.add_feedback(feedback_result=feedback_result)

    def get_app(self, app_id: Optional[str] = None) -> JSON:
        """
        Look up a app from the database.
        """

        return self.db.get_app(app_id)

    def get_apps(self) -> Iterable[JSON]:
        """
        Look up all apps from the database.
        """

        return self.db.get_apps()

    def get_records_and_feedback(self, app_ids: List[str]):
        """
        Get records, their feeback results, and feedback names from the
        database. Pass an empty list of app_ids to return all.

        ```python
        tru.get_records_and_feedback(app_ids=[])
        ```
        """

        df, feedback_columns = self.db.get_records_and_feedback(app_ids)

        return df, feedback_columns

    def get_leaderboard(self, app_ids: List[str]):
        """
        Get a leaderboard by app id from the
        database. Pass an empty list of app_ids to return all.

        ```python
        tru.get_leaderboard(app_ids=[])
        ```
        """
        df, feedback_cols = self.db.get_records_and_feedback(app_ids)

        col_agg_list = feedback_cols + ['latency', 'total_cost']

        leaderboard = df.groupby('app_id')[col_agg_list].mean().sort_values(
            by=feedback_cols, ascending=False
        )

        return leaderboard

    def start_evaluator(self,
                        restart=False,
                        fork=False) -> Union[Process, Thread]:
        """
        Start a deferred feedback function evaluation thread.
        """

        assert not fork, "Fork mode not yet implemented."

        if self.evaluator_proc is not None:
            if restart:
                self.stop_evaluator()
            else:
                raise RuntimeError(
                    "Evaluator is already running in this process."
                )

        if not fork:
            self.evaluator_stop = threading.Event()

        def runloop():
            assert self.evaluator_stop is not None

            while fork or not self.evaluator_stop.is_set():
                futures = Feedback.evaluate_deferred(tru=self)

                if len(futures) > 0:
                    print(
                        f"{UNICODE_YIELD}{UNICODE_YIELD}{UNICODE_YIELD} Started {len(futures)} deferred feedback functions."
                    )
                    wait(futures)
                    print(
                        f"{UNICODE_CHECK}{UNICODE_CHECK}{UNICODE_CHECK} Finished evaluating deferred feedback functions."
                    )

                if fork:
                    sleep(10)
                else:
                    self.evaluator_stop.wait(10)

            print("Evaluator stopped.")

        if fork:
            proc = Process(target=runloop)
        else:
            proc = Thread(target=runloop)
            proc.daemon = True

        # Start a persistent thread or process that evaluates feedback functions.

        self.evaluator_proc = proc
        proc.start()

        return proc

    def stop_evaluator(self):
        """
        Stop the deferred feedback evaluation thread.
        """

        if self.evaluator_proc is None:
            raise RuntimeError("Evaluator not running this process.")

        if isinstance(self.evaluator_proc, Process):
            self.evaluator_proc.terminate()

        elif isinstance(self.evaluator_proc, Thread):
            self.evaluator_stop.set()
            self.evaluator_proc.join()
            self.evaluator_stop = None

        self.evaluator_proc = None

    def stop_dashboard(self, force: bool = False) -> None:
        """
        Stop existing dashboard(s) if running.

        Args:

            - force: bool: Also try to find any other dashboard processes not
              started in this notebook and shut them down too.

        Raises:

            - ValueError: Dashboard is not running.
        """
        if Tru.dashboard_proc is None:
            if not force:
                raise ValueError(
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
            Tru.dashboard_proc.kill()
            Tru.dashboard_proc = None

    def run_dashboard_in_jupyter(self):
        """
        Experimental approach to attempt to display the dashboard inside a
        jupyter notebook. Relies on the `streamlit_jupyter` package.
        """
        # EXPERIMENTAL
        # TODO: check for jupyter

        logger.warning(
            "Running dashboard inside a notebook is an experimental feature and may not work well."
        )

        from streamlit_jupyter import StreamlitPatcher
        StreamlitPatcher().jupyter()
        from trulens_eval import Leaderboard

        Leaderboard.main()

    def run_dashboard(
        self, force: bool = False, _dev: Optional[Path] = None
    ) -> Process:
        """
        Run a streamlit dashboard to view logged results and apps.

        Args:

            - force: bool: Stop existing dashboard(s) first.

            - _dev: Optional[Path]: If given, run dashboard with the given
              PYTHONPATH. This can be used to run the dashboard from outside of
              its pip package installation folder.

        Raises:

            - ValueError: Dashboard is already running.

        Returns:

            - Process: Process containing streamlit dashboard.
        """

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
        leaderboard_path = pkg_resources.resource_filename(
            'trulens_eval', 'Leaderboard.py'
        )

        if Tru.dashboard_proc is not None:
            print("Dashboard already running at path:", Tru.dashboard_urls)
            return Tru.dashboard_proc

        env_opts = {}
        if _dev is not None:
            env_opts['env'] = os.environ
            env_opts['env']['PYTHONPATH'] = str(_dev)

        proc = subprocess.Popen(
            [
                "streamlit", "run", "--server.headless=True", leaderboard_path,
                "--", "--database-url",
                self.db.engine.url.render_as_string(hide_password=False)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **env_opts
        )

        started = threading.Event()
        tunnel_started = threading.Event()
        if is_notebook():
            out_stdout, out_stderr = setup_widget_stdout_stderr()
        else:
            out_stdout = None
            out_stderr = None

        IN_COLAB = 'google.colab' in sys.modules
        if IN_COLAB:
            tunnel_proc = subprocess.Popen(
                ["npx", "localtunnel", "--port", "8501"],
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
                        line = line.replace(":8501", "")
                        if out is not None:
                            out.append_stdout(line)
                        else:
                            print(line)
                        Tru.dashboard_urls = line  # store the url when dashboard is started
                else:
                    if "Network URL: " in line:
                        url = line.split(": ")[1]
                        url = url.rstrip()
                        print(f"Dashboard started at {url} .")
                        started.set()
                        Tru.dashboard_urls = line  # store the url when dashboard is started
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

        Tru.dashboard_proc = proc

        wait_period = DASHBOARD_START_TIMEOUT
        if IN_COLAB:
            # Need more time to setup 2 processes tunnel and dashboard
            wait_period = wait_period * 3
        if not started.wait(timeout=wait_period
                           ):  # This might not work on windows.
            raise RuntimeError(
                "Dashboard failed to start in time. "
                "Please inspect dashboard logs for additional information."
            )

        return proc

    start_dashboard = run_dashboard
