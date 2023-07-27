import logging
from multiprocessing import Process
import os
from pathlib import Path
import subprocess
import sys
import threading
from threading import Thread
from time import sleep
from typing import Iterable, List, Optional, Sequence, Union

import pkg_resources

from trulens_eval.db import JSON
from trulens_eval.db import LocalSQLite
from trulens_eval.feedback import Feedback
from trulens_eval.schema import AppDefinition
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import Record
from trulens_eval.util import SingletonPerName
from trulens_eval.util import TP
from trulens_eval.utils.notebook_utils import is_notebook
from trulens_eval.utils.notebook_utils import setup_widget_stdout_stderr
from trulens_eval.utils.text import UNICODE_CHECK
from trulens_eval.utils.text import UNICODE_YIELD

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

    All data is logged to the current working directory to default.sqlite.
    """
    DEFAULT_DATABASE_FILE = "default.sqlite"

    # Process or Thread of the deferred feedback function evaluator.
    evaluator_proc = None

    # Process of the dashboard app.
    dashboard_proc = None

    def Chain(self, chain, **kwargs):
        """
        Create a TruChain with database managed by self.
        """

        from trulens_eval.tru_chain import TruChain

        return TruChain(tru=self, app=chain, **kwargs)

    def Llama(self, engine, **kwargs):
        """
        Create a llama_index engine with database managed by self.
        """

        from trulens_eval.tru_llama import TruLlama

        return TruLlama(tru=self, app=engine, **kwargs)

    def __init__(self):
        """
        TruLens instrumentation, logging, and feedback functions for apps.
        Creates a local database 'default.sqlite' in current working directory.
        """

        if hasattr(self, "db"):
            # Already initialized by SingletonByName mechanism.
            return

        self.db = LocalSQLite(filename=Path(Tru.DEFAULT_DATABASE_FILE))

    def reset_database(self):
        """
        Reset the database. Clears all tables.
        """

        self.db.reset_database()

    def migrate_database(self):
        """
        Migrates the database. 
        """

        self.db.migrate_database()

    def add_record(self, record: Optional[Record] = None, **kwargs):
        """
        Add a record to the database.

        Parameters:
        
        - record: Record

        - **kwargs: Record fields.
            
        Returns:
            RecordID: Unique record identifier.

        """

        if record is None:
            record = Record(**kwargs)
        else:
            record.update(**kwargs)

        return self.db.insert_record(record=record)

    def run_feedback_functions(
        self,
        record: Record,
        feedback_functions: Sequence[Feedback],
        app: Optional[AppDefinition] = None,
    ) -> Sequence[JSON]:
        """
        Run a collection of feedback functions and report their result.

        Parameters:

            record (Record): The record on which to evaluate the feedback
            functions.

            app (App, optional): The app that produced the given record.
            If not provided, it is looked up from the given database `db`.

            feedback_functions (Sequence[Feedback]): A collection of feedback
            functions to evaluate.

        Returns nothing.
        """

        app_id = record.app_id

        if app is None:
            app = self.db.get_app(app_id=app_id)
            if app is None:
                raise RuntimeError(
                    "App {app_id} not present in db. "
                    "Either add it with `tru.add_app` or provide `app_json` to `tru.run_feedback_functions`."
                )

        else:
            assert app_id == app.app_id, "Record was produced by a different app."

            if self.db.get_app(app_id=app.app_id) is None:
                logger.warn(
                    "App {app_id} was not present in database. Adding it."
                )
                self.add_app(app=app)

        evals = []

        for func in feedback_functions:
            evals.append(
                TP().promise(lambda f: f.run(app=app, record=record), func)
            )

        evals = map(lambda p: p.get(), evals)

        return list(evals)

    def add_app(self, app: AppDefinition) -> None:
        """
        Add a app to the database.        
        """

        self.db.insert_app(app=app)

    def add_feedback(
        self, feedback_result: FeedbackResult = None, **kwargs
    ) -> None:
        """
        Add a single feedback result to the database.
        """

        if feedback_result is None:
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

        # TODO: unserialize
        return self.db.get_app(app_id)

    def get_records_and_feedback(self, app_ids: List[str]):
        """
        Get records, their feeback results, and feedback names from the database.
        """

        df, feedback_columns = self.db.get_records_and_feedback(app_ids)

        return df, feedback_columns

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

        from trulens_eval.feedback import Feedback

        if not fork:
            self.evaluator_stop = threading.Event()

        def runloop():
            while fork or not self.evaluator_stop.is_set():
                #print(
                #    "Looking for things to do. Stop me with `tru.stop_evaluator()`.",
                #    end=''
                #)
                started_count = Feedback.evaluate_deferred(tru=self)

                if started_count > 0:
                    print(
                        f"{UNICODE_YIELD}{UNICODE_YIELD}{UNICODE_YIELD} Started {started_count} deferred feedback functions."
                    )
                    TP().finish()
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

        if Tru.dashboard_proc is not None:
            raise ValueError(
                "Dashboard already running. "
                "Run tru.stop_dashboard() to stop existing dashboard."
            )

        print("Starting dashboard ...")

        # Create .streamlit directory if it doesn't exist
        streamlit_dir = os.path.join(os.getcwd(), '.streamlit')
        os.makedirs(streamlit_dir, exist_ok=True)

        # Create config.toml file
        config_path = os.path.join(streamlit_dir, 'config.toml')
        with open(config_path, 'w') as f:
            f.write('[theme]\n')
            f.write('primaryColor="#0A2C37"\n')
            f.write('backgroundColor="#FFFFFF"\n')
            f.write('secondaryBackgroundColor="F5F5F5"\n')
            f.write('textColor="#0A2C37"\n')
            f.write('font="sans serif"\n')

        cred_path = os.path.join(streamlit_dir, 'credentials.toml')
        with open(cred_path, 'w') as f:
            f.write('[general]\n')
            f.write('email=""\n')

        #run leaderboard with subprocess
        leaderboard_path = pkg_resources.resource_filename(
            'trulens_eval', 'Leaderboard.py'
        )

        env_opts = {}
        if _dev is not None:
            env_opts['env'] = os.environ
            env_opts['env']['PYTHONPATH'] = str(_dev)

        proc = subprocess.Popen(
            ["streamlit", "run", "--server.headless=True", leaderboard_path],
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
                else:
                    if "Network URL: " in line:
                        url = line.split(": ")[1]
                        url = url.rstrip()
                        print(f"Dashboard started at {url} .")
                        started.set()
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
