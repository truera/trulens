from datetime import datetime
import logging
from multiprocessing import Process
import os
from pathlib import Path
import subprocess
from threading import Thread
import threading
from time import sleep
from typing import Iterable, List, Optional, Sequence, Union

import pkg_resources

from trulens_eval.tru_db import JSON
from trulens_eval.tru_db import LocalSQLite
from trulens_eval.tru_feedback import Feedback
from trulens_eval.schema import FeedbackResult, Model, Record
from trulens_eval.util import TP, SingletonPerName


class Tru(SingletonPerName):
    """
    Tru is the main class that provides an entry points to trulens-eval. Tru lets you:

    * Log chain prompts and outputs
    * Log chain Metadata
    * Run and log feedback functions
    * Run streamlit dashboard to view experiment results

    All data is logged to the current working directory to default.sqlite.
    """
    DEFAULT_DATABASE_FILE = "default.sqlite"

    # Process or Thread of the deferred feedback function evaluator.
    evaluator_proc = None

    # Process of the dashboard app.
    dashboard_proc = None

    def Chain(self, **kwargs):
        """
        Create a TruChain with database managed by self.
        """

        from trulens_eval.tru_chain import TruChain

        return TruChain(tru=self, **kwargs)

    def __init__(self):
        """
        TruLens instrumentation, logging, and feedback functions for chains.
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
        chain: Optional[Model] = None,
    ) -> Sequence[JSON]:
        """
        Run a collection of feedback functions and report their result.

        Parameters:

            record_json (JSON): The record on which to evaluate the feedback
            functions.

            chain_json (JSON, optional): The chain that produced the given record.
            If not provided, it is looked up from the given database `db`.

            feedback_functions (Sequence[Feedback]): A collection of feedback
            functions to evaluate.

        Returns nothing.
        """

        chain_id = record.chain_id

        if chain is None:
            chain = self.db.get_chain(chain_id=chain_id)
            if chain is None:
                raise RuntimeError(
                    "Chain {chain_id} not present in db. "
                    "Either add it with `tru.add_chain` or provide `chain_json` to `tru.run_feedback_functions`."
                )

        else:
            assert chain_id == chain.chain_id, "Record was produced by a different chain."

            if self.db.get_chain(chain_id=chain.chain_id) is None:
                logger.warn(
                    "Chain {chain_id} was not present in database. Adding it."
                )
                self.add_chain(chain=chain)

        evals = []

        for func in feedback_functions:
            evals.append(
                TP().promise(
                    lambda f: f.run_on_record(chain=chain, record=record), func
                )
            )

        evals = map(lambda p: p.get(), evals)

        return list(evals)

    def add_chain(self, chain: Model) -> None:
        """
        Add a chain to the database.        
        """

        self.db.insert_chain(chain=chain)

    def add_feedback(self, result: FeedbackResult, **kwargs) -> None:
        """
        Add a single feedback result to the database.
        """

        if result is None:
            result = FeedbackResult(**kwargs)
        else:
            result.update(**kwargs)

        self.db.insert_feedback(result=result)

    def add_feedbacks(self, results: Iterable[FeedbackResult]) -> None:
        """
        Add multiple feedback results to the database.
        """

        for result in results:
            self.add_feedback(result=result)

    def get_chain(self, chain_id: Optional[str] = None) -> JSON:
        """
        Look up a chain from the database.
        """

        # TODO: unserialize
        return self.db.get_chain(chain_id)

    def get_records_and_feedback(self, chain_ids: List[str]):
        """
        Get records, their feeback results, and feedback names from the database.
        """

        df, feedback_columns = self.db.get_records_and_feedback(chain_ids)

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

        from trulens_eval.tru_feedback import Feedback

        if not fork:
            self.evaluator_stop = threading.Event()

        def runloop():
            while fork or not self.evaluator_stop.is_set():
                print(
                    "Looking for things to do. Stop me with `tru.stop_evaluator()`.",
                    end=''
                )
                Feedback.evaluate_deferred(tru=self)
                TP().finish(timeout=10)
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
        """Stop existing dashboard if running.

        Raises:
            ValueError: Dashboard is already running.
        """
        if Tru.dashboard_proc is None:
            if not force:
                raise ValueError(
                    "Dashboard not running in this workspace. "
                    "You may be able to shut other instances by setting the `force` flag."
                )

            else:
                print("Force stopping dashboard ...")
                import psutil
                import pwd
                import os
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
        self, force: bool, _dev: Optional[Path] = None
    ) -> Process:
        """ Runs a streamlit dashboard to view logged results and chains

        Raises:
            ValueError: Dashboard is already running.

        Returns:
            Process: Process containing streamlit dashboard.
        """

        if force:
            self.stop_dashboard(force=force)

        if Tru.dashboard_proc is not None:
            raise ValueError(
                "Dashboard already running. Run tru.stop_dashboard() to stop existing dashboard."
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
            **env_opts
        )

        Tru.dashboard_proc = proc

        return proc

    start_dashboard = run_dashboard
