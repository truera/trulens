"""
Public interfaces.
"""

from datetime import datetime
import logging
import os
import subprocess
from typing import Iterable, List, Optional, Sequence

import pkg_resources

from trulens_eval.tru_db import JSON
from trulens_eval.tru_db import LocalSQLite
from trulens_eval.tru_db import TruDB
from trulens_eval.tru_feedback import Feedback
from trulens_eval.util import TP, SingletonPerName


class Tru(SingletonPerName):
    DEFAULT_DATABASE_FILE = "default.sqlite"

    deferred_feedback_evaluator_started = False

    def __init__(self, db: Optional[TruDB] = None):
        """
        TruLens instrumentation, logging, and feedback functions for chains.
        
        Parameters:
        
            db (TruDB, optional): Target database. Default database is an SQLite
            db at Tru.DEFAULT_DATABASE_FILE if not provided.
        """

        if hasattr(self, "db"):
            # Already initialized by SingletonByName mechanism.
            return

        self.db = db or LocalSQLite(Tru.DEFAULT_DATABASE_FILE)


    def start_deferred_feedback_evaluator(self):
        if self.deferred_feedback_evaluator_started:
            raise RuntimeError("Evaluator is already running in this process.")

        from trulens_eval.tru_feedback import Feedback

        # Start a persistent thread that evaluates feedback functions.
        Feedback.start_evaluator(tru=self)

        self.deferred_feedback_evaluator_started = True


    def add_record(
        self,
        prompt: str,
        response: str,
        record_json: JSON,
        tags: Optional[str] = None,
        ts: Optional[int] = None,
        total_tokens: Optional[int] = None,
        total_cost: Optional[float] = None,
    ):
        """
        Add a record to the database.

        Parameters:

            prompt (str): Chain input or "prompt".

            response (str): Chain output or "response".

            record_json (JSON): Record as produced by `TruChain.call_with_record`.

            tags (str, optional): Additional metadata to include with the record.

            ts (int, optional): Timestamp of record creation.

            total_tokens (int, optional): The number of tokens generated in
            producing the response.

            total_cost (float, optional): The cost of producing the response.

        Returns:
            str: Unique record identifier.

        """
        if not ts:
            ts = datetime.now()

        chain_id = record_json['chain_id']

        record_id = self.db.insert_record(
            chain_id=chain_id,
            input=prompt,
            output=response,
            record_json=record_json,
            ts=ts,
            tags=tags,
            total_tokens=total_tokens,
            total_cost=total_cost
        )

        return record_id


    def run_feedback_functions(
        self,
        record_json: JSON,
        feedback_functions: Sequence['Feedback'],
        chain_json: Optional[JSON] = None,
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

        chain_id = record_json['chain_id']

        if chain_json is None:
            chain_json = self.db.get_chain(chain_id=chain_id)
            if chain_json is None:
                raise RuntimeError(
                    "Chain {chain_id} not present in db. "
                    "Either add it with `tru.add_chain` or provide `chain_json` to `tru.run_feedback_functions`."
                )

        else:
            assert chain_id == chain_json[
                'chain_id'], "Record was produced by a different chain."

            if self.db.get_chain(chain_id=chain_json['chain_id']) is None:
                logging.warn(
                    "Chain {chain_id} was not present in database. Adding it."
                )
                self.add_chain(chain_json=chain_json)

        evals = []

        for func in feedback_functions:
            evals.append(
                TP().promise(
                    lambda f: f.
                    run_on_record(chain_json=chain_json, record_json=record_json),
                    func
                )
            )

        evals = map(lambda p: p.get(), evals)

        return list(evals)


    def add_chain(
        self,
        chain_json: JSON,
        chain_id: Optional[str] = None
    ) -> None:

        self.db.insert_chain(chain_id=chain_id, chain_json=chain_json)


    def add_feedback(self, result_json: JSON) -> None:

        if 'record_id' not in result_json or result_json['record_id'] is None:
            raise RuntimeError("Result does not include record_id. "
                            "To log feedback, log the record first using `tru.add_record`.")

        self.db.insert_feedback(result_json=result_json, status=2)


    def add_feedbacks(self, result_jsons: Iterable[JSON]) -> None:

        for result_json in result_jsons:
            self.add_feedback(result_json=result_json)


    def get_chain(self, chain_id: str) -> JSON:

        return self.db.get_chain(chain_id)


    def get_records_and_feedback(self, chain_ids: List[str]):

        df, feedback_columns = self.db.get_records_and_feedback(chain_ids)

        return df, feedback_columns


    def run_dashboard(self) -> None:

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

        #run leaderboard with subprocess
        leaderboard_path = pkg_resources.resource_filename(
            'trulens_eval', 'Leaderboard.py'
        )

        subprocess.Popen(["streamlit", "run", leaderboard_path])

