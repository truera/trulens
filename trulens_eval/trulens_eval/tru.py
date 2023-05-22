"""
Public interfaces.
"""

from datetime import datetime
import logging
import sqlite3
import subprocess
from typing import Callable, Dict, List, Optional, Sequence

import pkg_resources

from trulens_eval.tru_db import JSON, json_str_of_obj
from trulens_eval.tru_db import LocalSQLite
from trulens_eval.tru_db import TruDB
from trulens_eval.tru_feedback import Feedback
from trulens_eval.util import TP

lms = LocalSQLite()


def to_json(details):
    return json_str_of_obj(details)


def add_record(
    prompt: str,
    response: str,
    record_json: JSON,
    tags: Optional[str] = None,
    ts: Optional[int] = None,
    total_tokens: Optional[int] = None,
    total_cost: Optional[float] = None,
    db: Optional[TruDB] = None
):
    """_summary_

    Parameters:

        prompt (str): _description_

        response (str): _description_

        details (str, optional): _description_. Defaults to None.

        tags (str, optional): _description_. Defaults to None.

        ts (int, optional): _description_. Defaults to None.

        total_tokens (int, optional): _description_. Defaults to None.

        total_cost (float, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if not ts:
        ts = datetime.now()

    db = db or lms

    chain_id = record_json['chain_id']

    record_id = db.insert_record(
        chain_id, prompt, response, record_json, ts, tags, total_tokens,
        total_cost
    )

    return record_id


def run_feedback_function(
    prompt: str, response: str, feedback_functions: Callable[[str, str], str]
):

    # Run feedback functions
    eval = {}
    for f in feedback_functions:
        eval[f.__name__] = f(prompt, response)
    return eval


def run_feedback_functions(
    record_json: JSON,
    feedback_functions: Sequence['Feedback'],
    chain_json: Optional[JSON] = None,
    db: Optional[TruDB] = None
) -> Sequence[JSON]:
    
    # Run a collection of feedback functions and report their result.
    
    db = db or lms

    chain_id = record_json['chain_id']

    if chain_json is None:
        chain_json = db.get_chain(chain_id=chain_id)
        if chain_json is None:
            raise RuntimeError("Chain {chain_id} not present in db. Either add it with `tru.add_chain` or provide `chain_json` to `tru.run_feedback_functions`.")
        
    else:
        assert chain_id == chain_json['chain_id'], "Record was produced by a different chain."

        if db.get_chain(chain_id=chain_json['chain_id']) is None:
            logging.warn("Chain {chain_id} was not present in database. Adding it.")
            add_chain(chain_json=chain_json)

    evals = []

    for func in feedback_functions:
        evals.append(TP().promise(
            lambda f: f.
            run_on_record(chain_json=chain_json, record_json=record_json), func
        ))

    evals = map(lambda p: p.get(), evals)

    return list(evals)


def add_chain(
    chain_json: JSON,
    chain_id: Optional[str] = None,
    db: Optional[TruDB] = None
) -> None:
    db = db or lms

    db.insert_chain(chain_id=chain_id, chain_json=chain_json)


def add_feedback(result_json: JSON, db: Optional[TruDB] = None):
    db = db or lms

    db.insert_feedback(result_json=result_json, status=2)


def get_chain(chain_id, db: Optional[TruDB] = None):
    db = db or lms

    return db.get_chain(chain_id)


def get_records_and_feedback(chain_ids: List[str], db: Optional[TruDB] = None):
    db = db or lms

    df, feedback_columns = db.get_records_and_feedback(chain_ids)

    return df, feedback_columns


def run_dashboard():
    leaderboard_path = pkg_resources.resource_filename(
        'trulens_eval', 'Leaderboard.py'
    )

    subprocess.Popen(["streamlit", "run", leaderboard_path])
    return None
