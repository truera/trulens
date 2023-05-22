"""
Public interfaces.
"""

from datetime import datetime
import logging
import os
import subprocess
from typing import Callable, List, Optional, Sequence

import pkg_resources

from trulens_eval.tru_db import JSON
from trulens_eval.tru_db import json_str_of_obj
from trulens_eval.tru_db import LocalSQLite
from trulens_eval.tru_db import TruDB
from trulens_eval.tru_feedback import Feedback
from trulens_eval.util import TP

lms = LocalSQLite()


def to_json(details):
    return json_str_of_obj(details)


deferred_feedback_evaluator_started = False

def start_deferred_feedback_evaluator(db: Optional[TruDB] = None):
    global deferred_feedback_evaluator_started

    db = db or lms

    if deferred_feedback_evaluator_started:
        raise RuntimeError("Evaluator is already running in this process.")

    from trulens_eval.tru_feedback import Feedback

    # Start a persistent thread that evaluates feedback functions.
    Feedback.start_evaluator(db=db)

    deferred_feedback_evaluator_started = True


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
            raise RuntimeError(
                "Chain {chain_id} not present in db. "
                "Either add it with `tru.add_chain` or provide `chain_json` to `tru.run_feedback_functions`."
            )

    else:
        assert chain_id == chain_json[
            'chain_id'], "Record was produced by a different chain."

        if db.get_chain(chain_id=chain_json['chain_id']) is None:
            logging.warn(
                "Chain {chain_id} was not present in database. Adding it."
            )
            add_chain(chain_json=chain_json)

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
    return None
