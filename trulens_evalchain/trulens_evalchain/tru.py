from datetime import datetime
import json
from typing import Callable, List
import subprocess

import pandas as pd

from trulens_evalchain.tru_db import json_default
from trulens_evalchain.tru_db import LocalSQLite

lms = LocalSQLite()

import sqlite3


def init_db(db_name):

    # Connect to the database
    conn = sqlite3.connect(f'{db_name}.db')
    c = conn.cursor()

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    return None


def to_json(details):
    return json.dumps(details, default=json_default)


def add_data(
    chain_id: str,
    prompt: str,
    response: str,
    details: str = None,
    tags: str = None,
    ts: int = None,
    total_tokens: int = None,
    total_cost: float = None
):
    if not ts:
        ts = datetime.now()

    record_id = lms.insert_record(
        chain_id, prompt, response, to_json(details), ts, tags, total_tokens,
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


def add_feedback(record_id: str, eval: dict):
    lms.insert_feedback(record_id, eval)


def get_chain(chain_id):
    return lms.get_chain(chain_id)


def get_records_and_feedback(chain_ids: List[str]):
    df_records, df_feedback = lms.get_records_and_feedback(chain_ids)
    return df_records, df_feedback


def run_dashboard():
    subprocess.Popen(["streamlit", "run", 'trulens_evalchain/Leaderboard.py'])
    return None
