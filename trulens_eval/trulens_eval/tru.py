from datetime import datetime
import os
import sqlite3
import subprocess
from typing import Callable, Dict, List, Optional, Sequence

import pkg_resources

from trulens_eval.tru_db import json_str_of_obj
from trulens_eval.tru_db import LocalSQLite
from trulens_eval.tru_db import TruDB
from trulens_eval.tru_feedback import Feedback
from trulens_eval.util import TP

lms = LocalSQLite()


def init_db(db_name):

    # Connect to the database
    conn = sqlite3.connect(f'{db_name}.db')
    c = conn.cursor()

    # Commit changes and close the connection
    conn.commit()
    conn.close()
    return None


def to_json(details):
    return json_str_of_obj(details)


def add_data(
    chain_id: str,
    prompt: str,
    response: str,
    record: Dict = None,
    tags: str = None,
    ts: int = None,
    total_tokens: int = None,
    total_cost: float = None,
    db: Optional[TruDB] = None
):
    """_summary_

    Parameters:

        chain_id (str): _description_

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

    if db is None:
        db = lms

    record_id = db.insert_record(
        chain_id, prompt, response, record, ts, tags, total_tokens, total_cost
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
    chain: 'TruChain', record: Dict, feedback_functions: Sequence['Feedback']
):

    # Run feedback functions
    evals = {}

    for func in feedback_functions:
        evals[
            func.name
        ] = TP().promise(lambda f: f.run(chain=chain, record=record), func)

    for name, promise in evals.items():
        temp = promise.get()
        print(f"{name}={temp}")
        evals[name] = temp

    return evals


def add_feedback(record_id: str, eval: dict, db: Optional[TruDB] = None):
    if db is None:
        db = lms

    db.insert_feedback(record_id, eval)


def get_chain(chain_id, db: Optional[TruDB] = None):
    if db is None:
        db = lms

    return lms.get_chain(chain_id)


def get_records_and_feedback(chain_ids: List[str], db: Optional[TruDB] = None):
    if db is None:
        db = lms

    df_records, df_feedback = lms.get_records_and_feedback(chain_ids)
    return df_records, df_feedback


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
