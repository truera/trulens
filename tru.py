from datetime import datetime
import json
import sqlite3
from typing import Callable, List

import pandas as pd

from tru_db import LocalModelStore

lms = LocalModelStore()


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

    def to_json(details):
        return json.dumps(
            details, default=lambda o: f"<{o.__class__.__name__}>"
        )

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


def get_model(chain_id):
    return lms.get_model(chain_id)


def get_records_and_feedback(chain_ids: List[str]):
    df_records, df_feedback = lms.get_records_and_feedback(chain_ids)
    return df_records, df_feedback
