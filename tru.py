from datetime import datetime
import sqlite3
from typing import Callable, List
import pandas as pd

from tru_db import LocalModelStore


lms = LocalModelStore()


def add_data(
    model_id: str,
    prompt: str,
    response: str,
    details: str = None,
    tags: str = None,
    ts: int = None
):
    if not ts:
        ts = datetime.now()

    record_id = lms.insert_record(model_id, prompt, response, details, ts, tags)
    return record_id


def run_feedback_function(prompt: str, response: str, feedback_functions: Callable[[str, str], str]):

    # Run feedback functions
    eval = {}
    for f in feedback_functions:
        eval[f.__name__] = f(prompt, response)
    return eval


def add_feedback(record_id: str, eval: dict):
    lms.insert_feedback(record_id, eval)


def get_model(model_id):
    return lms.get_model(model_id)

def get_records_and_feedback(model_ids: List[str]):
    df_records, df_feedback = lms.get_records_and_feedback(model_ids)
    return df_records, df_feedback
