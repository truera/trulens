from datetime import datetime
import sqlite3

import tru_feedback

db_name = 'llm_quality.db'


def add_data(
    model_id,
    prompt,
    template,
    response,
    tags,
    feedback,
    prompt_id=None,
    ts=None
):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Create table if it does not exist
    c.execute(
        '''CREATE TABLE IF NOT EXISTS llm_calls
                    (record_id TEXT, model_id TEXT, prompt_id TEXT, prompt TEXT, template TEXT, response TEXT, tags TEXT, feedback TEXT, ts INTEGER)'''
    )
    feedback_str = str(feedback)
    if not ts:
        ts = datetime.now()

    record_id = f'{model_id}_{prompt_id}_{ts}'
    c.execute(
        "INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            record_id, model_id, prompt_id, prompt, template, response, tags,
            feedback_str, ts
        )
    )
    # Commit changes and close the connection
    conn.commit()
    conn.close()

    return record_id


def run_feedback_function(prompt, response, feedback_functions):

    # Run feedback functions
    eval = {}
    for f in feedback_functions:
        eval[f.__name__] = f(prompt, response)

    eval_str = str(eval)

    return eval_str


def add_feedback(record_id, eval_str):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()
    # Create table if it does not exist
    c.execute(
        '''CREATE TABLE IF NOT EXISTS llm_feedback_functions
                    (record_id TEXT, feedback_functions)'''
    )

    c.execute(
        "INSERT INTO llm_feedback_functions VALUES (?, ?)",
        (record_id, eval_str)
    )

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    return False
