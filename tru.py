import sqlite3
import tru_feedback
from datetime import datetime
db_name = 'llm_quality.db'

def add_data(model_id, request, template, response, tags, feedback, feedback_functions, ts=None):
    conn = sqlite3.connect(db_name)
    c = conn.cursor()

    # Create table if it does not exist
    c.execute('''CREATE TABLE IF NOT EXISTS llm_calls
                    (id TEXT, request TEXT, template TEXT, response TEXT, tags TEXT, feedback TEXT, feedback_functions TEXT, ts INTEGER)''')

    # Run feedback functions
    eval = {f: tru_feedback.FEEDBACK_FUNCTIONS[f](request, response) for f in feedback_functions}
    feedback_str = str(feedback)
    eval_str = str(eval)
    if not ts:
        ts = datetime.now()

    # Save the function call to the database
    c.execute("INSERT INTO llm_calls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (model_id, request, template, response, tags, feedback_str, eval_str, ts))

    # Commit changes and close the connection
    conn.commit()
    conn.close()

    return False