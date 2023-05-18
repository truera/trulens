from datetime import datetime
import json
from multiprocessing.pool import ThreadPool
from queue import Queue
import sqlite3
from time import sleep
from typing import Any, Callable, Dict, List, Sequence

import pandas as pd
import requests
from tqdm.auto import tqdm

from trulens_evalchain.keys import HUGGINGFACE_HEADERS
from trulens_evalchain.tru_db import json_default
from trulens_evalchain.tru_db import LocalSQLite
from trulens_evalchain.tru_chain import TruChain
from trulens_evalchain.tru_feedback import Feedback

lms = LocalSQLite()

thread_pool = ThreadPool(processes=8)

class Endpoint():
    def __init__(self, name: str, rpm: float = 60, retries: int = 3, post_headers = None):
        """
        
        Args:
        - name: str -- api name / identifier.
        - rpm: float -- requests per minute.
        - retries: int -- number of retries before failure.
        """

        self.rpm = rpm
        self.retries = retries
        self.pace = Queue(maxsize=rpm / 6.0) # 10 second's worth of accumulated api
        self.tqdm = tqdm(desc=f"{name}", unit="request")
        self.name = name
        self.post_headers = post_headers
        
        self._start_pacer()

    def pace_me(self):
        """
        Block until we can make a request to this endpoint.
        """

        self.pace.get()
        self.tqdm.update(1)
        return

    def post(self, url, payload) -> Any:
        extra = dict()
        if self.post_headers is not None:
            extra['headers'] = self.post_headers

        self.pace_me()
        ret = requests.post(url, json=payload, **extra)

        j = ret.json()

        # Huggingface public api sometimes tells us that a model is loading and how long to wait:
        if "estimated_time" in j:
            wait_time = j['estimated_time']
            print(f"WARNING: Waiting for {j} ({wait_time}) second(s).")
            sleep(wait_time)
            return self.post(url, payload)

        assert isinstance(j, Sequence) and len(j) > 0, f"Post did not return a sequence: {j}"

        return j[0]
    

    def run_me(self, thunk):
        """
        Run the given thunk, returning itse output, on pace with the api.
        Retries request multiple times if self.retries > 0.
        """
        
        retries = self.retries + 1
        retry_delay = 2.0

        while retries > 0:
            try:
                self.pace_me()
                ret = thunk()                
                return ret
            except Exception as e:
                retries -= 1
                print(f"WARNING: {self.name} request failed {type(e)}={e}. Retries={retries}.")
                if retries > 0:
                    sleep(retry_delay)
                    retry_delay *= 2

        raise RuntimeError(f"API {self.name} request failed {self.retries+1} time(s).")

    def _start_pacer(self):
        def keep_pace():
            while True:
                sleep(60.0 / self.rpm)
                self.pace.put(True)
        
        thread_pool.apply_async(keep_pace)

endpoint_openai = Endpoint(name="openai", rpm=60)
endpoint_huggingface = Endpoint(name="huggingface", rpm=60, post_headers=HUGGINGFACE_HEADERS)
endpoint_cohere = Endpoint(name="cohere", rpm=60)

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
    record: Dict = None,
    tags: str = None,
    ts: int = None,
    total_tokens: int = None,
    total_cost: float = None
):
    if not ts:
        ts = datetime.now()

    record_id = lms.insert_record(
        chain_id, prompt, response, record, ts, tags, total_tokens,
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
    chain: TruChain,
    record: Dict,
    feedback_functions: Sequence[Feedback]
):

    # Run feedback functions
    evals = {}
    
    for f in feedback_functions:
        evals[f.name] = f.run(chain=chain, record=record)

    return evals


def add_feedback(record_id: str, eval: dict):
    lms.insert_feedback(record_id, eval)


def get_chain(chain_id):
    return lms.get_chain(chain_id)


def get_records_and_feedback(chain_ids: List[str]):
    df_records, df_feedback = lms.get_records_and_feedback(chain_ids)
    return df_records, df_feedback
