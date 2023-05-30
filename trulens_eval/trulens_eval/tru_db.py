import abc
import json
import logging
from pathlib import Path
import sqlite3
from typing import (Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple,
                    Union)

from frozendict import frozendict
from merkle_json import MerkleJson
import pandas as pd
import pydantic

from trulens_eval.schema import Chain
from trulens_eval.schema import FeedbackDefinition
from trulens_eval.schema import FeedbackResult
from trulens_eval.schema import Record
from trulens_eval.util import _project
from trulens_eval.util import GetItem
from trulens_eval.util import JSON
from trulens_eval.util import JSON_BASES
from trulens_eval.util import json_str_of_obj
from trulens_eval.util import JSONPath
from trulens_eval.util import noserio
from trulens_eval.util import obj_id_of_obj
from trulens_eval.util import UNCIODE_YIELD
from trulens_eval.util import UNICODE_CHECK

mj = MerkleJson()
NoneType = type(None)



# Typing for type hints.
Query = JSONPath

# Instance for constructing queries for record json like `Record.chain.llm`.
RecordQuery = Query().attr("_record")

# Instance for constructing queries for chain json.
ChainQuery = Query().attr("_chain")


def get_calls(record_json: JSON) -> Iterable[JSON]:
    """
    Iterate over the call parts of the record.
    """

    for q in TruDB.all_queries(record_json):
        if q._path[-1] == "_call":
            yield q


def get_calls_by_stack(record_json: JSON) -> Dict[Tuple[str, ...], JSON]:
    """
    Get a dictionary mapping chain call stack to the call information.
    """

    def frozen_frame(frame):
        frame['path'] = tuple(frame['path'])
        return frozendict(frame)

    ret = dict()
    for c in get_calls(record_json):
        obj = TruDB.project(c, record_json=record_json, chain_json=None)
        if isinstance(obj, Sequence):
            for o in obj:
                call_stack = tuple(map(frozen_frame, o['chain_stack']))
                if call_stack not in ret:
                    ret[call_stack] = []
                ret[call_stack].append(o)
        else:
            call_stack = tuple(map(frozen_frame, obj['chain_stack']))
            if call_stack not in ret:
                ret[call_stack] = []
            ret[call_stack].append(obj)

    return ret


def get_calls(record_json: JSON) -> Iterable[JSON]:
    """
    Iterate over the call parts of the record.
    """

    for q in TruDB.all_queries(record_json):
        if q._path[-1] == "_call":
            yield q


def get_calls_by_stack(record_json: JSON) -> Dict[Tuple[str, ...], JSON]:
    """
    Get a dictionary mapping chain call stack to the call information.
    """

    def frozen_frame(frame):
        frame['path'] = tuple(frame['path'])
        return frozendict(frame)

    ret = dict()
    for c in get_calls(record_json):
        obj = TruDB.project(c, record_json=record_json, chain_json=None)
        if isinstance(obj, Sequence):
            for o in obj:
                call_stack = tuple(map(frozen_frame, o['chain_stack']))
                if call_stack not in ret:
                    ret[call_stack] = []
                ret[call_stack].append(o)
        else:
            call_stack = tuple(map(frozen_frame, obj['chain_stack']))
            if call_stack not in ret:
                ret[call_stack] = []
            ret[call_stack].append(obj)

    return ret


def query_of_path(path: List[Union[str, int]]) -> JSONPath:
    """
    Convert the given path to a query object.
    """

    if path[0] == "_record":
        ret = RecordQuery
        path = path[1:]
    elif path[0] == "_chain":
        ret = ChainQuery
        path = path[1:]
    else:
        ret = JSONPath()

    for attr in path:
        ret = getattr(ret, attr)

    return ret


def jsonify(obj: Any, dicted=None) -> JSON:
    """
    Convert the given object into types that can be serialized in json.
    """

    dicted = dicted or dict()

    if isinstance(obj, JSON_BASES):
        return obj

    if id(obj) in dicted:
        return {'_CIRCULAR_REFERENCE': id(obj)}

    new_dicted = {k: v for k, v in dicted.items()}

    if isinstance(obj, Dict):
        temp = {}
        new_dicted[id(obj)] = temp
        temp.update({k: jsonify(v, dicted=new_dicted) for k, v in obj.items()})
        return temp

    elif isinstance(obj, Sequence):
        temp = []
        new_dicted[id(obj)] = temp
        for x in (jsonify(v, dicted=new_dicted) for v in obj):
            temp.append(x)
        return temp

    elif isinstance(obj, Set):
        temp = []
        new_dicted[id(obj)] = temp
        for x in (jsonify(v, dicted=new_dicted) for v in obj):
            temp.append(x)
        return temp

    elif isinstance(obj, pydantic.BaseModel):
        temp = {}
        new_dicted[id(obj)] = temp
        temp.update(
            {
                k: jsonify(getattr(obj, k), dicted=new_dicted)
                for k in obj.__fields__
            }
        )
        return temp

    else:
        logging.debug(
            f"Don't know how to jsonify an object '{str(obj)[0:32]}' of type '{type(obj)}'."
        )
        return noserio(obj)


class TruDB(abc.ABC):

    # Use TinyDB queries for looking up parts of records/models and/or filtering
    # on those parts.

    @abc.abstractmethod
    def reset_database(self):
        """
        Delete all data.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_record(
        self, chain_id: str, input: str, output: str, record: Record, ts: int,
        tags: str, total_tokens: int, total_cost: float
    ) -> int:
        """
        Insert a new `record` into db, indicating its `model` as well. Return
        record id.

        Args:

        - chain_id: str - the chain id of the chain that generated `record`.

        - input: str - the main chain input.

        - output: str - the main chain output.

        - record: Record - the full record of the execution of a chain.

        - ts: int - timestamp.

        - tags: str - additional metadata to store alongside the record.

        - total_tokens: int - number of tokens generated in process of chain
          evaluation.

        - total_cost: float - the cost of chain evaluation.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_chain(self, chain: Chain) -> str:
        """
        Insert a new `chain` into db under the given `chain_id`. 

        Args:
        - chain: Chain - Chain definition. 
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback_def(self, feedback: FeedbackDefinition):
        """
        Insert a feedback definition into the db.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback(
        self,
        record_id: str,
        feedback_id: str,
        last_ts: Optional[int] = None,  # "last timestamp"
        status: Optional[int] = None,
        result: Optional[FeedbackResult] = None,
        total_cost: Optional[float] = None,
        total_tokens: Optional[int] = None,
    ) -> str:
        """
        Insert a feedback record into the db.

        Args:

        - record_id: Optional[str] - the record id that produced the feedback,
          if any.

        - feedback_id: Optional[str] - the feedback id of the feedback function
          definition that produced this result, if any.

        - last_ts: Optional[int] - timestamp of the last update to this record.

        - status: Optional[int] - the status of the result used for referred
          execution.

        - result: Optional[FeedbackResult] - the result itself.

        - total_cost: Optional[float] - 

        - total_tokens: Optional[int] - 
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def get_records_and_feedback(
        self, chain_ids: List[str]
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        """
        Get the records logged for the given set of `chain_ids` (otherwise all)
        alongside the names of the feedback function columns listed the
        dataframe.
        """
        raise NotImplementedError()

    @staticmethod
    def project(
        query: JSONPath,
        record_json: JSON,
        chain_json: JSON,
        obj: Optional[JSON] = None
    ):
        path = query.path

        if path[0] == GetItem("_record"):
            if len(path) == 1:
                return record_json
            return _project(path=path[1:], obj=record_json)

        elif path[0] == GetItem("_chain"):
            if len(path) == 1:
                return chain_json

            return _project(path=path[1:], obj=chain_json)
        else:
            return _project(path=path, obj=obj)


class LocalSQLite(TruDB):

    TABLE_RECORDS = "records"
    TABLE_FEEDBACKS = "feedbacks"
    TABLE_FEEDBACK_DEFS = "feedback_defs"
    TABLE_CHAINS = "chains"

    def __init__(self, filename: Optional[Path] = 'default.sqlite'):
        """
        Database locally hosted using SQLite.

        Args
        
        - filename: Optional[Path] -- location of sqlite database dump
          file. It will be created if it does not exist.

        """

        self.filename = filename
        self._build_tables()

    def __str__(self) -> str:
        return f"SQLite({self.filename})"

    # TruDB requirement
    def reset_database(self) -> None:
        self._clear_tables()
        self._build_tables()

    def _clear_tables(self) -> None:
        conn, c = self._connect()

        # Create table if it does not exist
        c.execute(f'''DELETE FROM {self.TABLE_RECORDS}''')
        c.execute(f'''DELETE FROM {self.TABLE_FEEDBACKS}''')
        c.execute(f'''DELETE FROM {self.TABLE_FEEDBACK_DEFS}''')
        c.execute(f'''DELETE FROM {self.TABLE_CHAINS}''')
        self._close(conn)

    def _build_tables(self):
        conn, c = self._connect()

        # Create table if it does not exist
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_RECORDS} (
                record_id TEXT,
                chain_id TEXT,
                input TEXT,
                output TEXT,
                record_json TEXT,
                tags TEXT,
                ts INTEGER,
                total_tokens INTEGER,
                total_cost REAL,
                PRIMARY KEY (record_id, chain_id)
            )'''
        )
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_FEEDBACKS} (
                record_id TEXT,
                feedback_id TEXT,
                last_ts INTEGER,
                status INTEGER,
                result_json TEXT,
                total_tokens INTEGER,
                total_cost REAL,
                PRIMARY KEY (record_id, feedback_id)
            )'''
        )
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_FEEDBACK_DEFS} (
                feedback_id TEXT PRIMARY KEY,
                feedback_json TEXT
            )'''
        )
        c.execute(
            f'''CREATE TABLE IF NOT EXISTS {self.TABLE_CHAINS} (
                chain_id TEXT PRIMARY KEY,
                chain_json TEXT
            )'''
        )
        self._close(conn)

    def _connect(self) -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        return conn, c

    def _close(self, conn: sqlite3.Connection) -> None:
        conn.commit()
        conn.close()

    # TruDB requirement
    def insert_record(
        self, chain_id: str, input: str, output: str, record: Record, ts: int,
        tags: str, total_tokens: int, total_cost: float
    ) -> str:

        record_json = record.dict()

        assert isinstance(
            record_json, Dict
        ), f"Attempting to add a record that is not a dict, is {type(record_json)} instead."

        conn, c = self._connect()

        record_id = obj_id_of_obj(obj=record_json, prefix="record")
        record_json['record_id'] = record_id
        record_str = json_str_of_obj(record_json)

        c.execute(
            f"INSERT INTO {self.TABLE_RECORDS} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record_id, chain_id, input, output, record_str, tags, ts,
                total_tokens, total_cost
            )
        )
        self._close(conn)

        print(
            f"{UNICODE_CHECK} record {record_id} from {chain_id} -> {self.filename}"
        )

        return record_id

    # TruDB requirement
    def insert_chain(self, chain: Chain) -> str:

        chain_json = chain.dict()

        chain_id = chain_json['chain_id']
        chain_str = json_str_of_obj(chain_json)

        conn, c = self._connect()
        c.execute(
            f"INSERT OR IGNORE INTO {self.TABLE_CHAINS} VALUES (?, ?)",
            (chain_id, chain_str)
        )
        self._close(conn)

        print(f"{UNICODE_CHECK} chain {chain_id} -> {self.filename}")

        return chain_id

    def insert_feedback_def(self, feedback: FeedbackDefinition) -> None:
        """
        Insert a feedback definition into the database.
        """

        feedback_json = feedback.dict()

        feedback_id = feedback_json['feedback_id']
        feedback_str = json_str_of_obj(feedback_json)

        conn, c = self._connect()
        c.execute(
            f"INSERT OR REPLACE INTO {self.TABLE_FEEDBACK_DEFS} VALUES (?, ?)",
            (feedback_id, feedback_str)
        )
        self._close(conn)

        print(f"{UNICODE_CHECK} feedback def. {feedback_id} -> {self.filename}")

    def get_feedback_defs(
        self, feedback_id: Optional[str] = None
    ) -> pd.DataFrame:

        clause = ""
        args = ()
        if feedback_id is not None:
            clause = "WHERE feedback_id=?"
            args = (feedback_id,)

        query = f"""
            SELECT
                feedback_id, feedback_json
            FROM {self.TABLE_FEEDBACK_DEFS}
            {clause}
        """

        conn, c = self._connect()
        c.execute(query, args)
        rows = c.fetchall()
        self._close(conn)

        from trulens_eval.tru_feedback import Feedback

        df_rows = []

        for row in rows:
            row = list(row)
            row[1] = Feedback.of_json(json.loads(row[1]))
            df_rows.append(row)

        return pd.DataFrame(rows, columns=['feedback_id', 'feedback'])

    def insert_feedback(
        self,
        record_id: Optional[str] = None,
        feedback_id: Optional[str] = None,
        last_ts: Optional[int] = None,  # "last timestamp"
        status: Optional[int] = None,
        result: Optional[FeedbackResult] = None,
        total_cost: Optional[float] = None,
        total_tokens: Optional[int] = None,
    ) -> None:
        """
        Insert a record-feedback link to db or update an existing one.
        """

        if result is not None:
            result_json = result.dict()

        if record_id is None or feedback_id is None:
            assert result is not None, "`result` needs to be given if `record_id` or `feedback_id` are not provided."
            record_id = result_json['record_id']
            feedback_id = result_json['feedback_id']

        last_ts = last_ts or 0
        status = status or 0
        result_json = result_json or dict()
        total_cost = total_cost = 0.0
        total_tokens = total_tokens or 0
        result_str = json_str_of_obj(result_json)

        conn, c = self._connect()
        c.execute(
            f"INSERT OR REPLACE INTO {self.TABLE_FEEDBACKS} VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                record_id, feedback_id, last_ts, status, result_str,
                total_tokens, total_cost
            )
        )
        self._close(conn)

        if status == 2:
            print(
                f"{UNICODE_CHECK} feedback {feedback_id} on {record_id} -> {self.filename}"
            )
        else:
            print(
                f"{UNCIODE_YIELD} feedback {feedback_id} on {record_id} -> {self.filename}"
            )

    def get_feedback(
        self,
        record_id: Optional[str] = None,
        feedback_id: Optional[str] = None,
        status: Optional[int] = None,
        last_ts_before: Optional[int] = None
    ) -> pd.DataFrame:

        clauses = []
        vars = []
        if record_id is not None:
            clauses.append("record_id=?")
            vars.append(record_id)
        if feedback_id is not None:
            clauses.append("feedback_id=?")
            vars.append(feedback_id)
        if status is not None:
            if isinstance(status, Sequence):
                clauses.append(
                    "status in (" + (",".join(["?"] * len(status))) + ")"
                )
                for v in status:
                    vars.append(v)
            else:
                clauses.append("status=?")
                vars.append(status)
        if last_ts_before is not None:
            clauses.append("last_ts<=?")
            vars.append(last_ts_before)

        where_clause = " AND ".join(clauses)
        if len(where_clause) > 0:
            where_clause = " AND " + where_clause

        query = f"""
            SELECT
                f.record_id, f.feedback_id, f.last_ts, f.status,
                f.result_json, f.total_cost, f.total_tokens,
                fd.feedback_json, r.record_json, c.chain_json
            FROM {self.TABLE_FEEDBACKS} f 
                JOIN {self.TABLE_FEEDBACK_DEFS} fd
                JOIN {self.TABLE_RECORDS} r
                JOIN {self.TABLE_CHAINS} c
            WHERE f.feedback_id=fd.feedback_id
                AND r.record_id=f.record_id
                AND r.chain_id=c.chain_id
                {where_clause}
        """

        conn, c = self._connect()
        c.execute(query, vars)
        rows = c.fetchall()
        self._close(conn)

        from trulens_eval.tru_feedback import Feedback

        df_rows = []
        for row in rows:
            row = list(row)
            row[4] = json.loads(row[4])  # result_json
            row[7] = json.loads(row[7])  # feedback_json
            row[8] = json.loads(row[8])  # record_json
            row[9] = json.loads(row[9])  # chain_json

            df_rows.append(row)

        return pd.DataFrame(
            df_rows,
            columns=[
                'record_id', 'feedback_id', 'last_ts', 'status', 'result_json',
                'total_cost', 'total_tokens', 'feedback_json', 'record_json',
                'chain_json'
            ]
        )

    def get_chain(self, chain_id: str) -> JSON:
        conn, c = self._connect()
        c.execute(
            f"SELECT chain_json FROM {self.TABLE_CHAINS} WHERE chain_id=?",
            (chain_id,)
        )
        result = c.fetchone()[0]
        conn.close()

        return json.loads(result)

    def get_records_and_feedback(
        self, chain_ids: List[str]
    ) -> Tuple[pd.DataFrame, Sequence[str]]:
        # This returns all models if the list of chain_ids is empty.
        conn, c = self._connect()
        query = f"""
            SELECT r.record_id, f.result_json
            FROM {self.TABLE_RECORDS} r 
            LEFT JOIN {self.TABLE_FEEDBACKS} f
                ON r.record_id = f.record_id
            """
        if len(chain_ids) > 0:
            chain_id_list = ', '.join('?' * len(chain_ids))
            query = query + f" WHERE r.chain_id IN ({chain_id_list})"

        c.execute(query)
        rows = c.fetchall()
        conn.close()

        df_results = pd.DataFrame(
            rows, columns=[description[0] for description in c.description]
        )

        if len(df_results) == 0:
            return df_results, []

        conn, c = self._connect()
        query = f"""
            SELECT DISTINCT r.*, c.chain_json
            FROM {self.TABLE_RECORDS} r 
            JOIN {self.TABLE_CHAINS} c
                ON r.chain_id = c.chain_id
            """
        if len(chain_ids) > 0:
            chain_id_list = ', '.join('?' * len(chain_ids))
            query = query + f" WHERE r.chain_id IN ({chain_id_list})"

        c.execute(query)
        rows = c.fetchall()
        conn.close()

        df_records = pd.DataFrame(
            rows, columns=[description[0] for description in c.description]
        )

        if len(df_records) == 0:
            return df_records, []

        # Apply the function to the 'data' column to convert it into separate columns
        df_results['result_json'] = df_results['result_json'].apply(
            lambda d: {} if d is None else json.loads(d)
        )

        if "record_id" not in df_results.columns:
            return df_results, []

        df_results = df_results.groupby("record_id").agg(
            lambda dicts: {key: val for d in dicts for key, val in d.items()}
        ).reset_index()

        df_results = df_results['result_json'].apply(pd.Series)

        result_cols = [
            col for col in df_results.columns
            if col not in ['feedback_id', 'record_id', '_success', "_error"]
        ]

        if len(df_results) == 0 or len(result_cols) == 0:
            return df_records, []

        assert "record_id" in df_results.columns
        assert "record_id" in df_records.columns

        combined_df = df_records.merge(df_results, on=['record_id'])
        combined_df = combined_df.drop(
            columns=set(["_success", "_error"]
                       ).intersection(set(combined_df.columns))
        )

        return combined_df, result_cols
