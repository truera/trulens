import abc
import json
import logging
from pathlib import Path
import sqlite3
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
)

from merkle_json import MerkleJson
import pandas as pd
import pydantic
from tinydb import Query as TinyQuery
from tinydb.queries import QueryInstance as TinyQueryInstance
from trulens_eval.util import UNCIODE_YIELD, UNICODE_CHECK

mj = MerkleJson()
NoneType = type(None)

JSON_BASES = (str, int, float, NoneType)
JSON_BASES_T = Union[str, int, float, NoneType]
# JSON = (List, Dict) + JSON_BASES
# JSON_T = Union[JSON_BASES_T, List, Dict]
JSON = Dict

def is_empty(obj):
    try:
        return len(obj) == 0
    except Exception:
        return False


def is_noserio(obj):
    """
    Determines whether the given json object represents some non-serializable
    object. See `noserio`.
    """
    return isinstance(obj, dict) and "_NON_SERIALIZED_OBJECT" in obj


def noserio(obj, **extra: Dict) -> dict:
    """
    Create a json structure to represent a non-serializable object. Any
    additional keyword arguments are included.
    """

    inner = {
        "id": id(obj),
        "class": obj.__class__.__name__,
        "module": obj.__class__.__module__,
        "bases": list(map(lambda b: b.__name__, obj.__class__.__bases__))
    }
    inner.update(extra)

    return {'_NON_SERIALIZED_OBJECT': inner}


def obj_id_of_obj(obj: dict, prefix="obj"):
    """
    Create an id from a json-able structure/definition. Should produce the same
    name if definition stays the same.
    """

    return f"{prefix}_hash_{mj.hash(obj)}"


def json_str_of_obj(obj: Any) -> str:
    """
    Encode the given json object as a string.
    """
    return json.dumps(obj, default=json_default)


def json_default(obj: Any) -> str:
    """
    Produce a representation of an object which cannot be json-serialized.
    """

    if isinstance(obj, pydantic.BaseModel):
        try:
            return json.dumps(obj.dict())
        except Exception as e:
            return noserio(obj, exception=e)

    # Intentionally not including much in this indicator to make sure the model
    # hashing procedure does not get randomized due to something here.

    return noserio(obj)


# Typing for type hints.
Query = TinyQuery

# Instance for constructing queries for record json like `Record.chain.llm`.
Record = Query()._record

# Instance for constructing queries for chain json.
Chain = Query()._chain

# Type of conditions, constructed from query/record like `Record.chain != None`.
Condition = TinyQueryInstance


def query_of_path(path: List[Union[str, int]]) -> Query:
    if path[0] == "_record":
        ret = Record
        path = path[1:]
    elif path[0] == "_chain":
        ret = Chain
        path = path[1:]
    else:
        ret = Query()

    for attr in path:
        ret = getattr(ret, attr)

    return ret


def path_of_query(query: Query) -> List[Union[str, int]]:
    return query._path


class TruDB(abc.ABC):

    # Use TinyDB queries for looking up parts of records/models and/or filtering
    # on those parts.

    @abc.abstractmethod
    def reset_database(self):
        """Delete all data."""

        raise NotImplementedError()

    @abc.abstractmethod
    def select(
        self,
        *query: Tuple[Query],
        where: Optional[Condition] = None
    ) -> pd.DataFrame:
        """
        Select `query` fields from the records database, filtering documents
        that do not match the `where` condition.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_record(
        self, chain_id: str, input: str, output: str, record_json: JSON,
        ts: int, tags: str, total_tokens: int, total_cost: float
    ) -> int:
        """
        Insert a new `record` into db, indicating its `model` as well. Return
        record id.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_chain(
        self, chain_json: JSON, chain_id: Optional[str] = None
    ) -> str:
        """
        Insert a new `chain` into db under the given `chain_id`. If name not
        provided, generate a name from chain definition. Return the name.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback_def(self, feedback_json: dict):
        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback(
        self,
        record_id: str,
        feedback_id: str,
        last_ts: Optional[int] = None,  # "last timestamp"
        status: Optional[int] = None,
        result_json: Optional[JSON] = None,
        total_cost: Optional[float] = None,
        total_tokens: Optional[int] = None,
    ) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_records_and_feedback(
        self, chain_ids: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError()

    @staticmethod
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
            temp.update(
                {
                    k: TruDB.jsonify(v, dicted=new_dicted)
                    for k, v in obj.items()
                }
            )
            return temp

        elif isinstance(obj, Sequence):
            temp = []
            new_dicted[id(obj)] = temp
            for x in (TruDB.jsonify(v, dicted=new_dicted) for v in obj):
                temp.append(x)
            return temp

        elif isinstance(obj, Set):
            temp = []
            new_dicted[id(obj)] = temp
            for x in (TruDB.jsonify(v, dicted=new_dicted) for v in obj):
                temp.append(x)
            return temp

        elif isinstance(obj, pydantic.BaseModel):
            temp = {}
            new_dicted[id(obj)] = temp
            temp.update(
                {
                    k: TruDB.jsonify(getattr(obj, k), dicted=new_dicted)
                    for k in obj.__fields__
                }
            )
            return temp

        else:
            logging.debug(
                f"Don't know how to jsonify an object '{str(obj)[0:32]}' of type '{type(obj)}'."
            )
            return noserio(obj)

    @staticmethod
    def leaf_queries(obj_json: JSON, query: Query = None) -> Iterable[Query]:
        """
        Get all queries for the given object that select all of its leaf values.
        """

        query = query or Record

        if isinstance(obj_json, (str, int, float, NoneType)):
            yield query

        elif isinstance(obj_json, Dict):
            for k, v in obj_json.items():
                sub_query = query[k]
                for res in TruDB.leaf_queries(obj_json[k], sub_query):
                    yield res

        elif isinstance(obj_json, Sequence):
            for i, v in enumerate(obj_json):
                sub_query = query[i]
                for res in TruDB.leaf_queries(obj_json[i], sub_query):
                    yield res

        else:
            yield query

    @staticmethod
    def all_queries(obj: Any, query: Query = None) -> Iterable[Query]:
        """
        Get all queries for the given object.
        """

        query = query or Record

        if isinstance(obj, (str, int, float, NoneType)):
            yield query

        elif isinstance(obj, pydantic.BaseModel):
            yield query

            for k in obj.__fields__:
                v = getattr(obj, k)
                sub_query = query[k]
                for res in TruDB.all_queries(v, sub_query):
                    yield res

        elif isinstance(obj, Dict):
            yield query

            for k, v in obj.items():
                sub_query = query[k]
                for res in TruDB.all_queries(obj[k], sub_query):
                    yield res

        elif isinstance(obj, Sequence):
            yield query

            for i, v in enumerate(obj):
                sub_query = query[i]
                for res in TruDB.all_queries(obj[i], sub_query):
                    yield res

        else:
            yield query

    @staticmethod
    def all_objects(obj: Any,
                    query: Query = None) -> Iterable[Tuple[Query, Any]]:
        """
        Get all queries for the given object.
        """

        query = query or Record

        if isinstance(obj, (str, int, float, NoneType)):
            yield (query, obj)

        elif isinstance(obj, pydantic.BaseModel):
            yield (query, obj)

            for k in obj.__fields__:
                v = getattr(obj, k)
                sub_query = query[k]
                for res in TruDB.all_objects(v, sub_query):
                    yield res

        elif isinstance(obj, Dict):
            yield (query, obj)

            for k, v in obj.items():
                sub_query = query[k]
                for res in TruDB.all_objects(obj[k], sub_query):
                    yield res

        elif isinstance(obj, Sequence):
            yield (query, obj)

            for i, v in enumerate(obj):
                sub_query = query[i]
                for res in TruDB.all_objects(obj[i], sub_query):
                    yield res

        else:
            yield (query, obj)

    @staticmethod
    def leafs(obj: Any) -> Iterable[Tuple[str, Any]]:
        for q in TruDB.leaf_queries(obj):
            path_str = TruDB._query_str(q)
            val = TruDB._project(q._path, obj)
            yield (path_str, val)

    @staticmethod
    def matching_queries(obj: Any, match: Callable) -> Iterable[Query]:
        for q in TruDB.all_queries(obj):
            val = TruDB._project(q._path, obj)
            if match(q, val):
                yield q

    @staticmethod
    def matching_objects(obj: Any,
                         match: Callable) -> Iterable[Tuple[Query, Any]]:
        for q, val in TruDB.all_objects(obj):
            if match(q, val):
                yield (q, val)

    @staticmethod
    def _query_str(query: Query) -> str:

        def render(ks):
            if len(ks) == 0:
                return ""

            first = ks[0]
            if len(ks) > 1:
                rest = ks[1:]
            else:
                rest = ()

            if isinstance(first, str):
                return f".{first}{render(rest)}"
            elif isinstance(first, int):
                return f"[{first}]{render(rest)}"
            else:
                RuntimeError(
                    f"Don't know how to render path element {first} of type {type(first)}."
                )

        return "Record" + render(query._path)

    @staticmethod
    def set_in_json(query: Query, in_json: JSON, val: JSON) -> JSON:
        return TruDB._set_in_json(query._path, in_json=in_json, val=val)

    @staticmethod
    def _set_in_json(path, in_json: JSON, val: JSON) -> JSON:
        if len(path) == 0:
            if isinstance(in_json, Dict):
                assert isinstance(val, Dict)
                in_json = {k: v for k, v in in_json.items()}
                in_json.update(val)
                return in_json

            assert in_json is None, f"Cannot set non-None json object: {in_json}"

            return val

        if len(path) == 1:
            first = path[0]
            rest = []
        else:
            first = path[0]
            rest = path[1:]

        if isinstance(first, str):
            if isinstance(in_json, Dict):
                in_json = {k: v for k, v in in_json.items()}
                if not first in in_json:
                    in_json[first] = None
            elif in_json is None:
                in_json = {first: None}
            else:
                raise RuntimeError(
                    f"Do not know how to set path {path} in {in_json}."
                )

            in_json[first] = TruDB._set_in_json(
                path=rest, in_json=in_json[first], val=val
            )
            return in_json

        elif isinstance(first, int):
            if isinstance(in_json, Sequence):
                # In case it is some immutable sequence. Also copy.
                in_json = list(in_json)
            elif in_json is None:
                in_json = []
            else:
                raise RuntimeError(
                    f"Do not know how to set path {path} in {in_json}."
                )

            while len(in_json) <= first:
                in_json.append(None)

            in_json[first] = TruDB._set_in_json(
                path=rest, in_json=in_json[first], val=val
            )
            return in_json

        else:
            raise RuntimeError(
                f"Do not know how to set path {path} in {in_json}."
            )

    @staticmethod
    def project(
        query: Query,
        record_json: JSON,
        chain_json: JSON,
        obj: Optional[JSON] = None
    ):
        path = query._path
        if path[0] == "_record":
            if len(path) == 1:
                return record_json
            return TruDB._project(path=path[1:], obj=record_json)
        elif path[0] == "_chain":
            if len(path) == 1:
                return chain_json
            return TruDB._project(path=path[1:], obj=chain_json)
        else:
            return TruDB._project(path=path, obj=obj)

    @staticmethod
    def _project(path: List, obj: Any):
        if len(path) == 0:
            return obj

        first = path[0]
        if len(path) > 1:
            rest = path[1:]
        else:
            rest = ()

        if isinstance(first, str):
            if isinstance(obj, pydantic.BaseModel):
                if not hasattr(obj, first):
                    logging.warn(
                        f"Cannot project {str(obj)[0:32]} with path {path} because {first} is not an attribute here."
                    )
                    return None
                return TruDB._project(path=rest, obj=getattr(obj, first))

            elif isinstance(obj, Dict):
                if first not in obj:
                    logging.warn(
                        f"Cannot project {str(obj)[0:32]} with path {path} because {first} is not a key here."
                    )
                    return None
                return TruDB._project(path=rest, obj=obj[first])

            else:
                logging.warn(
                    f"Cannot project {str(obj)[0:32]} with path {path} because object is not a dict or model."
                )
                return None

        elif isinstance(first, int):
            if not isinstance(obj, Sequence) or first >= len(obj):
                logging.warn(
                    f"Cannot project {str(obj)[0:32]} with path {path}."
                )
                return None

            return TruDB._project(path=rest, obj=obj[first])
        else:
            raise RuntimeError(
                f"Don't know how to locate element with key of type {first}"
            )


class LocalSQLite(TruDB):

    TABLE_RECORDS = "records"
    TABLE_FEEDBACKS = "feedbacks"
    TABLE_FEEDBACK_DEFS = "feedback_defs"
    TABLE_CHAINS = "chains"

    def __str__(self):
        return f"SQLite({self.filename})"

    def reset_database(self):
        self._clear_tables()
        self._build_tables()

    def _clear_tables(self):
        conn, c = self._connect()

        # Create table if it does not exist
        c.execute(
            f'''DELETE FROM {self.TABLE_RECORDS}'''
        )
        c.execute(
            f'''DELETE FROM {self.TABLE_FEEDBACKS}'''
        )
        c.execute(
            f'''DELETE FROM {self.TABLE_FEEDBACK_DEFS}'''
        )
        c.execute(
            f'''DELETE FROM {self.TABLE_CHAINS}'''
        )
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

    def __init__(self, filename: Optional[Path] = 'default.sqlite'):
        self.filename = filename
        self._build_tables()

    def _connect(self):
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        return conn, c

    def _close(self, conn):
        conn.commit()
        conn.close()

    # TruDB requirement
    def insert_record(
        self, chain_id: str, input: str, output: str, record_json: dict,
        ts: int, tags: str, total_tokens: int, total_cost: float
    ) -> int:
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

        print(f"{UNICODE_CHECK} record {record_id} from {chain_id} -> {self.filename}")

        return record_id

    # TruDB requirement
    def insert_chain(
        self, chain_json: dict, chain_id: Optional[str] = None
    ) -> str:
        chain_id = chain_id or chain_json['chain_id'] or obj_id_of_obj(
            obj=chain_json, prefix="chain"
        )
        chain_str = json_str_of_obj(chain_json)

        conn, c = self._connect()
        c.execute(
            f"INSERT OR IGNORE INTO {self.TABLE_CHAINS} VALUES (?, ?)",
            (chain_id, chain_str)
        )
        self._close(conn)

        print(f"{UNICODE_CHECK} chain {chain_id} -> {self.filename}")

        return chain_id

    def insert_feedback_def(self, feedback_json: dict):
        """
        Insert a feedback definition into the database.
        """

        feedback_id = feedback_json['feedback_id']
        feedback_str = json_str_of_obj(feedback_json)

        conn, c = self._connect()
        c.execute(
            f"INSERT OR REPLACE INTO {self.TABLE_FEEDBACK_DEFS} VALUES (?, ?)",
            (feedback_id, feedback_str)
        )
        self._close(conn)

        print(f"{UNICODE_CHECK} feedback def. {feedback_id} -> {self.filename}")

    def get_feedback_defs(self, feedback_id: Optional[str] = None):
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
        result_json: Optional[dict] = None,
        total_cost: Optional[float] = None,
        total_tokens: Optional[int] = None,
    ):
        """
        Insert a record-feedback link to db or update an existing one.
        """

        if record_id is None or feedback_id is None:
            assert result_json is not None, "`result_json` needs to be given if `record_id` or `feedback_id` are not provided."
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
            print(f"{UNICODE_CHECK} feedback {feedback_id} on {record_id} -> {self.filename}")
        else:
            print(f"{UNCIODE_YIELD} feedback {feedback_id} on {record_id} -> {self.filename}")

    def get_feedback(
        self,
        record_id: Optional[str] = None,
        feedback_id: Optional[str] = None,
        status: Optional[int] = None,
        last_ts_before: Optional[int] = None
    ):

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

    # TO REMOVE:
    # TruDB requirement
    def select(
        self,
        *query: Tuple[Query],
        where: Optional[Condition] = None
    ) -> pd.DataFrame:
        raise NotImplementedError
        """
        # get the record json dumps from sql
        record_strs = ...  # TODO(shayak)

        records: Sequence[Dict] = map(json.loads, record_strs)

        db = LocalTinyDB()  # in-memory db if filename not provided
        for record in records:
            db.insert_record(chain_id=record['chain_id'], record=record)

        return db.select(*query, where)
        """

    def get_chain(self, chain_id: str) -> JSON:
        conn, c = self._connect()
        c.execute(
            f"SELECT chain_json FROM {self.TABLE_CHAINS} WHERE chain_id=?",
            (chain_id,)
        )
        result = c.fetchone()[0]
        conn.close()

        return json.loads(result)

    def get_records_and_feedback(self, chain_ids: List[str]) -> Tuple[pd.DataFrame, Sequence[str]]:
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
        df_results['result_json'] = df_results['result_json'].apply(lambda d: {} if d is None else json.loads(d)) 

        if "record_id" not in df_results.columns:
            return df_results, []

        df_results = df_results.groupby("record_id").agg(
            lambda dicts: {key: val for d in dicts for key, val in d.items()}
        ).reset_index()
        
        df_results = df_results['result_json'].apply(pd.Series)

        result_cols = [col for col in df_results.columns if col not in ['feedback_id', 'record_id', '_success', "_error"]]
        
        if len(df_results) == 0 or len(result_cols) == 0:
            return df_records, []

        assert "record_id" in df_results.columns
        assert "record_id" in df_records.columns

        combined_df = df_records.merge(df_results, on=['record_id'])

        return combined_df, result_cols
