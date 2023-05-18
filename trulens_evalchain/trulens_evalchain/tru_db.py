import abc
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
import uuid

import langchain
from merkle_json import MerkleJson
import pandas as pd
import pydantic
from tinydb import Query as TinyQuery
from tinydb import TinyDB
from tinydb.queries import QueryInstance as TinyQueryInstance
from tinydb.storages import MemoryStorage
from tinydb.table import Document
from tinydb.table import Table

mj = MerkleJson()
NoneType = type(None)


def is_empty(obj):
    try:
        return len(obj) == 0
    except Exception:
        return False

def is_noserio(obj):
    return isinstance(obj, dict) and "_NON_SERIALIZED_OBJECT" in obj

def noserio(obj, **extra: Dict) -> dict:
    inner = {
                "class": obj.__class__.__name__,
                "module": obj.__class__.__module__,
                "bases": list(map(lambda b: b.__name__, obj.__class__.__bases__))
            }
    inner.update(extra)

    return {
        '_NON_SERIALIZED_OBJECT': inner
    }


def obj_id_of_obj(obj: dict, prefix="obj"):
    """
    Create an id from a json-able structure/definition. Should produce the same
    name if definition stays the same.
    """

    return f"{prefix}_hash_{mj.hash(obj)}"


def json_str_of_obj(obj: Any) -> str:
    return json.dumps(obj, default=json_default)


def json_default(obj: Any) -> str:
    """
    Produce a representation of an object which cannot be json-serialized.
    """

    if isinstance(obj, pydantic.BaseModel):  #langchain.schema.Document):
        return json.dumps(obj.dict())

    # Intentionally not including much in this indicator to make sure the model
    # hashing procedure does not get randomized due to something here.

    return noserio(obj)


# Typing for type hints.
Query = TinyQuery

# Instance for constructing queries like `Record.chain.llm`.
Record = Query()

# Type of conditions, constructed from query/record like `Record.chain != None`.
Condition = TinyQueryInstance


class TruDB(abc.ABC):

    # Use TinyDB queries for looking up parts of records/models and/or filtering
    # on those parts.

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
        self, chain_id: str, record: dict, *args, **kwargs
    ) -> str:
        """
        Insert a new `record` into db, indicating its `model` as well. Return
        record id.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_chain(self, chain_id: str, chain: dict) -> str:
        """
        Insert a new `chain` into db under the given `chain_id`. If name not
        provided, generate a name from chain definition. Return the name.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_feedback(
        self, chain_id: str, record_id: int, feedback: dict
    ) -> str:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_records_and_feedback(
        self, chain_ids: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError()

    @staticmethod
    def dictify(obj: Any,
                dicted=None) -> Union[str, int, float, NoneType, List, Dict]:
        """
        Convert the given object into types that can be serialized in json.
        """

        dicted = dicted or dict()

        if isinstance(obj, (str, int, float, NoneType)):
            # dicted[id(obj)] = obj
            return obj

        if id(obj) in dicted:
            return {'_CIRCULAR_REFERENCE': id(obj)}

            # return dicted[id(obj)]

        new_dicted = {k: v for k, v in dicted.items()}

        if isinstance(obj, Dict):
            temp = {}
            new_dicted[id(obj)] = temp    
            temp.update(
                {k: TruDB.dictify(v, dicted=new_dicted) for k, v in obj.items()}
            )
            return temp

        elif isinstance(obj, Sequence):
            temp = []
            new_dicted[id(obj)] = temp
            for x in (TruDB.dictify(v, dicted=new_dicted) for v in obj):
                temp.append(x)
            return temp

        elif isinstance(obj, Set):
            temp = []
            new_dicted[id(obj)] = temp
            for x in (TruDB.dictify(v, dicted=new_dicted) for v in obj):
                temp.append(x)
            return temp

        elif isinstance(obj, pydantic.BaseModel):
            temp = {}
            new_dicted[id(obj)] = temp
            temp.update(
                {
                    k: TruDB.dictify(getattr(obj, k), dicted=new_dicted)
                    for k in obj.__fields__
                }
            )
            return temp

        else:
            print(
                f"WARNING: Don't know how to dictify an object '{str(obj)[0:32]}' of type '{type(obj)}'."
            )
            return noserio(obj)
            #raise RuntimeError(
            #    f"Don't know how to dictify an object '{str(obj)[0:32]}' of type '{type(obj)}'."
            #)

    @staticmethod
    def leaf_queries(obj: Any, query: Query = None) -> Iterable[Query]:
        """
        Get all queries for the given object that select all of its leaf values.
        """

        query = query or Record

        if isinstance(obj, (str, int, float, NoneType)):
            yield query

        elif isinstance(obj, Dict):
            for k, v in obj.items():
                sub_query = query[k]
                for res in TruDB.leaf_queries(obj[k], sub_query):
                    yield res

        elif isinstance(obj, Sequence):
            for i, v in enumerate(obj):
                sub_query = query[i]
                for res in TruDB.leaf_queries(obj[i], sub_query):
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
            val = TruDB.project(q, obj)
            yield (path_str, val)

    @staticmethod
    def matching_queries(obj: Any, match: Callable) -> Iterable[Query]:
        for q in TruDB.all_queries(obj):
            val = TruDB.project(q, obj)
            if match(q, val):
                yield q

    @staticmethod
    def matching_objects(obj: Any,
                         match: Callable) -> Iterable[Tuple[Query, Any]]:
        for q, val in TruDB.all_objects(obj):
            if match(q, val):
                yield (q, val)

    @staticmethod
    def _query_str(query: Query):

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
    def project(query: Query, obj: Any):
        return TruDB._project(query._path, obj)

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
                    print(
                        f"WARNING: Cannot project {str(obj)[0:32]} with path {path} because {first} is not an attribute here."
                    )
                    return None
                return TruDB._project(path=rest, obj=getattr(obj, first))

            elif isinstance(obj, Dict):
                if first not in obj:
                    print(
                        f"WARNING: Cannot project {str(obj)[0:32]} with path {path} because {first} is not a key here."
                    )
                    return None
                return TruDB._project(path=rest, obj=obj[first])

            else:
                print(
                    f"WARNING: Cannot project {str(obj)[0:32]} with path {path} because object is not a dict or model."
                )
                return None

        elif isinstance(first, int):
            if not isinstance(obj, Sequence) or first >= len(obj):
                print(
                    f"WARNING: Cannot project {str(obj)[0:32]} with path {path}."
                )
                return None

            return TruDB._project(path=rest, obj=obj[first])
        else:
            raise RuntimeError(
                f"Don't know how to locate element with key of type {first}"
            )


class LocalTinyDB(TruDB):

    def __init__(self, filename: Optional[Path] = None):

        if filename is not None:
            self.filename = Path(filename)
            self.db: TinyDB = TinyDB(filename, indent=4, default=json_default)
        else:
            self.filename = None
            print("WARNING: db is memory-only. It will not persist.")
            self.db: TinyDB = TinyDB(storage=MemoryStorage)

        self.records: Table = self.db.table("records")
        self.records.document_id_class = int

        self.chains: Table = self.db.table("chains")
        self.chains.document_id_class = str

        self.feedbacks: Table = self.db.table("feedbacks")
        self.feedbacks.document_id_class = int

    # TruDB requirement
    def insert_record(self, chain_id: str, record: dict) -> int:
        record['chain_id'] = chain_id
        record_id = self.records._get_next_id()
        record['record_id'] = record_id

        return self.records.insert(Document(doc_id=record_id, value=record))

    # TruDB requirement
    def insert_chain(self, chain: dict, chain_id: Optional[str] = None) -> str:
        chain_id = chain_id or obj_id_of_obj(obj=chain, prefix="chain")

        if self.chains.contains(doc_id=chain_id):
            print(f"WARNING: chain {chain_id} already exists in {self.chains}.")
            self.chains.update(Document(doc_id=chain_id, value=chain))
        else:
            self.chains.insert(Document(doc_id=chain_id, value=chain))

        return chain_id

    # TruDB requirement
    def insert_feedback(
        self, chain_id: str, record_id: int, feedback: dict
    ) -> str:
        feedback['record_id'] = record_id
        feedback['chain_id'] = chain_id
        feedback_id = self.feedbacks._get_next_id()
        feedback['feedback_id'] = feedback_id

        self.feedbacks.insert(Document(doc_id=feedback_id, value=feedback))

    # TruDB requirement
    def get_records_and_feedback(
        self, chain_ids: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # This returns all models if the list of chain_ids is empty
        # conn, c = self._connect()
        # query = "SELECT l.*, f.feedback FROM records l LEFT JOIN feedback f on l.record_id = f.record_id"

        queries = [Record.chain_id, Record.record_id, Record]
        queries_records = queries + [
            Record.chain._call.args.inputs, Record.chain._call.rets
        ]
        queries_feedbacks = queries + [Record.feedback_id]

        where = None
        if len(chain_ids) > 0:
            where = Record.chain_id in chain_ids

        df = self.select(*queries_records, where=where, table=self.records)
        df = df.rename(columns={'Record': 'record'})

        df_feedback = self.select(
            *queries_feedbacks, where=where, table=self.feedbacks
        )
        df_feedback = df_feedback.rename(columns={'Record': 'feedback'})

        #df = pd.DataFrame(
        #    rows, columns=[description[0] for description in c.description]
        #)

        # Apply the function to the 'data' column to convert it into separate columns
        """
        for col in ['record',
                    'feedback',
                    'Record.chain._call.args.inputs', 
                    'Record.chain._call.rets'
                    ]:
            if col in df_feedback.columns:
                df_feedback[col] = df_feedback[col].apply(str_dict_to_series)
            if col in df.columns:
                df[col] = df[col].apply(str_dict_to_series)
        """

        return df, df_feedback

    # TruDB requirement
    def select(
        self,
        *query: Tuple[Query],
        where: Optional[Condition] = None,
        table: Table = None
    ):
        if isinstance(query, Query):
            queries = [query]
        else:
            queries = query

        table = table if table is not None else self.records

        return self._select(table=table, queries=queries, where=where)

    @staticmethod
    def _select(
        table: Table,
        queries: List[Query],
        where: Optional[Condition] = None,
    ) -> pd.DataFrame:
        rows = []

        if where is not None:
            table_rows = table.search(where)
        else:
            table_rows = table.all()

        for row in table_rows:
            vals = [TruDB.project(query=q, obj=row) for q in queries]
            rows.append(vals)

        rename_map = {
            "Record.chain_id": "chain_id",
            "Record.record_id": "record_id",
            "Record.feedback_id": "feedback_id",
        }

        cols = list(map(TruDB._query_str, queries))

        if len(rows) == 0:
            df = pd.DataFrame([], columns=cols)
            df = df.rename(columns=rename_map)
            print(df)
            return df

        df = pd.DataFrame(rows, columns=cols)
        df = df.rename(columns=rename_map)

        return df


class LocalSQLite(TruDB):

    def __init__(self, filename: Optional[Path] = 'llm_quality.db'):
        self.filename = filename
        conn, c = self._connect()

        # Create table if it does not exist
        c.execute(
            '''CREATE TABLE IF NOT EXISTS records
                (record_id TEXT PRIMARY KEY, chain_id TEXT, input TEXT, output TEXT, details TEXT, tags TEXT, ts INTEGER, total_tokens INTEGER, total_cost REAL)'''
        )
        #TODO(Shayak): Make this not be a primary key and allow multiple rows on the same text
        c.execute(
            '''CREATE TABLE IF NOT EXISTS feedback
                    (record_id TEXT PRIMARY KEY, feedback TEXT)'''
        )
        c.execute(
            '''CREATE TABLE IF NOT EXISTS chains
                    (chain_id TEXT PRIMARY KEY, chain TEXT)'''
        )
        self._close(conn)

    def _connect(self):
        conn = sqlite3.connect(self.filename)
        c = conn.cursor()
        return conn, c

    def _close(self, conn):
        conn.commit()
        conn.close()

    # TruDB requirement
    def insert_record(
        self, chain_id: str, input: str, output: str, record: dict, ts: int,
        tags: str, total_tokens: int, total_cost: float
    ) -> int:
        assert isinstance(record, Dict), f"Attempting to add a record that is not a dict, is {type(record)} instead."

        conn, c = self._connect()
        
        record_str = json_str_of_obj(record)
        record_id = str(uuid.uuid4())

        # Main chain input and output are these but these may be dicts or
        # otherwise, depending on the wrapped chain.
        # overall_inputs: Union[
        #     Dict[str, Any],
        #     Any] = self.project(Record.chain._call.args.inputs, record)
        # overall_outputs: Dict[
        #     str, Any] = self.project(Record.chain.call.rets, record)

        c.execute(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                record_id, chain_id, input, output, record_str, tags, ts,
                total_tokens, total_cost
            )
        )
        self._close(conn)
        return record_id  # record_id

    # TruDB requirement
    def insert_chain(self, chain: dict, chain_id: Optional[str] = None) -> str:
        chain_id = chain_id or obj_id_of_obj(obj=chain, prefix="chain")
        chain_str = json_str_of_obj(chain)

        conn, c = self._connect()
        c.execute("INSERT INTO chains VALUES (?, ?)", (chain_id, chain_str))
        self._close(conn)

        return chain_id

    def insert_feedback(self, record_id: str, feedback: dict):
        feedback_str = json_str_of_obj(feedback)

        conn, c = self._connect()
        c.execute(
            "INSERT INTO feedback VALUES (?, ?)", (record_id, feedback_str)
        )
        self._close(conn)

    # TruDB requirement
    def select(
        self,
        *query: Tuple[Query],
        where: Optional[Condition] = None
    ) -> pd.DataFrame:

        # get the record json dumps from sql
        record_strs = ...  # TODO(shayak)

        records: Sequence[Dict] = map(json.loads, record_strs)

        db = LocalTinyDB()  # in-memory db if filename not provided
        for record in records:
            db.insert_record(chain_id=record['chain_id'], record=record)

        return db.select(*query, where)

    def get_chain(self, chain_id: str):
        conn, c = self._connect()
        c.execute("SELECT model FROM records WHERE chain_id=?", (chain_id,))
        result = c.fetchone()
        conn.close()

        return result

    def get_records_and_feedback(self, chain_ids: List[str]):
        # This returns all models if the list of chain_ids is empty
        conn, c = self._connect()
        query = "SELECT l.*, f.feedback FROM records l LEFT JOIN feedback f on l.record_id = f.record_id"

        if len(chain_ids) > 0:
            chain_id_list = ', '.join('?' * len(chain_ids))
            query = query + f" WHERE chain_id IN ({chain_id_list})"

        c.execute(query)
        rows = c.fetchall()
        conn.close()

        df = pd.DataFrame(
            rows, columns=[description[0] for description in c.description]
        )

        def str_dict_to_series(str_dict):
            if not str_dict:
                str_dict = "{}"
            dict_obj = eval(str_dict)
            return pd.Series(dict_obj)

        # Apply the function to the 'data' column to convert it into separate columns
        df_feedback = df['feedback'].apply(str_dict_to_series)

        return df, df_feedback
