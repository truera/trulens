import abc
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

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
import sqlite3
import uuid
import pandas as pd

mj = MerkleJson()
NoneType = type(None)


def model_name_of_model(model: dict):
    """
    Create a model name from model definition. Should produce the same name if
    definition stays the same.
    """

    return f"model_hash_{mj.hash(model)}"


def json_default(obj: Any) -> str:
    """
    Produce a representation of an object which cannot be json-serialized.
    """

    if isinstance(obj, pydantic.BaseModel):  #langchain.schema.Document):
        return json.dumps(obj.dict())

    # Intentionally not including much in this indicator to make sure the model
    # hashing procedure does not get randomized due to something here.

    return f"NON-SERIALIZED OBJECT: type={type(obj)}"


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
    def insert_record(self, model_name: str, record: dict, *args, **kwargs) -> int:
        """
        Insert a new `record` into db, indicating its `model` as well. Return
        record id.
        """

        raise NotImplementedError()

    @abc.abstractmethod
    def insert_model(self, model_name: str, model: dict) -> str:
        """
        Insert a new `model` into db under the given `model_name`. If name not
        provided, generate a name from model definition. Return the name.
        """

        raise NotImplementedError()

    @staticmethod
    def dictify(obj: Any) -> Union[str, int, float, NoneType, List, Dict]:
        """
        Convert the given object into types that can be serialized in json.
        """

        if isinstance(obj, (str, int, float, NoneType)):
            return obj

        elif isinstance(obj, Dict):
            return {k: TruDB.dictify(v) for k, v in obj.items()}

        elif isinstance(obj, Sequence):
            return [TruDB.dictify(v) for v in obj]

        elif isinstance(obj, pydantic.BaseModel):
            return TruDB.dictify(obj.dict())

        else:
            raise RuntimeError(
                f"Don't know how to dictify an object of type {type(obj)}."
            )

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
    def leafs(obj: Any) -> Iterable[Tuple[str, Any]]:
        for q in TruDB.leaf_queries(obj):
            path_str = TruDB._query_str(q)
            val = TruDB.project(q, obj)
            yield (path_str, val)

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
            if not isinstance(obj, Dict) or first not in obj:
                return None

            return TruDB._project(path=rest, obj=obj[first])

        elif isinstance(first, int):
            if not isinstance(obj, Sequence) or first >= len(obj):
                return None

            return TruDB._project(path=rest, obj=obj[first])
        else:
            raise RuntimeError(
                f"Don't know how to locate element with key of type {first}"
            )


class TruTinyDB(TruDB):

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

        self.models: Table = self.db.table("models")
        self.models.document_id_class = str

    # TruDB requirement
    def insert_record(self, model_name: str, record: dict) -> int:
        record['model_name'] = model_name
        return self.records.insert(record)

    # TruDB requirement
    def insert_model(
        self, model: dict, model_name: Optional[str] = None
    ) -> str:
        model_name = model_name or model_name_of_model(model)

        if self.models.contains(doc_id=model_name):
            print(
                f"WARNING: model {model_name} already exists in {self.models}."
            )
            self.models.update(Document(doc_id=model_name, value=model))
        else:
            self.models.insert(Document(doc_id=model_name, value=model))

        return model_name

    # TruDB requirement
    def select(self, *query: Tuple[Query], where: Optional[Condition] = None):
        if isinstance(query, Query):
            queries = [query]
        else:
            queries = query

        return self._select(table=self.records, queries=queries, where=where)

    @staticmethod
    def _select(
        table: Table,
        queries: List[Query],
        where: Optional[Condition] = None
    ) -> pd.DataFrame:
        rows = []

        if where is not None:
            table_rows = table.search(where)
        else:
            table_rows = table.all()

        for row in table_rows:
            vals = [TruDB.project(query=q, obj=row) for q in queries]
            rows.append(vals)

        return pd.DataFrame(rows, columns=map(TruDB._query_str, queries))


class LocalModelStore(TruDB):
   
    def __init__(self, db_name: Optional[Path] = 'llm_quality.db'):
        self.db_name = db_name
        conn, c = self._connect()

        # Create table if it does not exist
        c.execute(
            '''CREATE TABLE IF NOT EXISTS records
                        (record_id TEXT PRIMARY KEY, model_id TEXT, input TEXT, output TEXT, details TEXT, tags TEXT, ts INTEGER)'''
        )
        #TODO(Shayak): Make this not be a primary key and allow multiple rows on the same text
        c.execute(
        '''CREATE TABLE IF NOT EXISTS feedback
                    (record_id TEXT PRIMARY KEY, feedback TEXT)''' 
        )
        c.execute(
        '''CREATE TABLE IF NOT EXISTS models
                    (model_id TEXT PRIMARY KEY, model TEXT)'''
        )
        self._close(conn)

    def _connect(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()
        return conn, c
    
    def _close(self, conn):
        conn.commit()
        conn.close()
        

    # TruDB requirement
    def insert_record(self, model_name: str, input: str, output: str, record: dict, ts: int, tags:str) -> int:
        conn, c = self._connect()
        record_str = json.dumps(record, default=json_default)
        record_id = str(uuid.uuid4())


        # Main chain input and output are these but these may be dicts or
        # otherwise, depending on the wrapped chain.
        # overall_inputs: Union[
        #     Dict[str, Any],
        #     Any] = self.project(Record.chain._call.args.inputs, record)
        # overall_outputs: Dict[
        #     str, Any] = self.project(Record.chain.call.rets, record)
        
        c.execute(
            "INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?)", (
                record_id, model_name, input, output, record_str, tags, ts
            )
        )
        self._close(conn)
        return record_id  # record_id

    # TruDB requirement
    def insert_model(
        self, model: dict, model_name: Optional[str] = None
    ) -> str:
        model_name = model_name or model_name_of_model(model)
        model_str = json.dumps(model, default=json_default)
        conn, c = self._connect()
        c.execute(
            "INSERT INTO models VALUES (?, ?)", (
                model_name, model_str
            )
        )
        self._close(conn)
        return model_name
    
    def insert_feedback(self, record_id: str, feedback: dict):
        feedback_str = str(feedback)
        conn, c = self._connect()
        c.execute(
            "INSERT INTO feedback VALUES (?, ?)", (
                record_id, feedback_str
            )
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

        db = TruTinyDB()  # in-memory db if filename not provided
        for record in records:
            db.insert_record(model_name=record['model_name'], record=record)

        return db.select(*query, where)
    

    def get_model(self, model_id: str):
        conn, c = self._connect()
        c.execute("SELECT model FROM records WHERE model_id=?", (model_id,))
        result = c.fetchone()
        conn.close()


    def get_records_and_feedback(self, model_ids: List[str]):
        conn, c = self._connect()
        query = "SELECT l.*, f.feedback FROM records l LEFT JOIN feedback f on l.record_id = f.record_id"
        if len(model_ids) > 0:
            model_id_list = ', '.join('?' * len(model_ids))
            query = query + f" WHERE model_id IN ({model_id_list})"
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
