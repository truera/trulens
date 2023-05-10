import abc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from merkle_json import MerkleJson
import pandas as pd
from tinydb import Query as TinyQuery
from tinydb import TinyDB
from tinydb.queries import QueryInstance as TinyQueryInstance
from tinydb.storages import MemoryStorage
from tinydb.table import Document
from tinydb.table import Table

mj = MerkleJson()


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

    # Intentionally not including much in this indicator to make sure the model
    # hashing procedure does not get randomized due to something here.

    return f"NON-SERIALIZED OBJECT: type={type(obj)}"


Query = TinyQuery  # for typing
Record = Query()  # for constructing
Condition = TinyQueryInstance  # type of conditions, constructed from query/record like `Record.chain != None``


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
    def insert_record(self, model_name: str, record: dict) -> int:
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
        self.models.insert(Document(doc_id=model_name, value=model))
        return model_name

    # TruDB requirement
    def select(self, *query: Tuple[Query], where: Optional[Query] = None):
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


class TruSQL(TruDB):

    def __init__(self):

        # TODO(shayak)

        ...

    # TruDB requirement
    def insert_record(self, model_name: str, record: dict) -> int:
        record['model_name'] = model_name
        record_str = json.dumps(record, default=json_default)

        # TODO(shayak)

        return 42  # record_id

    # TruDB requirement
    def insert_model(
        self, model: dict, model_name: Optional[str] = None
    ) -> str:
        model_name = model_name or model_name_of_model(model)
        model_str = json.dumps(model, default=json_default)

        # TODO(shayak)

        return model_name

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
