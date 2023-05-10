import abc
import json
from pathlib import Path
from typing import Any, Optional

from merkle_json import MerkleJson
from tinydb import TinyDB
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


class TruDB(abc.ABC):

    @abc.abstractmethod
    def insert_record(self, model_name: str, record: dict) -> int:
        """
        Insert a new `record` into db, indicating its `model` as well. Return
        record id.
        """

        raise NotImplementedError()

    def insert_model(self, model_name: str, model: dict) -> str:
        """
        Insert a new `model` into db under the given `model_name`. If name not
        provided, generate a name from model definition. Return the name.
        """

        raise NotImplementedError()


class TruTinyDB(TruDB):

    def __init__(self, filename: Path):
        self.filename = Path(filename)

        self.db: TinyDB = TinyDB(filename, indent=4, default=json_default)

        self.records: Table = self.db.table("records")
        self.records.document_id_class = int

        self.models: Table = self.db.table("models")
        self.models.document_id_class = str

    def insert_record(self, model_name: str, record: dict) -> int:
        record['model_name'] = model_name
        return self.records.insert(record)

    def insert_model(
        self, model: dict, model_name: Optional[str] = None
    ) -> str:
        model_name = model_name or model_name_of_model(model)
        self.models.insert(Document(doc_id=model_name, value=model))
        return model_name


class TruSQL(TruDB):

    def __init__(self):
        pass

    def insert_record(self, model_name: str, record: dict) -> int:
        record['model_name'] = model_name
        record_str = json.dumps(record, default=json_default)

        # TODO(shayak)

        return 42  # record_id

    def insert_model(
        self, model: dict, model_name: Optional[str] = None
    ) -> str:
        model_name = model_name or model_name_of_model(model)
        model_str = json.dumps(model, default=json_default)

        # TODO(shayak)

        return model_name
