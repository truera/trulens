"""Serializable dataset-related classes."""

from __future__ import annotations

import datetime
import logging
from typing import Hashable, Optional

import pydantic
from trulens.core.schema import types as mod_types_schema
from trulens.core.utils import pyschema
from trulens.core.utils import serial
from trulens.core.utils.json import jsonify
from trulens.core.utils.json import obj_id_of_obj

logger = logging.getLogger(__name__)


class Dataset(pyschema.WithClassInfo, serial.SerialModel, Hashable):
    """Serialized fields of a dataset."""

    dataset_id: mod_types_schema.DatasetID  # str

    name: str

    meta: mod_types_schema.Metadata  # dict

    ts: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now
    )

    def __init__(
        self,
        name: str,
        dataset_id: Optional[mod_types_schema.DatasetID] = None,
        meta: Optional[mod_types_schema.Metadata] = None,
        **kwargs,
    ):
        kwargs["dataset_id"] = "temporary"  # will be updated below
        kwargs["name"] = name
        kwargs["meta"] = meta if meta is not None else {}

        if dataset_id is None:
            dataset_id = obj_id_of_obj(jsonify(self), prefix="dataset")

        self.dataset_id = dataset_id

        super().__init__(**kwargs)

    def __hash__(self):
        return hash(self.dataset_id)


# HACK013: Need these if using __future__.annotations .
Dataset.model_rebuild()
