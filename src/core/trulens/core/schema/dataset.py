"""Serializable dataset-related classes."""

from __future__ import annotations

import logging
from typing import Hashable, Optional

from trulens.core.schema import types as mod_types_schema
from trulens.core.utils import serial
from trulens.core.utils.json import jsonify
from trulens.core.utils.json import obj_id_of_obj

logger = logging.getLogger(__name__)


class Dataset(serial.SerialModel, Hashable):
    """The class that holds the metadata of a dataset stored in the DB."""

    dataset_id: mod_types_schema.DatasetID  # str

    name: str

    meta: mod_types_schema.Metadata  # dict

    def __init__(
        self,
        name: str,
        dataset_id: Optional[mod_types_schema.DatasetID] = None,
        meta: Optional[mod_types_schema.Metadata] = None,
        **kwargs,
    ):
        kwargs["name"] = name
        kwargs["meta"] = meta if meta is not None else {}
        super().__init__(
            dataset_id="temporary", **kwargs
        )  # dataset_id will be updated below

        if dataset_id is None:
            dataset_id = obj_id_of_obj(jsonify(self), prefix="dataset")

        self.dataset_id = dataset_id

    def __hash__(self):
        return hash(self.dataset_id)


# HACK013: Need these if using __future__.annotations .
Dataset.model_rebuild()
