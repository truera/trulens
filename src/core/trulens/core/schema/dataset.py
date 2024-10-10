"""Serializable dataset-related classes."""

from __future__ import annotations

import logging
from typing import Hashable, Optional

from trulens.core.schema import types as types_schema
from trulens.core.utils import json as json_utils
from trulens.core.utils import serial as serial_utils

logger = logging.getLogger(__name__)


class Dataset(serial_utils.SerialModel, Hashable):
    """The class that holds the metadata of a dataset stored in the DB."""

    dataset_id: types_schema.DatasetID  # str
    """The unique identifier for the dataset."""

    name: str
    """The name of the dataset."""

    meta: types_schema.Metadata  # dict
    """Metadata associated with the dataset."""

    def __init__(
        self,
        name: str,
        dataset_id: Optional[types_schema.DatasetID] = None,
        meta: Optional[types_schema.Metadata] = None,
        **kwargs,
    ):
        kwargs["name"] = name
        kwargs["meta"] = meta if meta is not None else {}
        super().__init__(
            dataset_id="temporary", **kwargs
        )  # dataset_id will be updated below

        if dataset_id is None:
            dataset_id = json_utils.obj_id_of_obj(
                json_utils.jsonify(self), prefix="dataset"
            )

        self.dataset_id = dataset_id

    def __hash__(self):
        return hash(self.dataset_id)


# HACK013: Need these if using __future__.annotations .
Dataset.model_rebuild()
