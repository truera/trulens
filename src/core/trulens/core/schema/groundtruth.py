"""Serializable groundtruth-related classes."""

from __future__ import annotations

import logging
from typing import Dict, Hashable, Optional, Sequence

from trulens.core.schema import types as mod_types_schema
from trulens.core.utils import serial
from trulens.core.utils.json import jsonify
from trulens.core.utils.json import obj_id_of_obj

logger = logging.getLogger(__name__)


class GroundTruth(serial.SerialModel, Hashable):
    """The class that represents a single ground truth data entry."""

    ground_truth_id: mod_types_schema.GroundTruthID  # str
    """The unique identifier for the ground truth."""

    query: str
    """The query for which the ground truth is provided."""

    query_id: Optional[str] = None
    """Unique identifier for the query."""

    expected_response: Optional[str] = (
        None  # expected response can be empty in GT datasets
    )
    """The expected response for the query."""

    expected_chunks: Optional[Sequence[Dict]] = None
    """Expected chunks for the ground truth."""

    meta: Optional[mod_types_schema.Metadata] = (
        None  # TODO: which naming are we exactly going with - meta vs metadata?
    )
    """Metadata for the ground truth."""

    dataset_id: mod_types_schema.DatasetID  # str
    """The dataset ID to which this ground truth belongs.
    See [Dataset.dataset_id][trulens.core.schema.dataset.Dataset.dataset_id]."""

    def __init__(
        self,
        dataset_id: mod_types_schema.DatasetID,
        query: str,
        query_id: Optional[str] = None,
        expected_response: Optional[str] = None,
        expected_chunks: Optional[Sequence[Dict]] = None,
        meta: Optional[mod_types_schema.Metadata] = None,
        ground_truth_id: Optional[mod_types_schema.GroundTruthID] = None,
        **kwargs,
    ):
        kwargs["query"] = query
        kwargs["query_id"] = query_id
        kwargs["dataset_id"] = dataset_id
        kwargs["expected_response"] = expected_response
        kwargs["expected_chunks"] = expected_chunks
        kwargs["meta"] = meta if meta is not None else {}

        super().__init__(
            ground_truth_id="temporary", **kwargs
        )  # will be updated below

        if ground_truth_id is None:
            ground_truth_id = obj_id_of_obj(
                jsonify(self), prefix="ground_truth"
            )
        self.ground_truth_id = ground_truth_id

    def __hash__(self):
        return hash(self.ground_truth_id)


# HACK013: Need these if using __future__.annotations .
GroundTruth.model_rebuild()
