"""Serializable event-related classes."""

from __future__ import annotations

from datetime import datetime
import logging
from typing import Any, Dict, Hashable

from trulens.core.schema import types as types_schema
from trulens.core.utils import serial as serial_utils

logger = logging.getLogger(__name__)


class Event(serial_utils.SerialModel, Hashable):
    """The class that represents a single event data entry."""

    event_id: types_schema.EventID  # str
    """
    The unique identifier for the event. This is just the span_id.
    """

    record: Dict[str, Any]
    """
    For a span, this is an object that includes:
    - name: the function/procedure that emitted the data
    - kind: SPAN_KIND_TRULENS
    - parent_span_id: the unique identifier for the parent span
    - status: STATUS_CODE_ERROR when the span corresponds to an unhandled exception. Otherwise, STATUS_CODE_UNSET.
    """

    record_attributes: Dict[str, Any]
    """
    Attributes of the record that can either come from the user, or based on the TruLens semantic conventions.
    """

    record_type: str
    """
    Specifies the kind of record specified by this row. This will always be "SPAN" for TruLens.
    """

    resource_attributes: Dict[str, Any]
    """
    Reserved.
    """

    start_timestamp: datetime
    """
    The timestamp when the span started. This is a UNIX timestamp in milliseconds.
    Note: The Snowflake event table uses the TIMESTAMP_NTZ data type for this column.
    """

    timestamp: datetime
    """
    The timestamp when the span concluded. This is a UNIX timestamp in milliseconds.
    Note: The Snowflake event table uses the TIMESTAMP_NTZ data type for this column.
    """

    trace: Dict[str, Any]

    def __init__(
        self,
        event_id: types_schema.EventID,
        **kwargs,
    ):
        super().__init__(event_id=event_id, **kwargs)
        self.event_id = event_id

    def __hash__(self):
        return hash(self.event_id)
