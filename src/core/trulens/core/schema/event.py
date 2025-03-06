"""Serializable event-related classes."""

from __future__ import annotations

from datetime import datetime
import enum
import logging
from typing import Any, Dict, Hashable

from trulens.core.utils import serial as serial_utils
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class EventRecordType(enum.Enum):
    """The enumeration of the possible record types for an event."""

    SPAN = "SPAN"


class Trace(TypedDict):
    """The type hint for a trace dictionary."""

    trace_id: str
    parent_id: str
    span_id: str


class Event(serial_utils.SerialModel, Hashable):
    """The class that represents a single event data entry."""

    event_id: str
    """
    The unique identifier for the event.
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

    record_type: EventRecordType
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

    trace: Trace
    """
    The trace context information for the span.
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def __hash__(self):
        return self.trace["span_id"]
