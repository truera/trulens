"""Serializable event-related classes."""

from __future__ import annotations

import datetime
import logging
from typing import Any, Dict, Hashable, Optional

import pydantic
from trulens.core.schema import types as types_schema
from trulens.core.utils import serial as serial_utils

logger = logging.getLogger(__name__)


class Event(serial_utils.SerialModel, Hashable):
    """The class that represents a single event data entry."""

    event_id: types_schema.EventID  # str
    """The unique identifier for the event."""

    record: Dict[str, Any]
    record_attributes: Dict[str, Any]
    record_type: str
    resource_attributes: Dict[str, Any]
    start_timestamp: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now
    )

    timestamp: datetime.datetime = pydantic.Field(
        default_factory=datetime.datetime.now
    )

    trace: Dict[str, Any]

    def __init__(
        self,
        event_id: types_schema.EventID,
        event_type: str,
        event_time: datetime.datetime,
        event_data: Optional[Dict] = None,
        meta: Optional[types_schema.Metadata] = None,
        **kwargs,
    ):
        kwargs["event_id"] = event_id
        kwargs["event_type"] = event_type
        kwargs["event_time"] = event_time
        kwargs["event_data"] = event_data
        kwargs["meta"] = meta if meta is not None else {}

    def __hash__(self):
        return hash(self.event_id)
