"""Conversation schema definitions."""

import datetime
from typing import List, Optional

import pydantic
from trulens.core.schema import types as types_schema
from trulens.core.utils import serial as serial_utils


class Conversation(serial_utils.SerialModel):
    """An ordered group of records belonging to one conversation."""

    conversation_id: types_schema.ConversationID
    app_id: types_schema.AppID
    record_ids: List[types_schema.RecordID]
    ts: datetime.datetime
    meta: Optional[serial_utils.JSON] = None
    num_turns: int = 0
    """Number of records in the conversation."""

    duration: float = 0.0
    """Elapsed seconds from the first record's start to the last record's end."""

    llm_calls: int = 0
    """Number of generation spans across the conversation."""

    models_used: List[str] = pydantic.Field(default_factory=list)
    """Distinct model names used by generation spans."""

    total_tokens: int = 0
    """Total tokens reported across the conversation's records."""

    total_cost: float = 0.0
    """Total application cost reported across the conversation's records."""

    tool_calls: int = 0
    """Number of TOOL and MCP spans across the conversation."""

    tool_errors: int = 0
    """Number of tool spans carrying an ``error.type`` attribute."""
