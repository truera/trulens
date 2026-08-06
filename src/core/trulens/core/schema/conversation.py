"""Conversation schema definitions."""

import datetime
from typing import List, Optional

from trulens.core.schema import types as types_schema
from trulens.core.utils import serial as serial_utils


class Conversation(serial_utils.SerialModel):
    """An ordered group of records belonging to one conversation."""

    conversation_id: types_schema.ConversationID
    app_id: types_schema.AppID
    record_ids: List[types_schema.RecordID]
    ts: datetime.datetime
    meta: Optional[serial_utils.JSON] = None
