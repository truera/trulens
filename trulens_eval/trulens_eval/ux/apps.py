# Code in support of the Apps.py page.

from typing import Optional
import pydantic

from trulens_eval.utils.serial import JSON


class ChatRecord(pydantic.BaseModel):
    human: Optional[str] = None
    computer: Optional[str] = None
    record_json: Optional[JSON] = None
    app_json: JSON