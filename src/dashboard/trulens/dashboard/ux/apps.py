# Code in support of the Apps.py page.

from typing import ClassVar, Optional

import pydantic
from trulens.core import app as mod_app
from trulens.core.utils.serial import JSON


class ChatRecord(pydantic.BaseModel):
    model_config: ClassVar[dict] = dict(arbitrary_types_allowed=True)

    # Human input
    human: Optional[str] = None

    # Computer response
    computer: Optional[str] = None

    # Jsonified record. Available only after the app is run on human input and
    # produced a computer output.
    record_json: Optional[JSON] = None

    # The final app state for continuing the session.
    app: mod_app.App

    # The state of the app as was when this record was produced.
    app_json: JSON
