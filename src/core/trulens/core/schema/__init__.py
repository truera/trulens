"""Serializable Classes

Note: Only put classes which can be serialized in this module.

## Classes with non-serializable variants

Many of the classes defined here extending serial.SerialModel are meant to be
serialized into json. Most are extended with non-serialized fields in other files.

| Serializable       | Non-serializable        |
| ------------------ | ----------------------- |
| [AppDefinition][trulens.core.schema.app.AppDefinition] | [App][trulens.core.app.App], Tru{Chain, Llama, ...} |
| [FeedbackDefinition][trulens.core.schema.feedback.FeedbackDefinition] | [Feedback][trulens.core.Feedback] |

`AppDefinition.app` is the JSON-ized version of a wrapped app while `App.app` is the
actual wrapped app. We can thus inspect the contents of a wrapped app without
having to construct it. Additionally, JSONized objects like `AppDefinition.app`
feature information about the encoded object types in the dictionary under the
`util.py:CLASS_INFO` key.

"""

from trulens.core.schema import app as app_schema
from trulens.core.schema import base as base_schema
from trulens.core.schema import feedback as feedback_schema
from trulens.core.schema import record as record_schema
from trulens.core.schema import (
    select as select_schema,  # noqa: F401 needed for model_update below
)
from trulens.core.schema import types as types_schema  # noqa: F401 same

"""
from trulens.core.schema.app import AppDefinition
from trulens.core.schema.feedback import FeedbackDefinition
from trulens.core.schema.feedback import FeedbackMode
from trulens.core.schema.feedback import FeedbackResult, FeedbackCall
from trulens.core.schema.record import Record, RecordAppCall, RecordAppCallMethod
from trulens.core.schema.select import Select
from trulens.core.schema.base import Cost, Perf
"""

"""
__all__ = [
    "AppDefinition",
    "Select",
    "FeedbackDefinition",
    "FeedbackResult",
    "FeedbackMode",
    "Record",
]
"""

# HACK013: Need these if using __future__.annotations .

# Moved all the `model_rebuild` here to make sure they are executed after all of
# the types mentioned inside them are defined.

record_schema.RecordAppCallMethod.model_rebuild()
record_schema.Record.model_rebuild()
record_schema.RecordAppCall.model_rebuild()

feedback_schema.FeedbackResult.model_rebuild()
feedback_schema.FeedbackCall.model_rebuild()
feedback_schema.FeedbackDefinition.model_rebuild()

app_schema.AppDefinition.model_rebuild()

base_schema.Cost.model_rebuild()
base_schema.Perf.model_rebuild()
